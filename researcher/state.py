"""
SQLite-backed state store (WAL mode).
Read methods are synchronous — call from the asyncio event loop directly
(reads are fast; no network I/O).  All writes go through WriterActor.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    topic       TEXT NOT NULL,
    start_time  TEXT NOT NULL,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS urls_seen (
    url           TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    fetch_timestamp TEXT NOT NULL,
    status        TEXT NOT NULL,
    PRIMARY KEY (url, session_id)
);

CREATE TABLE IF NOT EXISTS findings (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    cycle_num       INTEGER NOT NULL,
    finding_text    TEXT NOT NULL,
    source_url      TEXT NOT NULL,
    fetch_timestamp TEXT NOT NULL,
    source_hash     TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    relevance_score INTEGER,
    quality_type    TEXT,
    embedding_id    TEXT,
    section         TEXT
);

CREATE TABLE IF NOT EXISTS sources (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    url                 TEXT NOT NULL,
    title               TEXT,
    source_type         TEXT,
    fetch_date          TEXT,
    novel_findings_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS queries_used (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    cycle_num   INTEGER NOT NULL,
    query_text  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS leads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    lead_text   TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    cycle_num   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS event_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_urls_session     ON urls_seen(session_id);
CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id);
CREATE INDEX IF NOT EXISTS idx_sources_session  ON sources(session_id);
CREATE INDEX IF NOT EXISTS idx_queries_session  ON queries_used(session_id);
"""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_finding_id(finding_text: str, source_url: str) -> str:
    key = f"{finding_text.strip()}\x00{source_url}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


class StateReader:
    """Thread-safe read-only view of the state DB.  One connection per thread."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            c = sqlite3.connect(self.db_path, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
        return self._local.conn

    # ── reads ──────────────────────────────────────────────────────────────

    def url_seen(self, url: str, session_id: str) -> bool:
        row = self._conn().execute(
            "SELECT 1 FROM urls_seen WHERE url=? AND session_id=?",
            (url, session_id),
        ).fetchone()
        return row is not None

    def get_findings(self, session_id: str) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM findings WHERE session_id=? ORDER BY cycle_num, id",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_sources(self, session_id: str) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM sources WHERE session_id=? ORDER BY fetch_date DESC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_queries(self, session_id: str) -> list[str]:
        rows = self._conn().execute(
            "SELECT query_text FROM queries_used WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [r["query_text"] for r in rows]

    def get_leads(self, session_id: str) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM leads WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> Optional[dict]:
        row = self._conn().execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def count_findings(self, session_id: str) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) FROM findings WHERE session_id=?", (session_id,)
        ).fetchone()
        return row[0]

    def count_sources(self, session_id: str) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) FROM sources WHERE session_id=?", (session_id,)
        ).fetchone()
        return row[0]


class StateWriter:
    """Synchronous write operations — called exclusively from WriterActor."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=True)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ── write primitives ───────────────────────────────────────────────────

    def create_session(self, session_id: str, topic: str, config: dict) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions VALUES (?, ?, ?, ?)",
            (session_id, topic, _now_utc(), json.dumps(config)),
        )
        self.conn.commit()

    def mark_url_seen(self, url: str, session_id: str, status: str = "success") -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO urls_seen VALUES (?, ?, ?, ?)",
            (url, session_id, _now_utc(), status),
        )
        self.conn.commit()

    def insert_finding(self, finding: dict) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO findings
               (id, session_id, cycle_num, finding_text, source_url,
                fetch_timestamp, source_hash, source_type, relevance_score,
                quality_type, embedding_id, section)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                finding["id"],
                finding["session_id"],
                finding["cycle_num"],
                finding["finding_text"],
                finding["source_url"],
                finding.get("fetch_timestamp", _now_utc()),
                finding["source_hash"],
                finding["source_type"],
                finding.get("relevance_score"),
                finding.get("quality_type"),
                finding.get("embedding_id"),
                finding.get("section"),
            ),
        )
        self.conn.commit()

    def insert_source(self, source: dict) -> None:
        self.conn.execute(
            """INSERT INTO sources
               (session_id, url, title, source_type, fetch_date, novel_findings_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                source["session_id"],
                source["url"],
                source.get("title"),
                source.get("source_type"),
                source.get("fetch_date", _now_utc()),
                source.get("novel_findings_count", 0),
            ),
        )
        self.conn.commit()

    def insert_query(self, session_id: str, cycle_num: int, query_text: str) -> None:
        self.conn.execute(
            "INSERT INTO queries_used (session_id, cycle_num, query_text) VALUES (?, ?, ?)",
            (session_id, cycle_num, query_text),
        )
        self.conn.commit()

    def insert_lead(self, session_id: str, lead_text: str, source_url: str, cycle_num: int) -> None:
        self.conn.execute(
            "INSERT INTO leads (session_id, lead_text, source_url, cycle_num) VALUES (?, ?, ?, ?)",
            (session_id, lead_text, source_url, cycle_num),
        )
        self.conn.commit()

    def log_event(self, session_id: str, event_type: str, payload: object = None) -> None:
        self.conn.execute(
            "INSERT INTO event_log (session_id, timestamp, event_type, payload_json) VALUES (?, ?, ?, ?)",
            (session_id, _now_utc(), event_type, json.dumps(payload)),
        )
        self.conn.commit()

    def write_file_atomic(self, path: str, content: str) -> None:
        p = Path(path)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(p)

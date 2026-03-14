"""Tests for SQLite state layer (StateWriter + StateReader)."""
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

from researcher.state import StateReader, StateWriter, make_finding_id


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def writer(db_path):
    return StateWriter(db_path)


@pytest.fixture
def reader(db_path, writer):
    # writer creates the schema; reader opens the same file
    return StateReader(db_path)


# ── session ───────────────────────────────────────────────────────────────────

class TestSession:
    def test_create_and_get(self, writer, reader):
        writer.create_session("s1", "test topic", {"key": "val"})
        s = reader.get_session("s1")
        assert s is not None
        assert s["topic"] == "test topic"

    def test_create_idempotent(self, writer, reader):
        writer.create_session("s1", "topic A", {})
        writer.create_session("s1", "topic B", {})  # OR IGNORE
        assert reader.get_session("s1")["topic"] == "topic A"

    def test_missing_session_returns_none(self, reader):
        assert reader.get_session("nope") is None

    def test_list_sessions(self, writer, reader):
        writer.create_session("s1", "topic 1", {})
        writer.create_session("s2", "topic 2", {})
        sessions = reader.list_sessions()
        ids = {s["id"] for s in sessions}
        assert {"s1", "s2"} <= ids


# ── running_summary ───────────────────────────────────────────────────────────

class TestRunningSummary:
    def test_save_and_get(self, writer, reader):
        writer.create_session("s1", "topic", {})
        writer.save_running_summary("s1", "some summary text")
        assert reader.get_running_summary("s1") == "some summary text"

    def test_update_overwrite(self, writer, reader):
        writer.create_session("s1", "topic", {})
        writer.save_running_summary("s1", "v1")
        writer.save_running_summary("s1", "v2")
        assert reader.get_running_summary("s1") == "v2"

    def test_empty_before_save(self, writer, reader):
        writer.create_session("s1", "topic", {})
        assert reader.get_running_summary("s1") == ""


# ── url dedup ─────────────────────────────────────────────────────────────────

class TestURLDedup:
    def test_unseen_url_returns_false(self, reader):
        assert reader.url_seen("https://example.com", "s1") is False

    def test_seen_url_returns_true(self, writer, reader):
        writer.create_session("s1", "topic", {})
        writer.mark_url_seen("https://example.com", "s1")
        assert reader.url_seen("https://example.com", "s1") is True

    def test_url_scoped_to_session(self, writer, reader):
        writer.create_session("s1", "topic", {})
        writer.mark_url_seen("https://example.com", "s1")
        assert reader.url_seen("https://example.com", "s2") is False


# ── findings ─────────────────────────────────────────────────────────────────

class TestFindings:
    def _finding(self, fid="f1", session="s1", text="A finding"):
        return {
            "id": fid,
            "session_id": session,
            "cycle_num": 0,
            "finding_text": text,
            "source_url": "https://example.com",
            "fetch_timestamp": "2024-01-01T00:00:00",
            "source_hash": "abc123",
            "source_type": "web",
            "relevance_score": 3,
            "quality_type": "research",
            "embedding_id": fid,
            "section": None,
            "conflicting": 0,
        }

    def test_insert_and_get(self, writer, reader):
        writer.create_session("s1", "t", {})
        writer.insert_finding(self._finding())
        findings = reader.get_findings("s1")
        assert len(findings) == 1
        assert findings[0]["finding_text"] == "A finding"

    def test_insert_idempotent(self, writer, reader):
        writer.create_session("s1", "t", {})
        writer.insert_finding(self._finding())
        writer.insert_finding(self._finding())  # OR IGNORE
        assert len(reader.get_findings("s1")) == 1

    def test_update_finding_sections(self, writer, reader):
        writer.create_session("s1", "t", {})
        writer.insert_finding(self._finding("f1"))
        writer.insert_finding(self._finding("f2", text="Another"))
        writer.update_finding_sections({"f1": "Theme A", "f2": "Theme B"})
        findings = {f["id"]: f for f in reader.get_findings("s1")}
        assert findings["f1"]["section"] == "Theme A"
        assert findings["f2"]["section"] == "Theme B"

    def test_update_sections_empty_dict_is_safe(self, writer):
        writer.create_session("s1", "t", {})
        writer.update_finding_sections({})  # should not raise


# ── sources ───────────────────────────────────────────────────────────────────

class TestSources:
    def test_insert_and_get(self, writer, reader):
        writer.create_session("s1", "t", {})
        writer.insert_source({
            "session_id": "s1",
            "url": "https://github.com/foo/bar",
            "title": "foo/bar",
            "source_type": "github",
            "fetch_date": "2024-01-01",
            "novel_findings_count": 3,
            "stars": 1500,
            "description": "A cool project",
            "language": "Python",
        })
        sources = reader.get_sources("s1")
        assert len(sources) == 1
        s = sources[0]
        assert s["stars"] == 1500
        assert s["description"] == "A cool project"
        assert s["language"] == "Python"

    def test_stars_can_be_null(self, writer, reader):
        writer.create_session("s1", "t", {})
        writer.insert_source({
            "session_id": "s1",
            "url": "https://example.com",
            "source_type": "web",
        })
        sources = reader.get_sources("s1")
        assert sources[0]["stars"] is None


# ── migration on old schema ───────────────────────────────────────────────────

class TestMigration:
    def test_old_db_migrates_cleanly(self, tmp_path):
        db = str(tmp_path / "old.db")
        # Create a DB without the new columns
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                start_time TEXT NOT NULL,
                config_json TEXT NOT NULL
            );
            CREATE TABLE sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT,
                source_type TEXT,
                fetch_date TEXT,
                novel_findings_count INTEGER DEFAULT 0
            );
            CREATE TABLE findings (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                cycle_num INTEGER NOT NULL,
                finding_text TEXT NOT NULL,
                source_url TEXT NOT NULL,
                fetch_timestamp TEXT NOT NULL,
                source_hash TEXT NOT NULL,
                source_type TEXT NOT NULL,
                relevance_score INTEGER,
                quality_type TEXT,
                embedding_id TEXT,
                section TEXT
            );
        """)
        conn.execute("INSERT INTO sessions VALUES ('s1', 'topic', '2024-01-01', '{}')")
        conn.commit()
        conn.close()

        # StateWriter should migrate without error
        sw = StateWriter(db)
        sw.save_running_summary("s1", "migrated")
        sw.insert_source({
            "session_id": "s1",
            "url": "x",
            "stars": 10,
            "description": "d",
            "language": "Go",
        })

        sr = StateReader(db)
        assert sr.get_running_summary("s1") == "migrated"
        assert sr.get_sources("s1")[0]["stars"] == 10


# ── make_finding_id ───────────────────────────────────────────────────────────

class TestMakeFindingId:
    def test_deterministic(self):
        fid1 = make_finding_id("some finding", "https://example.com")
        fid2 = make_finding_id("some finding", "https://example.com")
        assert fid1 == fid2

    def test_different_text_gives_different_id(self):
        assert make_finding_id("a", "url") != make_finding_id("b", "url")

    def test_different_url_gives_different_id(self):
        assert make_finding_id("text", "url1") != make_finding_id("text", "url2")

    def test_length_is_32(self):
        assert len(make_finding_id("text", "url")) == 32

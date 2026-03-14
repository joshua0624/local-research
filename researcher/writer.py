"""
Single async writer actor — all DB writes and file I/O go through here,
eliminating race conditions.

Usage:
    actor = WriterActor(db_path, output_dir)
    await actor.start()
    await actor.insert_finding(finding)
    ...
    await actor.stop()
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .state import StateWriter

log = logging.getLogger(__name__)


# ── write request types ───────────────────────────────────────────────────────

@dataclass
class _Req:
    kind: str
    payload: Any = None
    _done: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class WriterActor:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._queue: asyncio.Queue[Optional[_Req]] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._writer: Optional[StateWriter] = None

    async def start(self) -> None:
        self._writer = StateWriter(self._db_path)
        self._task = asyncio.create_task(self._consume(), name="writer-actor")

    async def stop(self) -> None:
        await self._queue.put(None)
        if self._task:
            await self._task

    async def _consume(self) -> None:
        while True:
            req = await self._queue.get()
            if req is None:
                break
            try:
                self._dispatch(req)
                if not req._done.done():
                    req._done.set_result(None)
            except Exception as exc:
                log.exception("WriterActor error processing %s", req.kind)
                if not req._done.done():
                    req._done.set_exception(exc)

    def _dispatch(self, req: _Req) -> None:
        w = self._writer
        k = req.kind
        p = req.payload
        if k == "create_session":
            w.create_session(**p)
        elif k == "mark_url_seen":
            w.mark_url_seen(**p)
        elif k == "insert_finding":
            w.insert_finding(p)
        elif k == "insert_source":
            w.insert_source(p)
        elif k == "insert_query":
            w.insert_query(**p)
        elif k == "insert_lead":
            w.insert_lead(**p)
        elif k == "log_event":
            w.log_event(**p)
        elif k == "write_file":
            w.write_file_atomic(**p)
        elif k == "save_running_summary":
            w.save_running_summary(**p)
        elif k == "update_finding_sections":
            w.update_finding_sections(p)
        else:
            raise ValueError(f"Unknown write kind: {k}")

    async def _enqueue(self, kind: str, payload: Any) -> None:
        req = _Req(kind=kind, payload=payload)
        await self._queue.put(req)
        await req._done  # back-pressure: wait for the write to complete

    # ── public async API ──────────────────────────────────────────────────

    async def create_session(self, session_id: str, topic: str, config: dict) -> None:
        await self._enqueue("create_session", dict(session_id=session_id, topic=topic, config=config))

    async def mark_url_seen(self, url: str, session_id: str, status: str = "success") -> None:
        await self._enqueue("mark_url_seen", dict(url=url, session_id=session_id, status=status))

    async def insert_finding(self, finding: dict) -> None:
        await self._enqueue("insert_finding", finding)

    async def insert_source(self, source: dict) -> None:
        await self._enqueue("insert_source", source)

    async def insert_query(self, session_id: str, cycle_num: int, query_text: str) -> None:
        await self._enqueue("insert_query", dict(session_id=session_id, cycle_num=cycle_num, query_text=query_text))

    async def insert_lead(self, session_id: str, lead_text: str, source_url: str, cycle_num: int) -> None:
        await self._enqueue("insert_lead", dict(session_id=session_id, lead_text=lead_text, source_url=source_url, cycle_num=cycle_num))

    async def log_event(self, session_id: str, event_type: str, payload: object = None) -> None:
        await self._enqueue("log_event", dict(session_id=session_id, event_type=event_type, payload=payload))

    async def write_file(self, path: str, content: str) -> None:
        await self._enqueue("write_file", dict(path=path, content=content))

    async def save_running_summary(self, session_id: str, summary: str) -> None:
        await self._enqueue("save_running_summary", dict(session_id=session_id, summary=summary))

    async def update_finding_sections(self, assignments: dict) -> None:
        await self._enqueue("update_finding_sections", assignments)

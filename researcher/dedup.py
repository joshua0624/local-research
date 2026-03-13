"""
Phase 1: URL-level dedup backed by SQLite.
Phase 3 will extend this with semantic embedding similarity + contradiction detection.
"""
from __future__ import annotations

from .state import StateReader


class URLDedup:
    def __init__(self, reader: StateReader):
        self._reader = reader

    def is_seen(self, url: str, session_id: str) -> bool:
        return self._reader.url_seen(url, session_id)

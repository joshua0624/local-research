"""Abstract fetcher interface and shared result type."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FetchResult:
    url: str
    source_type: str           # "web" | "reddit" | "github"
    content: str = ""
    title: str = ""
    author: Optional[str] = None
    date: Optional[str] = None   # ISO-8601 string or None
    word_count: int = 0
    error: Optional[str] = None

    def ok(self) -> bool:
        return self.error is None and self.word_count > 0


class BaseFetcher(ABC):
    @abstractmethod
    async def fetch(self, url: str) -> FetchResult: ...

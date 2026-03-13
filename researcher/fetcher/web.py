"""
Web fetcher: httpx for transport, trafilatura for extraction, BS4 as fallback.
Optional Crawl4AI fallback for JS-heavy pages (only if installed and enabled).
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

import httpx
from aiolimiter import AsyncLimiter

from .base import BaseFetcher, FetchResult

try:
    import trafilatura
    from trafilatura.settings import use_config as _tf_use_config

    _tf_cfg = _tf_use_config()
    _tf_cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

try:
    import dateparser
    _HAS_DATEPARSER = True
except ImportError:
    _HAS_DATEPARSER = False

log = logging.getLogger(__name__)

# Block obviously irrelevant or paywalled domains
_BLOCKED_DOMAINS = frozenset(
    [
        "wsj.com",
        "nytimes.com",
        "ft.com",
        "bloomberg.com",
        "paywallsite.com",
    ]
)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_date(date_str: Optional[str]) -> Optional[str]:
    """Parse a date string to ISO-8601 UTC. Returns None on failure."""
    if not date_str:
        return None
    if _HAS_DATEPARSER:
        parsed = dateparser.parse(
            date_str, settings={"RETURN_AS_TIMEZONE_AWARE": True, "PREFER_DAY_OF_MONTH": "first"}
        )
        if parsed:
            return parsed.astimezone(timezone.utc).isoformat()
    # Fallback: look for YYYY-MM-DD in the string
    m = re.search(r"(\d{4}-\d{2}-\d{2})", date_str)
    if m:
        return m.group(1) + "T00:00:00+00:00"
    return None


def _is_within_age(date_iso: Optional[str], max_age_months: int) -> bool:
    """Return True if date is within max_age_months, or if date is unknown."""
    if not date_iso:
        return True
    try:
        dt = datetime.fromisoformat(date_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_months * 30)
        return dt >= cutoff
    except Exception:
        return True


def _extract_with_bs4(html: str, url: str) -> tuple[str, str, Optional[str], Optional[str]]:
    """Return (text, title, author, date) using BeautifulSoup."""
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    # Strip scripts and style
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Try to find main content areas
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(id=re.compile(r"content|main|article", re.I))
        or soup.find(class_=re.compile(r"content|main|article|post", re.I))
        or soup.body
    )
    text = main.get_text(separator="\n", strip=True) if main else ""

    # Author heuristic
    author_tag = soup.find(class_=re.compile(r"author|byline", re.I))
    author = author_tag.get_text(strip=True) if author_tag else None

    # Date heuristic
    date_tag = soup.find("time") or soup.find(class_=re.compile(r"date|published|posted", re.I))
    raw_date = None
    if date_tag:
        raw_date = date_tag.get("datetime") or date_tag.get_text(strip=True)

    return text, title, author, _parse_date(raw_date)


class WebFetcher(BaseFetcher):
    def __init__(
        self,
        rate_limit: float = 2.5,    # seconds between fetches
        max_concurrent: int = 5,
        max_age_months: int = 6,
        timeout: float = 15.0,
    ):
        self.max_age_months = max_age_months
        self._limiter = AsyncLimiter(max_rate=1, time_period=rate_limit)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = httpx.AsyncClient(
            headers=_DEFAULT_HEADERS,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            http2=False,
        )

    def _is_blocked(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return any(host.endswith(d) for d in _BLOCKED_DOMAINS)

    async def fetch(self, url: str) -> FetchResult:
        if self._is_blocked(url):
            return FetchResult(url=url, source_type="web", error="blocked domain")

        async with self._semaphore:
            async with self._limiter:
                return await self._fetch_inner(url)

    async def _fetch_inner(self, url: str) -> FetchResult:
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return FetchResult(url=url, source_type="web", error=f"HTTP {exc.response.status_code}")
        except httpx.RequestError as exc:
            return FetchResult(url=url, source_type="web", error=str(exc))

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return FetchResult(url=url, source_type="web", error=f"non-HTML content-type: {content_type}")

        html = resp.text
        text, title, author, date = await self._extract(html, url)

        # Date filter (level 2)
        if date and not _is_within_age(date, self.max_age_months):
            return FetchResult(url=url, source_type="web", error=f"too old: {date}")

        words = text.split()
        return FetchResult(
            url=url,
            source_type="web",
            content=text,
            title=title,
            author=author,
            date=date,
            word_count=len(words),
        )

    async def _extract(self, html: str, url: str) -> tuple[str, str, Optional[str], Optional[str]]:
        """Try trafilatura first, fall back to BS4."""
        if _HAS_TRAFILATURA:
            try:
                text = await asyncio.to_thread(
                    trafilatura.extract,
                    html,
                    include_comments=False,
                    include_tables=True,
                    output_format="txt",
                    config=_tf_cfg,
                )
                meta = await asyncio.to_thread(trafilatura.extract_metadata, html)
                if text and len(text.split()) >= 50:
                    title = (meta.title if meta and meta.title else url)
                    author = (meta.author if meta else None)
                    raw_date = (meta.date if meta else None)
                    return text, title, author, _parse_date(raw_date)
            except Exception as exc:
                log.debug("trafilatura failed for %s: %s", url, exc)

        if _HAS_BS4:
            return _extract_with_bs4(html, url)

        return "", url, None, None

    async def aclose(self) -> None:
        await self._client.aclose()

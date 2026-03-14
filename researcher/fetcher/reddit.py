"""
Reddit fetcher: httpx JSON endpoint primary, asyncpraw secondary (if credentials configured).

Fetches post body + top comments for post URLs, subreddit digest for sub URLs.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import urlparse

import httpx
from aiolimiter import AsyncLimiter

from .base import BaseFetcher, FetchResult
from .circuit_breaker import CircuitBreaker

try:
    import asyncpraw
    _HAS_ASYNCPRAW = True
except ImportError:
    _HAS_ASYNCPRAW = False

log = logging.getLogger(__name__)

_MIN_POST_SCORE = 3
_MIN_COMMENT_SCORE = 2
_MAX_COMMENTS = 50
_REDDIT_HOST = "www.reddit.com"
_CB_TRIP_STATUSES = frozenset([429, 500, 502, 503, 504])

_REDDIT_HEADERS = {
    "User-Agent": "LocalResearcher/1.0 (research tool; not affiliated with Reddit)",
    "Accept": "application/json",
}


def _parse_reddit_url(url: str) -> dict:
    """Parse a Reddit URL into components.

    Returns dict with keys: type ("post"|"subreddit"|"unknown"),
                             subreddit, post_id
    """
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]

    result: dict = {"type": "unknown", "subreddit": None, "post_id": None}

    if len(parts) >= 4 and parts[0] == "r" and parts[2] == "comments":
        result["type"] = "post"
        result["subreddit"] = parts[1]
        result["post_id"] = parts[3]
    elif len(parts) >= 2 and parts[0] == "r":
        result["type"] = "subreddit"
        result["subreddit"] = parts[1]
    elif len(parts) == 1 and parts[0] not in ("search", "user", "message"):
        # Bare subreddit name like reddit.com/LocalLLaMA — unlikely but handle gracefully
        result["type"] = "unknown"

    return result


def _format_post(post_data: dict, comments: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"# {post_data.get('title', 'Untitled')}")
    lines.append(
        f"r/{post_data.get('subreddit', '')} | "
        f"Score: {post_data.get('score', 0)} | "
        f"Comments: {post_data.get('num_comments', 0)}"
    )

    selftext = (post_data.get("selftext") or "").strip()
    if selftext and selftext not in ("[deleted]", "[removed]"):
        lines.append("")
        lines.append(selftext)

    if comments:
        lines.append("\n## Top Comments")
        for c in comments:
            body = (c.get("body") or "").strip()
            if body and body not in ("[deleted]", "[removed]"):
                lines.append(f"\n[Score: {c.get('score', 0)}] {body}")

    return "\n".join(lines)


def _extract_comments(children: list, depth: int = 0) -> list[dict]:
    """Recursively extract comments with score >= threshold, sorted by score."""
    comments: list[dict] = []
    for child in children:
        if child.get("kind") != "t1":
            continue
        c = child.get("data", {})
        score = c.get("score", 0)
        if score < _MIN_COMMENT_SCORE:
            continue
        body = (c.get("body") or "").strip()
        if not body or body in ("[deleted]", "[removed]"):
            continue
        comments.append({"body": body, "score": score})
        if depth < 2:
            replies = c.get("replies", {})
            if isinstance(replies, dict):
                reply_children = replies.get("data", {}).get("children", [])
                comments.extend(_extract_comments(reply_children, depth + 1))
    comments.sort(key=lambda x: x["score"], reverse=True)
    return comments[:_MAX_COMMENTS]


class RedditFetcher(BaseFetcher):
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: str = "LocalResearcher/1.0",
        rate_limit_per_min: float = 60.0,
        max_age_months: int = 6,
        timeout: float = 15.0,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.max_age_months = max_age_months
        self._client_id = client_id
        self._client_secret = client_secret
        self._user_agent = user_agent
        self._cb = circuit_breaker

        # 60 req/min burst, smoothed
        self._limiter = AsyncLimiter(max_rate=rate_limit_per_min, time_period=60.0)

        # ETag cache: url → etag
        self._etag_cache: dict[str, str] = {}

        self._client = httpx.AsyncClient(
            headers={**_REDDIT_HEADERS, "User-Agent": user_agent},
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )

        self._praw: Optional[object] = None

    def _is_recent(self, created_utc: Optional[float]) -> bool:
        if not created_utc:
            return True
        dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_months * 30)
        return dt >= cutoff

    async def fetch(self, url: str) -> FetchResult:
        # Strip query params and trailing slash for cleanliness
        url = url.split("?")[0].rstrip("/")
        parsed = _parse_reddit_url(url)

        if parsed["type"] == "post":
            return await self._fetch_post(url, parsed)
        elif parsed["type"] == "subreddit":
            return await self._fetch_subreddit(url, parsed)
        else:
            return FetchResult(
                url=url, source_type="reddit", error="unrecognized Reddit URL format"
            )

    async def _fetch_post(self, url: str, parsed: dict) -> FetchResult:
        async with self._limiter:
            return await self._fetch_post_json(url, parsed)

    async def _fetch_post_json(self, url: str, parsed: dict) -> FetchResult:
        if self._cb and self._cb.is_open(_REDDIT_HOST):
            return FetchResult(url=url, source_type="reddit", error="circuit open: reddit")

        json_url = (
            f"https://www.reddit.com/r/{parsed['subreddit']}"
            f"/comments/{parsed['post_id']}.json?limit=100&raw_json=1"
        )

        headers: dict[str, str] = {}
        etag = self._etag_cache.get(json_url)
        if etag:
            headers["If-None-Match"] = etag

        try:
            resp = await self._client.get(json_url, headers=headers)
        except httpx.RequestError as exc:
            return FetchResult(url=url, source_type="reddit", error=str(exc))

        if resp.status_code == 304:
            return FetchResult(url=url, source_type="reddit", error="not-modified")
        if resp.status_code in _CB_TRIP_STATUSES:
            if self._cb:
                self._cb.record_failure(_REDDIT_HOST)
            msg = "rate-limited" if resp.status_code == 429 else f"HTTP {resp.status_code}"
            return FetchResult(url=url, source_type="reddit", error=msg)
        if resp.status_code != 200:
            return FetchResult(
                url=url, source_type="reddit", error=f"HTTP {resp.status_code}"
            )

        new_etag = resp.headers.get("ETag")
        if new_etag:
            self._etag_cache[json_url] = new_etag

        if self._cb:
            self._cb.record_success(_REDDIT_HOST)

        try:
            data = resp.json()
        except Exception as exc:
            return FetchResult(url=url, source_type="reddit", error=f"JSON error: {exc}")

        if not isinstance(data, list) or len(data) < 1:
            return FetchResult(url=url, source_type="reddit", error="unexpected structure")

        try:
            post_data = data[0]["data"]["children"][0]["data"]
        except (KeyError, IndexError):
            return FetchResult(url=url, source_type="reddit", error="missing post data")

        if not self._is_recent(post_data.get("created_utc")):
            return FetchResult(url=url, source_type="reddit", error="too old")

        score = post_data.get("score", 0)
        if score < _MIN_POST_SCORE:
            return FetchResult(url=url, source_type="reddit", error=f"low score: {score}")

        comments: list[dict] = []
        if len(data) >= 2:
            children = data[1].get("data", {}).get("children", [])
            comments = _extract_comments(children)

        text = _format_post(post_data, comments)

        created = post_data.get("created_utc")
        date_str = (
            datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
            if created
            else None
        )

        return FetchResult(
            url=url,
            source_type="reddit",
            content=text,
            title=post_data.get("title", ""),
            author=post_data.get("author"),
            date=date_str,
            word_count=len(text.split()),
        )

    async def _fetch_subreddit(self, url: str, parsed: dict) -> FetchResult:
        """Fetch top posts from a subreddit as a digest."""
        if self._cb and self._cb.is_open(_REDDIT_HOST):
            return FetchResult(url=url, source_type="reddit", error="circuit open: reddit")

        async with self._limiter:
            sub = parsed["subreddit"]
            json_url = (
                f"https://www.reddit.com/r/{sub}/top.json"
                f"?t=month&limit=10&raw_json=1"
            )

            try:
                resp = await self._client.get(json_url)
            except httpx.RequestError as exc:
                return FetchResult(url=url, source_type="reddit", error=str(exc))

            if resp.status_code in _CB_TRIP_STATUSES:
                if self._cb:
                    self._cb.record_failure(_REDDIT_HOST)
                return FetchResult(
                    url=url, source_type="reddit", error=f"HTTP {resp.status_code}"
                )
            if resp.status_code != 200:
                return FetchResult(
                    url=url, source_type="reddit", error=f"HTTP {resp.status_code}"
                )
            if self._cb:
                self._cb.record_success(_REDDIT_HOST)

            try:
                data = resp.json()
                children = data["data"]["children"]
            except Exception as exc:
                return FetchResult(url=url, source_type="reddit", error=f"parse error: {exc}")

            lines = [f"# Top posts from r/{sub}\n"]
            count = 0
            for child in children:
                post = child.get("data", {})
                if post.get("score", 0) < _MIN_POST_SCORE:
                    continue
                if not self._is_recent(post.get("created_utc")):
                    continue
                lines.append(f"## {post.get('title', 'Untitled')}")
                lines.append(f"Score: {post.get('score', 0)}")
                selftext = (post.get("selftext") or "").strip()
                if selftext and selftext not in ("[deleted]", "[removed]"):
                    lines.append(selftext[:500])
                lines.append("")
                count += 1

            if count == 0:
                return FetchResult(
                    url=url, source_type="reddit", error="no qualifying posts"
                )

            text = "\n".join(lines)
            return FetchResult(
                url=url,
                source_type="reddit",
                content=text,
                title=f"r/{sub} — top posts",
                word_count=len(text.split()),
            )

    async def aclose(self) -> None:
        await self._client.aclose()
        if _HAS_ASYNCPRAW and self._praw is not None:
            await self._praw.close()

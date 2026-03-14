"""
GitHub fetcher: REST API via httpx.

Fetches repo README + metadata for repo URLs, issue body + comments for issue URLs.
Respects ETag caching and rate-limit headers.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional
from urllib.parse import urlparse

import httpx

from .base import BaseFetcher, FetchResult
from .circuit_breaker import CircuitBreaker

log = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"
_GITHUB_HOST = "api.github.com"
_DEFAULT_ACCEPT = "application/vnd.github+json"
_API_VERSION = "2022-11-28"
_CB_TRIP_STATUSES = frozenset([429, 500, 502, 503, 504])


def _parse_github_url(url: str) -> dict:
    """Parse a GitHub URL into components.

    Returns dict with keys: type ("repo"|"issue"|"file"|"unknown"),
                             owner, repo, issue_num
    """
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]

    result: dict = {"type": "unknown", "owner": None, "repo": None, "issue_num": None}

    if len(parts) == 0:
        return result
    if len(parts) == 1:
        # github.com/owner — user/org page, not useful to fetch via API
        return result

    result["owner"] = parts[0]
    result["repo"] = parts[1]

    if len(parts) == 2:
        result["type"] = "repo"
    elif len(parts) >= 4 and parts[2] == "issues":
        try:
            result["issue_num"] = int(parts[3])
            result["type"] = "issue"
        except ValueError:
            result["type"] = "repo"
    elif len(parts) >= 3 and parts[2] in ("blob", "tree", "raw"):
        result["type"] = "file"
    else:
        result["type"] = "repo"

    return result


class GitHubFetcher(BaseFetcher):
    def __init__(
        self,
        token: Optional[str] = None,
        courtesy_delay: float = 0.5,
        max_age_months: int = 6,
        timeout: float = 20.0,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.max_age_months = max_age_months
        self._courtesy_delay = courtesy_delay
        self._last_request_time: float = 0.0
        self._cb = circuit_breaker

        resolved_token = token or os.environ.get("GITHUB_TOKEN")

        headers = {
            "Accept": _DEFAULT_ACCEPT,
            "X-GitHub-Api-Version": _API_VERSION,
            "User-Agent": "LocalResearcher/1.0",
        }
        if resolved_token:
            headers["Authorization"] = f"Bearer {resolved_token}"

        self._client = httpx.AsyncClient(
            base_url=_GITHUB_API_BASE,
            headers=headers,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        )

        # ETag cache: path → etag string
        self._etag_cache: dict[str, str] = {}

    def _is_recent(self, date_str: Optional[str]) -> bool:
        if not date_str:
            return True
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.max_age_months * 30)
            return dt >= cutoff
        except Exception:
            return True

    async def _get(
        self, path: str, params: Optional[dict] = None
    ) -> Optional[httpx.Response]:
        """Rate-limited GET with ETag support and circuit-breaker guard."""
        if self._cb and self._cb.is_open(_GITHUB_HOST):
            log.warning("CircuitBreaker: rejecting GitHub request for %s (circuit open)", path)
            return None

        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._courtesy_delay:
            import asyncio
            await asyncio.sleep(self._courtesy_delay - elapsed)

        headers: dict[str, str] = {}
        etag = self._etag_cache.get(path)
        if etag:
            headers["If-None-Match"] = etag

        try:
            resp = await self._client.get(path, params=params, headers=headers)
        except httpx.RequestError as exc:
            log.warning("GitHub request error for %s: %s", path, exc)
            return None
        finally:
            self._last_request_time = time.monotonic()

        new_etag = resp.headers.get("ETag")
        if new_etag:
            self._etag_cache[path] = new_etag

        remaining = resp.headers.get("X-RateLimit-Remaining")
        if remaining and int(remaining) < 100:
            log.warning("GitHub rate limit low: %s remaining", remaining)

        if self._cb:
            if resp.status_code in _CB_TRIP_STATUSES:
                self._cb.record_failure(_GITHUB_HOST)
            elif resp.status_code < 400:
                self._cb.record_success(_GITHUB_HOST)

        return resp

    async def fetch(self, url: str) -> FetchResult:
        url = url.split("?")[0].rstrip("/")
        parsed = _parse_github_url(url)

        if parsed["type"] == "repo":
            return await self._fetch_repo(url, parsed)
        elif parsed["type"] == "issue":
            return await self._fetch_issue(url, parsed)
        elif parsed["type"] == "file":
            return await self._fetch_file(url, parsed)
        else:
            return FetchResult(
                url=url, source_type="github", error="unrecognized GitHub URL format"
            )

    async def _fetch_repo(self, url: str, parsed: dict) -> FetchResult:
        owner, repo = parsed["owner"], parsed["repo"]
        api_path = f"/repos/{owner}/{repo}"

        resp = await self._get(api_path)
        if resp is None:
            return FetchResult(url=url, source_type="github", error="request failed")
        if resp.status_code == 304:
            return FetchResult(url=url, source_type="github", error="not-modified")
        if resp.status_code == 404:
            return FetchResult(url=url, source_type="github", error="repo not found")
        if resp.status_code == 403:
            return FetchResult(url=url, source_type="github", error="forbidden (rate limit?)")
        if resp.status_code != 200:
            return FetchResult(
                url=url, source_type="github", error=f"HTTP {resp.status_code}"
            )

        try:
            repo_data = resp.json()
        except Exception:
            return FetchResult(url=url, source_type="github", error="JSON parse error")

        stars = repo_data.get("stargazers_count", 0)
        pushed_at = repo_data.get("pushed_at")

        if not self._is_recent(pushed_at):
            return FetchResult(
                url=url, source_type="github", error=f"not recently updated: {pushed_at}"
            )

        lines: list[str] = []
        lines.append(f"# {repo_data.get('full_name', f'{owner}/{repo}')}")
        lines.append(
            f"**Stars:** {stars} | "
            f"**Language:** {repo_data.get('language') or 'N/A'} | "
            f"**Updated:** {pushed_at or 'N/A'}"
        )
        desc = (repo_data.get("description") or "").strip()
        if desc:
            lines.append(f"\n**Description:** {desc}")
        topics = repo_data.get("topics", [])
        if topics:
            lines.append(f"**Topics:** {', '.join(topics)}")

        # Fetch README
        readme_resp = await self._get(f"/repos/{owner}/{repo}/readme")
        has_readme = False
        if readme_resp and readme_resp.status_code == 200:
            try:
                readme_data = readme_resp.json()
                raw_content = readme_data.get("content", "")
                readme_text = base64.b64decode(raw_content).decode("utf-8", errors="replace")
                readme_text = readme_text[:4000]
                lines.append(f"\n## README\n{readme_text}")
                has_readme = True
            except Exception as exc:
                log.debug("README decode failed for %s/%s: %s", owner, repo, exc)

        # Filter: 0 stars + no README → not useful
        if stars == 0 and not has_readme:
            return FetchResult(
                url=url, source_type="github", error="0 stars and no README"
            )

        # Fetch top issues (by comment count)
        issues_resp = await self._get(
            f"/repos/{owner}/{repo}/issues",
            params={"state": "open", "sort": "comments", "direction": "desc", "per_page": 5},
        )
        if issues_resp and issues_resp.status_code == 200:
            try:
                issues = issues_resp.json()
                if issues:
                    lines.append("\n## Open Issues (most discussed)")
                    for issue in issues[:5]:
                        num = issue.get("number")
                        title = issue.get("title", "")
                        n_comments = issue.get("comments", 0)
                        lines.append(f"- #{num}: {title} ({n_comments} comments)")
                        body = (issue.get("body") or "").strip()[:300]
                        if body:
                            lines.append(f"  {body}")
            except Exception:
                pass

        text = "\n".join(lines)
        return FetchResult(
            url=url,
            source_type="github",
            content=text,
            title=repo_data.get("full_name", f"{owner}/{repo}"),
            date=pushed_at,
            word_count=len(text.split()),
            metadata={
                "stars": stars,
                "description": desc,
                "language": repo_data.get("language") or "",
            },
        )

    async def _fetch_issue(self, url: str, parsed: dict) -> FetchResult:
        owner, repo, issue_num = parsed["owner"], parsed["repo"], parsed["issue_num"]
        api_path = f"/repos/{owner}/{repo}/issues/{issue_num}"

        resp = await self._get(api_path)
        if resp is None:
            return FetchResult(url=url, source_type="github", error="request failed")
        if resp.status_code == 304:
            return FetchResult(url=url, source_type="github", error="not-modified")
        if resp.status_code == 404:
            return FetchResult(url=url, source_type="github", error="issue not found")
        if resp.status_code != 200:
            return FetchResult(
                url=url, source_type="github", error=f"HTTP {resp.status_code}"
            )

        try:
            issue_data = resp.json()
        except Exception:
            return FetchResult(url=url, source_type="github", error="JSON parse error")

        if not self._is_recent(issue_data.get("created_at")):
            return FetchResult(url=url, source_type="github", error="issue too old")

        lines: list[str] = []
        lines.append(f"# Issue #{issue_num}: {issue_data.get('title', '')}")
        lines.append(
            f"**Repo:** {owner}/{repo} | "
            f"**State:** {issue_data.get('state', '')} | "
            f"**Comments:** {issue_data.get('comments', 0)}"
        )
        body = (issue_data.get("body") or "").strip()
        if body:
            lines.append(f"\n{body}")

        # Fetch comments
        comments_resp = await self._get(
            f"/repos/{owner}/{repo}/issues/{issue_num}/comments",
            params={"per_page": 10},
        )
        if comments_resp and comments_resp.status_code == 200:
            try:
                comments = comments_resp.json()
                if comments:
                    lines.append("\n## Comments")
                    for comment in comments[:10]:
                        author = (comment.get("user") or {}).get("login", "unknown")
                        c_body = (comment.get("body") or "").strip()[:500]
                        if c_body:
                            lines.append(f"\n**{author}:** {c_body}")
            except Exception:
                pass

        text = "\n".join(lines)
        return FetchResult(
            url=url,
            source_type="github",
            content=text,
            title=f"#{issue_num}: {issue_data.get('title', '')}",
            date=issue_data.get("created_at"),
            word_count=len(text.split()),
        )

    async def _fetch_file(self, url: str, parsed: dict) -> FetchResult:
        """Fetch raw file content from GitHub."""
        # Convert blob URL → raw URL
        raw_url = (
            url.replace("github.com", "raw.githubusercontent.com")
               .replace("/blob/", "/")
        )
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    raw_url,
                    headers={"User-Agent": "LocalResearcher/1.0"},
                    follow_redirects=True,
                )
            if resp.status_code != 200:
                return FetchResult(
                    url=url, source_type="github", error=f"HTTP {resp.status_code}"
                )
            content = resp.text[:5000]
            return FetchResult(
                url=url,
                source_type="github",
                content=content,
                title=url.split("/")[-1],
                word_count=len(content.split()),
            )
        except Exception as exc:
            return FetchResult(url=url, source_type="github", error=str(exc))

    async def aclose(self) -> None:
        await self._client.aclose()

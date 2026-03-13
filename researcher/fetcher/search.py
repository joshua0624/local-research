"""
SearXNG search interface.
Returns a list of result dicts: {url, title, snippet, source_type}.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)

# Map max_age_months to SearXNG time_range values
_TIME_RANGE: dict[int, str] = {
    1: "month",
    3: "year",
    6: "year",
}


def _time_range_for(max_age_months: int) -> Optional[str]:
    if max_age_months <= 1:
        return "month"
    if max_age_months <= 12:
        return "year"
    return None


def _classify_source(url: str) -> str:
    if "reddit.com" in url:
        return "reddit"
    if "github.com" in url:
        return "github"
    return "web"


class SearXNGSearcher:
    def __init__(self, base_url: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": "LocalResearcher/1.0"},
            follow_redirects=True,
        )

    async def search(
        self,
        query: str,
        max_results: int = 10,
        max_age_months: int = 6,
        language: str = "en",
    ) -> list[dict]:
        """Return up to max_results search results as dicts."""
        params = {
            "q": query,
            "format": "json",
            "language": language,
            "engines": "google,bing,duckduckgo",
            "categories": "general",
        }
        time_range = _time_range_for(max_age_months)
        if time_range:
            params["time_range"] = time_range

        try:
            resp = await self._client.get(f"{self.base_url}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            log.warning("SearXNG request failed for %r: %s", query, exc)
            return []
        except Exception as exc:
            log.warning("SearXNG unexpected error for %r: %s", query, exc)
            return []

        results = []
        for item in data.get("results", [])[:max_results]:
            url = item.get("url", "")
            if not url:
                continue
            results.append(
                {
                    "url": url,
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "source_type": _classify_source(url),
                    "published_date": item.get("publishedDate"),
                }
            )
        log.debug("SearXNG: %d results for %r", len(results), query)
        return results

    async def aclose(self) -> None:
        await self._client.aclose()

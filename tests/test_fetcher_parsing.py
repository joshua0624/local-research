"""
Tests for pure parsing/logic in the fetcher modules — no HTTP calls.

Covers:
  - Reddit URL parsing and comment extraction
  - GitHub URL parsing
  - WebFetcher domain blocking and age filtering
  - Circuit breaker integration (blocked host short-circuits)
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from researcher.fetcher.base import FetchResult
from researcher.fetcher.circuit_breaker import CircuitBreaker
from researcher.fetcher.reddit import (
    RedditFetcher,
    _extract_comments,
    _format_post,
    _parse_reddit_url,
)
from researcher.fetcher.github import _parse_github_url
from researcher.fetcher.web import WebFetcher, _is_within_age


# ── Reddit URL parsing ────────────────────────────────────────────────────────

class TestParseRedditUrl:
    def test_post_url(self):
        r = _parse_reddit_url("https://www.reddit.com/r/LocalLLaMA/comments/abc123/my_post")
        assert r["type"] == "post"
        assert r["subreddit"] == "LocalLLaMA"
        assert r["post_id"] == "abc123"

    def test_subreddit_url(self):
        r = _parse_reddit_url("https://www.reddit.com/r/ollama")
        assert r["type"] == "subreddit"
        assert r["subreddit"] == "ollama"

    def test_unknown_url(self):
        r = _parse_reddit_url("https://reddit.com/search?q=llm")
        assert r["type"] == "unknown"

    def test_post_url_with_trailing_slash(self):
        r = _parse_reddit_url("https://www.reddit.com/r/MachineLearning/comments/xyz/title/")
        assert r["type"] == "post"


# ── Reddit comment extraction ─────────────────────────────────────────────────

class TestExtractComments:
    def _make_comment(self, body, score, replies=None):
        c = {"kind": "t1", "data": {"body": body, "score": score, "replies": {}}}
        if replies:
            c["data"]["replies"] = {
                "data": {"children": [self._make_comment(r["body"], r["score"]) for r in replies]}
            }
        return c

    def test_filters_by_score(self):
        children = [
            self._make_comment("good comment", score=5),
            self._make_comment("bad comment", score=1),  # below _MIN_COMMENT_SCORE=2
        ]
        result = _extract_comments(children)
        assert len(result) == 1
        assert result[0]["body"] == "good comment"

    def test_filters_deleted(self):
        children = [
            self._make_comment("[deleted]", score=10),
            self._make_comment("[removed]", score=10),
            self._make_comment("real comment", score=5),
        ]
        result = _extract_comments(children)
        assert len(result) == 1
        assert result[0]["body"] == "real comment"

    def test_sorted_by_score_descending(self):
        children = [
            self._make_comment("low", score=3),
            self._make_comment("high", score=100),
            self._make_comment("mid", score=50),
        ]
        result = _extract_comments(children)
        assert result[0]["body"] == "high"
        assert result[1]["body"] == "mid"

    def test_non_t1_items_ignored(self):
        children = [
            {"kind": "t3", "data": {"body": "post", "score": 100}},
            self._make_comment("comment", score=10),
        ]
        result = _extract_comments(children)
        assert len(result) == 1

    def test_empty_list(self):
        assert _extract_comments([]) == []


# ── GitHub URL parsing ────────────────────────────────────────────────────────

class TestParseGithubUrl:
    def test_repo_url(self):
        r = _parse_github_url("https://github.com/langchain-ai/langchain")
        assert r["type"] == "repo"
        assert r["owner"] == "langchain-ai"
        assert r["repo"] == "langchain"

    def test_issue_url(self):
        r = _parse_github_url("https://github.com/owner/repo/issues/42")
        assert r["type"] == "issue"
        assert r["issue_num"] == 42

    def test_file_url(self):
        r = _parse_github_url("https://github.com/owner/repo/blob/main/README.md")
        assert r["type"] == "file"

    def test_tree_url(self):
        r = _parse_github_url("https://github.com/owner/repo/tree/main/src")
        assert r["type"] == "file"

    def test_user_page(self):
        r = _parse_github_url("https://github.com/octocat")
        assert r["type"] == "unknown"

    def test_root_url(self):
        r = _parse_github_url("https://github.com")
        assert r["type"] == "unknown"


# ── WebFetcher domain blocking ────────────────────────────────────────────────

class TestWebFetcherBlocking:
    def test_blocked_domain_returns_error(self):
        fetcher = WebFetcher()
        result = asyncio.get_event_loop().run_until_complete(
            fetcher.fetch("https://wsj.com/some-article")
        )
        assert result.error == "blocked domain"
        assert result.content == ""

    def test_non_blocked_domain_not_blocked(self):
        # Just verify the blocking check doesn't incorrectly flag good domains
        fetcher = WebFetcher()
        assert not fetcher._is_blocked("https://example.com/article")
        assert not fetcher._is_blocked("https://arxiv.org/abs/2301.00001")

    def test_subdomain_of_blocked_is_blocked(self):
        fetcher = WebFetcher()
        assert fetcher._is_blocked("https://markets.ft.com/data")


# ── WebFetcher age filtering ──────────────────────────────────────────────────

class TestIsWithinAge:
    def test_recent_date_passes(self):
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        assert _is_within_age(recent, max_age_months=6) is True

    def test_old_date_fails(self):
        assert _is_within_age("2020-01-01T00:00:00+00:00", max_age_months=6) is False

    def test_none_date_passes(self):
        assert _is_within_age(None, max_age_months=6) is True

    def test_border_date(self):
        from datetime import datetime, timezone, timedelta
        # Exactly at the cutoff boundary — just outside
        old = (datetime.now(timezone.utc) - timedelta(days=6 * 30 + 1)).isoformat()
        assert _is_within_age(old, max_age_months=6) is False


# ── Circuit breaker integration in fetchers ───────────────────────────────────

class TestCircuitBreakerInWebFetcher:
    def test_open_circuit_returns_error_without_http(self):
        cb = CircuitBreaker(threshold=1, pause_minutes=10.0)
        # Trip it
        cb.record_failure("example.com")
        fetcher = WebFetcher(circuit_breaker=cb)

        result = asyncio.get_event_loop().run_until_complete(
            fetcher.fetch("https://example.com/page")
        )
        assert result.error is not None
        assert "circuit open" in result.error

    def test_closed_circuit_does_not_block(self):
        cb = CircuitBreaker(threshold=5)
        cb.record_failure("other-host.com")  # different host, no effect
        fetcher = WebFetcher(circuit_breaker=cb)
        # Is not blocked (would fail on network, but not on circuit)
        assert not cb.is_open("example.com")


class TestCircuitBreakerInRedditFetcher:
    def test_open_circuit_returns_error(self):
        cb = CircuitBreaker(threshold=1)
        cb.record_failure("www.reddit.com")
        fetcher = RedditFetcher(circuit_breaker=cb)

        result = asyncio.get_event_loop().run_until_complete(
            fetcher.fetch("https://www.reddit.com/r/LocalLLaMA/comments/abc/title")
        )
        assert result.error is not None
        assert "circuit open" in result.error

    def test_429_trips_circuit(self):
        """A 429 response should increment the failure counter."""
        cb = CircuitBreaker(threshold=3)
        fetcher = RedditFetcher(circuit_breaker=cb, rate_limit_per_min=10000)

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {}

        with patch.object(fetcher._client, "get", new=AsyncMock(return_value=mock_resp)):
            asyncio.get_event_loop().run_until_complete(
                fetcher.fetch("https://www.reddit.com/r/sub/comments/abc/title")
            )

        assert cb.failure_count("www.reddit.com") == 1

    def test_success_resets_counter(self):
        """A 200 response should reset the failure counter."""
        cb = CircuitBreaker(threshold=3)
        cb.record_failure("www.reddit.com")
        cb.record_failure("www.reddit.com")
        assert cb.failure_count("www.reddit.com") == 2

        fetcher = RedditFetcher(circuit_breaker=cb, rate_limit_per_min=10000)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {}
        mock_resp.json.return_value = [
            {"data": {"children": [{"data": {
                "title": "Test Post", "score": 10, "subreddit": "sub",
                "selftext": "body", "num_comments": 5,
                "created_utc": 9999999999,  # far future → always recent
                "author": "user",
            }}]}},
            {"data": {"children": []}},
        ]

        with patch.object(fetcher._client, "get", new=AsyncMock(return_value=mock_resp)):
            asyncio.get_event_loop().run_until_complete(
                fetcher.fetch("https://www.reddit.com/r/sub/comments/abc/title")
            )

        assert cb.failure_count("www.reddit.com") == 0

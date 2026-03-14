"""Tests for pure utility functions in researcher.brain (no LLM calls)."""
from __future__ import annotations

import pytest

from researcher.brain import _parse_json, trigram_similarity


# ── _parse_json ───────────────────────────────────────────────────────────────

class TestParseJson:
    def test_direct_json_object(self):
        result = _parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_direct_json_array(self):
        result = _parse_json('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_code_fenced_json(self):
        text = '```json\n{"findings": ["one", "two"]}\n```'
        result = _parse_json(text)
        assert result == {"findings": ["one", "two"]}

    def test_code_fenced_no_lang(self):
        text = '```\n["query one", "query two"]\n```'
        result = _parse_json(text)
        assert result == ["query one", "query two"]

    def test_json_object_embedded_in_prose(self):
        # Object with only scalar values (avoids the array-first regex match)
        text = 'Here is the result:\n{"saturated": false, "cycle_count": 3}\nDone.'
        result = _parse_json(text)
        assert result["saturated"] is False
        assert result["cycle_count"] == 3

    def test_json_array_embedded_in_prose(self):
        text = 'Based on the topic, here are queries:\n["q1", "q2", "q3"]\nHope that helps!'
        result = _parse_json(text)
        assert result == ["q1", "q2", "q3"]

    def test_whitespace_stripped(self):
        result = _parse_json('   {"x": 1}   ')
        assert result == {"x": 1}

    def test_nested_structure(self):
        text = '{"findings": ["f1", "f2"], "relevance_score": 4, "quality_type": "research"}'
        result = _parse_json(text)
        assert result["relevance_score"] == 4
        assert len(result["findings"]) == 2

    def test_raises_on_unparseable(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            _parse_json("This is just plain text with no JSON at all.")

    def test_raises_on_empty_string(self):
        with pytest.raises((ValueError, Exception)):
            _parse_json("")


# ── trigram_similarity ────────────────────────────────────────────────────────

class TestTrigramSimilarity:
    def test_identical_strings(self):
        assert trigram_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_completely_different(self):
        sim = trigram_similarity("abcdefgh", "zyxwvuts")
        assert sim == pytest.approx(0.0)

    def test_empty_strings(self):
        # short strings have no trigrams → 0.0
        assert trigram_similarity("", "") == pytest.approx(0.0)
        assert trigram_similarity("ab", "ab") == pytest.approx(0.0)

    def test_partial_overlap(self):
        sim = trigram_similarity("local llm agents", "llm coding agents")
        assert 0.0 < sim < 1.0

    def test_symmetric(self):
        a = "open source llm"
        b = "llm open weights"
        assert trigram_similarity(a, b) == pytest.approx(trigram_similarity(b, a))

    def test_case_insensitive(self):
        assert trigram_similarity("Hello World", "hello world") == pytest.approx(1.0)

    def test_high_similarity_near_duplicate_queries(self):
        a = "best open source LLM models 2024"
        b = "best open source LLM models 2024 performance"
        # Should be fairly similar
        assert trigram_similarity(a, b) > 0.7

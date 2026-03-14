"""Tests for FindingsStore (in-memory store + markdown renderer)."""
from __future__ import annotations

import pytest

from researcher.findings import FindingsStore


def make_finding(fid="f1", text="Test finding", source_type="web", section=None, score=3):
    return {
        "id": fid,
        "finding_text": text,
        "source_url": "https://example.com",
        "source_type": source_type,
        "fetch_timestamp": "2024-01-15T00:00:00",
        "quality_type": "research",
        "relevance_score": score,
        "section": section,
        "conflicting": 0,
    }


def make_source(url="https://example.com", source_type="web", **kwargs):
    base = {
        "url": url,
        "title": url.split("/")[-1],
        "source_type": source_type,
        "fetch_date": "2024-01-15",
        "novel_findings_count": 1,
        "stars": None,
        "description": None,
        "language": None,
    }
    base.update(kwargs)
    return base


def render(store):
    return store.render("test topic", "sess1", "2024-01-01 00:00 UTC", "max_age=6m")


# ── counts ────────────────────────────────────────────────────────────────────

class TestCounts:
    def test_empty_store(self):
        fs = FindingsStore()
        assert fs.finding_count() == 0
        assert fs.source_count() == 0

    def test_counts_increment(self):
        fs = FindingsStore()
        fs.add_finding(make_finding())
        fs.add_finding(make_finding("f2"))
        fs.add_source(make_source())
        assert fs.finding_count() == 2
        assert fs.source_count() == 1


# ── update_sections ───────────────────────────────────────────────────────────

class TestUpdateSections:
    def test_assigns_section(self):
        fs = FindingsStore()
        fs.add_finding(make_finding("f1"))
        fs.update_sections({"f1": "Model Architecture"})
        assert fs.findings[0]["section"] == "Model Architecture"

    def test_unknown_id_is_ignored(self):
        fs = FindingsStore()
        fs.add_finding(make_finding("f1"))
        fs.update_sections({"f99": "Phantom"})
        assert fs.findings[0]["section"] is None

    def test_partial_assignment(self):
        fs = FindingsStore()
        fs.add_finding(make_finding("f1"))
        fs.add_finding(make_finding("f2"))
        fs.update_sections({"f1": "Theme A"})
        assert fs.findings[0]["section"] == "Theme A"
        assert fs.findings[1]["section"] is None


# ── render — header ───────────────────────────────────────────────────────────

class TestRenderHeader:
    def test_contains_topic(self):
        md = render(FindingsStore())
        assert "test topic" in md

    def test_no_findings_placeholder(self):
        md = render(FindingsStore())
        assert "No findings yet" in md

    def test_exec_summary_placeholder(self):
        md = render(FindingsStore())
        assert "Not yet generated" in md

    def test_exec_summary_shown_when_set(self):
        fs = FindingsStore()
        fs.set_executive_summary("This is the summary.")
        assert "This is the summary." in render(fs)


# ── render — findings ─────────────────────────────────────────────────────────

class TestRenderFindings:
    def test_finding_text_appears(self):
        fs = FindingsStore()
        fs.add_finding(make_finding(text="LLMs are fast now"))
        assert "LLMs are fast now" in render(fs)

    def test_conflict_tag_shown(self):
        fs = FindingsStore()
        f = make_finding()
        f["conflicting"] = 1
        fs.add_finding(f)
        assert "conflicts with another finding" in render(fs)

    def test_no_conflict_tag_by_default(self):
        fs = FindingsStore()
        fs.add_finding(make_finding())
        assert "conflicts" not in render(fs)

    def test_sections_grouped(self):
        fs = FindingsStore()
        fs.add_finding(make_finding("f1", text="Finding A", section="Theme X"))
        fs.add_finding(make_finding("f2", text="Finding B", section="Theme Y"))
        md = render(fs)
        assert "### Theme X" in md
        assert "### Theme Y" in md

    def test_no_section_headers_for_single_section(self):
        fs = FindingsStore()
        fs.add_finding(make_finding("f1", section="Theme X"))
        fs.add_finding(make_finding("f2", section="Theme X"))
        md = render(fs)
        # Only one section → no ### headers
        assert "### Theme X" not in md


# ── render — notable projects ─────────────────────────────────────────────────

class TestRenderNotableProjects:
    def test_no_github_sources_no_table(self):
        fs = FindingsStore()
        fs.add_source(make_source(source_type="web"))
        assert "Notable Projects" not in render(fs)

    def test_github_source_shows_table(self):
        fs = FindingsStore()
        fs.add_source(make_source(
            url="https://github.com/foo/bar",
            source_type="github",
            title="foo/bar",
        ))
        md = render(fs)
        assert "Notable Projects" in md
        assert "foo/bar" in md

    def test_stars_and_description_shown(self):
        fs = FindingsStore()
        fs.add_source(make_source(
            url="https://github.com/foo/bar",
            source_type="github",
            title="foo/bar",
            stars=4200,
            description="A very useful tool",
            language="Rust",
        ))
        md = render(fs)
        assert "4,200" in md
        assert "A very useful tool" in md
        assert "Rust" in md

    def test_sorted_by_stars_descending(self):
        fs = FindingsStore()
        fs.add_source(make_source(
            url="https://github.com/a/low",
            source_type="github",
            title="a/low",
            stars=10,
        ))
        fs.add_source(make_source(
            url="https://github.com/b/high",
            source_type="github",
            title="b/high",
            stars=9999,
        ))
        md = render(fs)
        assert md.index("b/high") < md.index("a/low")


# ── render — leads ────────────────────────────────────────────────────────────

class TestRenderLeads:
    def test_leads_section_appears(self):
        fs = FindingsStore()
        fs.add_lead({"lead_text": "Check out project X", "source_url": "https://example.com", "cycle_num": 1})
        md = render(fs)
        assert "Leads to Follow Up" in md
        assert "Check out project X" in md

    def test_no_leads_no_section(self):
        assert "Leads to Follow Up" not in render(FindingsStore())


# ── render — research log ─────────────────────────────────────────────────────

class TestRenderResearchLog:
    def test_cycle_log_appears(self):
        fs = FindingsStore()
        fs.add_cycle_log({
            "cycle_num": 1,
            "timestamp": "2024-01-15 10:00:00 UTC",
            "queries": ["query one", "query two"],
            "sources_fetched": 5,
            "novel_findings": 3,
            "skipped": 2,
        })
        md = render(fs)
        assert "Research Log" in md
        assert "Cycle 1" in md
        assert "query one" in md
        assert "Novel findings: 3" in md

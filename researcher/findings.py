"""
In-memory findings store + markdown renderer.
Populated from SQLite at startup; updated in memory during a session.
findings.md is rendered from scratch each cycle via the writer actor.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class FindingsStore:
    def __init__(self) -> None:
        self._findings: list[dict] = []
        self._sources: list[dict] = []
        self._leads: list[dict] = []
        self._running_summary: str = ""
        self._executive_summary: str = ""
        self._cycle_log: list[dict] = []

    # ── mutations ─────────────────────────────────────────────────────────

    def update_sections(self, assignments: dict) -> None:
        """Apply section assignments {finding_id: section_name} to in-memory findings."""
        for f in self._findings:
            sec = assignments.get(f["id"])
            if sec:
                f["section"] = sec

    def add_finding(self, finding: dict) -> None:
        self._findings.append(finding)

    def add_source(self, source: dict) -> None:
        self._sources.append(source)

    def add_lead(self, lead: dict) -> None:
        self._leads.append(lead)

    def set_running_summary(self, text: str) -> None:
        self._running_summary = text

    def set_executive_summary(self, text: str) -> None:
        self._executive_summary = text

    def add_cycle_log(self, entry: dict) -> None:
        self._cycle_log.append(entry)

    # ── reads ─────────────────────────────────────────────────────────────

    @property
    def running_summary(self) -> str:
        return self._running_summary

    @property
    def findings(self) -> list[dict]:
        return list(self._findings)

    def finding_count(self) -> int:
        return len(self._findings)

    def source_count(self) -> int:
        return len(self._sources)

    # ── markdown renderer ─────────────────────────────────────────────────

    def render(
        self,
        topic: str,
        session_id: str,
        start_time: str,
        config_summary: str,
    ) -> str:
        now = _now_utc_str()
        fc = len(self._findings)
        sc = len(self._sources)

        lines: list[str] = []

        # Header
        lines += [
            f"# Research Findings: {topic}",
            "",
            f"**Session:** {start_time} — {now}  "
            f"| **Sources processed:** {sc}  "
            f"| **Findings:** {fc}  "
            f"| **Config:** {config_summary}",
            "",
        ]

        # Executive Summary
        lines += ["## Executive Summary", ""]
        if self._executive_summary:
            lines += [self._executive_summary, ""]
        else:
            lines += ["*Not yet generated. Use `synthesize` or wait for auto-generation.*", ""]

        # Key Findings
        lines += ["## Key Findings", ""]
        if not self._findings:
            lines += ["*No findings yet.*", ""]
        else:
            # Group by section if available; otherwise flat list
            by_section: dict[str, list[dict]] = {}
            for f in self._findings:
                sec = f.get("section") or "General"
                by_section.setdefault(sec, []).append(f)

            for section, items in by_section.items():
                if len(by_section) > 1:
                    lines += [f"### {section}", ""]
                for f in items:
                    conflict_tag = " `⚠ conflicts with another finding`" if f.get("conflicting") else ""
                    lines += [f"- **Finding:** {f['finding_text']}{conflict_tag}"]
                    url = f.get("source_url", "")
                    src_type = f.get("source_type", "web")
                    date = f.get("fetch_timestamp", "")[:10]
                    quality = f.get("quality_type", "")
                    rel = f.get("relevance_score", "")
                    lines += [
                        f"  - *Source:* [{url}]({url}) ({src_type}, {date})"
                        + (f" · {quality}" if quality else "")
                        + (f" · relevance {rel}/5" if rel else ""),
                        "",
                    ]

        # Notable Projects (GitHub sources with stars)
        github_sources = [s for s in self._sources if s.get("source_type") == "github"]
        if github_sources:
            lines += [
                "## Notable Projects & Tools",
                "",
                "| Project | Stars | Language | What It Does | Findings |",
                "|---------|-------|----------|-------------|----------|",
            ]
            # Sort by stars descending, then novel findings descending
            sorted_gh = sorted(
                github_sources,
                key=lambda s: (s.get("stars") or 0, s.get("novel_findings_count") or 0),
                reverse=True,
            )
            for s in sorted_gh[:20]:
                url = s.get("url", "")
                title = s.get("title") or url.split("/")[-1]
                stars = s.get("stars")
                stars_str = f"{stars:,}" if stars is not None else "—"
                lang = s.get("language") or "—"
                desc = (s.get("description") or "—")[:80]
                nf = s.get("novel_findings_count", 0)
                lines.append(f"| [{title}]({url}) | {stars_str} | {lang} | {desc} | {nf} |")
            lines.append("")

        # Leads to Follow Up
        if self._leads:
            lines += ["## Leads to Follow Up", ""]
            for lead in self._leads:
                src = lead.get("source_url", "")
                lines.append(f"- [ ] {lead['lead_text']} — mentioned in [{src}]({src})")
            lines.append("")

        # Research Log
        if self._cycle_log:
            lines += ["## Research Log", ""]
            for entry in self._cycle_log:
                ts = entry.get("timestamp", "")
                cn = entry.get("cycle_num", "?")
                queries = ", ".join(f'"{q}"' for q in entry.get("queries", []))
                lines += [
                    f"### Cycle {cn} — {ts}",
                    f"- Queries: {queries}",
                    f"- Sources fetched: {entry.get('sources_fetched', 0)}  "
                    f"| Novel findings: {entry.get('novel_findings', 0)}  "
                    f"| Skipped (dup/filtered): {entry.get('skipped', 0)}",
                    "",
                ]

        # Sources Consulted
        if self._sources:
            lines += ["## Sources Consulted", ""]
            for s in self._sources:
                url = s.get("url", "")
                date = (s.get("fetch_date") or "")[:10]
                nf = s.get("novel_findings_count", 0)
                flag = "✓" if nf > 0 else "–"
                lines.append(f"- {flag} [{url}]({url}) ({date}) — {nf} novel findings")
            lines.append("")

        return "\n".join(lines)

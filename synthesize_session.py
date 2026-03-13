#!/usr/bin/env python3
"""
Offline synthesis script — reads findings from researcher.db and generates
a comprehensive summary using the local LLM.

The existing in-loop synthesize() caps at 60 findings.  This script handles
large sessions via hierarchical summarization: findings are chunked, each chunk
is condensed by the LLM, then a final synthesis is produced across the
condensed chunks.

Usage:
    python synthesize_session.py                          # most recent session
    python synthesize_session.py --session abc12345
    python synthesize_session.py --db /path/to/researcher.db --output summary.md
"""
from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from pathlib import Path

import yaml

# ── allow running from repo root without installing the package ───────────────
sys.path.insert(0, str(Path(__file__).parent))
from researcher.brain import Brain  # noqa: E402

_HERE = Path(__file__).parent
_CONFIG_PATH = _HERE / "researcher" / "config.yaml"
_PROMPTS_DIR = _HERE / "researcher" / "prompts"

CHUNK_SIZE = 40  # findings per batch before hierarchical summarization


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_session(db_path: str, session_id: str | None) -> tuple[str, str]:
    """Return (session_id, topic) — uses most recent session if none specified."""
    con = sqlite3.connect(db_path)
    try:
        if session_id:
            row = con.execute(
                "SELECT id, topic FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not row:
                raise SystemExit(f"Session '{session_id}' not found in {db_path}")
        else:
            row = con.execute(
                "SELECT id, topic FROM sessions ORDER BY start_time DESC LIMIT 1"
            ).fetchone()
            if not row:
                raise SystemExit(f"No sessions found in {db_path}")
        return row[0], row[1]
    finally:
        con.close()


def _load_findings(db_path: str, session_id: str) -> list[dict]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT finding_text, source_url, section, relevance_score, cycle_num
            FROM findings
            WHERE session_id = ?
            ORDER BY cycle_num, id
            """,
            (session_id,),
        ).fetchall()
        return [
            {
                "finding_text": r[0],
                "source_url": r[1],
                "section": r[2] or "General",
                "relevance_score": r[3],
                "cycle_num": r[4],
            }
            for r in rows
        ]
    finally:
        con.close()


def _load_sources(db_path: str, session_id: str) -> list[dict]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT url, title, novel_findings_count
            FROM sources
            WHERE session_id = ?
            ORDER BY novel_findings_count DESC
            """,
            (session_id,),
        ).fetchall()
        return [{"url": r[0], "title": r[1] or r[0], "count": r[2]} for r in rows]
    finally:
        con.close()


def _load_leads(db_path: str, session_id: str) -> list[str]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT DISTINCT lead_text FROM leads WHERE session_id = ? LIMIT 30",
            (session_id,),
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


# ── LLM helpers ───────────────────────────────────────────────────────────────

_BATCH_PROMPT = """\
You are a research assistant condensing a batch of raw findings into a tight summary.

**Research topic:** {topic}

**Findings batch ({n} items):**
{findings_list}

Write a condensed summary (8–12 bullet points) capturing the most important
claims, facts, and patterns from this batch.  Be specific — include names,
numbers, and technical details.  Skip obvious or generic statements.

Output plain text bullet points only (no headers, no JSON).
"""

_FINAL_PROMPT = """\
You are a research assistant writing a comprehensive executive summary.

**Research topic:** {topic}
**Total findings processed:** {total}
**Sessions / cycles:** {cycles}

**Condensed batch summaries:**
{batch_summaries}

Write a thorough executive summary (5–8 paragraphs) that:
1. States the main conclusions and themes across all findings
2. Highlights the most significant and well-supported findings, with specifics
3. Notes contradictions, open questions, or areas of uncertainty
4. Points to the most important sources, projects, or tools discovered
5. Identifies what the research did NOT find or where coverage is thin

Style: factual, specific, technical audience.  Cite names, numbers, claims.
Do NOT reproduce every finding — synthesize and interpret.
Output plain text only (no markdown headers, no JSON).
"""


async def _condense_chunk(brain: Brain, topic: str, chunk: list[dict]) -> str:
    lines = [
        f"- {f['finding_text']} (source: {f['source_url']})"
        for f in chunk
    ]
    prompt = _BATCH_PROMPT.format(
        topic=topic,
        n=len(chunk),
        findings_list="\n".join(lines),
    )
    return (await brain._call("heavy", prompt)).strip()


async def _final_synthesis(
    brain: Brain,
    topic: str,
    batch_summaries: list[str],
    total_findings: int,
    num_cycles: int,
) -> str:
    combined = "\n\n---\n\n".join(
        f"Batch {i + 1}:\n{s}" for i, s in enumerate(batch_summaries)
    )
    prompt = _FINAL_PROMPT.format(
        topic=topic,
        total=total_findings,
        cycles=num_cycles,
        batch_summaries=combined,
    )
    return (await brain._call("heavy", prompt)).strip()


# ── output rendering ──────────────────────────────────────────────────────────

def _render(
    output_path: str,
    topic: str,
    session_id: str,
    summary: str,
    findings: list[dict],
    sources: list[dict],
    leads: list[str],
) -> None:
    lines: list[str] = []

    lines += [f"# Research Summary: {topic}", ""]
    lines += [f"**Session:** `{session_id}`  ", f"**Total findings:** {len(findings)}  ",
               f"**Sources processed:** {len(sources)}", ""]

    lines += ["## Executive Summary", ""]
    lines.append(summary)
    lines.append("")

    # Findings grouped by section
    sections: dict[str, list[dict]] = {}
    for f in findings:
        sections.setdefault(f["section"], []).append(f)

    lines += ["## Findings by Section", ""]
    for section, items in sections.items():
        lines += [f"### {section}", ""]
        for f in items:
            src = f["source_url"]
            lines.append(f"- {f['finding_text']}  \n  *Source: {src}*")
        lines.append("")

    # Top sources
    lines += ["## Top Sources", ""]
    for s in sources[:30]:
        count_str = f" ({s['count']} findings)" if s["count"] else ""
        lines.append(f"- [{s['title']}]({s['url']}){count_str}")
    lines.append("")

    # Leads
    if leads:
        lines += ["## Follow-up Leads", ""]
        for lead in leads:
            lines.append(f"- {lead}")
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Written to {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text())
    brain = Brain(config, str(_PROMPTS_DIR))

    session_id, topic = _load_session(args.db, args.session)
    print(f"Session: {session_id}  |  Topic: {topic}")

    findings = _load_findings(args.db, session_id)
    sources = _load_sources(args.db, session_id)
    leads = _load_leads(args.db, session_id)
    num_cycles = max((f["cycle_num"] for f in findings), default=0) + 1

    print(f"Loaded {len(findings)} findings from {len(sources)} sources across {num_cycles} cycles")

    if not findings:
        raise SystemExit("No findings found for this session.")

    # Hierarchical synthesis
    chunks = [findings[i : i + CHUNK_SIZE] for i in range(0, len(findings), CHUNK_SIZE)]
    print(f"Condensing {len(chunks)} chunk(s) of up to {CHUNK_SIZE} findings each…")

    batch_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i + 1}/{len(chunks)} ({len(chunk)} findings)…")
        summary = await _condense_chunk(brain, topic, chunk)
        batch_summaries.append(summary)

    print("Generating final executive summary…")
    final = await _final_synthesis(brain, topic, batch_summaries, len(findings), num_cycles)

    _render(args.output, topic, session_id, final, findings, sources, leads)


def main() -> None:
    p = argparse.ArgumentParser(description="Offline synthesis from researcher.db")
    p.add_argument("--db", default="researcher.db", help="SQLite DB path")
    p.add_argument("--session", default=None, help="Session ID (default: most recent)")
    p.add_argument("--output", "-o", default="summary.md", help="Output markdown file")
    p.add_argument("--config", default=str(_CONFIG_PATH), help="Config YAML path")
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Second-pass analysis of a completed LocalResearch session.

Pipeline:
  LOAD → EMBED → MAP (LLM per chunk) → REDUCE (programmatic) →
  NOVELTY (numpy) → CONTRADICTIONS (light LLM) → SYNTHESIS (heavy LLM) → RENDER

Usage:
    python second_pass.py                            # most recent session
    python second_pass.py --session abc12345
    python second_pass.py --embed-cache emb.npy      # save/reuse embeddings on reruns
    python second_pass.py --no-embed                 # skip novelty scoring
    python second_pass.py --synthesis-only           # skip MAP/REDUCE, load intermediate JSON
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import math

import yaml

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

sys.path.insert(0, str(Path(__file__).parent))
from researcher.brain import Brain, _parse_json, trigram_similarity  # noqa: E402

_HERE = Path(__file__).parent
_CONFIG_PATH = _HERE / "researcher" / "config.yaml"
_PROMPTS_DIR = _HERE / "researcher" / "prompts"

console = Console()


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_session(db_path: str, session_id: str | None) -> tuple[str, str]:
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


def _load_findings_full(db_path: str, session_id: str) -> list[dict]:
    """Load all findings with source title via JOIN."""
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT
                f.id, f.finding_text, f.source_url, f.source_type,
                f.relevance_score, f.quality_type, f.fetch_timestamp,
                f.cycle_num, f.section, f.conflicting,
                s.title AS source_title
            FROM findings f
            LEFT JOIN (
                SELECT url, MAX(title) AS title
                FROM sources
                WHERE session_id = ?
                GROUP BY url
            ) s ON f.source_url = s.url
            WHERE f.session_id = ?
            ORDER BY f.cycle_num, f.id
            """,
            (session_id, session_id),
        ).fetchall()
        cols = [
            "id", "finding_text", "source_url", "source_type",
            "relevance_score", "quality_type", "fetch_timestamp",
            "cycle_num", "section", "conflicting", "source_title",
        ]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        con.close()


def _load_sources_full(db_path: str, session_id: str) -> list[dict]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT url, MAX(title) AS title, source_type,
                   SUM(novel_findings_count) AS total_findings,
                   MAX(stars) AS stars, MAX(description) AS description
            FROM sources
            WHERE session_id = ?
            GROUP BY url
            ORDER BY total_findings DESC
            """,
            (session_id,),
        ).fetchall()
        cols = ["url", "title", "source_type", "total_findings", "stars", "description"]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        con.close()


# ── Embedding helpers ─────────────────────────────────────────────────────────

async def _embed_all(
    brain: Brain,
    findings: list[dict],
    cache_path: str | None,
    progress: Progress,
) -> dict[str, list[float]]:
    """Embed all findings. Returns {finding_id -> vector}. Loads/saves numpy cache."""
    ids = [f["id"] for f in findings]
    texts = [f["finding_text"] for f in findings]

    if cache_path and Path(cache_path).exists():
        data: dict[str, list[float]] = json.loads(Path(cache_path).read_text())
        if len(data) == len(findings):
            console.print(f"[dim]Loaded {len(data)} embeddings from cache ({cache_path})[/dim]")
            return data
        console.print(
            f"[yellow]Cache has {len(data)} entries but session has {len(findings)} findings "
            f"— re-embedding[/yellow]"
        )

    batch_size = brain.config.get("embed_batch_size", 32)
    task = progress.add_task("Embedding findings…", total=len(texts))
    all_vecs: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = await brain.embed_texts(batch)
        all_vecs.extend(vecs)
        progress.advance(task, len(batch))

    result = {fid: vec for fid, vec in zip(ids, all_vecs)}

    if cache_path:
        Path(cache_path).write_text(json.dumps(result), encoding="utf-8")
        console.print(f"[dim]Saved embeddings to {cache_path}[/dim]")

    return result


# ── MAP phase ─────────────────────────────────────────────────────────────────

async def _map_chunk(
    brain: Brain,
    topic: str,
    chunk: list[dict],
    chunk_idx: int,
    prompt_template: str,
) -> list[dict]:
    """Run one MAP LLM call on a chunk. Returns list of EntityMention dicts."""
    lines = []
    for i, f in enumerate(chunk):
        text = f["finding_text"][:150].replace("\n", " ")
        lines.append(f"{i}: {text}")

    prompt = (
        prompt_template
        .replace("{topic}", topic)
        .replace("{n}", str(len(chunk)))
        .replace("{n_minus_1}", str(len(chunk) - 1))
        .replace("{findings}", "\n".join(lines))
    )

    raw = await brain._call("medium", prompt, temperature=0.2)
    try:
        parsed = _parse_json(raw)
        if not isinstance(parsed, list):
            raise ValueError("expected list")
    except Exception as exc:
        console.print(f"[yellow]  MAP chunk {chunk_idx} parse failed: {exc}[/yellow]")
        return []

    mentions: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        entity_type = str(item.get("entity_type", "")).strip().lower()
        if not name or entity_type not in ("model", "tool", "workflow", "hardware"):
            continue

        indices = item.get("finding_indices", [])
        finding_ids: list[str] = []
        source_urls: list[str] = []
        for idx in indices:
            if isinstance(idx, int) and 0 <= idx < len(chunk):
                finding_ids.append(chunk[idx]["id"])
                source_urls.append(chunk[idx]["source_url"])

        mentions.append({
            "entity_type": entity_type,
            "name": name,
            "claims": [str(c).strip() for c in item.get("claims", []) if str(c).strip()],
            "finding_ids": finding_ids,
            "source_urls": list(dict.fromkeys(source_urls)),
        })

    return mentions


async def _map_phase(
    brain: Brain,
    topic: str,
    findings: list[dict],
    chunk_size: int,
    progress: Progress,
) -> list[dict]:
    prompt_template = (_PROMPTS_DIR / "map_extract.md").read_text()
    chunks = [findings[i : i + chunk_size] for i in range(0, len(findings), chunk_size)]
    task = progress.add_task("MAP extraction…", total=len(chunks))
    all_mentions: list[dict] = []

    for i, chunk in enumerate(chunks):
        mentions = await _map_chunk(brain, topic, chunk, i, prompt_template)
        all_mentions.extend(mentions)
        progress.advance(task, 1)
        console.print(
            f"  [dim]Chunk {i + 1}/{len(chunks)}: extracted {len(mentions)} entity mentions[/dim]"
        )

    return all_mentions


# ── REDUCE phase ──────────────────────────────────────────────────────────────

def _normalize_name(name: str) -> str:
    """Normalize entity name for grouping and alias detection."""
    n = name.lower().strip()
    n = re.sub(r"[-_./]+", " ", n)
    for prefix in (
        "meta ", "openai ", "mistral ai ", "google ", "anthropic ",
        "hugging face ", "huggingface ", "microsoft ", "alibaba ",
    ):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return re.sub(r"\s+", " ", n).strip()


def _dedup_claims(claims: list[str]) -> list[str]:
    """Remove claims that are >75% trigram-similar to an already-accepted claim."""
    kept: list[str] = []
    for c in claims:
        if not any(trigram_similarity(c, k) > 0.75 for k in kept):
            kept.append(c)
    return kept


def _reduce_phase(mentions: list[dict]) -> list[dict]:
    """Merge all EntityMentions into MergedEntity list by (type, normalized_name)."""
    # Group by (entity_type, normalized_name)
    groups: dict[tuple[str, str], list[dict]] = {}
    for m in mentions:
        key = (m["entity_type"], _normalize_name(m["name"]))
        groups.setdefault(key, []).append(m)

    # Alias resolution: merge groups with trigram similarity > 0.80
    items = sorted(groups.items(), key=lambda x: (-len(x[1]), x[0][1]))
    canonical_keys: list[tuple[str, str]] = []
    key_map: dict[tuple[str, str], tuple[str, str]] = {}

    for key, _ in items:
        etype, norm = key
        canonical = next(
            (ck for ck in canonical_keys
             if ck[0] == etype and trigram_similarity(norm, ck[1]) > 0.80),
            None,
        )
        if canonical:
            key_map[key] = canonical
        else:
            canonical_keys.append(key)
            key_map[key] = key

    # Merge into canonical groups
    canonical_groups: dict[tuple[str, str], list[dict]] = {}
    for key, ms in groups.items():
        canonical_groups.setdefault(key_map[key], []).extend(ms)

    # Build MergedEntity list
    entities: list[dict] = []
    for (etype, norm), ms in canonical_groups.items():
        # Pick most frequently used display name
        name_counts: dict[str, int] = {}
        for m in ms:
            name_counts[m["name"]] = name_counts.get(m["name"], 0) + 1
        display_name = max(name_counts, key=lambda n: name_counts[n])

        all_claims: list[str] = []
        all_finding_ids: list[str] = []
        all_source_urls: list[str] = []
        for m in ms:
            all_claims.extend(m["claims"])
            all_finding_ids.extend(m["finding_ids"])
            all_source_urls.extend(m["source_urls"])

        entities.append({
            "entity_type": etype,
            "name": display_name,
            "normalized_name": norm,
            "claims": _dedup_claims(all_claims),
            "corroborating_sources": list(dict.fromkeys(all_source_urls)),
            "confidence_score": min(5, len(dict.fromkeys(all_source_urls))),
            "contradictions": [],
            "finding_ids": list(dict.fromkeys(all_finding_ids)),
            "novelty_scores": [],
            "prose_summary": "",
        })

    entities.sort(key=lambda e: (-e["confidence_score"], e["name"].lower()))
    return entities


# ── Novelty scoring ───────────────────────────────────────────────────────────

def _l2_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _normalize(v: list[float]) -> list[float]:
    n = _l2_norm(v)
    return [x / n for x in v] if n > 0 else v


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _novelty_scores(
    embeddings: dict[str, list[float]],
    findings: list[dict],
    top_n: int,
) -> tuple[dict[str, float], list[dict]]:
    """
    Score each finding by cosine distance from the global corpus centroid.
    Returns (score_map, top_n novel findings sorted by score desc).
    Uses numpy if available, otherwise pure Python.
    """
    # Exclude low-relevance findings (off-topic tangents) before scoring
    relevant = [f for f in findings if (f.get("relevance_score") or 0) >= 4]
    ids = [f["id"] for f in relevant if f["id"] in embeddings]
    if not ids:
        return {}, []

    if _NUMPY:
        mat = np.array([embeddings[fid] for fid in ids], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        mat_norm = mat / norms
        centroid = mat_norm.mean(axis=0)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        dists = (1.0 - mat_norm @ centroid_norm).tolist()
    else:
        # Pure Python path — ~1-2s for 1300 × 768-dim vectors
        vecs = [_normalize(embeddings[fid]) for fid in ids]
        dim = len(vecs[0])
        centroid_raw = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
        centroid_norm = _normalize(centroid_raw)
        dists = [1.0 - _dot(v, centroid_norm) for v in vecs]

    score_map = {fid: float(d) for fid, d in zip(ids, dists)}

    finding_by_id = {f["id"]: f for f in relevant}
    scored = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    novel = []
    for fid, score in scored[:top_n]:
        if fid in finding_by_id:
            novel.append({**finding_by_id[fid], "novelty_score": score})

    return score_map, novel


def _assign_novelty_to_entities(
    entities: list[dict],
    score_map: dict[str, float],
) -> None:
    """Attach per-finding novelty scores to each entity in-place."""
    for e in entities:
        e["novelty_scores"] = [score_map.get(fid, 0.0) for fid in e["finding_ids"]]


# ── Contradiction detection ───────────────────────────────────────────────────

async def _contradiction_scan(
    brain: Brain,
    entities: list[dict],
    progress: Progress,
) -> None:
    """
    For entities with > 5 claims, check high-similarity claim pairs for contradictions
    using the light model. Modifies entities in-place.
    """
    contra_template = (_PROMPTS_DIR / "contradiction_check.md").read_text()
    candidates = [e for e in entities if len(e["claims"]) > 5]
    task = progress.add_task("Contradiction scan…", total=len(candidates))

    for entity in candidates:
        claims = entity["claims"]
        pairs_checked = 0

        for i in range(len(claims)):
            if pairs_checked >= 20:
                break
            for j in range(i + 1, len(claims)):
                if pairs_checked >= 20:
                    break
                sim = trigram_similarity(claims[i], claims[j])
                # Only check pairs that overlap enough to be about the same aspect
                # but aren't identical (0.2 < sim < 0.75)
                if not (0.15 < sim < 0.75):
                    continue

                prompt = (
                    contra_template
                    .replace("{finding_a}", claims[i])
                    .replace("{source_a}", entity["name"])
                    .replace("{finding_b}", claims[j])
                    .replace("{source_b}", entity["name"])
                )
                raw = await brain._call("light", prompt, temperature=0.1)
                try:
                    result = _parse_json(raw)
                    if isinstance(result, dict) and result.get("contradicts"):
                        entity["contradictions"].append({
                            "claim_a": claims[i],
                            "claim_b": claims[j],
                            "explanation": result.get("explanation", ""),
                        })
                except Exception:
                    pass
                pairs_checked += 1

        progress.advance(task, 1)


# ── Synthesis phase ───────────────────────────────────────────────────────────

async def _synthesis_phase(
    brain: Brain,
    topic: str,
    entities: list[dict],
    progress: Progress,
) -> None:
    """Generate prose summaries for all entities in-place."""
    synth_template = (_PROMPTS_DIR / "entity_synthesis.md").read_text()
    task = progress.add_task("Synthesizing entities…", total=len(entities))

    for entity in entities:
        # Prefer claims from findings with highest novelty scores
        claims = entity["claims"]
        scores = entity.get("novelty_scores") or []
        if scores and len(scores) == len(claims):
            paired = sorted(zip(claims, scores), key=lambda x: x[1], reverse=True)
            top_claims = [c for c, _ in paired[:30]]
        else:
            top_claims = claims[:30]

        contradictions_text = "None"
        if entity["contradictions"]:
            lines = [
                f'- "{ct["claim_a"]}" vs "{ct["claim_b"]}" — {ct["explanation"]}'
                for ct in entity["contradictions"][:5]
            ]
            contradictions_text = "\n".join(lines)

        prompt = (
            synth_template
            .replace("{topic}", topic)
            .replace("{entity_type}", entity["entity_type"])
            .replace("{entity_name}", entity["name"])
            .replace("{confidence_score}", str(entity["confidence_score"]))
            .replace("{source_count}", str(len(entity["corroborating_sources"])))
            .replace("{claims}", "\n".join(f"- {c}" for c in top_claims))
            .replace("{contradictions}", contradictions_text)
        )

        entity["prose_summary"] = (await brain._call("heavy", prompt, temperature=0.4)).strip()
        progress.advance(task, 1)
        console.print(
            f"  [dim]{entity['entity_type']:10} {entity['name'][:55]}[/dim]"
        )


# ── Report rendering ──────────────────────────────────────────────────────────

def _confidence_bar(score: int) -> str:
    return "█" * score + "░" * (5 - score) + f" {score}/5"


def _render_report(
    output_path: str,
    topic: str,
    session_id: str,
    findings: list[dict],
    sources: list[dict],
    entities: list[dict],
    novel_findings: list[dict],
    top_n_novel: int,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    def ln(text: str = "") -> None:
        lines.append(text)

    # Pre-build lookup structures
    finding_index: dict[str, int] = {f["id"]: i + 1 for i, f in enumerate(findings)}
    fid_to_entities: dict[str, list[str]] = {}
    for e in entities:
        for fid in e["finding_ids"]:
            fid_to_entities.setdefault(fid, []).append(f"{e['entity_type']}:{e['name']}")
    novel_scores: dict[str, float] = {nf["id"]: nf["novelty_score"] for nf in novel_findings}

    # ── Header ──
    ln(f"# Second-Pass Analysis: {topic}")
    ln()
    ln(f"**Session:** `{session_id}`  ")
    ln(f"**Analysis date:** {now}  ")
    ln(f"**Total findings:** {len(findings)}  ")
    ln(f"**Sources:** {len(sources)}  ")
    ln(f"**Entities extracted:** {len(entities)}")
    ln()
    ln("---")
    ln()

    # ── Table of Contents ──
    ln("## Contents")
    ln()
    ln("1. [Model Leaderboard](#model-leaderboard)")
    ln("2. [High-Confidence Findings](#high-confidence-findings)")
    ln("3. [Novel & Unique Findings](#novel--unique-findings)")
    ln("4. [Models — Detailed Summaries](#models--detailed-summaries)")
    ln("5. [Tools & Frameworks](#tools--frameworks)")
    ln("6. [Workflow Patterns](#workflow-patterns)")
    ln("7. [Hardware & Infrastructure](#hardware--infrastructure)")
    ln("8. [Contradiction Register](#contradiction-register)")
    ln("9. [All Findings Index](#all-findings-index)")
    ln()
    ln("---")
    ln()

    # ── 1. Model Leaderboard ──
    ln("## Model Leaderboard")
    ln()
    models = [e for e in entities if e["entity_type"] == "model"]
    if models:
        ln("| Rank | Model | Confidence | Sources | Claims | Contradictions |")
        ln("|------|-------|------------|---------|--------|----------------|")
        for i, e in enumerate(models, 1):
            flag = " ⚠" if e["contradictions"] else ""
            ln(
                f"| {i} | **{e['name']}** | {_confidence_bar(e['confidence_score'])} | "
                f"{len(e['corroborating_sources'])} | {len(e['claims'])} | "
                f"{len(e['contradictions'])}{flag} |"
            )
    else:
        ln("*No model entities extracted.*")
    ln()
    ln("---")
    ln()

    # ── 2. High-Confidence Findings ──
    ln("## High-Confidence Findings")
    ln()
    ln("*Findings from entities backed by 4+ independent sources, or relevance score 5/5.*")
    ln()

    high_conf_fids: set[str] = set()
    for e in entities:
        if e["confidence_score"] >= 4:
            high_conf_fids.update(e["finding_ids"])
    for f in findings:
        if (f.get("relevance_score") or 0) >= 5:
            high_conf_fids.add(f["id"])

    finding_by_id = {f["id"]: f for f in findings}
    high_conf = sorted(
        [finding_by_id[fid] for fid in high_conf_fids if fid in finding_by_id],
        key=lambda f: -(f.get("relevance_score") or 0),
    )

    if high_conf:
        for f in high_conf[:100]:
            num = finding_index.get(f["id"], "?")
            tags = ", ".join(fid_to_entities.get(f["id"], []))
            ln(f"**[#{num}]** {f['finding_text']}")
            ln(
                f"*{f['source_url']} · {f.get('source_type', '')} · "
                f"relevance {f.get('relevance_score', '?')}/5"
                + (f" · {tags}" if tags else "") + "*"
            )
            ln()
    else:
        ln("*No high-confidence findings found.*")
        ln()

    ln("---")
    ln()

    # ── 3. Novel & Unique Findings ──
    ln("## Novel & Unique Findings")
    ln()
    if novel_findings:
        ln(
            f"*Top {min(top_n_novel, len(novel_findings))} findings by cosine distance "
            f"from the corpus centroid — these are the most peripheral/unique findings.*"
        )
        ln()
        for i, f in enumerate(novel_findings[:top_n_novel], 1):
            score = f.get("novelty_score", 0.0)
            num = finding_index.get(f["id"], "?")
            ln(f"**[#{num}]** *(novelty {score:.3f})* {f['finding_text']}")
            ln(f"*{f['source_url']} · {f.get('source_type', '')}*")
            ln()
    else:
        ln("*Novelty scoring was skipped (`--no-embed`).*")
        ln()

    ln("---")
    ln()

    # ── 4-7. Per-category detailed summaries ──
    categories = [
        ("model",    "## Models — Detailed Summaries"),
        ("tool",     "## Tools & Frameworks"),
        ("workflow", "## Workflow Patterns"),
        ("hardware", "## Hardware & Infrastructure"),
    ]

    for etype, header in categories:
        ln(header)
        ln()
        cat = [e for e in entities if e["entity_type"] == etype]
        if not cat:
            ln(f"*No {etype} entities extracted.*")
            ln()
            ln("---")
            ln()
            continue

        for e in cat:
            ln(f"### {e['name']}")
            ln()
            ln(
                f"**Confidence:** {_confidence_bar(e['confidence_score'])} &nbsp;|&nbsp; "
                f"**Sources:** {len(e['corroborating_sources'])} &nbsp;|&nbsp; "
                f"**Claims:** {len(e['claims'])}"
            )
            ln()

            if e["prose_summary"]:
                ln(e["prose_summary"])
                ln()

            if e["contradictions"]:
                ln(f"**⚠ Contradictions detected ({len(e['contradictions'])}):**")
                for ct in e["contradictions"]:
                    a = ct["claim_a"][:120]
                    b = ct["claim_b"][:120]
                    ln(f'- *"{a}"* **vs** *"{b}"*')
                    if ct["explanation"]:
                        ln(f'  → {ct["explanation"]}')
                ln()

            ln("**Sources:**")
            for url in e["corroborating_sources"][:10]:
                ln(f"- {url}")
            ln()

            # Finding cross-references
            ref_nums = [
                f"#{finding_index[fid]}"
                for fid in e["finding_ids"][:30]
                if fid in finding_index
            ]
            if ref_nums:
                ln(f"**Findings:** {', '.join(ref_nums)}")
                ln()

        ln("---")
        ln()

    # ── 8. Contradiction Register ──
    ln("## Contradiction Register")
    ln()

    conflicting = [f for f in findings if f.get("conflicting")]
    all_contradictions: list[dict] = []
    for e in entities:
        for ct in e["contradictions"]:
            all_contradictions.append({**ct, "entity": e["name"]})

    if all_contradictions:
        ln("### Detected via LLM scan")
        ln()
        ln("| Entity | Claim A | Claim B | Notes |")
        ln("|--------|---------|---------|-------|")
        for ct in all_contradictions:
            a = ct["claim_a"][:80].replace("|", "\\|")
            b = ct["claim_b"][:80].replace("|", "\\|")
            note = ct.get("explanation", "")[:80].replace("|", "\\|")
            ln(f"| **{ct['entity']}** | {a} | {b} | {note} |")
        ln()
    else:
        ln("*No contradictions detected by LLM scan.*")
        ln()

    if conflicting:
        ln(f"### Pre-flagged during live run ({len(conflicting)} findings)")
        ln()
        for f in conflicting:
            num = finding_index.get(f["id"], "?")
            ln(f"- **[#{num}]** {f['finding_text']}  ")
            ln(f"  *{f['source_url']}*")
        ln()

    ln("---")
    ln()

    # ── 9. All Findings Index ──
    ln("## All Findings Index")
    ln()
    ln(f"*{len(findings)} findings, ordered by discovery cycle.*")
    ln()

    for i, f in enumerate(findings, 1):
        tags = ", ".join(fid_to_entities.get(f["id"], ["—"]))
        novelty_str = ""
        if f["id"] in novel_scores:
            novelty_str = f" · novelty {novel_scores[f['id']]:.3f}"

        text = f["finding_text"]
        display = text[:200] + ("…" if len(text) > 200 else "")
        ln(f"**#{i}** {display}")
        ln(
            f"*{f['source_url']} · {f.get('source_type', '')} · "
            f"relevance {f.get('relevance_score', '?')}/5 · {tags}{novelty_str}*"
        )
        ln()

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    console.print(f"\n[bold green]Written → {output_path}[/bold green]")


# ── Main ──────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(Path(args.config).read_text())
    brain = Brain(config, str(_PROMPTS_DIR))

    session_id, topic = _load_session(args.db, args.session)
    console.print(f"[bold]Session:[/bold] {session_id}  [bold]Topic:[/bold] {topic}")

    findings = _load_findings_full(args.db, session_id)
    sources = _load_sources_full(args.db, session_id)
    console.print(
        f"Loaded [bold]{len(findings)}[/bold] findings from [bold]{len(sources)}[/bold] sources"
    )

    if not findings:
        raise SystemExit("No findings found for this session.")

    intermediate_path = args.output.replace(".md", "_entities.json")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # ── EMBED ──
        embeddings: dict[str, list[float]] = {}
        novel_findings: list[dict] = []
        score_map: dict[str, float] = {}

        if not args.no_embed:
            embeddings = await _embed_all(brain, findings, args.embed_cache, progress)
            score_map, novel_findings = _novelty_scores(embeddings, findings, args.top_n_novel)
            console.print(
                f"Novelty scored [bold]{len(score_map)}[/bold] findings; "
                f"top [bold]{len(novel_findings)}[/bold] surfaced"
            )

        # ── MAP + REDUCE ──
        entities: list[dict] = []

        if not args.synthesis_only:
            all_mentions = await _map_phase(brain, topic, findings, args.chunk_size, progress)
            console.print(f"MAP: [bold]{len(all_mentions)}[/bold] entity mentions extracted")

            entities = _reduce_phase(all_mentions)
            console.print(f"REDUCE: [bold]{len(entities)}[/bold] unique entities after merge")

            if score_map:
                _assign_novelty_to_entities(entities, score_map)

            # ── CONTRADICTIONS ──
            await _contradiction_scan(brain, entities, progress)
            n_contra = sum(len(e["contradictions"]) for e in entities)
            console.print(f"Contradictions detected: [bold]{n_contra}[/bold]")

            # Save intermediate so --synthesis-only can resume
            Path(intermediate_path).write_text(
                json.dumps(entities, indent=2, default=str), encoding="utf-8"
            )
            console.print(f"[dim]Intermediate saved → {intermediate_path}[/dim]")

        else:
            if Path(intermediate_path).exists():
                entities = json.loads(Path(intermediate_path).read_text())
                console.print(
                    f"[dim]Loaded {len(entities)} entities from {intermediate_path}[/dim]"
                )
            else:
                console.print(
                    f"[yellow]--synthesis-only: no intermediate file at {intermediate_path}[/yellow]"
                )

        # ── SYNTHESIS ──
        if entities:
            await _synthesis_phase(brain, topic, entities, progress)

    # ── RENDER ──
    _render_report(
        args.output, topic, session_id, findings, sources,
        entities, novel_findings, args.top_n_novel,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Second-pass analysis of a LocalResearch session")
    p.add_argument("--db", default="researcher.db")
    p.add_argument("--session", default=None, help="Session ID (default: most recent)")
    p.add_argument("--output", "-o", default="second_pass.md")
    p.add_argument("--config", default=str(_CONFIG_PATH))
    p.add_argument("--chunk-size", type=int, default=100, dest="chunk_size",
                   help="Findings per MAP chunk (default: 100)")
    p.add_argument("--top-n-novel", type=int, default=50, dest="top_n_novel",
                   help="Novel findings to surface in report (default: 50)")
    p.add_argument("--embed-cache", default=None, dest="embed_cache",
                   help="Path to .json file for saving/loading embeddings across runs")
    p.add_argument("--no-embed", action="store_true", dest="no_embed",
                   help="Skip embedding and novelty scoring entirely")
    p.add_argument("--synthesis-only", action="store_true", dest="synthesis_only",
                   help="Skip MAP/REDUCE, load existing intermediate JSON and re-synthesize")
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()

"""
All LLM interactions.  Uses LiteLLM so any Ollama model can be swapped via config.

Model routing:
  heavy  → heavy summarization (30B)
  medium → query gen, reflection, summary updates (7–14B)
  light  → novelty/contradiction checks (3B)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import litellm

log = logging.getLogger(__name__)

# Suppress litellm's noisy success prints
litellm.success_callback = []
litellm.failure_callback = []
litellm.set_verbose = False


# ── JSON parsing helpers ──────────────────────────────────────────────────────

def _parse_json(text: str) -> Any:
    """Try to parse JSON from an LLM response that may have extra prose."""
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Find JSON array
    arr = re.search(r"(\[[\s\S]*\])", text)
    if arr:
        try:
            return json.loads(arr.group(1))
        except json.JSONDecodeError:
            pass

    # Find JSON object
    obj = re.search(r"(\{[\s\S]*\})", text, re.DOTALL)
    if obj:
        try:
            return json.loads(obj.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:300]!r}")


# ── Trigram similarity for query dedup ───────────────────────────────────────

def _trigrams(s: str) -> set[str]:
    s = s.lower().strip()
    return {s[i : i + 3] for i in range(max(0, len(s) - 2))} if len(s) >= 3 else set()


def trigram_similarity(a: str, b: str) -> float:
    t1, t2 = _trigrams(a), _trigrams(b)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


# ── Brain ─────────────────────────────────────────────────────────────────────

class Brain:
    def __init__(self, config: dict, prompts_dir: str):
        self.config = config
        self._prompts = self._load_prompts(Path(prompts_dir))
        self._llm_log: list[dict] = []

    def _load_prompts(self, d: Path) -> dict[str, str]:
        names = ["query_gen", "summarize", "novelty_check", "contradiction_check", "reflect", "synthesize"]
        return {n: (d / f"{n}.md").read_text() for n in names}

    # ── LLM call ─────────────────────────────────────────────────────────

    async def _call(
        self,
        model_key: str,
        prompt: str,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        model = self.config["models"][model_key]
        temp = temperature if temperature is not None else self.config["temperature"]
        api_base = self.config["ollama_base_url"]
        max_retries = self.config.get("llm_max_retries", 3)
        timeout = self.config.get("llm_timeout", 180)

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            api_base=api_base,
            timeout=timeout,
            stream=False,
        )
        if seed is not None:
            kwargs["seed"] = seed

        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(max_retries):
            try:
                resp = await litellm.acompletion(**kwargs)
                text = resp.choices[0].message.content or ""
                if self.config.get("log_llm_calls"):
                    self._llm_log.append(
                        {"model": model, "prompt_head": prompt[:200], "response_head": text[:200]}
                    )
                return text
            except Exception as exc:
                last_exc = exc
                wait = self.config.get("backoff_base", 2.0) ** attempt
                log.warning(
                    "LLM call failed (attempt %d/%d, model=%s): %s — retrying in %.1fs",
                    attempt + 1, max_retries, model, exc, wait,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)

        raise last_exc

    # ── Public API ────────────────────────────────────────────────────────

    async def generate_queries(
        self,
        topic: str,
        prior_queries: list[str],
        running_summary: str,
        knowledge_gaps: list[str],
        n: int = 7,
    ) -> list[str]:
        max_age = f"{self.config.get('max_age_months', 6)} months"
        prompt = self._prompts["query_gen"].format(
            topic=topic,
            n=n,
            prior_queries="\n".join(f"- {q}" for q in prior_queries) or "None",
            knowledge_gaps="\n".join(f"- {g}" for g in knowledge_gaps) or "None",
            max_age=max_age,
        )
        raw = await self._call("medium", prompt)
        try:
            queries = _parse_json(raw)
            if not isinstance(queries, list):
                raise ValueError("expected list")
        except Exception as exc:
            log.warning("query_gen parse failed: %s — falling back to line split", exc)
            queries = [line.strip().strip('"') for line in raw.splitlines() if line.strip()]

        # Filter out queries too similar to prior ones
        threshold = self.config.get("query_trigram_similarity", 0.70)
        all_prior = list(prior_queries)
        accepted: list[str] = []
        for q in queries:
            if not isinstance(q, str) or not q.strip():
                continue
            if any(trigram_similarity(q, p) >= threshold for p in all_prior):
                log.debug("Skipping similar query: %r", q)
                continue
            accepted.append(q)
            all_prior.append(q)

        return accepted[:n]

    async def summarize(
        self,
        content: str,
        source_url: str,
        source_type: str,
        source_date: Optional[str],
        topic: str,
        running_summary: str,
    ) -> dict:
        max_chars = self.config.get("max_content_chars", 8000)
        prompt = self._prompts["summarize"].format(
            topic=topic,
            running_summary=running_summary or "No findings yet.",
            source_url=source_url,
            source_type=source_type,
            source_date=source_date or "Unknown",
            content=content[:max_chars],
        )
        raw = await self._call("heavy", prompt)
        try:
            result = _parse_json(raw)
            if not isinstance(result, dict):
                raise ValueError("expected dict")
        except Exception as exc:
            log.warning("summarize parse failed for %s: %s", source_url, exc)
            return {"findings": [], "relevance_score": 1, "quality_type": "unknown", "new_leads": []}

        return {
            "findings": result.get("findings", []),
            "relevance_score": int(result.get("relevance_score", 1)),
            "quality_type": result.get("quality_type", "unknown"),
            "new_leads": result.get("new_leads", []),
        }

    async def update_running_summary(
        self, topic: str, current_summary: str, new_findings: list[str]
    ) -> str:
        if not new_findings:
            return current_summary
        findings_text = "\n".join(f"- {f}" for f in new_findings)
        prompt = (
            f"Update the running research summary for topic: {topic}\n\n"
            f"Current summary:\n{current_summary or 'No summary yet.'}\n\n"
            f"New findings to integrate:\n{findings_text}\n\n"
            f"Write a compressed summary (max 300 words) that:\n"
            f"1. Captures the most important findings so far\n"
            f"2. Integrates the new findings\n"
            f"3. Highlights key themes and patterns\n"
            f"4. Notes any important gaps or conflicts\n\n"
            f"Be concise and factual. Output plain text only."
        )
        return (await self._call("medium", prompt)).strip()

    async def reflect(
        self, topic: str, running_summary: str, prior_queries: list[str], cycle_count: int
    ) -> dict:
        prompt = self._prompts["reflect"].format(
            topic=topic,
            running_summary=running_summary or "No findings yet.",
            prior_queries="\n".join(f"- {q}" for q in prior_queries) or "None",
            cycle_count=cycle_count,
        )
        raw = await self._call("medium", prompt)
        try:
            result = _parse_json(raw)
            if not isinstance(result, dict):
                raise ValueError("expected dict")
        except Exception as exc:
            log.warning("reflect parse failed: %s", exc)
            return {
                "well_covered": [],
                "knowledge_gaps": [],
                "new_queries": [],
                "saturated": False,
            }

        # Deduplicate new_queries against prior_queries
        threshold = self.config.get("query_trigram_similarity", 0.70)
        new_q: list[str] = []
        for q in result.get("new_queries", []):
            if isinstance(q, str) and q.strip():
                if not any(trigram_similarity(q, p) >= threshold for p in prior_queries):
                    new_q.append(q)

        return {
            "well_covered": result.get("well_covered", []),
            "knowledge_gaps": result.get("knowledge_gaps", []),
            "new_queries": new_q[:3],
            "saturated": bool(result.get("saturated", False)),
        }

    async def synthesize(self, topic: str, findings: list[dict]) -> str:
        lines = [
            f"- {f['finding_text']} (source: {f['source_url']})"
            for f in findings[:60]
        ]
        prompt = self._prompts["synthesize"].format(
            topic=topic,
            findings_list="\n".join(lines) or "No findings yet.",
        )
        return (await self._call("heavy", prompt)).strip()

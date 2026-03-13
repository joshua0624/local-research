# LocalResearch

Async local LLM research assistant. Searches web/Reddit/GitHub, summarizes, deduplicates, builds a growing findings document. Runs on Ollama (Qwen 30B) on M4 Max.

## Key Docs

- `docs/BUILD_SPEC.md` — Full build spec with architecture, phases, and decisions
- `docs/FEEDBACK_CONDENSED.md` — Reviewer feedback (already merged into BUILD_SPEC)

## Architecture

Three components: **Orchestrator** (async state machine), **Fetcher** (source-specific async scrapers), **Brain** (LLM calls). Single writer actor serializes all DB/file I/O. Producer-consumer pipeline: fetchers → asyncio.Queue → LLM consumer.

## Stack

Python 3.12+, asyncio, Ollama (Qwen 30B + nomic-embed-text), LiteLLM, httpx, PRAW, trafilatura (Crawl4AI fallback), SearXNG, ChromaDB, SQLite (WAL mode), Rich, dateparser, aiolimiter.

## Critical Design Rules

- **Never pass full findings doc to LLM.** Use a compressed Running Research Summary for prompt context.
- **Structured state, rendered markdown.** Store findings in SQLite/JSON. Render `findings.md` from scratch each cycle.
- **Dedup = URL check + embedding similarity + contradiction check.** Similarity >0.85 triggers a small-model contradiction check before discarding.
- **All I/O through writer actor.** Single async coroutine handles all DB writes, ChromaDB upserts, and file writes via queue.
- **Model routing:** 30B for summarization, 7-14B for query gen, 3B for novelty/contradiction checks.

## Build Phases

1. Core loop with web search only (orchestrator, brain, SearXNG, web fetcher, state store, CLI)
2. Reddit & GitHub fetchers
3. Semantic dedup & steering
4. Polish (session resume, clustering, executive summary)

# LocalResearch — Build Specification

## What This Is

A local LLM-powered research assistant that iteratively searches the web, Reddit, and GitHub, then summarizes, deduplicates, and builds a growing findings document in markdown. Runs entirely on a MacBook Pro M4 Max with Ollama. Designed for interactive 30-minute to multi-hour sessions where the user steers the research in real-time via CLI.

**Good for:** Surveying fast-moving fields, finding scattered community knowledge, building state-of-the-art overviews, discovering unknown projects/tools.

**Not good for:** Deep multi-source reasoning, paywalled content, extremely niche topics with little public content.

---

## Infrastructure

- **Hardware:** MacBook Pro M4 Max, 36GB unified memory
- **Model serving:** Ollama
- **Primary model:** Qwen 3/3.5 ~30B at Q4_K_M (~20GB + context headroom, 15-25 tok/s)
- **Embedding model:** nomic-embed-text via Ollama (8192 context, same ecosystem, better technical terminology than MiniLM)

### Model Routing

Use smaller models for lightweight tasks to save compute and thermal cycles:

| Task | Model Size |
|------|-----------|
| Heavy summarization | 30B |
| Query generation | 7-14B |
| Novelty/contradiction check | 3B or 1-2B |

### Libraries

| Purpose | Library |
|---------|---------|
| LLM interface | LiteLLM |
| Reddit API | PRAW (with JSON endpoint fallback) |
| GitHub API | PyGithub or httpx |
| Web scraping (primary) | trafilatura or BeautifulSoup |
| Web scraping (JS fallback) | Crawl4AI (only when primary returns empty) |
| Search engine | SearXNG (self-hosted Docker) — not duckduckgo-search (rate-limits aggressively) |
| Vector store | ChromaDB (or FAISS in-memory with checkpoints if Chroma gets slow) |
| Embeddings | nomic-embed-text via Ollama |
| Persistence | SQLite (WAL mode) |
| CLI display | Rich |
| Async HTTP | httpx.AsyncClient (connection pooling, keep-alive) |
| Date parsing | dateparser |
| Rate limiting | aiolimiter |

### API Keys (all free)

- Reddit API credentials (reddit.com/prefs/apps)
- GitHub personal access token (5,000 req/hr authenticated vs 60 unauthenticated)
- SearXNG: self-hosted, no key needed

---

## Architecture

Three components:

| Component | Role |
|-----------|------|
| **Orchestrator** | Async state machine managing the research loop, state, and steering input |
| **Fetcher** | Source-specific async scrapers returning clean text with metadata |
| **Brain** | LLM interactions: query gen, summarization, novelty check, reflection |

### File Structure

```
researcher/
├── main.py                  # Entry point, CLI, steering input handler
├── orchestrator.py          # Async research loop state machine
├── brain.py                 # All LLM interactions
├── fetcher/
│   ├── __init__.py
│   ├── base.py              # Abstract fetcher interface
│   ├── reddit.py            # PRAW + JSON endpoints
│   ├── github.py            # GitHub API
│   ├── web.py               # trafilatura/BS4 primary, Crawl4AI fallback
│   └── search.py            # SearXNG interface
├── dedup.py                 # Two-layer dedup: URL + semantic with contradiction check
├── findings.py              # Structured state store + markdown renderer
├── state.py                 # SQLite persistence, provenance tracking
├── writer.py                # Single async writer actor for all DB/file I/O
├── config.yaml              # Default settings
└── prompts/
    ├── query_gen.md
    ├── summarize.md
    ├── novelty_check.md
    ├── contradiction_check.md
    ├── reflect.md
    └── synthesize.md
```

---

## The Research Loop

### Startup

```
python main.py --topic "local LLM coding agents" --max-age 3m

Optional flags:
  --sources reddit,github,web    (default: all)
  --subreddits ollama,LocalLLaMA
  --github-orgs langchain-ai
  --max-age 3m                   (default: 6m)
  --output findings.md
  --resume session-id
  --deterministic                (low temp, fixed seed for reproducibility)
```

### Phase 1: Initial Query Generation

LLM generates 5-8 diverse search queries from the research topic. Use a 7-14B model for this.

### Phase 2: Search & Discover

For each query, search via SearXNG. Filter by date, classify by source type (Reddit/GitHub/web), check against seen-URLs in SQLite, add new URLs to the async processing queue.

### Phase 3: Fetch & Read (Async Producers)

Async producer coroutines fetch URLs concurrently via the appropriate fetcher. Each returns clean text + metadata (date, author, source URL, source type). Results go into an `asyncio.Queue` for the consumer.

**By source:**

| Source | Fetcher | What to Extract | Rate Limit |
|--------|---------|----------------|------------|
| Reddit | PRAW + `.json` endpoint | Post title/body, top 30-50 comments (score > 2), post score, date, subreddit | 60 req/min, 1s delay |
| GitHub | REST API | README, repo metadata (stars, language, pushed_at, description), recent issues + top comments | 5,000 req/hr, 0.5s courtesy delay |
| Web | trafilatura/BS4, Crawl4AI fallback | Article title, author, date, body as markdown | 2-3s between fetches |

Cache Reddit/GitHub ETags and If-Modified-Since headers to skip unchanged content. Record GitHub commit SHA via `pushed_at` to detect drift across sessions.

### Phase 4: Summarize & Extract (Async Consumer)

The LLM consumer processes fetched text from the queue while producers continue fetching. For each source, the 30B model extracts:
- Key findings (bullet points)
- Relevance score (1-5)
- Source quality type (research, tutorial, discussion, opinion)
- New search leads (URLs, project names, topics worth following)

**Context management:** Pass only the compressed Running Research Summary (executive summary + one-sentence findings list) as context — never the full master document. This keeps prompt size stable across long sessions.

The Running Research Summary also serves as the primary input to the final `synthesize` call. The synthesize prompt receives the full running summary (which covers all findings) plus the top-N highest-relevance findings as grounding examples. `synthesize_top_n` (default 20) is configurable. This avoids the context scaling problem for long sessions with hundreds of findings.

### Phase 5: Novelty Check (Two-Layer Dedup)

**Layer 1 — URL dedup:** SQLite lookup. Skip already-visited URLs. Fast, zero-cost.

**Layer 2 — Semantic dedup with contradiction detection:**
1. Compute embedding of each finding via nomic-embed-text
2. Compare against ChromaDB collection of all existing findings
3. Below 0.70 similarity: novel — include
4. 0.70-0.85: similar but potentially new angle — include with note
5. Above 0.85: run a fast LLM check (3B model): *"Does the new finding contradict, update, or add critical nuance to the existing one?"*
   - If yes: keep both, mark as conflicting with timestamps and source links
   - If no: discard as duplicate

Batch embedding operations 32-64 at a time. Use deterministic IDs (hash of normalized text + source ID) to prevent reprocessing.

### Phase 6: Update State & Render

The master document is stored as **structured JSON/SQLite**, not edited markdown. Each novel finding is written via the single async writer actor (see Concurrency below). After writes complete, render `findings.md` from scratch.

**Finding record schema:**
- finding_text, source_url, fetch_timestamp, source_hash, source_type
- commit_ref (GitHub), embedding_id, relevance_score, quality_type
- section/theme (assigned during clustering)

Keep an append-only event log for rollback.

### Phase 7: Reflect & Generate New Queries

Pass the **Running Research Summary** (not the full document) + original topic + list of prior queries to the LLM. It outputs:
- What's well-covered
- Remaining knowledge gaps
- 2-3 new queries to fill gaps
- Whether the topic is saturated

Before executing new queries, check trigram similarity against all prior queries. Skip if >70% similar. After 3 failed attempts to generate a distinct query, signal saturation.

If new queries: loop to Phase 2. If saturated: notify user, pause for steering.

### Phase 8: Steering Input

Between cycles, check for user commands via async stdin monitoring:

```
focus {topic}    — Narrow to a subtopic
ignore {topic}   — Exclude from future results
add {source}     — Add URL or subreddit
broaden          — Expand search scope
status           — Print progress summary
synthesize       — Regenerate Executive Summary
pause / go       — Pause/resume loop
done             — Final synthesis and exit
```

---

## Output Document Structure

Rendered from structured state each cycle:

```markdown
# Research Findings: {topic}

**Session:** {start} — {last_updated} | **Sources:** {count} | **Config:** max_age={max_age}

## Executive Summary
{Regenerated every 5 cycles or on user request}

## Key Findings

### {Theme cluster — auto-generated after 10+ findings}
- **Finding:** {insight}
  - *Source:* [{title}]({url}) ({type}, {date})

## Notable Projects & Tools
| Project | URL | Stars | What It Does | Relevance |
|---------|-----|-------|-------------|-----------|

## Leads to Follow Up
- [ ] {lead} — mentioned in {source}

## Research Log
### Cycle N — {timestamp}
- Queries: {list} | Sources: {count} | Novel: {count} | Dupes: {count}

## Sources Consulted
{All URLs with date and novel-finding flag}
```

Section clustering: first findings go into a flat list. After 10+ accumulate, the LLM clusters them into themed sections. New findings route to the appropriate section or create a new one.

---

## Async Pipeline Design

The system uses an async producer-consumer architecture to keep the GPU/NPU busy while network I/O is in flight.

```
[SearXNG] ──→ [Fetcher Pool (async producers)] ──→ asyncio.Queue ──→ [LLM Consumer]
                                                                          │
                                                                    [Writer Actor] ──→ SQLite / ChromaDB / findings.md
```

**Writer actor:** A single coroutine consumes write requests from an `asyncio.Queue`. All DB inserts, ChromaDB upserts, and file writes go through it. This eliminates race conditions. Use atomic file writes (tmp → rename) for markdown output. SQLite in WAL mode with short transactions.

---

## Rate Limiting & Safety

- Per-source token-bucket limiters via `aiolimiter` or `asyncio.Semaphore` + per-host counters
- Exponential backoff with jitter + circuit-breaker: after N 429/5xx responses, pause host for T minutes
- Honor `robots.txt` and `Retry-After` headers
- Config toggle for aggressive vs. polite modes
- Per-domain allow/disallow list; skip paywalled content by default, log for manual review
- Headless browsers (Crawl4AI fallback) sandboxed with strict CPU/memory/timeout limits; hard kill after 20s

### Source Quality Filters

Skip before processing:
- Reddit posts with score < 3
- GitHub repos with 0 stars and no README
- Web pages with < 200 words of content
- Failed/errored fetches

### Saturation Detection

- Last 10 sources yielded 0 novel findings → saturated
- Last 5 cycles all had < 2 novel findings → diminishing returns
- On detection: notify user, pause, wait for steering input

---

## Date Filtering (Three Levels)

1. **Search engine (coarse):** SearXNG date parameters, PRAW `time_filter`, GitHub `pushed:>` qualifier
2. **Metadata check (precise):** After fetch, before LLM — check post date / `pushed_at` / `<time>` tags / URL date patterns / HTTP Last-Modified against `max_age`. Use `dateparser` with multiple heuristics. Normalize all to UTC.
3. **Content-level (fallback):** Prompt the LLM to flag apparently outdated content when metadata is missing. Soft filter — flags, doesn't block.

---

## Data Quality

### Provenance

Every finding stores immutably: source URL, fetch timestamp, source hash, commit ref (GitHub), embedding ID, relevance score, quality type. This is non-negotiable for traceability.

### Text Normalization

- Unicode NFC normalization on all ingested text
- Language detection via `fastlangdetect`; optionally route non-English to translation before summarization

### Embedding Index Maintenance

- Deterministic IDs (hash of normalized text + source ID) prevent duplicates on reprocessing
- Background compaction/re-clustering of vector index
- Retention policy: keep last N months or top M findings; archive older to cold store

---

## Observability

### Logging

Log for every LLM call: prompt, model name, temperature, seed (if set), context snippets. Store alongside generated output.

### Metrics

Emit per-cycle: sources/sec, novel_per_cycle, avg_fetch_latency, error rates. Push to local Prometheus or log file.

### Testing

- Unit tests for fetchers using recorded fixtures
- Integration tests with playback mode (recorded HTTP responses)
- Fuzz tests for malformed HTML handling

---

## Build Order

### Phase 1: Core Loop — Web Search Only

- Async orchestrator state machine
- Brain module (query gen, summarize, novelty check prompts)
- SearXNG search integration
- Web fetcher (trafilatura/BS4 primary, Crawl4AI fallback)
- Structured state store + markdown renderer
- URL-level dedup via SQLite
- Writer actor for serialized I/O
- Basic CLI with Rich status output
- Config file, date filtering
- Running Research Summary for context management

**Milestone:** Working system that researches via web search and reads articles. Tune prompts, validate the loop.

### Phase 2: Reddit & GitHub Fetchers

- Reddit fetcher (PRAW + JSON comments)
- GitHub fetcher (API: repos, READMEs, issues)
- Source-type classification from URLs
- ETag/If-Modified-Since caching
- Source-specific date filtering

**Milestone:** System routes to the right fetcher by URL. All three source types work.

### Phase 3: Semantic Dedup & Steering

- ChromaDB + nomic-embed-text integration
- Batched embedding computation
- Contradiction detection via small model
- Async steering input handler + command parsing
- Saturation detection
- Knowledge gap reflection

**Milestone:** Intelligent dedup and interactive steering. This is where it becomes a research assistant.

### Phase 4: Polish

- Session persistence (resume interrupted sessions)
- Executive Summary regeneration
- Themed section clustering
- Notable Projects table auto-generation
- Leads to Follow Up extraction
- Per-cycle research log with stats

**Milestone:** Complete v1.

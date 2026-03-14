"""
Async research loop — state machine coordinating search, fetch, LLM, and writes.

Pipeline:
  SearXNG → fetch_queue → [WebFetcher producers]
                       → asyncio.Queue → [LLM consumer]
                                      → WriterActor → SQLite / findings.md
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .brain import Brain
from .dedup import Disposition, DedupeResult, SemanticDedup, URLDedup
from .fetcher import FetchResult, GitHubFetcher, RedditFetcher, SearXNGSearcher, WebFetcher
from .fetcher.circuit_breaker import CircuitBreaker
from .findings import FindingsStore
from .state import StateReader, make_finding_id
from .writer import WriterActor

log = logging.getLogger(__name__)
console = Console()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class ResearchOrchestrator:
    def __init__(
        self,
        topic: str,
        config: dict,
        session_id: str,
        output_path: str,
        prompts_dir: str,
        db_path: str,
        deterministic: bool = False,
        is_resume: bool = False,
        quiet: bool = False,
    ):
        self.topic = topic
        self.config = config
        self.session_id = session_id
        self.output_path = output_path
        self.deterministic = deterministic
        self.is_resume = is_resume
        self._quiet = quiet

        # Components
        self.brain = Brain(config, prompts_dir)
        self.writer = WriterActor(db_path)
        self.reader = StateReader(db_path)
        self.dedup = URLDedup(self.reader)
        self.semantic_dedup = SemanticDedup(self.brain, config, session_id)
        self.searcher = SearXNGSearcher(config["searxng_url"])

        cb = CircuitBreaker(
            threshold=config.get("circuit_breaker_threshold", 5),
            pause_minutes=config.get("circuit_breaker_pause_minutes", 10),
        )
        self.web_fetcher = WebFetcher(
            rate_limit=config.get("web_fetch_delay", 2.5),
            max_concurrent=config.get("max_concurrent_fetches", 5),
            max_age_months=config.get("max_age_months", 6),
            circuit_breaker=cb,
        )
        self.reddit_fetcher = RedditFetcher(
            client_id=config.get("reddit_client_id"),
            client_secret=config.get("reddit_client_secret"),
            user_agent=config.get("reddit_user_agent", "LocalResearcher/1.0"),
            rate_limit_per_min=config.get("reddit_rate_limit", 60.0),
            max_age_months=config.get("max_age_months", 6),
            circuit_breaker=cb,
        )
        self.github_fetcher = GitHubFetcher(
            token=config.get("github_token"),
            courtesy_delay=config.get("github_courtesy_delay", 0.5),
            max_age_months=config.get("max_age_months", 6),
            circuit_breaker=cb,
        )
        self.store = FindingsStore()

        # Which source types to fetch (default: all)
        self._enabled_sources: set[str] = set(config.get("sources", ["web", "reddit", "github"]))

        # State
        self.cycle_num: int = 0
        self.queries: list[str] = []
        self.prior_queries: list[str] = []
        self.knowledge_gaps: list[str] = []
        self.saturated: bool = False
        self.paused: bool = False
        self.done: bool = False

        # Saturation tracking
        self._consecutive_zero_novel: int = 0  # sources with 0 novel findings
        self._novel_per_cycle: list[int] = []

        # Steering
        self._steering_queue: asyncio.Queue[str] = asyncio.Queue()
        self._focus: Optional[str] = None
        self._ignore: list[str] = []

        # Live panel tracking
        self._current_phase: str = "idle"
        self._current_url: Optional[str] = None
        self._urls_remaining: int = 0
        self._urls_total: int = 0
        self._cycle_start_time: Optional[datetime] = None
        self._pending_commands: list[str] = []

        # Start time
        self._start_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.writer.start()

        console.print(f"\n[bold cyan]LocalResearcher[/bold cyan] — topic: [bold]{self.topic}[/bold]")
        console.print(f"Session ID: {self.session_id}")
        console.print(f"Output: {self.output_path}")
        console.print(f"Sources: {', '.join(sorted(self._enabled_sources))}\n")

        # Seed extra sources from CLI flags
        self._seed_urls: list[dict] = []
        for sub in self.config.get("seed_subreddits", []):
            sub = sub.lstrip("r/").strip()
            self._seed_urls.append({
                "url": f"https://www.reddit.com/r/{sub}",
                "title": f"r/{sub}",
                "snippet": "",
                "source_type": "reddit",
            })
        for org in self.config.get("seed_github_orgs", []):
            self._seed_urls.append({
                "url": f"https://github.com/{org}",
                "title": org,
                "snippet": "",
                "source_type": "github",
            })
        if self._seed_urls:
            console.print(f"[cyan]Seeding {len(self._seed_urls)} extra source(s).[/cyan]")

        if self.is_resume:
            # Resume: restore state from DB, then generate continuation queries
            await self._restore_session()
            console.print("[yellow]Generating continuation queries…[/yellow]")
            self.queries = await self.brain.generate_queries(
                self.topic,
                prior_queries=self.prior_queries,
                running_summary=self.store.running_summary,
                knowledge_gaps=self.knowledge_gaps,
                n=self.config.get("max_queries", 7),
            )
            if not self.queries:
                console.print("[dim]No new queries after restore — using topic directly.[/dim]")
                self.queries = [self.topic]
            await self.writer.log_event(self.session_id, "session_resume", {
                "topic": self.topic, "cycle_num": self.cycle_num
            })
        else:
            # Fresh start
            await self.writer.create_session(self.session_id, self.topic, self.config)
            await self.writer.log_event(self.session_id, "session_start", {"topic": self.topic})
            console.print("[yellow]Generating initial queries…[/yellow]")
            self.queries = await self.brain.generate_queries(
                self.topic,
                prior_queries=[],
                running_summary="",
                knowledge_gaps=[],
                n=self.config.get("max_queries", 7),
            )
            if not self.queries:
                console.print("[red]Brain returned no queries — using topic directly.[/red]")
                self.queries = [self.topic]

        console.print(f"[green]Generated {len(self.queries)} queries.[/green]")
        for q in self.queries:
            console.print(f"  • {q}")

        # Main loop — wrapped in Rich Live for status panel
        async def _refresh_live(live: Live) -> None:
            while not (self.saturated or self.done):
                live.update(self._make_status_panel())
                live.refresh()
                await asyncio.sleep(0.5)
            live.update(self._make_status_panel())
            live.refresh()

        live_ctx = (
            Live(self._make_status_panel(), console=console, auto_refresh=False, screen=False)
            if not self._quiet else None
        )
        if live_ctx:
            live_ctx.start(refresh=False)
        refresh_task = asyncio.create_task(_refresh_live(live_ctx)) if live_ctx else None

        try:
            while not self.saturated and not self.done:
                await self._run_cycle()
                if not self.done:
                    await self._process_steering()
                self.cycle_num += 1

            self._current_phase = "complete"

            # Final output
            if not self.done:
                await self._synthesize_and_write()
            console.print("\n[bold green]Research session complete.[/bold green]")
            console.print(f"Output: {self.output_path}")
        finally:
            if refresh_task:
                refresh_task.cancel()
                try:
                    await refresh_task
                except asyncio.CancelledError:
                    pass
            if live_ctx:
                live_ctx.stop()

        await self.searcher.aclose()
        await self.web_fetcher.aclose()
        await self.reddit_fetcher.aclose()
        await self.github_fetcher.aclose()
        await self.writer.stop()

    # ── resume ────────────────────────────────────────────────────────────

    async def _restore_session(self) -> None:
        """Reload findings, sources, leads, queries and running_summary from SQLite."""
        console.print(f"[yellow]Restoring session {self.session_id}…[/yellow]")

        findings = self.reader.get_findings(self.session_id)
        sources = self.reader.get_sources(self.session_id)
        leads = self.reader.get_leads(self.session_id)
        prior_queries = self.reader.get_queries(self.session_id)
        running_summary = self.reader.get_running_summary(self.session_id)

        for f in findings:
            self.store.add_finding(f)
        for s in sources:
            self.store.add_source(s)
        for lead in leads:
            self.store.add_lead(lead)

        self.prior_queries = list(prior_queries)
        if running_summary:
            self.store.set_running_summary(running_summary)

        # Restore cycle counter
        if findings:
            self.cycle_num = max(f["cycle_num"] for f in findings) + 1

        console.print(
            f"[green]Restored: {len(findings)} findings, {len(sources)} sources, "
            f"{len(prior_queries)} prior queries. Resuming from cycle {self.cycle_num + 1}.[/green]"
        )

        # Re-populate ChromaDB for semantic dedup
        if findings:
            console.print(
                f"[yellow]Re-embedding {len(findings)} findings for semantic dedup…[/yellow]"
            )
            await self.semantic_dedup.upsert_batch(findings)

    # ── cycle ─────────────────────────────────────────────────────────────

    async def _run_cycle(self) -> None:
        self._current_phase = "searching"
        self._cycle_start_time = datetime.now(timezone.utc)
        console.rule(f"[bold]Cycle {self.cycle_num + 1}[/bold]")

        # Phase 2: Search
        console.print("[yellow]Searching…[/yellow]")
        urls_to_fetch = await self._search()
        console.print(f"[cyan]{len(urls_to_fetch)} new URLs to fetch[/cyan]")

        if not urls_to_fetch:
            console.print("[dim]No new URLs this cycle — checking saturation…[/dim]")
            self._novel_per_cycle.append(0)
            await self._check_saturation()
            return

        # Record queries
        for q in self.queries:
            await self.writer.insert_query(self.session_id, self.cycle_num, q)
        self.prior_queries.extend(self.queries)

        self._current_phase = "fetching"
        self._urls_total = len(urls_to_fetch)
        self._urls_remaining = len(urls_to_fetch)
        self._current_url = None

        # Phases 3 & 4: Fetch + Summarize (async producer-consumer)
        novel_count, skipped_count = await self._fetch_and_summarize(urls_to_fetch)

        self._novel_per_cycle.append(novel_count)
        console.print(
            f"[green]Cycle {self.cycle_num + 1} done:[/green] "
            f"{novel_count} novel findings, {skipped_count} skipped"
        )

        # Section clustering (after 10+ findings, every cluster_every cycles)
        cluster_threshold = self.config.get("cluster_threshold", 10)
        cluster_every = self.config.get("cluster_every_cycles", 5)
        total_findings = self.store.finding_count()
        if (
            total_findings >= cluster_threshold
            and (self.cycle_num + 1) % cluster_every == 0
        ):
            await self._cluster_sections()

        # Phase 6: Render findings
        await self._render()
        await self.writer.log_event(
            self.session_id,
            "cycle_complete",
            {"cycle": self.cycle_num, "novel": novel_count, "skipped": skipped_count},
        )

        # Auto executive summary regen every N cycles
        regen_every = self.config.get("summary_regen_every", 5)
        if (self.cycle_num + 1) % regen_every == 0 and self.store.finding_count() > 0:
            await self._synthesize_and_write()

        # Phase 7: Reflect & generate new queries
        if not self.done:
            await self._reflect()

        # Saturation check
        await self._check_saturation()

    # ── search ────────────────────────────────────────────────────────────

    async def _search(self) -> list[dict]:
        max_results = self.config.get("max_results_per_query", 10)
        max_age = self.config.get("max_age_months", 6)
        seen: list[dict] = []

        # Inject seed URLs on first cycle only
        if self.cycle_num == 0 and self._seed_urls:
            for r in self._seed_urls:
                url = r["url"]
                if not self.dedup.is_seen(url, self.session_id) and r.get("source_type") in self._enabled_sources:
                    seen.append(r)
            self._seed_urls = []  # consume once

        for query in self.queries:
            results = await self.searcher.search(query, max_results=max_results, max_age_months=max_age)
            for r in results:
                url = r["url"]
                if self.dedup.is_seen(url, self.session_id):
                    continue
                source_type = r.get("source_type", "web")
                if source_type not in self._enabled_sources:
                    await self.writer.mark_url_seen(url, self.session_id, f"skipped_source_{source_type}")
                    continue
                if self._focus and self._focus.lower() not in url.lower() + r.get("snippet", "").lower():
                    continue
                if any(ig.lower() in url.lower() + r.get("snippet", "").lower() for ig in self._ignore):
                    continue
                seen.append(r)

        # Deduplicate within this batch
        seen_urls: set[str] = set()
        unique = []
        for r in seen:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                unique.append(r)
        return unique

    # ── fetch & summarize ─────────────────────────────────────────────────

    async def _fetch_and_summarize(
        self, urls_to_fetch: list[dict]
    ) -> tuple[int, int]:
        fetch_queue: asyncio.Queue[Optional[tuple[dict, FetchResult]]] = asyncio.Queue()
        novel_count = 0
        skipped_count = 0
        min_words = self.config.get("min_content_words", 200)

        # Producers: fetch concurrently; always put exactly one item (even on exception)
        async def producer(url_info: dict) -> None:
            source_type = url_info.get("source_type", "web")
            fetcher = (
                self.reddit_fetcher if source_type == "reddit"
                else self.github_fetcher if source_type == "github"
                else self.web_fetcher
            )
            try:
                self._current_url = url_info["url"]
                result = await fetcher.fetch(url_info["url"])
            except Exception as exc:
                result = FetchResult(
                    url=url_info["url"], source_type=source_type, error=str(exc)
                )
            await fetch_queue.put((url_info, result))

        tasks = [asyncio.create_task(producer(u)) for u in urls_to_fetch]

        # Consumer: processes exactly one item per URL — no sentinels needed
        async def consumer() -> None:
            nonlocal novel_count, skipped_count
            for _ in range(len(urls_to_fetch)):
                url_info, result = await fetch_queue.get()
                self._urls_remaining -= 1

                if result.error:
                    log.debug("Fetch error for %s: %s", result.url, result.error)
                    await self.writer.mark_url_seen(result.url, self.session_id, f"error:{result.error[:50]}")
                    skipped_count += 1
                    continue

                if result.word_count < min_words:
                    log.debug("Too short (%d words): %s", result.word_count, result.url)
                    await self.writer.mark_url_seen(result.url, self.session_id, "filtered_short")
                    skipped_count += 1
                    continue

                console.print(f"  [dim]↓ Fetched:[/dim] {result.url[:80]} ({result.word_count} words)")

                # LLM: summarize
                self._current_phase = "summarizing"
                self._current_url = result.url
                summary = await self.brain.summarize(
                    content=result.content,
                    source_url=result.url,
                    source_type=result.source_type,
                    source_date=result.date,
                    topic=self.topic,
                    running_summary=self.store.running_summary,
                )
                self._current_phase = "fetching"

                if summary["relevance_score"] < 2:
                    log.debug("Low relevance (%d): %s", summary["relevance_score"], result.url)
                    await self.writer.mark_url_seen(result.url, self.session_id, "filtered_irrelevant")

                    skipped_count += 1
                    continue

                new_findings = []
                s_hash = _source_hash(result.content)

                # Collect all finding texts and batch-embed them in one call
                raw_findings = [
                    f for f in summary.get("findings", [])
                    if isinstance(f, str) and f.strip()
                ]
                if raw_findings:
                    try:
                        embeddings = await self.brain.embed_texts(raw_findings)
                    except Exception as exc:
                        log.warning(
                            "Embedding failed for %s: %s — falling back to URL-only dedup",
                            result.url, exc,
                        )
                        embeddings = [None] * len(raw_findings)
                else:
                    embeddings = []

                for finding_text, embedding in zip(raw_findings, embeddings):
                    fid = make_finding_id(finding_text, result.url)

                    # Semantic dedup check
                    if embedding is not None:
                        dedup = await self.semantic_dedup.check(
                            finding_text=finding_text,
                            finding_id=fid,
                            source_url=result.url,
                            embedding=embedding,
                        )
                    else:
                        dedup = DedupeResult(disposition=Disposition.NOVEL)

                    if dedup.disposition == Disposition.DUPLICATE:
                        log.debug(
                            "Dedup: discarding duplicate (dist=%.3f): %s…",
                            dedup.distance or 0.0, finding_text[:60],
                        )
                        skipped_count += 1
                        continue

                    # Annotate text for near-duplicate but new-angle findings
                    stored_text = finding_text
                    if dedup.disposition == Disposition.NEW_ANGLE:
                        stored_text = f"{finding_text} *(similar to existing finding — included as new angle)*"

                    finding = {
                        "id": fid,
                        "session_id": self.session_id,
                        "cycle_num": self.cycle_num,
                        "finding_text": stored_text,
                        "source_url": result.url,
                        "fetch_timestamp": _now_utc(),
                        "source_hash": s_hash,
                        "source_type": result.source_type,
                        "relevance_score": summary["relevance_score"],
                        "quality_type": summary["quality_type"],
                        "embedding_id": fid,
                        "conflicting": 1 if dedup.disposition == Disposition.KEEP_BOTH else 0,
                    }

                    await self.writer.insert_finding(finding)

                    # Add to ChromaDB using the pre-computed embedding
                    if embedding is not None:
                        self.semantic_dedup.upsert(
                            finding_id=fid,
                            finding_text=finding_text,
                            embedding=embedding,
                            metadata={"source_url": result.url, "session_id": self.session_id},
                        )

                    self.store.add_finding(finding)
                    new_findings.append(stored_text)
                    novel_count += 1

                    if dedup.disposition == Disposition.KEEP_BOTH:
                        console.print(
                            f"  [yellow]⚠[/yellow] Conflicting finding kept "
                            f"(dist={dedup.distance:.3f}): {finding_text[:60]}"
                        )
                    elif dedup.disposition == Disposition.NEW_ANGLE:
                        console.print(
                            f"  [cyan]~[/cyan] New-angle finding "
                            f"(dist={dedup.distance:.3f}): {finding_text[:60]}"
                        )

                # Leads
                for lead in summary.get("new_leads", [])[:5]:
                    if isinstance(lead, str) and lead.strip():
                        await self.writer.insert_lead(
                            self.session_id, lead, result.url, self.cycle_num
                        )
                        self.store.add_lead(
                            {"lead_text": lead, "source_url": result.url, "cycle_num": self.cycle_num}
                        )

                # Source record
                source_rec = {
                    "session_id": self.session_id,
                    "url": result.url,
                    "title": result.title,
                    "source_type": result.source_type,
                    "fetch_date": result.date or _now_utc(),
                    "novel_findings_count": len(new_findings),
                    "stars": result.metadata.get("stars"),
                    "description": result.metadata.get("description"),
                    "language": result.metadata.get("language"),
                }
                await self.writer.insert_source(source_rec)
                self.store.add_source(source_rec)
                await self.writer.mark_url_seen(result.url, self.session_id, "success")

                # Update running summary
                if new_findings:
                    self._consecutive_zero_novel = 0
                    updated = await self.brain.update_running_summary(
                        self.topic, self.store.running_summary, new_findings
                    )
                    self.store.set_running_summary(updated)
                    await self.writer.save_running_summary(self.session_id, updated)
                    console.print(
                        f"  [green]+{len(new_findings)} findings[/green] from {result.url[:60]}"
                    )
                else:
                    self._consecutive_zero_novel += 1
                    console.print(f"  [dim]0 findings[/dim] from {result.url[:60]}")

        consumer_task = asyncio.create_task(consumer())
        await asyncio.gather(*tasks)
        await consumer_task
        self._current_url = None

        # Cycle log
        self.store.add_cycle_log(
            {
                "cycle_num": self.cycle_num + 1,
                "timestamp": _now_utc()[:19].replace("T", " ") + " UTC",
                "queries": list(self.queries),
                "sources_fetched": len(urls_to_fetch),
                "novel_findings": novel_count,
                "skipped": skipped_count,
            }
        )

        return novel_count, skipped_count

    # ── section clustering ────────────────────────────────────────────────

    async def _cluster_sections(self) -> None:
        self._current_phase = "clustering"
        console.print("[yellow]Clustering findings into sections…[/yellow]")
        assignments = await self.brain.cluster_findings(self.topic, self.store.findings)
        if assignments:
            self.store.update_sections(assignments)
            await self.writer.update_finding_sections(assignments)
            unique_sections = len(set(assignments.values()))
            console.print(
                f"[green]Clustered {len(assignments)} findings into "
                f"{unique_sections} sections.[/green]"
            )

    # ── render ────────────────────────────────────────────────────────────

    async def _render(self) -> None:
        sources_str = ",".join(sorted(self._enabled_sources))
        config_summary = (
            f"max_age={self.config.get('max_age_months', 6)}m, "
            f"sources={sources_str}"
        )
        md = self.store.render(
            topic=self.topic,
            session_id=self.session_id,
            start_time=self._start_time,
            config_summary=config_summary,
        )
        await self.writer.write_file(self.output_path, md)

    # ── reflect ───────────────────────────────────────────────────────────

    async def _reflect(self) -> None:
        self._current_phase = "reflecting"
        console.print("[yellow]Reflecting on findings…[/yellow]")
        result = await self.brain.reflect(
            topic=self.topic,
            running_summary=self.store.running_summary,
            prior_queries=self.prior_queries,
            cycle_count=self.cycle_num + 1,
        )

        self.knowledge_gaps = result.get("knowledge_gaps", [])
        new_queries = result.get("new_queries", [])

        if result.get("saturated") and not new_queries:
            console.print("[yellow]Brain signals topic is saturated.[/yellow]")
            self.saturated = True
            return

        if not new_queries:
            console.print("[dim]No new queries from reflection.[/dim]")
            self.queries = []
            return

        self.queries = new_queries
        console.print(f"[cyan]New queries:[/cyan]")
        for q in self.queries:
            console.print(f"  • {q}")

    # ── saturation detection ──────────────────────────────────────────────

    async def _check_saturation(self) -> None:
        zero_window = self.config.get("saturation_zero_novel_window", 10)
        low_cycles = self.config.get("saturation_low_novel_cycles", 5)
        low_threshold = self.config.get("saturation_low_novel_threshold", 2)

        if self._consecutive_zero_novel >= zero_window:
            console.print(
                f"[yellow]Saturation: last {zero_window} sources yielded 0 novel findings.[/yellow]"
            )
            self.saturated = True
            return

        recent = self._novel_per_cycle[-low_cycles:]
        if (
            len(recent) >= low_cycles
            and all(n < low_threshold for n in recent)
        ):
            console.print(
                f"[yellow]Saturation: last {low_cycles} cycles each had < {low_threshold} novel findings.[/yellow]"
            )
            self.saturated = True

    # ── synthesize ────────────────────────────────────────────────────────

    async def _synthesize_and_write(self) -> None:
        self._current_phase = "synthesizing"
        console.print("[yellow]Generating executive summary…[/yellow]")
        summary = await self.brain.synthesize(self.topic, self.store.findings, self.store.running_summary)
        self.store.set_executive_summary(summary)
        await self._render()
        console.print("[green]Executive summary written.[/green]")

    # ── steering ─────────────────────────────────────────────────────────

    def _make_status_panel(self) -> Panel:
        """Build the Rich status Panel; called every 0.5s by the Live refresh task."""
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", no_wrap=True)
        grid.add_column()

        # Cycle + elapsed
        if self._cycle_start_time is not None:
            elapsed_s = int((datetime.now(timezone.utc) - self._cycle_start_time).total_seconds())
            elapsed_str = f"{elapsed_s // 60}m {elapsed_s % 60}s"
        else:
            elapsed_str = "—"
        grid.add_row("Cycle", f"{self.cycle_num + 1}  ({elapsed_str} elapsed)")

        # Phase
        phase_colors = {
            "idle": "dim", "searching": "yellow", "fetching": "cyan",
            "summarizing": "magenta", "reflecting": "yellow",
            "clustering": "blue", "synthesizing": "green", "complete": "bold green",
        }
        phase_color = phase_colors.get(self._current_phase, "white")
        grid.add_row("Phase", f"[{phase_color}]{self._current_phase}[/{phase_color}]")

        # Current URL + remaining count
        if self._current_url or self._urls_total > 0:
            url_display = ""
            if self._current_url:
                url_display = (self._current_url[:70] + "…") if len(self._current_url) > 70 else self._current_url
            frac = f"  [{self._urls_remaining}/{self._urls_total} remaining]" if self._urls_total > 0 else ""
            grid.add_row("Fetching", f"[dim]{url_display}[/dim]{frac}")

        # Findings / sources / novel this cycle
        novel_this_cycle = self._novel_per_cycle[-1] if self._novel_per_cycle else 0
        grid.add_row(
            "Findings",
            f"{self.store.finding_count()} total  |  {self.store.source_count()} sources  |  "
            f"[green]+{novel_this_cycle}[/green] this cycle",
        )

        # Zero-novel streak
        zero_window = self.config.get("saturation_zero_novel_window", 10)
        streak_color = "yellow" if self._consecutive_zero_novel >= zero_window // 2 else "dim"
        grid.add_row(
            "Zero-novel streak",
            f"[{streak_color}]{self._consecutive_zero_novel}/{zero_window}[/{streak_color}]",
        )

        # Status flags (only when set)
        flags: list[str] = []
        if self.saturated:
            flags.append("[yellow]SATURATED[/yellow]")
        if self.paused:
            flags.append("[yellow]PAUSED[/yellow]")
        if self.done:
            flags.append("[green]DONE[/green]")
        if flags:
            grid.add_row("Status", "  ".join(flags))

        # Queries (first 3)
        if self.queries:
            q_lines = "\n".join(f"  • {q[:80]}" for q in self.queries[:3])
            if len(self.queries) > 3:
                q_lines += f"\n  … +{len(self.queries) - 3} more"
            grid.add_row("Queries", q_lines)

        # Focus / ignore (only when set)
        if self._focus:
            grid.add_row("Focus", f"[cyan]{self._focus}[/cyan]")
        if self._ignore:
            grid.add_row("Ignoring", f"[dim]{len(self._ignore)} term(s): {', '.join(self._ignore[:3])}[/dim]")

        # Pending commands (only when non-empty)
        if self._pending_commands:
            cmds = ", ".join(repr(c) for c in self._pending_commands[:4])
            grid.add_row("Queued cmds", f"[dim]{cmds}[/dim]")

        # Writer queue depth (only when backlogged)
        q_depth = self.writer._queue.qsize()
        if q_depth > 0:
            grid.add_row("Writer queue", f"[dim]{q_depth} pending[/dim]")

        return Panel(grid, title="[bold]LocalResearcher[/bold]", border_style="cyan", padding=(0, 1))

    async def handle_command(self, cmd: str) -> None:
        """Called from the stdin reader task."""
        self._pending_commands.append(cmd)
        await self._steering_queue.put(cmd)

    async def _process_steering(self) -> None:
        self._pending_commands.clear()
        while not self._steering_queue.empty():
            cmd = await self._steering_queue.get()
            await self._execute_command(cmd)

    async def _execute_command(self, cmd: str) -> None:
        parts = cmd.strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        if verb == "focus":
            self._focus = arg
            console.print(f"[cyan]Focus set: {arg}[/cyan]")
        elif verb == "ignore":
            self._ignore.append(arg)
            console.print(f"[cyan]Ignore added: {arg}[/cyan]")
        elif verb == "broaden":
            self._focus = None
            self._ignore.clear()
            console.print("[cyan]Search scope broadened.[/cyan]")
        elif verb == "status":
            console.print(
                f"Cycle: {self.cycle_num + 1} | "
                f"Findings: {self.store.finding_count()} | "
                f"Sources: {self.store.source_count()} | "
                f"Saturated: {self.saturated}"
            )
        elif verb == "synthesize":
            await self._synthesize_and_write()
        elif verb == "pause":
            self.paused = True
            console.print("[yellow]Paused. Type 'go' to resume.[/yellow]")
            while self.paused:
                await asyncio.sleep(0.5)
                while not self._steering_queue.empty():
                    c = await self._steering_queue.get()
                    if c.strip().lower() == "go":
                        self.paused = False
                        console.print("[green]Resumed.[/green]")
                        break
        elif verb == "go":
            self.paused = False
        elif verb == "done":
            console.print("[yellow]Stopping after this cycle…[/yellow]")
            self.done = True
            await self._synthesize_and_write()
        elif verb == "add":
            if arg:
                # Add URL to pending fetch (insert directly into a temporary list)
                console.print(f"[cyan]Will add URL/source: {arg}[/cyan]")
                # Enqueue for next cycle by injecting into queries
                self.queries.append(f'site:{arg}' if '.' in arg and ' ' not in arg else arg)
        elif verb == "help":
            console.print(
                "[bold]Commands:[/bold] focus <topic>, ignore <topic>, broaden, "
                "status, synthesize, pause, go, done, add <url/source>, help"
            )
        else:
            console.print(f"[red]Unknown command: {cmd!r}. Type 'help' for commands.[/red]")

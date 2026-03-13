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

from .brain import Brain
from .dedup import URLDedup
from .fetcher import FetchResult, SearXNGSearcher, WebFetcher
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
    ):
        self.topic = topic
        self.config = config
        self.session_id = session_id
        self.output_path = output_path
        self.deterministic = deterministic

        # Components
        self.brain = Brain(config, prompts_dir)
        self.writer = WriterActor(db_path)
        self.reader = StateReader(db_path)
        self.dedup = URLDedup(self.reader)
        self.searcher = SearXNGSearcher(config["searxng_url"])
        self.web_fetcher = WebFetcher(
            rate_limit=config.get("web_fetch_delay", 2.5),
            max_concurrent=config.get("max_concurrent_fetches", 5),
            max_age_months=config.get("max_age_months", 6),
        )
        self.store = FindingsStore()

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

        # Start time
        self._start_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.writer.start()
        await self.writer.create_session(
            self.session_id, self.topic, self.config
        )
        await self.writer.log_event(self.session_id, "session_start", {"topic": self.topic})

        console.print(f"\n[bold cyan]LocalResearcher[/bold cyan] — topic: [bold]{self.topic}[/bold]")
        console.print(f"Session ID: {self.session_id}")
        console.print(f"Output: {self.output_path}\n")

        # Phase 1: initial query generation
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

        console.print(f"[green]Generated {len(self.queries)} initial queries.[/green]")
        for q in self.queries:
            console.print(f"  • {q}")

        # Main loop
        while not self.saturated and not self.done:
            await self._run_cycle()
            if not self.done:
                await self._process_steering()
            self.cycle_num += 1

        # Final output
        if not self.done:
            await self._synthesize_and_write()
        console.print("\n[bold green]Research session complete.[/bold green]")
        console.print(f"Output: {self.output_path}")

        await self.searcher.aclose()
        await self.web_fetcher.aclose()
        await self.writer.stop()

    # ── cycle ─────────────────────────────────────────────────────────────

    async def _run_cycle(self) -> None:
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

        # Phases 3 & 4: Fetch + Summarize (async producer-consumer)
        novel_count, skipped_count = await self._fetch_and_summarize(urls_to_fetch)

        self._novel_per_cycle.append(novel_count)
        console.print(
            f"[green]Cycle {self.cycle_num + 1} done:[/green] "
            f"{novel_count} novel findings, {skipped_count} skipped"
        )

        # Phase 6: Render findings
        await self._render()
        await self.writer.log_event(
            self.session_id,
            "cycle_complete",
            {"cycle": self.cycle_num, "novel": novel_count, "skipped": skipped_count},
        )

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

        for query in self.queries:
            results = await self.searcher.search(query, max_results=max_results, max_age_months=max_age)
            for r in results:
                url = r["url"]
                if self.dedup.is_seen(url, self.session_id):
                    continue
                if r.get("source_type") != "web":
                    # Phase 1: web only — skip reddit/github URLs
                    await self.writer.mark_url_seen(url, self.session_id, "skipped_not_web")
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
            try:
                result = await self.web_fetcher.fetch(url_info["url"])
            except Exception as exc:
                result = FetchResult(url=url_info["url"], source_type="web", error=str(exc))
            await fetch_queue.put((url_info, result))

        tasks = [asyncio.create_task(producer(u)) for u in urls_to_fetch]

        # Consumer: processes exactly one item per URL — no sentinels needed
        async def consumer() -> None:
            nonlocal novel_count, skipped_count
            for _ in range(len(urls_to_fetch)):
                url_info, result = await fetch_queue.get()

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
                summary = await self.brain.summarize(
                    content=result.content,
                    source_url=result.url,
                    source_type=result.source_type,
                    source_date=result.date,
                    topic=self.topic,
                    running_summary=self.store.running_summary,
                )

                if summary["relevance_score"] < 2:
                    log.debug("Low relevance (%d): %s", summary["relevance_score"], result.url)
                    await self.writer.mark_url_seen(result.url, self.session_id, "filtered_irrelevant")

                    skipped_count += 1
                    continue

                new_findings = []
                s_hash = _source_hash(result.content)

                for finding_text in summary.get("findings", []):
                    if not isinstance(finding_text, str) or not finding_text.strip():
                        continue
                    fid = make_finding_id(finding_text, result.url)
                    finding = {
                        "id": fid,
                        "session_id": self.session_id,
                        "cycle_num": self.cycle_num,
                        "finding_text": finding_text,
                        "source_url": result.url,
                        "fetch_timestamp": _now_utc(),
                        "source_hash": s_hash,
                        "source_type": result.source_type,
                        "relevance_score": summary["relevance_score"],
                        "quality_type": summary["quality_type"],
                    }
                    await self.writer.insert_finding(finding)
                    self.store.add_finding(finding)
                    new_findings.append(finding_text)
                    novel_count += 1

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
                }
                await self.writer.insert_source(source_rec)
                self.store.add_source(source_rec)
                await self.writer.mark_url_seen(result.url, self.session_id, "success")

                # Update running summary
                if new_findings:
                    updated = await self.brain.update_running_summary(
                        self.topic, self.store.running_summary, new_findings
                    )
                    self.store.set_running_summary(updated)
                    console.print(
                        f"  [green]+{len(new_findings)} findings[/green] from {result.url[:60]}"
                    )
                else:
                    console.print(f"  [dim]0 findings[/dim] from {result.url[:60]}")

        consumer_task = asyncio.create_task(consumer())
        await asyncio.gather(*tasks)
        await consumer_task

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

    # ── render ────────────────────────────────────────────────────────────

    async def _render(self) -> None:
        config_summary = (
            f"max_age={self.config.get('max_age_months', 6)}m, "
            f"sources=web"
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
        console.print("[yellow]Generating executive summary…[/yellow]")
        summary = await self.brain.synthesize(self.topic, self.store.findings)
        self.store.set_executive_summary(summary)
        await self._render()
        console.print("[green]Executive summary written.[/green]")

    # ── steering ─────────────────────────────────────────────────────────

    async def handle_command(self, cmd: str) -> None:
        """Called from the stdin reader task."""
        await self._steering_queue.put(cmd)

    async def _process_steering(self) -> None:
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

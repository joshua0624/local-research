"""
Entry point for LocalResearcher.

Usage:
    python -m researcher.main --topic "local LLM coding agents" --max-age 3

Optional flags:
    --sources web,reddit,github     comma-separated list (default: all three)
    --subreddits ollama,LocalLLaMA  extra subreddits to seed
    --github-orgs langchain-ai      extra GitHub orgs to seed
    --max-age 6                     months (default: 6)
    --output findings.md
    --db researcher.db
    --deterministic                 low temp, fixed seed
    --quiet                         suppress progress output
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

import yaml
from rich.console import Console

from .orchestrator import ResearchOrchestrator

console = Console()

_HERE = Path(__file__).parent
_DEFAULT_CONFIG = _HERE / "config.yaml"
_DEFAULT_PROMPTS = _HERE / "prompts"


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _setup_logging(config: dict) -> None:
    log_file = config.get("log_file", "researcher.log")
    level = logging.DEBUG if config.get("log_llm_calls") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="researcher",
        description="Local LLM research assistant — searches the web and builds a findings document.",
    )
    p.add_argument("--topic", "-t", default=None, help="Research topic (required unless --resume)")
    p.add_argument("--max-age", type=int, default=None, metavar="MONTHS", help="Max source age in months")
    p.add_argument("--output", "-o", default=None, help="Output markdown file (default: findings.md)")
    p.add_argument("--db", default="researcher.db", help="SQLite database path")
    p.add_argument("--config", default=str(_DEFAULT_CONFIG), help="Config YAML path")
    p.add_argument("--deterministic", action="store_true", help="Low temperature, fixed seed")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    p.add_argument(
        "--sources",
        default=None,
        help="Comma-separated source types to use: web,reddit,github (default: all)",
    )
    p.add_argument(
        "--subreddits",
        default=None,
        help="Comma-separated subreddits to seed (e.g. ollama,LocalLLaMA)",
    )
    p.add_argument(
        "--github-orgs",
        default=None,
        dest="github_orgs",
        help="Comma-separated GitHub orgs/users to seed (e.g. langchain-ai)",
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="SESSION_ID",
        help="Resume a prior session by its session ID",
    )
    p.add_argument(
        "--sessions",
        action="store_true",
        help="List recent sessions and exit",
    )
    return p.parse_args()


async def _stdin_reader(orchestrator: ResearchOrchestrator, stop_event: asyncio.Event) -> None:
    """Background task: reads commands from stdin and forwards to orchestrator."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    try:
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    except Exception as exc:
        logging.getLogger(__name__).warning("stdin reader could not connect: %s", exc)
        return
    while not stop_event.is_set():
        try:
            line_bytes = await reader.readline()
            if not line_bytes:
                break
            cmd = line_bytes.decode(errors="replace").strip()
            if cmd:
                await orchestrator.handle_command(cmd)
        except (asyncio.CancelledError, EOFError):
            break
        except Exception as exc:
            logging.getLogger(__name__).debug("stdin reader error: %s", exc)


async def _run(args: argparse.Namespace) -> None:
    from .state import StateReader

    config = _load_config(Path(args.config))
    _setup_logging(config)

    # --sessions: list prior sessions and exit
    if args.sessions:
        reader = StateReader(args.db)
        sessions = reader.list_sessions()
        if not sessions:
            console.print("[dim]No sessions found in {args.db}[/dim]")
        else:
            console.print(f"\n[bold]Sessions in {args.db}:[/bold]")
            for s in sessions:
                console.print(f"  [cyan]{s['id']}[/cyan]  {s['start_time'][:16]}  {s['topic']}")
        return

    # Resolve topic and session_id
    is_resume = bool(args.resume)
    if is_resume:
        reader = StateReader(args.db)
        session_data = reader.get_session(args.resume)
        if not session_data:
            console.print(
                f"[red]Session [bold]{args.resume}[/bold] not found in {args.db}[/red]"
            )
            sys.exit(1)
        topic = args.topic or session_data["topic"]
        session_id = args.resume
    else:
        if not args.topic:
            console.print("[red]--topic is required unless --resume is used.[/red]")
            sys.exit(1)
        topic = args.topic
        session_id = str(uuid.uuid4())[:8]

    # CLI overrides
    if args.max_age is not None:
        config["max_age_months"] = args.max_age
    if args.deterministic:
        config["temperature"] = config.get("deterministic_temperature", 0.1)
    if args.sources is not None:
        config["sources"] = [s.strip() for s in args.sources.split(",") if s.strip()]
    if args.subreddits is not None:
        config["seed_subreddits"] = [s.strip() for s in args.subreddits.split(",") if s.strip()]
    if args.github_orgs is not None:
        config["seed_github_orgs"] = [o.strip() for o in args.github_orgs.split(",") if o.strip()]

    output_path = args.output or config.get("output_file", "findings.md")

    if not args.quiet:
        mode = "[yellow]RESUMING[/yellow]" if is_resume else "starting"
        console.print(
            f"\n[bold]LocalResearcher[/bold]  {mode}  topic=[cyan]{topic}[/cyan]  "
            f"max_age={config['max_age_months']}m  session={session_id}"
        )
        console.print(
            "[dim]Type commands between cycles: help, focus <topic>, ignore <topic>, "
            "status, synthesize, pause, go, done[/dim]\n"
        )

    orchestrator = ResearchOrchestrator(
        topic=topic,
        config=config,
        session_id=session_id,
        output_path=output_path,
        prompts_dir=str(_DEFAULT_PROMPTS),
        db_path=args.db,
        deterministic=args.deterministic,
        is_resume=is_resume,
        quiet=args.quiet,
    )

    stop_event = asyncio.Event()
    stdin_task = asyncio.create_task(
        _stdin_reader(orchestrator, stop_event), name="stdin-reader"
    )

    try:
        await orchestrator.run()
    finally:
        stop_event.set()
        stdin_task.cancel()
        try:
            await stdin_task
        except asyncio.CancelledError:
            pass


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()

"""
Entry point for LocalResearcher.

Usage:
    python -m researcher.main --topic "local LLM coding agents" --max-age 3

Optional flags:
    --sources web                   (default: web; reddit/github in Phase 2)
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
    p.add_argument("--topic", "-t", required=True, help="Research topic")
    p.add_argument("--max-age", type=int, default=None, metavar="MONTHS", help="Max source age in months")
    p.add_argument("--output", "-o", default=None, help="Output markdown file (default: findings.md)")
    p.add_argument("--db", default="researcher.db", help="SQLite database path")
    p.add_argument("--config", default=str(_DEFAULT_CONFIG), help="Config YAML path")
    p.add_argument("--deterministic", action="store_true", help="Low temperature, fixed seed")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
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
                console.print(f"[dim]Command received: {cmd}[/dim]")
        except (asyncio.CancelledError, EOFError):
            break
        except Exception as exc:
            logging.getLogger(__name__).debug("stdin reader error: %s", exc)


async def _run(args: argparse.Namespace) -> None:
    config = _load_config(Path(args.config))
    _setup_logging(config)

    # CLI overrides
    if args.max_age is not None:
        config["max_age_months"] = args.max_age
    if args.deterministic:
        config["temperature"] = config.get("deterministic_temperature", 0.1)

    output_path = args.output or config.get("output_file", "findings.md")
    session_id = str(uuid.uuid4())[:8]

    if not args.quiet:
        console.print(
            f"\n[bold]LocalResearcher[/bold]  topic=[cyan]{args.topic}[/cyan]  "
            f"max_age={config['max_age_months']}m  session={session_id}"
        )
        console.print(
            "[dim]Type commands between cycles: help, focus <topic>, ignore <topic>, "
            "status, synthesize, pause, go, done[/dim]\n"
        )

    orchestrator = ResearchOrchestrator(
        topic=args.topic,
        config=config,
        session_id=session_id,
        output_path=output_path,
        prompts_dir=str(_DEFAULT_PROMPTS),
        db_path=args.db,
        deterministic=args.deterministic,
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

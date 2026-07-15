"""Command-line interface for the Agentic Research Assistant.

Subcommands
-----------
* ``ingest``   — build/refresh the FAISS index from files or arXiv.
* ``research`` — run the multi-agent loop against an existing index.
* ``demo``     — ingest the bundled sample papers and run a sample question
                 (offline by default via ``--dry-run``).

Heavy imports are deferred into each handler so ``--help`` stays instant and
unrelated commands don't require optional dependencies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from .logging_utils import configure_logging

if TYPE_CHECKING:
    from rich.console import Console as _RichConsole

configure_logging()

_SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "examples" / "sample_papers"
_SAMPLE_QUESTION = (
    "How do retrieval-augmented and multi-agent approaches improve the "
    "reliability of LLM-generated scientific answers, and what gaps remain?"
)


# --------------------------------------------------------------------------- #
# Console helpers (degrade gracefully if `rich` is unavailable)
# --------------------------------------------------------------------------- #
class _Console:
    def __init__(self) -> None:
        self._rich: Optional[_RichConsole] = None
        try:
            from rich.console import Console

            self._rich = Console()
        except ImportError:  # pragma: no cover
            pass

    def print(self, *args, **kwargs) -> None:
        if self._rich:
            self._rich.print(*args, **kwargs)
        else:
            print(*args)

    def rule(self, text: str) -> None:
        if self._rich:
            self._rich.rule(text)
        else:
            print(f"\n=== {text} ===")

    def markdown(self, text: str) -> None:
        if self._rich:
            from rich.markdown import Markdown

            self._rich.print(Markdown(text))
        else:
            print(text)


console = _Console()


def _event_printer():
    def on_event(stage: str, info: dict) -> None:
        if stage == "retrieve":
            console.print(f"[bold cyan]→ retrieve[/]  query={info['query']!r}")
        elif stage == "retrieved":
            console.print(
                f"  found {info['new']} new chunk(s); {info['total']} in context"
            )
        elif stage == "summarize":
            console.print(f"[bold cyan]→ summarize[/]  over {info['chunks']} chunk(s)")
        elif stage == "evaluate":
            console.print(
                f"[bold cyan]→ evaluate[/]  iter={info['iteration']} "
                f"verdict={info['sufficiency']} coverage={info['coverage']:.2f}"
            )
        elif stage == "compose":
            console.print("[bold green]✓ composing report[/]")

    return on_event


def _settings(index_dir: Optional[str], top_k: Optional[int], max_iterations: Optional[int]):
    from .config import get_settings

    update: Dict[str, Any] = {}
    if index_dir:
        update["index_dir"] = Path(index_dir)
    if top_k:
        update["top_k"] = top_k
    if max_iterations:
        update["max_iterations"] = max_iterations
    return get_settings().model_copy(update=update)


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def _cmd_ingest(args: argparse.Namespace) -> int:
    from .factory import build_index
    from .ingestion.loaders import fetch_arxiv, load_paths

    settings = _settings(args.index_dir, None, None)

    papers = []
    if args.paths:
        console.print(f"Loading {len(args.paths)} path(s) from disk…")
        papers.extend(load_paths([Path(p) for p in args.paths]))
    if args.arxiv:
        console.print(f"Searching arXiv for: {args.arxiv!r} (max {args.arxiv_max})…")
        papers.extend(
            fetch_arxiv(args.arxiv, max_results=args.arxiv_max, download_dir=settings.raw_dir)
        )

    if not papers:
        console.print("[red]No documents loaded.[/] Provide file paths or --arxiv.")
        return 1

    store = build_index(papers, settings, dry_run=args.dry_run)
    store.save(settings.index_dir)
    console.print(
        f"[green]Indexed {len(papers)} paper(s) into {len(store)} chunk(s)[/] → "
        f"{settings.index_dir}  (backend: {store.embeddings.name})"
    )
    return 0


# --------------------------------------------------------------------------- #
# research
# --------------------------------------------------------------------------- #
def _run_research(question: str, settings, dry_run: bool, output: Optional[str], as_json: bool):
    from .factory import build_client, build_pipeline, load_index

    store = load_index(settings, dry_run=dry_run)
    console.print(f"Loaded index with {len(store)} chunk(s) from {settings.index_dir}\n")

    client = build_client(settings, dry_run=dry_run)
    pipeline = build_pipeline(store, client, settings, on_event=_event_printer())

    console.rule("Running agentic research loop")
    result = pipeline.run(question)
    console.rule("Report")

    if as_json:
        console.print(result.model_dump_json(indent=2))
    else:
        console.markdown(result.report)

    if output:
        Path(output).write_text(result.report, encoding="utf-8")
        console.print(f"\n[green]Report written to {output}[/]")

    if not dry_run and hasattr(client, "call_count"):
        console.print(
            f"\n[dim]API calls: {getattr(client, 'call_count', 0)} \u00b7 "
            f"input tokens: {getattr(client, 'input_tokens', 0)} \u00b7 "
            f"output tokens: {getattr(client, 'output_tokens', 0)}[/]"
        )
    return result


def _cmd_research(args: argparse.Namespace) -> int:
    settings = _settings(args.index_dir, args.top_k, args.max_iterations)
    question = " ".join(args.question).strip()
    if not question:
        console.print("[red]Please provide a research question.[/]")
        return 1
    _run_research(question, settings, args.dry_run, args.output, args.json)
    return 0


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #
def _cmd_demo(args: argparse.Namespace) -> int:
    from .factory import build_index

    dry_run = not args.use_api
    settings = _settings(args.index_dir or "data/index_demo", args.top_k, args.max_iterations)

    if not _SAMPLE_DIR.exists():
        console.print(f"[red]Sample papers not found at {_SAMPLE_DIR}[/]")
        return 1

    from .ingestion.loaders import load_paths

    console.print(f"Ingesting sample papers from {_SAMPLE_DIR}…")
    papers = load_paths([_SAMPLE_DIR])
    store = build_index(papers, settings, dry_run=dry_run)
    store.save(settings.index_dir)
    console.print(
        f"[green]Indexed {len(papers)} sample paper(s) into {len(store)} chunk(s)[/]"
        f"  (backend: {store.embeddings.name})\n"
    )

    mode = "OFFLINE dry-run (stub model)" if dry_run else "live Claude API"
    console.print(f"[bold]Mode:[/] {mode}")
    question = " ".join(args.question).strip() if args.question else _SAMPLE_QUESTION
    console.print(f"[bold]Question:[/] {question}\n")

    _run_research(question, settings, dry_run, args.output, args.json)
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-assistant",
        description="Multi-agent academic discovery over a FAISS-backed RAG pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_dry_run(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="Use offline hashing embeddings + stub model.",
        )

    p_ingest = sub.add_parser("ingest", help="Build/refresh the vector index.")
    p_ingest.add_argument("paths", nargs="*", help="Files or directories (.pdf/.txt/.md).")
    p_ingest.add_argument("--arxiv", help="arXiv search query to fetch and index.")
    p_ingest.add_argument("--arxiv-max", type=int, default=5, help="Max arXiv results.")
    p_ingest.add_argument("--index-dir", help="Where to store the index.")
    _add_dry_run(p_ingest)
    p_ingest.set_defaults(func=_cmd_ingest)

    p_research = sub.add_parser("research", help="Run the agentic loop on a question.")
    p_research.add_argument("question", nargs="+", help="The research question.")
    p_research.add_argument("--index-dir", help="Index to query.")
    p_research.add_argument("--top-k", type=int, help="Chunks retrieved per query.")
    p_research.add_argument("--max-iterations", type=int, help="Max reflection cycles.")
    p_research.add_argument("--output", "-o", help="Write the Markdown report to this file.")
    p_research.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    _add_dry_run(p_research)
    p_research.set_defaults(func=_cmd_research)

    p_demo = sub.add_parser("demo", help="Ingest bundled samples and run a sample question.")
    p_demo.add_argument("question", nargs="*", help="Optional custom question.")
    p_demo.add_argument("--use-api", action="store_true", help="Call the real Claude API.")
    p_demo.add_argument(
        "--dry-run",
        action="store_true",
        help="Run offline with the stub model (this is the default).",
    )
    p_demo.add_argument("--index-dir", help="Where to store the demo index.")
    p_demo.add_argument("--top-k", type=int, help="Chunks retrieved per query.")
    p_demo.add_argument("--max-iterations", type=int, help="Max reflection cycles.")
    p_demo.add_argument("--output", "-o", help="Write the Markdown report to this file.")
    p_demo.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        console.print("\n[yellow]Interrupted.[/]")
        return 130
    except Exception as exc:  # noqa: BLE001 - surface a clean message to the user
        console.print(f"[red]Error:[/] {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

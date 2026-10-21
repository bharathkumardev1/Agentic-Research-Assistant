"""Command line: ``research-assistant <command>``.

* ``ingest``    index papers from files or an arXiv search
* ``research``  answer a question from the index, with cited passages
* ``demo``      index the bundled sample papers and answer a sample question (offline unless --live)

Heavy imports happen inside each command, so ``--help`` is instant and a command never
needs another command's dependencies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .logging_utils import configure_logging

configure_logging()

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "examples" / "sample_papers"
SAMPLE_QUESTION = "How much does SciRAG reduce unsupported claims compared with a non-retrieval baseline?"


def _print_event(stage: str, info: dict) -> None:
    if stage == "retrieve":
        print(f"retrieve  {info['query']!r}: {info['new']} new passage(s), {info['total']} in context")
    elif stage == "answer":
        print(f"answer    cites {info['evidence'] or 'nothing'} of {info['passages']} passages")
    elif stage == "review":
        verdict = "sufficient" if info["sufficient"] else f"needs more: {info['missing']}"
        print(f"review    {verdict}")


def _settings(index_dir: Optional[str], top_k: Optional[int], max_rounds: Optional[int]):
    from .config import get_settings

    update: Dict[str, Any] = {}
    if index_dir:
        update["index_dir"] = Path(index_dir)
    if top_k:
        update["top_k"] = top_k
    if max_rounds:
        update["max_rounds"] = max_rounds
    return get_settings().model_copy(update=update)


def _cmd_ingest(args: argparse.Namespace) -> int:
    from .factory import build_index
    from .ingestion.loaders import fetch_arxiv, load_paths

    settings = _settings(args.index_dir, None, None)
    papers = []
    if args.paths:
        papers.extend(load_paths([Path(p) for p in args.paths]))
    if args.arxiv:
        papers.extend(fetch_arxiv(args.arxiv, max_results=args.arxiv_max, download_dir=settings.raw_dir))
    if not papers:
        print("No documents loaded. Give file paths or --arxiv.", file=sys.stderr)
        return 1
    store = build_index(papers, settings, dry_run=args.dry_run)
    store.save(settings.index_dir)
    print(f"Indexed {len(papers)} paper(s) into {len(store)} passages in {settings.index_dir} "
          f"({store.embeddings.name} embeddings)")
    return 0


def _run(question: str, settings, dry_run: bool, output: Optional[str], as_json: bool) -> int:
    from .factory import build_client, build_pipeline, load_index
    from .llm import Budget

    store = load_index(settings, dry_run=dry_run)
    client = build_client(settings, dry_run=dry_run, budget=Budget(settings.run_budget_usd))
    result = build_pipeline(store, client, settings, on_event=_print_event).run(question)
    print(result.model_dump_json(indent=2) if as_json else "\n" + result.report)
    if output:
        Path(output).write_text(result.report, encoding="utf-8")
    return 0


def _cmd_research(args: argparse.Namespace) -> int:
    question = " ".join(args.question).strip()
    if not question:
        print("Give a question.", file=sys.stderr)
        return 1
    return _run(question, _settings(args.index_dir, args.top_k, args.max_rounds), args.dry_run,
                args.output, args.json)


def _cmd_demo(args: argparse.Namespace) -> int:
    from .factory import build_index
    from .ingestion.loaders import load_paths

    dry_run = not args.live
    settings = _settings(args.index_dir or "data/index_demo", args.top_k, args.max_rounds)
    store = build_index(load_paths([SAMPLE_DIR]), settings, dry_run=dry_run)
    store.save(settings.index_dir)
    print(f"Indexed the sample papers into {len(store)} passages; "
          f"{'offline stub model' if dry_run else settings.answer_model}\n")
    question = " ".join(args.question).strip() if args.question else SAMPLE_QUESTION
    return _run(question, settings, dry_run, args.output, args.json)


def _cmd_qasper_download(args: argparse.Namespace) -> int:
    from .eval.qasper import download

    for name, digest in download(Path(args.dest)).items():
        print(f"{name}: sha256 {digest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-assistant",
                                     description="Answer questions about research papers, citing passages.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index papers")
    p.add_argument("paths", nargs="*", help="files or directories (.pdf, .txt, .md)")
    p.add_argument("--arxiv", help="arXiv search query to fetch and index")
    p.add_argument("--arxiv-max", type=int, default=5)
    p.add_argument("--index-dir")
    p.add_argument("--dry-run", action="store_true", help="hashing embeddings, no model download")
    p.set_defaults(func=_cmd_ingest)

    for name, func in (("research", _cmd_research), ("demo", _cmd_demo)):
        p = sub.add_parser(name, help="answer a question" if name == "research" else "run on the sample papers")
        p.add_argument("question", nargs="+" if name == "research" else "*")
        p.add_argument("--index-dir")
        p.add_argument("--top-k", type=int)
        p.add_argument("--max-rounds", type=int, help="1 runs a single pass with no review")
        p.add_argument("--output", "-o", help="write the Markdown report here")
        p.add_argument("--json", action="store_true", help="print the full result, trace included")
        if name == "research":
            p.add_argument("--dry-run", action="store_true", help="offline stub model and hashing embeddings")
        else:
            p.add_argument("--live", action="store_true", help="call the model instead of the offline stub")
        p.set_defaults(func=func)

    p = sub.add_parser("qasper-download", help="fetch the QASPER dev and test sets")
    p.add_argument("--dest", default="data/qasper")
    p.set_defaults(func=_cmd_qasper_download)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    import shlex

    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    args.command_line = shlex.join(["research-assistant", *argv])
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:  # noqa: BLE001 - a clean message instead of a traceback
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

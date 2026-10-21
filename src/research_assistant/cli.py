"""Command line: ``research-assistant <command>``.

* ``ingest``    index papers from files or an arXiv search
* ``research``  answer a question from the index, with cited passages
* ``demo``      index the bundled sample papers and answer a sample question (offline unless --live)
* ``qasper-download``, ``eval``, ``check-replay``, ``label-judge``, ``judge-agreement``  the QASPER evaluation

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


def _cmd_eval(args: argparse.Namespace) -> int:
    import json

    from .eval import qasper as Q
    from .eval.report import meta, write_markdown
    from .eval.run import run, sample
    from .eval.runlog import RunLog
    from .llm import Budget, OpenAIClient, Recorder
    from .rag.embeddings import get_embeddings

    settings = _settings(None, args.top_k, None)
    papers = Q.load(Path(args.data_dir) / Q.FILES[args.split])
    questions = sample(papers, args.sample, args.seed)
    recorder = Recorder(Path(args.record))
    log = RunLog(Path(args.log), write=not args.replay)
    budget = Budget(args.cap, spent_usd=recorder.recorded_spend_usd)
    if not args.replay:
        recorder.client = OpenAIClient(settings.require_api_key(), settings.call_timeout_seconds,
                                       settings.call_attempts, budget)
    embeddings = get_embeddings(settings.embedding_backend, settings.embedding_model_dir, settings.hashing_dim)

    def progress(row: dict) -> None:
        print(f"{row['question_id']}  {row['system']:<11}  F1 {row['answer_f1']:.2f}  evidence F1 "
              f"{row['evidence_f1']:.2f}  {row['verdict']:<9}  {row['status']}  ${row['cost_usd']:.4f}")

    res = run(papers, questions, recorder, embeddings, settings.answer_model, settings.review_model,
              settings.judge_model, settings.top_k, settings.run_deadline_seconds, settings.reasoning_effort or None,
              on_row=progress, log=log)
    res = {"meta": meta(args.command_line, settings, embeddings, Path(args.data_dir)), "split": args.split,
           "sample": args.sample, "seed": args.seed, "questions": len(questions),
           "calls": {"made": recorder.made, "replayed": recorder.replayed},
           "recorded_spend_usd": round(budget.spent_usd, 6), "cap_usd": args.cap, **res}
    out = Path(args.out or f"results/qasper-{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    write_markdown(out.parent, out.parent / "RESULTS.md")
    for system, s in res["summary"]["systems"].items():
        print(f"{system:<11} answer F1 {s['answer_f1']:.3f}  evidence F1 {s['evidence_f1']:.3f}  "
              f"judge {s['judge_accuracy']:.3f}  ${s['usd_per_question']:.4f}/question  p95 {s['seconds_p95']:.1f}s")
    if res["stopped"]:
        print(f"stopped early: {res['stopped']}", file=sys.stderr)
        return 2
    return 0


def _cmd_check_replay(args: argparse.Namespace) -> int:
    import json

    from .eval import qasper as Q
    from .eval.report import meta
    from .eval.run import differences, run, sample
    from .eval.runlog import RunLog
    from .llm import Recorder
    from .rag.embeddings import get_embeddings

    expected = json.loads(Path(args.results).read_text())
    m = expected["meta"]
    settings = _settings(None, m["top_k"], None).model_copy(update={
        "answer_model": m["answer_model"], "review_model": m["review_model"], "judge_model": m["judge_model"],
        "reasoning_effort": m["reasoning_effort"]})
    embeddings = get_embeddings(settings.embedding_backend, settings.embedding_model_dir, settings.hashing_dim)
    if embeddings.name != m["embeddings"]:
        print(f"{args.results} was made with {m['embeddings']} embeddings, not {embeddings.name}; "
              f"set EMBEDDING_BACKEND to match", file=sys.stderr)
        return 1
    papers = Q.load(Path(args.data_dir) / Q.FILES[expected["split"]])
    only = {(r["question_id"], r["system"]) for r in expected["rows"]}
    res = run(papers, sample(papers, expected["sample"], expected["seed"]), Recorder(Path(args.record)), embeddings,
              m["answer_model"], m["review_model"], m["judge_model"], m["top_k"], m["deadline_seconds"],
              m["reasoning_effort"] or None, log=RunLog(Path(args.log), write=False), only=only)
    res["meta"] = meta(args.command_line, settings, embeddings, Path(args.data_dir))
    diffs = differences(expected, res)
    for line in diffs[:50]:
        print(line, file=sys.stderr)
    if diffs:
        print(f"{len(diffs)} difference(s) from {args.results}", file=sys.stderr)
        return 1
    print(f"{args.results}: {len(res['rows'])} answers replayed with no key; every score matches")
    return 0


def _cmd_label_judge(args: argparse.Namespace) -> int:
    import csv
    import json

    from .eval import qasper as Q
    from .eval.run import label_sheet

    results = json.loads(Path(args.results).read_text())
    questions = {q.question_id: q for p in Q.load(Path(args.data_dir) / Q.FILES[results["split"]])
                 for q in p.questions}
    rows = label_sheet(results, questions, args.n)
    with open(args.out, "w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} answers to grade in {args.out}: put correct or incorrect in your_verdict")
    return 0


def _cmd_judge_agreement(args: argparse.Namespace) -> int:
    import json

    from .eval.report import write_markdown
    from .eval.run import agreement

    path = Path(args.results)
    results = json.loads(path.read_text())
    results["judge_check"] = {"labels": args.labels, **agreement(results, args.labels)}
    path.write_text(json.dumps(results, indent=1) + "\n")
    write_markdown(path.parent, path.parent / "RESULTS.md")
    print(json.dumps(results["judge_check"]))
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

    p = sub.add_parser("eval", help="the review loop against a single pass on QASPER")
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--sample", type=int, default=150, help="questions, drawn with --seed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data/qasper")
    p.add_argument("--top-k", type=int)
    p.add_argument("--record", default="results/qasper/calls.jsonl", help="model replies")
    p.add_argument("--log", default="results/qasper/runs.jsonl", help="searches and how each run stopped")
    p.add_argument("--replay", action="store_true", help="recorded replies only; no API key needed")
    p.add_argument("--cap", type=float, default=25.0, help="USD, over everything in the recording")
    p.add_argument("--out", help="default results/qasper-<split>.json")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("check-replay", help="rerun a results file from its recording and compare, no key needed")
    p.add_argument("results", help="e.g. results/qasper-test.json")
    p.add_argument("--data-dir", default="data/qasper")
    p.add_argument("--record", default="results/qasper/calls.jsonl")
    p.add_argument("--log", default="results/qasper/runs.jsonl")
    p.set_defaults(func=_cmd_check_replay)

    p = sub.add_parser("label-judge", help="sample judged answers for a person to grade")
    p.add_argument("--results", default="results/qasper-test.json")
    p.add_argument("--data-dir", default="data/qasper")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--out", default="results/qasper/labels.csv")
    p.set_defaults(func=_cmd_label_judge)

    p = sub.add_parser("judge-agreement", help="how often the judge agrees with your grades")
    p.add_argument("--results", default="results/qasper-test.json")
    p.add_argument("--labels", default="results/qasper/labels.csv")
    p.set_defaults(func=_cmd_judge_agreement)
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

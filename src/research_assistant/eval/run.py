"""QASPER evaluation: the review loop against a single pass, on the same questions.

For each sampled question, the paper's paragraphs are embedded with MiniLM and searched,
and both systems answer through the same pipeline: ``single pass`` (retrieve, answer) and
``loop`` (retrieve, answer, review, up to three rounds). Each answer is scored with the
official answer F1 and evidence F1, plus evidence precision and recall; a judge model says
correct or incorrect against the annotators' answers, since token F1 punishes a correct
answer phrased differently. The trace gives each question's seconds, tokens and dollars,
and how the run stopped.

Questions are a fixed-seed sample of a split, so the dev and test runs can be repeated
exactly. Every model call goes through a :class:`~research_assistant.llm.Recorder`, and
every search and every run's stop goes to a :class:`~research_assistant.eval.runlog.RunLog`:
a rerun replays what was recorded, ``--replay`` needs no key, and the spending cap covers
everything in the recording. Hitting the cap stops the run and keeps what was finished.
:func:`differences` compares a replay with the results file it should reproduce.
"""

from __future__ import annotations

import csv
import random
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from ..agents.answerer import AnswerAgent
from ..agents.evaluator import EvaluatorAgent
from ..agents.retriever import RetrieverAgent
from ..graph.workflow import ResearchPipeline
from ..llm import LLMClient, Request
from ..rag.embeddings import EmbeddingBackend
from ..rag.vector_store import VectorStore
from . import qasper as Q
from .runlog import LoggedRetriever, RecordedStop, RunLog

SYSTEMS = {"single pass": 1, "loop": 3}
BOOTSTRAP = 10_000
# Reasoning tokens count against the output allowance, and a reply cut short fails the call.
# Raising it later would change every request's hash and void the recording, so it's generous
# from the start; only the tokens actually produced are billed.
OUTPUT_TOKENS = 2048
JUDGE_SYSTEM = (
    "You grade an answer to a question about a research paper. You get the question, the "
    "answers that annotators who read the paper wrote, and the answer to grade. It is correct "
    "if it gives the same information as any one of the annotators' answers: wording, length "
    "and extra detail don't matter, and numbers may differ only by rounding. When an "
    "annotator's answer is Unanswerable, an answer saying the paper doesn't say is correct. "
    "It is incorrect if it contradicts every annotator, answers a different question, or "
    "hedges between options."
)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["correct", "incorrect"]},
                   "reason": {"type": "string"}},
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def sample(papers: List[Q.Paper], n: Optional[int], seed: int = 0) -> List[Q.Question]:
    """``n`` questions drawn with a fixed seed (all of them when n is None), in a stable order."""
    questions = [q for p in papers for q in p.questions]
    if n is not None and n < len(questions):
        questions = random.Random(seed).sample(questions, n)
    return sorted(questions, key=lambda q: (q.paper_id, q.question_id))


def judge_request(question: Q.Question, answer: str, model: str) -> Request:
    refs = "\n".join(f"- {a.text}" for a in question.answers)
    return Request(model=model, system=JUDGE_SYSTEM,
                   prompt=f"Question: {question.text}\n\nAnnotators' answers:\n{refs}\n\nAnswer to grade: {answer}",
                   max_output_tokens=OUTPUT_TOKENS, json_schema=JUDGE_SCHEMA, reasoning_effort="low")


def judge(client: LLMClient, question: Q.Question, answer: str, f1: float, model: str) -> Dict[str, Any]:
    """Exact matches count as correct without a call; anything else goes to the judge."""
    if f1 == 1.0:
        return {"verdict": "correct", "reason": "exact match", "judged": False}
    reply = client.complete(judge_request(question, answer, model))
    out = reply.json()
    return {"verdict": out["verdict"], "reason": out["reason"], "judged": True, "judge_model": reply.model,
            "judge_cost_usd": reply.cost_usd}


def pipeline(retriever: RetrieverAgent, client: LLMClient, answer_model: str, review_model: str,
             reasoning_effort: Optional[str], deadline_seconds: Optional[float]) -> ResearchPipeline:
    """The pipeline both systems run; the replay export builds it here too, so its requests match."""
    return ResearchPipeline(retriever, AnswerAgent(client, answer_model, OUTPUT_TOKENS, reasoning_effort),
                            EvaluatorAgent(client, review_model, OUTPUT_TOKENS, reasoning_effort),
                            max_rounds=max(SYSTEMS.values()), deadline_seconds=deadline_seconds)


def failure(row: Dict[str, Any]) -> str:
    """For a wrong answer, the first thing that went wrong, in pipeline order.

    Correct comes first: when annotators disagree, declining can match the one who said
    unanswerable, and that answer is right even though others answered."""
    if row["verdict"] == "correct":
        return "correct"
    if row["gold_unanswerable"]:
        return "answered an unanswerable question"
    if row["declined"]:
        return "declined an answerable question"
    if not row["gold_retrieved"]:
        return "evidence not retrieved"
    if not row["gold_cited"]:
        return "evidence retrieved, not cited"
    return "cited the evidence, answer wrong"


def run(papers: List[Q.Paper], questions: List[Q.Question], client: LLMClient, embeddings: EmbeddingBackend,
        answer_model: str, review_model: str, judge_model: str, top_k: int = 5,
        deadline_seconds: Optional[float] = 120.0, reasoning_effort: Optional[str] = "low",
        on_row: Optional[Callable[[Dict[str, Any]], None]] = None, log: Optional[RunLog] = None,
        only: Optional[Set[Tuple[str, str]]] = None) -> Dict[str, Any]:
    """``only`` limits the run to these (question id, system) pairs, which is how a replay
    covers exactly what a results file has, even when the cap stopped that run partway."""
    by_paper = {p.paper_id: p for p in papers}
    stores: Dict[str, VectorStore] = {}
    rows: List[Dict[str, Any]] = []
    stopped = None
    for q in questions:
        paper = by_paper[q.paper_id]
        if q.paper_id not in stores:
            stores[q.paper_id] = VectorStore(embeddings)
            stores[q.paper_id].add(paper.paragraphs)
        retriever = (LoggedRetriever(stores[q.paper_id], top_k, log, q.paper_id) if log is not None
                     else RetrieverAgent(stores[q.paper_id], top_k))
        pipe = pipeline(retriever, client, answer_model, review_model, reasoning_effort, deadline_seconds)
        text_of = {c.chunk_id: c.text for c in paper.paragraphs}
        gold = set().union(*(a.evidence for a in q.answers))
        for system, rounds in SYSTEMS.items():
            if only is not None and (q.question_id, system) not in only:
                continue
            # Both systems send the same first-round request, so the loop's first answer is the
            # single pass's answer replayed: the comparison isolates what the review rounds add.
            stop = log.stop(q.question_id, system) if log is not None else None
            t0 = time.perf_counter()
            result = pipe.run(q.text, max_rounds=rounds, trace=RecordedStop(stop) if stop else None)
            if result.status == "budget":
                stopped = f"spending cap reached at {q.question_id} ({system})"
                break
            if log is not None and stop is None:
                log.add_stop(q.question_id, system, result.status, result.trace["calls"])
            declined = result.answer.unanswerable
            answer = Q.UNANSWERABLE if declined else result.answer.answer
            evidence = [] if declined else [text_of[s.chunk_id] for s in result.cited]
            retrieved = {text_of[s.chunk_id] for s in result.sources}
            row: Dict[str, Any] = {
                "question_id": q.question_id, "paper_id": q.paper_id, "system": system, "answer": answer,
                "evidence_ids": [] if declined else [s.chunk_id for s in result.cited],
                **Q.score(q, answer, evidence),
                "gold_unanswerable": q.unanswerable, "declined": declined,
                "gold_retrieved": bool(gold & retrieved), "gold_cited": bool(gold & set(evidence)),
                "status": result.status, "rounds": result.rounds, "seconds": round(time.perf_counter() - t0, 3),
                "calls": result.trace["calls"], "input_tokens": result.trace["input_tokens"],
                "output_tokens": result.trace["output_tokens"], "cost_usd": result.trace["cost_usd"],
                "retries": result.trace["retries"],
            }
            row.update(judge(client, q, answer, row["answer_f1"], judge_model))
            row["failure"] = failure(row)
            rows.append(row)
            if on_row:
                on_row(row)
        if stopped:
            break
    return {"systems": SYSTEMS, "rows": rows, "summary": summarize(rows), "stopped": stopped}


def bootstrap_ci(values: np.ndarray, n: int = BOOTSTRAP, seed: int = 0) -> List[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return [float(lo), float(hi)]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"systems": {}, "loop_minus_single": {}}
    for system in SYSTEMS:
        rs = [r for r in rows if r["system"] == system]
        if not rs:
            continue
        def col(k: str, rs: List[Dict[str, Any]] = rs) -> np.ndarray:
            return np.array([float(r[k]) for r in rs])

        correct = np.array([r["verdict"] == "correct" for r in rs], dtype=float)
        unans = [r for r in rs if r["gold_unanswerable"]]
        answerable = [r for r in rs if not r["gold_unanswerable"]]
        seconds = col("seconds")
        out["systems"][system] = {
            "n": len(rs), "answer_f1": float(col("answer_f1").mean()), "answer_f1_ci95": bootstrap_ci(col("answer_f1")),
            "evidence_f1": float(col("evidence_f1").mean()), "evidence_f1_ci95": bootstrap_ci(col("evidence_f1")),
            "evidence_precision": float(col("evidence_precision").mean()),
            "evidence_recall": float(col("evidence_recall").mean()),
            "judge_accuracy": float(correct.mean()), "judge_accuracy_ci95": bootstrap_ci(correct),
            "answer_f1_by_type": {k: float(np.mean([r["answer_f1"] for r in rs if r["answer_kind"] == k]))
                                  for k in sorted({r["answer_kind"] for r in rs})},
            "declined_when_unanswerable": (sum(r["declined"] for r in unans) / len(unans)) if unans else None,
            "declined_when_answerable": (sum(r["declined"] for r in answerable) / len(answerable)) if answerable else None,
            "seconds_p50": float(np.percentile(seconds, 50)), "seconds_p95": float(np.percentile(seconds, 95)),
            "usd_per_question": float(col("cost_usd").mean() + np.mean([r.get("judge_cost_usd", 0.0) for r in rs])),
            "answering_usd_per_question": float(col("cost_usd").mean()),
            "calls_per_question": float(col("calls").mean()), "rounds_mean": float(col("rounds").mean()),
            "status": {s: sum(r["status"] == s for r in rs) for s in sorted({r["status"] for r in rs})},
            "failures": {f: sum(r["failure"] == f for r in rs) for f in sorted({r["failure"] for r in rs})},
        }
    pairs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        pairs.setdefault(r["question_id"], {})[r["system"]] = r
    both = [p for p in pairs.values() if len(p) == len(SYSTEMS)]
    def value(r: Dict[str, Any], metric: str) -> float:
        return float(r["verdict"] == "correct") if metric == "verdict" else float(r[metric])

    for metric in ("answer_f1", "evidence_f1", "verdict"):
        if not both:
            break
        d = np.array([value(p["loop"], metric) - value(p["single pass"], metric) for p in both])
        name = "judge_accuracy" if metric == "verdict" else metric
        out["loop_minus_single"][name] = {"diff": float(d.mean()), "ci95": bootstrap_ci(d),
                                          "loop_better": int((d > 0).sum()), "single_better": int((d < 0).sum())}
    return out


def label_sheet(results: Dict[str, Any], questions: Dict[str, Q.Question], n: int = 40, seed: int = 0
                ) -> List[Dict[str, str]]:
    """Judged answers, verdict hidden, for a person to grade; exact matches are left out."""
    judged = [r for r in results["rows"] if r.get("judged")]
    chosen = random.Random(seed).sample(judged, min(n, len(judged)))
    return [{"question_id": r["question_id"], "system": r["system"], "question": questions[r["question_id"]].text,
             "annotators": " | ".join(a.text for a in questions[r["question_id"]].answers),
             "answer": r["answer"], "your_verdict": ""} for r in chosen]


def agreement(results: Dict[str, Any], labels_csv: str) -> Dict[str, Any]:
    verdicts = {(r["question_id"], r["system"]): r["verdict"] for r in results["rows"]}
    with open(labels_csv, newline="", encoding="utf8") as fh:
        labels = [row for row in csv.DictReader(fh) if row["your_verdict"].strip()]
    pairs = [(row["your_verdict"].strip().lower(), verdicts[(row["question_id"], row["system"])]) for row in labels]
    if not pairs:
        return {"labeled": 0}
    po = sum(a == b for a, b in pairs) / len(pairs)
    pc = sum((sum(a == c for a, _ in pairs) / len(pairs)) * (sum(b == c for _, b in pairs) / len(pairs))
             for c in ("correct", "incorrect"))
    return {"labeled": len(pairs), "agreement": po, "cohens_kappa": (po - pc) / (1 - pc) if pc < 1 else 1.0,
            "judge_lenient": sum(a == "incorrect" and b == "correct" for a, b in pairs),
            "judge_strict": sum(a == "correct" and b == "incorrect" for a, b in pairs)}


TIMING = {"seconds", "seconds_p50", "seconds_p95"}  # wall time; a replay doesn't wait on the API


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= 1e-9
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a if k not in TIMING)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def differences(expected: Dict[str, Any], actual: Dict[str, Any]) -> List[str]:
    """Everything a replay changed relative to a results file, timings aside."""
    out = []
    want, got = expected["meta"].get("qasper_tarballs_sha256"), actual["meta"].get("qasper_tarballs_sha256")
    if want and got != want:
        out.append(f"QASPER tarballs: expected sha256 {want}, downloaded {got}")
    rows = {(r["question_id"], r["system"]): r for r in actual["rows"]}
    for r in expected["rows"]:
        key = (r["question_id"], r["system"])
        if key not in rows:
            out.append(f"{key}: not in the replay")
            continue
        out += [f"{key} {k}: expected {v!r}, replayed {rows[key].get(k)!r}"
                for k, v in r.items() if k not in TIMING and not _same(v, rows[key].get(k))]
    extra = set(rows) - {(r["question_id"], r["system"]) for r in expected["rows"]}
    out += [f"{key}: in the replay only" for key in sorted(extra)]
    for name, summary in expected["summary"]["systems"].items():
        replayed = actual["summary"]["systems"].get(name, {})
        out += [f"summary {name} {k}: expected {v!r}, replayed {replayed.get(k)!r}"
                for k, v in summary.items() if k not in TIMING and not _same(v, replayed.get(k))]
    for k, v in expected["summary"]["loop_minus_single"].items():
        if not _same(v, actual["summary"]["loop_minus_single"].get(k)):
            out.append(f"loop minus single {k}: expected {v!r}, replayed "
                       f"{actual['summary']['loop_minus_single'].get(k)!r}")
    return out

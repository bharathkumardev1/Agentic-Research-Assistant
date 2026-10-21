"""Recorded QASPER runs as one JSON file for the portfolio site's replay.

The site doesn't call a model. It plays back what happened: for each chosen question, every
round's search query, the passages it found, the answer and the passages it cited, the
review, and each round's model seconds and dollars, next to the single pass on the same
question and the annotators' answers. The runs are re-executed from the recorded replies,
so the file shows exactly what the evaluation scored, and building it needs no key.

Questions are picked to show the loop honestly: ones where review fixed a wrong answer,
ones where it broke a right one, and ones where it changed nothing, in question-id order.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..llm import LLMClient
from ..rag.embeddings import EmbeddingBackend
from ..rag.vector_store import VectorStore
from . import qasper as Q
from .run import SYSTEMS, pipeline
from .runlog import LoggedRetriever, RecordedStop, RunLog

OUTCOMES = ["review fixed it", "review broke it", "both wrong", "both right"]


def outcome(single: Dict[str, Any], loop: Dict[str, Any]) -> str:
    a, b = single["verdict"] == "correct", loop["verdict"] == "correct"
    if b and not a:
        return "review fixed it"
    if a and not b:
        return "review broke it"
    return "both right" if a else "both wrong"


def pick(results: Dict[str, Any], n: int = 6) -> List[str]:
    """Up to ``n`` question ids, taking each outcome in turn so every kind shows up."""
    pairs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in results["rows"]:
        pairs.setdefault(r["question_id"], {})[r["system"]] = r
    by_outcome: Dict[str, List[str]] = {}
    for qid in sorted(pairs):
        p = pairs[qid]
        if "single pass" in p and "loop" in p:
            by_outcome.setdefault(outcome(p["single pass"], p["loop"]), []).append(qid)
    picked: List[str] = []
    while len(picked) < n and any(by_outcome.get(k) for k in OUTCOMES):
        for k in OUTCOMES:
            if by_outcome.get(k) and len(picked) < n:
                picked.append(by_outcome[k].pop(0))
    return picked


def rounds(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One entry per round. Seconds are the recorded model time; search time on this
    machine isn't the original run's, so it's left out."""
    out: Dict[int, Dict[str, Any]] = {}
    for step in trace["steps"]:
        r = out.setdefault(step["round"], {"round": step["round"], "model_seconds": 0.0, "cost_usd": 0.0})
        r["model_seconds"] += sum(c["seconds"] for c in step["calls"])
        r["cost_usd"] += sum(c["cost_usd"] for c in step["calls"])
        info = step["info"]
        if step["node"] == "retrieve":
            r["query"], r["found"], r["new"] = info["query"], info["found"], info["new"]
        elif step["node"] == "answer":
            r["answer"] = {"text": info["answer"], "unanswerable": info["unanswerable"], "cited": info["cited"]}
        elif step["node"] == "review":
            r["review"] = {k: info[k] for k in ("sufficient", "grounded", "missing", "refined_query")}
    return [{**v, "model_seconds": round(v["model_seconds"], 3), "cost_usd": round(v["cost_usd"], 6)}
            for _, v in sorted(out.items())]


def export(papers: List[Q.Paper], results: Dict[str, Any], client: LLMClient, log: RunLog,
           embeddings: EmbeddingBackend, question_ids: List[str], top_k: int, reasoning_effort: str | None
           ) -> Dict[str, Any]:
    meta = results["meta"]
    questions = {q.question_id: q for p in papers for q in p.questions}
    by_paper = {p.paper_id: p for p in papers}
    rows = {(r["question_id"], r["system"]): r for r in results["rows"]}
    runs = []
    for qid in question_ids:
        q = questions[qid]
        paper = by_paper[q.paper_id]
        store = VectorStore(embeddings)
        store.add(paper.paragraphs)
        pipe = pipeline(LoggedRetriever(store, top_k, log, q.paper_id), client, meta["answer_model"],
                        meta["review_model"], reasoning_effort, None)
        systems: Dict[str, Any] = {}
        for system, max_rounds in SYSTEMS.items():
            row = rows[(qid, system)]
            stop = log.stop(qid, system) or {"status": row["status"], "calls": row["calls"]}
            result = pipe.run(q.text, max_rounds=max_rounds, trace=RecordedStop(stop))
            systems[system] = {
                "status": result.status, "rounds": rounds(result.trace),
                "final": {"answer": row["answer"], "cited": row["evidence_ids"]},
                "scores": {k: row[k] for k in ("answer_f1", "evidence_f1", "verdict", "failure")},
                "seconds": row["seconds"], "cost_usd": row["cost_usd"],
            }
            if result.status != row["status"] or [s.chunk_id for s in result.cited] != row["evidence_ids"]:
                raise ValueError(f"{qid} ({system}) replayed differently from the results file")
        seen = {cid for s in systems.values() for r in s["rounds"] for cid in r.get("found", [])}
        gold = set().union(*(a.evidence for a in q.answers))
        passages = {c.chunk_id: {"section": c.section, "text": c.text, "annotator_evidence": c.text in gold}
                    for c in paper.paragraphs if c.chunk_id in seen}
        runs.append({"question_id": qid, "paper_id": q.paper_id, "paper_title": paper.title,
                     "question": q.text, "annotators": [{"answer": a.text, "kind": a.kind} for a in q.answers],
                     "outcome": outcome(rows[(qid, "single pass")], rows[(qid, "loop")]),
                     "passages": passages, **systems})
    return {"source": {"command": meta["command"], "git_sha": meta["git_sha"], "date": meta["date"],
                       "answer_model": meta["answer_model"], "judge_model": meta["judge_model"],
                       "split": results["split"]},
            "summary": results["summary"], "runs": runs}

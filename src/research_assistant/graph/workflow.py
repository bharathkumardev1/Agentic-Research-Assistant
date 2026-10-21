"""The answer loop as a LangGraph graph.

    START -> retrieve -> answer --(single pass)--------------------------------> END
                ^          |
                |          +--> review --(sufficient, round cap, deadline)-----> END
                |                  |
                +--(refined query; stop if it finds no new passage)--+

Four things bound a run, and each one ends it with the best answer so far and a status
that says why it stopped:

* the round cap (``max_rounds``; 1 is the single-pass baseline),
* a refined query that finds no passage the run hasn't seen ("no new passages"): answering
  again from the same passages would only repeat the last answer,
* the deadline, checked before every model call (``deadline_seconds``),
* the budget, which the client enforces before every call.

Retries of individual calls happen in the client and show up in the trace.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from ..agents.answerer import AnswerAgent
from ..agents.evaluator import EvaluatorAgent
from ..agents.retriever import RetrieverAgent
from ..llm import BudgetExceeded
from ..logging_utils import get_logger, new_run_id
from ..report import render_report
from ..schemas import Answer, ResearchResult, RetrievedChunk, SourceRef
from ..trace import Trace
from .state import ResearchState

EventHook = Optional[Callable[[str, dict], None]]
logger = get_logger("workflow")
NO_ANSWER = Answer(answer="", evidence=[], unanswerable=True)


def merge_retrieved(existing: List[RetrievedChunk], new: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Union by chunk id, keeping each chunk's best score, best first."""
    best: dict[str, RetrievedChunk] = {}
    for item in [*existing, *new]:
        key = item.chunk.chunk_id
        if key not in best or item.score > best[key].score:
            best[key] = item
    return sorted(best.values(), key=lambda r: (-r.score, r.chunk.chunk_id))


def build_graph(retriever: RetrieverAgent, answerer: AnswerAgent, evaluator: EvaluatorAgent,
                on_event: EventHook = None):
    from langgraph.graph import END, START, StateGraph

    def emit(stage: str, info: dict) -> None:
        if on_event is not None:
            on_event(stage, info)

    def retrieve_node(state: ResearchState) -> dict:
        trace, rnd = state["trace"], state["round"] + 1
        with trace.step("retrieve", rnd) as step:
            found = retriever.retrieve(state["query"])
            before = {r.chunk.chunk_id for r in state.get("retrieved", [])}
            merged = merge_retrieved(state.get("retrieved", []), found)
            new = sum(r.chunk.chunk_id not in before for r in found)
            step.info = {"query": state["query"], "new": new, "total": len(merged)}
        emit("retrieve", step.info)
        update = {"retrieved": merged, "queries": [*state.get("queries", []), state["query"]], "round": rnd}
        if rnd > 1 and new == 0:  # the last answer already saw every passage this query found
            update.update(status="no new passages", round=rnd - 1)
        return update

    def answer_node(state: ResearchState) -> dict:
        trace, rnd = state["trace"], state["round"]
        if trace.expired():
            return {"status": "deadline"}
        with trace.step("answer", rnd) as step:
            try:
                answer, sources, reply = answerer.answer(state["question"], state["retrieved"])
            except BudgetExceeded:
                return {"status": "budget"}
            if reply is not None:
                step.calls.append(reply)
            step.info = {"passages": len(sources), "evidence": answer.evidence,
                         "unanswerable": answer.unanswerable}
        emit("answer", step.info)
        return {"answer": answer, "sources": sources}

    def review_node(state: ResearchState) -> dict:
        trace, rnd = state["trace"], state["round"]
        if trace.expired():
            return {"status": "deadline"}
        with trace.step("review", rnd) as step:
            try:
                review, reply = evaluator.evaluate(state["question"], state["answer"], state["sources"])
            except BudgetExceeded:
                return {"status": "budget"}
            step.calls.append(reply)
            step.info = review.model_dump()
        emit("review", step.info)
        update: dict = {"reviews": [*state.get("reviews", []), review]}
        if not review.sufficient and review.refined_query:
            update["query"] = review.refined_query
        return update

    def after_retrieve(state: ResearchState) -> str:
        return "end" if state.get("status") else "answer"

    def after_answer(state: ResearchState) -> str:
        if state.get("status") or state["max_rounds"] == 1:
            return "end"
        return "review"

    def after_review(state: ResearchState) -> str:
        if state.get("status"):
            return "end"
        review = state["reviews"][-1]
        if review.sufficient or not review.refined_query:
            return "end"
        if state["round"] >= state["max_rounds"]:
            return "cap"
        return "retrieve"

    def cap_node(state: ResearchState) -> dict:
        return {"status": "round cap"}

    graph = StateGraph(ResearchState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("review", review_node)
    graph.add_node("cap", cap_node)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges("retrieve", after_retrieve, {"answer": "answer", "end": END})
    graph.add_conditional_edges("answer", after_answer, {"review": "review", "end": END})
    graph.add_conditional_edges("review", after_review, {"retrieve": "retrieve", "cap": "cap", "end": END})
    graph.add_edge("cap", END)
    return graph.compile()


class ResearchPipeline:
    def __init__(self, retriever: RetrieverAgent, answerer: AnswerAgent, evaluator: EvaluatorAgent,
                 max_rounds: int = 3, deadline_seconds: Optional[float] = 120.0,
                 on_event: EventHook = None) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.max_rounds = max_rounds
        self.deadline_seconds = deadline_seconds
        self.app = build_graph(retriever, answerer, evaluator, on_event=on_event)

    def run(self, question: str, max_rounds: Optional[int] = None, trace: Optional[Trace] = None
            ) -> ResearchResult:
        run_id = new_run_id()
        rounds = max_rounds or self.max_rounds
        trace = trace or Trace(self.deadline_seconds)
        logger.info("run start: run_id=%s rounds<=%d question=%r", run_id, rounds, question)
        final = self.app.invoke(
            {"question": question, "query": question, "queries": [], "retrieved": [], "reviews": [],
             "round": 0, "max_rounds": rounds, "trace": trace},
            config={"recursion_limit": 3 * rounds + 4},
        )
        answer: Answer = final.get("answer") or NO_ANSWER
        sources: List[SourceRef] = final.get("sources", [])
        cited = [s for s in sources if s.marker in set(answer.evidence)]
        status = final.get("status") or "answered"
        logger.info("run done: run_id=%s status=%s rounds=%d", run_id, status, final.get("round", 0))
        result = ResearchResult(question=question, answer=answer, cited=cited, sources=sources,
                                reviews=final.get("reviews", []), queries=final.get("queries", []),
                                rounds=final.get("round", 0), status=status, trace=trace.to_dict())
        result.report = render_report(result)
        return result

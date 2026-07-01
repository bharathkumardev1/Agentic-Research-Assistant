"""Build and run the agentic research graph.

Topology::

    START → retrieve → summarize → evaluate ─┬─(needs_more & budget left)→ retrieve
                                              └─(sufficient | budget spent)→ compose → END

The evaluator's verdict drives the conditional edge, giving the system a
retrieve→reflect→retrieve loop bounded by ``max_iterations``.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from ..agents.evaluator import EvaluatorAgent
from ..agents.retriever import RetrieverAgent
from ..agents.summarizer import SummarizerAgent
from ..report import render_report
from ..schemas import ResearchResult, RetrievedChunk
from .state import ResearchState

EventHook = Optional[Callable[[str, dict], None]]


def _merge_retrieved(
    existing: List[RetrievedChunk], new: List[RetrievedChunk]
) -> List[RetrievedChunk]:
    """Union two lists of retrieved chunks by ``chunk_id``, keeping the best score."""
    best: dict[str, RetrievedChunk] = {}
    for item in [*existing, *new]:
        key = item.chunk.chunk_id
        if key not in best or item.score > best[key].score:
            best[key] = item
    return sorted(best.values(), key=lambda r: r.score, reverse=True)


def build_graph(
    retriever: RetrieverAgent,
    summarizer: SummarizerAgent,
    evaluator: EvaluatorAgent,
    on_event: EventHook = None,
):
    """Compile and return the LangGraph application for a given set of agents."""
    from langgraph.graph import END, START, StateGraph

    def emit(stage: str, info: dict) -> None:
        if on_event is not None:
            on_event(stage, info)

    def retrieve_node(state: ResearchState) -> dict:
        query = state["current_query"]
        emit("retrieve", {"query": query, "iteration": state.get("iteration", 0)})
        found = retriever.retrieve(query)
        merged = _merge_retrieved(state.get("retrieved", []), found)
        emit("retrieved", {"new": len(found), "total": len(merged)})
        return {
            "retrieved": merged,
            "queries": [*state.get("queries", []), query],
        }

    def summarize_node(state: ResearchState) -> dict:
        emit("summarize", {"chunks": len(state.get("retrieved", []))})
        summary, sources = summarizer.summarize(
            state["question"], state.get("retrieved", [])
        )
        return {"summary": summary, "sources": sources}

    def evaluate_node(state: ResearchState) -> dict:
        iteration = state.get("iteration", 0) + 1
        evaluation = evaluator.evaluate(
            state["question"], state["summary"], state.get("sources", [])
        )
        emit(
            "evaluate",
            {
                "iteration": iteration,
                "sufficiency": evaluation.sufficiency,
                "coverage": evaluation.coverage_score,
            },
        )
        update: dict = {"evaluation": evaluation, "iteration": iteration}
        if evaluation.sufficiency == "needs_more" and evaluation.refined_query:
            update["current_query"] = evaluation.refined_query
        return update

    def compose_node(state: ResearchState) -> dict:
        report = render_report(
            question=state["question"],
            summary=state["summary"],
            sources=state.get("sources", []),
            evaluation=state.get("evaluation"),
            queries=state.get("queries", []),
        )
        emit("compose", {"length": len(report)})
        return {"report": report}

    def route(state: ResearchState) -> str:
        evaluation = state["evaluation"]
        if (
            evaluation.sufficiency == "sufficient"
            or state.get("iteration", 0) >= state.get("max_iterations", 3)
            or not evaluation.refined_query
        ):
            return "compose"
        return "retrieve"

    graph = StateGraph(ResearchState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("compose", compose_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "summarize")
    graph.add_edge("summarize", "evaluate")
    graph.add_conditional_edges(
        "evaluate", route, {"retrieve": "retrieve", "compose": "compose"}
    )
    graph.add_edge("compose", END)
    return graph.compile()


class ResearchPipeline:
    """High-level facade: assemble the agents, run the graph, return a result."""

    def __init__(
        self,
        retriever: RetrieverAgent,
        summarizer: SummarizerAgent,
        evaluator: EvaluatorAgent,
        max_iterations: int = 3,
        on_event: EventHook = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.app = build_graph(retriever, summarizer, evaluator, on_event=on_event)

    def run(self, question: str) -> ResearchResult:
        """Execute the full agentic loop for ``question``."""
        initial: ResearchState = {
            "question": question,
            "current_query": question,
            "queries": [],
            "retrieved": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
        }
        # Allow enough supersteps for max_iterations full loops plus compose.
        final = self.app.invoke(
            initial, config={"recursion_limit": 4 * self.max_iterations + 5}
        )
        return ResearchResult(
            question=question,
            summary=final["summary"],
            sources=final.get("sources", []),
            evaluation=final.get("evaluation"),
            queries=final.get("queries", []),
            iterations=final.get("iteration", 0),
            report=final.get("report", ""),
        )

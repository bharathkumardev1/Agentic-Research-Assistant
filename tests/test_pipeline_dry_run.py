"""The answer loop end to end, offline: hashing embeddings and the stub model."""

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")

from research_assistant.config import Settings  # noqa: E402
from research_assistant.factory import build_client, build_index, build_pipeline  # noqa: E402
from research_assistant.schemas import PaperMetadata  # noqa: E402


def _papers():
    return [
        (PaperMetadata(paper_id="rag1", title="RAG Systems", source="rag1.txt"),
         "Abstract. Retrieval augmented generation grounds answers in retrieved passages. Methods. We "
         "build an index over chunked documents and retrieve the top passages for each query. Results. "
         "Grounding reduced hallucination by 40% on our benchmark."),
        (PaperMetadata(paper_id="agt1", title="Agent Loops", source="agt1.txt"),
         "Abstract. A reflective loop critiques and revises answers. Methods. A reviewer asks for "
         "another retrieval round when evidence is thin. Results. Iterative refinement improved answer "
         "completeness. Limitations. The loop adds latency and token cost."),
    ]


@pytest.fixture()
def settings():
    return Settings(max_rounds=3, top_k=2, hashing_dim=512, chunk_size=200, chunk_overlap=0)


def test_the_loop_reviews_refines_and_stops_when_the_reviewer_is_satisfied(settings):
    store = build_index(_papers(), settings, dry_run=True)
    events = []
    pipeline = build_pipeline(store, build_client(settings, dry_run=True), settings,
                              on_event=lambda s, info: events.append(s))
    result = pipeline.run("How much did grounding reduce hallucination?")
    assert result.status == "answered" and result.rounds == 2
    assert result.answer.answer and result.cited and len(result.queries) == 2
    assert events == ["retrieve", "answer", "review", "retrieve", "answer", "review"]
    assert [s["node"] for s in result.trace["steps"]] == ["retrieve", "answer", "review"] * 2
    assert result.trace["calls"] == 4 and "Stopped after 2 round(s): answered" in result.report


def test_single_pass_answers_once_without_a_review(settings):
    store = build_index(_papers(), settings, dry_run=True)
    pipeline = build_pipeline(store, build_client(settings, dry_run=True), settings)
    result = pipeline.run("What are the limitations?", max_rounds=1)
    assert result.rounds == 1 and result.reviews == [] and result.trace["calls"] == 1

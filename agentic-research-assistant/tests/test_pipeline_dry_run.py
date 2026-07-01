"""End-to-end test of the agentic loop in offline (dry-run) mode.

Exercises the full LangGraph topology with the hashing embeddings and the
stub LLM client: no API key, no model download, no network. Skipped when the
heavy optional dependencies are not installed.
"""

import pytest

pytest.importorskip("faiss", reason="faiss-cpu not installed")
pytest.importorskip("langgraph", reason="langgraph not installed")

from research_assistant.config import Settings  # noqa: E402
from research_assistant.factory import build_client, build_index, build_pipeline  # noqa: E402
from research_assistant.schemas import PaperMetadata  # noqa: E402


def _papers():
    return [
        (
            PaperMetadata(paper_id="rag1", title="RAG Systems", source="rag1.txt"),
            "Abstract. Retrieval augmented generation grounds answers in retrieved "
            "passages. Methods. We build a FAISS index over chunked documents and "
            "retrieve the top passages for each query. Results. Grounding reduced "
            "hallucination substantially on our benchmark. Limitations. We did not "
            "test on multilingual corpora.",
        ),
        (
            PaperMetadata(paper_id="agt1", title="Agent Loops", source="agt1.txt"),
            "Abstract. A reflective multi-agent loop critiques and revises drafts. "
            "Methods. An evaluator agent scores coverage and requests another "
            "retrieval round when evidence is thin. Results. Iterative refinement "
            "improved answer completeness. Limitations. The loop adds latency and "
            "token cost.",
        ),
    ]


@pytest.fixture()
def settings():
    return Settings(max_iterations=3, top_k=4, hashing_dim=512)


def test_dry_run_pipeline_produces_grounded_result(settings):
    store = build_index(_papers(), settings, dry_run=True)
    client = build_client(settings, dry_run=True)
    assert client.__class__.__name__ == "StubClaudeClient"

    events = []
    pipeline = build_pipeline(
        store, client, settings, on_event=lambda s, info: events.append(s)
    )
    result = pipeline.run("How do retrieval and multi-agent loops improve answers?")

    # The summary and structured fields are populated.
    assert result.summary.summary
    assert result.summary.methods
    assert result.summary.key_findings
    assert result.sources

    # The evaluator ran and the loop iterated at least once.
    assert result.evaluation is not None
    assert result.iterations >= 1

    # The stub forces one refinement, so we expect more than one query.
    assert len(result.queries) >= 2

    # The report was composed and contains the expected sections.
    assert "## Summary" in result.report
    assert "## References" in result.report

    # Key lifecycle events were emitted.
    assert "retrieve" in events
    assert "summarize" in events
    assert "evaluate" in events
    assert "compose" in events


def test_dry_run_respects_iteration_budget(settings):
    settings = settings.model_copy(update={"max_iterations": 1})
    store = build_index(_papers(), settings, dry_run=True)
    client = build_client(settings, dry_run=True)
    pipeline = build_pipeline(store, client, settings)
    result = pipeline.run("What are the limitations?")
    # With a budget of one iteration the loop must terminate promptly.
    assert result.iterations <= 1
    assert result.report

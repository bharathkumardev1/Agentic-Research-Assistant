"""The evaluator must judge grounding against the passages the summarizer cited, in full."""

from research_assistant.agents.evaluator import EvaluatorAgent
from research_assistant.context import number_sources
from research_assistant.schemas import Chunk, PaperSummary, RetrievedChunk

VERDICT = {
    "grounded": True,
    "coverage_score": 0.9,
    "sufficiency": "sufficient",
    "critique": "",
    "missing_aspects": [],
    "refined_query": None,
}


class RecordingClient:
    def __init__(self) -> None:
        self.prompts: list = []

    def complete_json(self, *, prompt: str, **_):
        self.prompts.append(prompt)
        return VERDICT


def _retrieved():
    # The supporting number sits well past the 180-character preview.
    text = "Background on the benchmark and its setup. " * 8 + "Accuracy rose from 61% to 74%."
    chunk = Chunk(chunk_id="p::0", paper_id="p", title="Paper", text=text, chunk_index=0)
    return [RetrievedChunk(chunk=chunk, score=0.9)]


def test_evaluator_sees_the_whole_cited_passage():
    retrieved = _retrieved()
    _, sources = number_sources(retrieved)
    assert "74%" not in sources[0].preview
    client = RecordingClient()
    summary = PaperSummary(summary="Accuracy rose from 61% to 74% [1].")
    EvaluatorAgent(client, model="m").evaluate("What changed?", summary, sources, retrieved)
    assert "Accuracy rose from 61% to 74%." in client.prompts[0]
    assert "[1] (source: \"Paper\")" in client.prompts[0]


def test_evaluator_falls_back_to_previews_without_passages():
    retrieved = _retrieved()
    _, sources = number_sources(retrieved)
    client = RecordingClient()
    EvaluatorAgent(client, model="m").evaluate("q", PaperSummary(summary="s"), sources)
    assert sources[0].preview in client.prompts[0]

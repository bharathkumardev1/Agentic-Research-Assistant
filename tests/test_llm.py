"""Tests for JSON extraction and the offline stub LLM client."""

import pytest

from research_assistant.llm import LLMError, StubClaudeClient, extract_json


def test_extract_plain_json():
    assert extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_extract_json_from_fenced_block():
    raw = 'Here you go:\n```json\n{"k": [1, 2, 3]}\n```\nThanks!'
    assert extract_json(raw) == {"k": [1, 2, 3]}


def test_extract_json_from_surrounding_prose():
    raw = 'Sure. {"ok": true} Let me know if you need more.'
    assert extract_json(raw) == {"ok": True}


def test_extract_json_raises_on_garbage():
    with pytest.raises(LLMError):
        extract_json("absolutely not json at all")


def test_stub_summary_has_required_keys():
    stub = StubClaudeClient()
    prompt = (
        "Research question: what works?\n\n"
        '[1] (source: "Paper A")\n'
        "This sentence describes a concrete method used in the study. "
        "This second sentence reports a measurable result of the experiment.\n\n"
        '[2] (source: "Paper B")\n'
        "Another lengthy sentence that should be long enough to be captured here."
    )
    out = stub.complete_json(model="m", system="summarize the sources", prompt=prompt)
    assert set(out) == {"summary", "methods", "key_findings", "research_gaps"}
    assert out["summary"].startswith("[dry-run]")
    assert "[1]" in out["summary"]


def test_stub_evaluation_alternates_then_terminates():
    stub = StubClaudeClient()
    first = stub.complete_json(model="m", system="evaluate the critique", prompt="x")
    second = stub.complete_json(model="m", system="evaluate the critique", prompt="x")
    assert first["sufficiency"] == "needs_more"
    assert first["refined_query"]
    assert second["sufficiency"] == "sufficient"
    assert second["refined_query"] is None


def test_stub_complete_returns_json_string():
    stub = StubClaudeClient()
    text = stub.complete(model="m", system="summarize", prompt="[1] something here.")
    parsed = extract_json(text)
    assert "summary" in parsed

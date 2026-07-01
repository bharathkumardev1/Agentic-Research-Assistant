"""Tests for the Pydantic data models."""

import pytest
from pydantic import ValidationError

from research_assistant.schemas import (
    Chunk,
    Evaluation,
    PaperSummary,
    SourceRef,
)


def _chunk(text: str = "hello world") -> Chunk:
    return Chunk(
        chunk_id="p::0", paper_id="p", title="T", text=text, chunk_index=0
    )


def test_chunk_short_truncates_and_collapses_whitespace():
    chunk = _chunk("  many\n\n   spaces   here  ")
    assert chunk.short() == "many spaces here"


def test_chunk_short_adds_ellipsis_when_truncated():
    chunk = _chunk("word " * 100)
    preview = chunk.short(20)
    assert len(preview) <= 21  # 20 chars + ellipsis
    assert preview.endswith("…")


def test_coverage_score_must_be_within_unit_interval():
    with pytest.raises(ValidationError):
        Evaluation(grounded=True, coverage_score=1.5, sufficiency="sufficient")
    with pytest.raises(ValidationError):
        Evaluation(grounded=True, coverage_score=-0.1, sufficiency="needs_more")


def test_sufficiency_is_constrained_to_known_values():
    with pytest.raises(ValidationError):
        Evaluation(grounded=True, coverage_score=0.5, sufficiency="maybe")


def test_evaluation_defaults():
    ev = Evaluation(grounded=True, coverage_score=0.8, sufficiency="sufficient")
    assert ev.critique == ""
    assert ev.missing_aspects == []
    assert ev.refined_query is None


def test_paper_summary_defaults_to_empty_lists():
    summary = PaperSummary(summary="text")
    assert summary.methods == []
    assert summary.key_findings == []
    assert summary.research_gaps == []


def test_source_ref_roundtrips_json():
    ref = SourceRef(marker=1, paper_id="p", title="T", chunk_id="p::0")
    restored = SourceRef.model_validate_json(ref.model_dump_json())
    assert restored == ref

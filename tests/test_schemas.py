"""The data models."""

import pytest
from pydantic import ValidationError

from research_assistant.schemas import Answer, Chunk, Review, SourceRef


def _chunk(text: str = "hello world") -> Chunk:
    return Chunk(chunk_id="p::0", paper_id="p", title="T", text=text, chunk_index=0)


def test_chunk_short_truncates_and_collapses_whitespace():
    assert _chunk("  many\n\n   spaces   here  ").short() == "many spaces here"
    preview = _chunk("word " * 100).short(20)
    assert len(preview) == 23 and preview.endswith("...")


def test_answer_and_review_validate_model_output():
    assert Answer.model_validate({"answer": "yes", "evidence": [2], "unanswerable": False}).evidence == [2]
    with pytest.raises(ValidationError):
        Answer.model_validate({"evidence": [1]})
    review = Review(sufficient=False, grounded=True)
    assert review.refined_query == "" and review.missing == ""


def test_source_ref_roundtrips_json():
    ref = SourceRef(marker=1, paper_id="p", title="T", chunk_id="p::0")
    assert SourceRef.model_validate_json(ref.model_dump_json()) == ref

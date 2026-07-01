"""Tests for source numbering / citation alignment."""

from research_assistant.context import number_sources
from research_assistant.schemas import Chunk, RetrievedChunk


def _retrieved(n: int):
    items = []
    for i in range(n):
        chunk = Chunk(
            chunk_id=f"p{i}::0",
            paper_id=f"p{i}",
            title=f"Paper {i}",
            text=f"This is the body of passage number {i}.",
            chunk_index=0,
            source=f"/tmp/p{i}.txt",
        )
        items.append(RetrievedChunk(chunk=chunk, score=1.0 - i * 0.1))
    return items


def test_empty_input_yields_empty_outputs():
    block, sources = number_sources([])
    assert block == ""
    assert sources == []


def test_markers_are_one_indexed_and_contiguous():
    block, sources = number_sources(_retrieved(3))
    assert [s.marker for s in sources] == [1, 2, 3]
    assert "[1]" in block and "[2]" in block and "[3]" in block


def test_sources_align_with_chunks():
    retrieved = _retrieved(2)
    _, sources = number_sources(retrieved)
    for src, item in zip(sources, retrieved):
        assert src.chunk_id == item.chunk.chunk_id
        assert src.title == item.chunk.title
        assert src.paper_id == item.chunk.paper_id
        assert src.preview  # non-empty preview


def test_context_block_contains_titles_and_text():
    block, _ = number_sources(_retrieved(1))
    assert "Paper 0" in block
    assert "body of passage number 0" in block

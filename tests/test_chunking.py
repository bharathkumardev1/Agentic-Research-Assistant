"""Tests for the boundary-aware chunker."""

from research_assistant.ingestion.chunking import chunk_paper, chunk_text
from research_assistant.schemas import PaperMetadata


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  \t ") == []


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("A short paragraph that easily fits.", chunk_size=1100)
    assert chunks == ["A short paragraph that easily fits."]


def test_every_chunk_respects_max_size():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_no_empty_chunks_are_emitted():
    text = "Para one.\n\n\n\nPara two.\n\n   \n\nPara three is a little longer here."
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    assert chunks
    assert all(c.strip() for c in chunks)


def test_consecutive_chunks_share_overlap_context():
    sentences = ". ".join(f"Sentence number {i} carries some content" for i in range(40))
    chunks = chunk_text(sentences, chunk_size=160, overlap=50)
    assert len(chunks) >= 2
    # At least one boundary should share a token between neighbours.
    shared = False
    for a, b in zip(chunks, chunks[1:]):
        tail = set(a.split()[-5:])
        head = set(b.split()[:5])
        if tail & head:
            shared = True
            break
    assert shared


def test_overlong_word_is_hard_split():
    giant = "x" * 500
    chunks = chunk_text(giant, chunk_size=100, overlap=0)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == giant


def test_invalid_parameters_raise():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=100, overlap=100)


def test_chunk_paper_sets_ids_and_metadata():
    meta = PaperMetadata(paper_id="abc123", title="A Title", source="/tmp/a.txt")
    text = " ".join(f"token{i}" for i in range(800))
    chunks = chunk_paper(meta, text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    for i, ch in enumerate(chunks):
        assert ch.chunk_id == f"abc123::{i}"
        assert ch.paper_id == "abc123"
        assert ch.title == "A Title"
        assert ch.chunk_index == i
        assert ch.source == "/tmp/a.txt"

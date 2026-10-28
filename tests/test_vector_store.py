"""Tests for the exact numpy vector store."""

import pytest

from research_assistant.rag.embeddings import HashingEmbeddings  # noqa: E402
from research_assistant.rag.vector_store import VectorStore  # noqa: E402
from research_assistant.schemas import Chunk  # noqa: E402


def _chunks():
    texts = [
        "Retrieval augmented generation grounds language models in documents.",
        "FAISS performs efficient nearest neighbour search over dense vectors.",
        "A multi-agent loop lets one agent critique another agent's output.",
        "Chocolate chip cookies need butter, sugar, flour and a hot oven.",
    ]
    return [
        Chunk(
            chunk_id=f"p{i}::0",
            paper_id=f"p{i}",
            title=f"Doc {i}",
            text=t,
            chunk_index=0,
        )
        for i, t in enumerate(texts)
    ]


def _store():
    store = VectorStore(HashingEmbeddings(dim=512))
    store.add(_chunks())
    return store


def test_add_grows_the_index():
    store = _store()
    assert len(store) == 4
    assert not store.is_empty


def test_search_returns_descending_scores():
    store = _store()
    results = store.search("vector nearest neighbour search with FAISS", k=3)
    assert results
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_finds_the_relevant_document_first():
    store = _store()
    results = store.search("efficient nearest neighbour search over vectors", k=4)
    assert results[0].chunk.paper_id == "p1"


def test_search_k_is_capped_at_corpus_size():
    store = _store()
    results = store.search("anything", k=100)
    assert len(results) == 4


def test_empty_store_returns_no_results():
    store = VectorStore(HashingEmbeddings(dim=64))
    assert store.search("query", k=5) == []


def test_save_and_load_roundtrip(tmp_path):
    store = _store()
    store.save(tmp_path)
    assert (tmp_path / "vectors.npy").exists()
    assert (tmp_path / "chunks.jsonl").exists()
    assert (tmp_path / "meta.json").exists()

    reloaded = VectorStore.load(tmp_path, HashingEmbeddings(dim=512))
    assert len(reloaded) == len(store)
    original = store.search("multi-agent critique loop", k=2)
    restored = reloaded.search("multi-agent critique loop", k=2)
    assert [r.chunk.chunk_id for r in restored] == [r.chunk.chunk_id for r in original]


def test_load_rejects_dim_mismatch(tmp_path):
    store = _store()
    store.save(tmp_path)
    with pytest.raises(ValueError):
        VectorStore.load(tmp_path, HashingEmbeddings(dim=128))


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path, HashingEmbeddings(dim=512))


def test_ties_at_the_cutoff_go_to_the_earlier_passage(monkeypatch):
    class Flat:
        name, dim = "flat", 2

        def embed(self, texts):
            import numpy as np

            return np.tile(np.array([[1.0, 0.0]], dtype="float32"), (len(texts), 1))

    store = VectorStore(Flat())
    store.add([Chunk(chunk_id=f"p::{i}", paper_id="p", title="t", text=f"t{i}", chunk_index=i)
               for i in range(6)])
    expect = ["p::0", "p::1", "p::2"]
    assert [r.chunk.chunk_id for r in store.search("q", k=3)] == expect

    # Another valid partition (what a different numpy build may return) must not change it.
    import numpy as np

    monkeypatch.setattr(np, "argpartition", lambda a, kth: np.arange(len(a))[::-1].copy())
    assert [r.chunk.chunk_id for r in store.search("q", k=3)] == expect

"""The embedding backends: the hashing fallback, and MiniLM against recorded vectors."""

import json
from pathlib import Path

import numpy as np
import pytest

from research_assistant.rag.embeddings import (
    HashingEmbeddings,
    get_embeddings,
)


def test_output_shape_matches_dim():
    emb = HashingEmbeddings(dim=64)
    out = emb.embed(["hello world", "another document here"])
    assert out.shape == (2, 64)
    assert out.dtype == np.float32


def test_vectors_are_unit_norm():
    emb = HashingEmbeddings(dim=128)
    out = emb.embed(["the quick brown fox", "lorem ipsum dolor sit amet"])
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embedding_is_deterministic():
    emb = HashingEmbeddings(dim=128)
    a = emb.embed(["reproducible text"])
    b = emb.embed(["reproducible text"])
    assert np.array_equal(a, b)


def test_similar_text_scores_higher_than_unrelated():
    emb = HashingEmbeddings(dim=512)
    vecs = emb.embed(
        [
            "retrieval augmented generation for question answering",
            "retrieval augmented generation improves question answering",
            "a recipe for chocolate chip cookies with butter",
        ]
    )
    sim_related = float(vecs[0] @ vecs[1])
    sim_unrelated = float(vecs[0] @ vecs[2])
    assert sim_related > sim_unrelated


def test_empty_string_produces_zero_vector():
    emb = HashingEmbeddings(dim=32)
    out = emb.embed([""])
    assert out.shape == (1, 32)
    # No tokens -> zero vector (normalisation leaves it at zero).
    assert np.allclose(out[0], 0.0)


def test_invalid_dim_raises():
    with pytest.raises(ValueError):
        HashingEmbeddings(dim=0)


def test_factory_selects_hashing_backend():
    emb = get_embeddings("hashing", hashing_dim=77)
    assert emb.name == "hashing"
    assert emb.dim == 77


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        get_embeddings("does-not-exist")


MODEL_DIR = Path("models/all-MiniLM-L6-v2")
GOLDEN = Path(__file__).parent / "fixtures" / "minilm_golden.json"


@pytest.mark.skipif(not (MODEL_DIR / "onnx" / "model.onnx").exists(), reason="MiniLM not downloaded")
def test_minilm_reproduces_the_recorded_vectors():
    from research_assistant.rag.embeddings import MiniLMEmbeddings

    golden = json.loads(GOLDEN.read_text())
    got = MiniLMEmbeddings(MODEL_DIR).embed(golden["texts"])
    assert got.shape == (3, 384)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(got[:, :8], np.array(golden["first8"]), atol=1e-4)


def test_minilm_refuses_weights_that_are_not_the_official_ones(tmp_path):
    from research_assistant.rag.embeddings import download_minilm

    src = tmp_path / "mirror"
    (src / "onnx").mkdir(parents=True)
    (src / "onnx" / "model.onnx").write_bytes(b"not the model")
    (src / "tokenizer.json").write_text("{}")
    with pytest.raises(RuntimeError, match="not the official"):
        download_minilm(tmp_path / "dest", base=src.as_uri())

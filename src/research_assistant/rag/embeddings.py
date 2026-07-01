"""Embedding backends.

Two interchangeable implementations are provided behind a common protocol:

* :class:`SentenceTransformerEmbeddings` — high-quality dense embeddings via
  ``sentence-transformers`` (the production default).
* :class:`HashingEmbeddings` — a deterministic, dependency-free bag-of-words
  hashing embedding used for offline tests and ``--dry-run`` demos, so the
  whole pipeline can run without downloading a model.

Both return L2-normalised ``float32`` vectors, which lets the FAISS
``IndexFlatIP`` index compute cosine similarity directly.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Anything that can turn a list of texts into a normalised matrix."""

    name: str
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array of unit vectors."""
        ...


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class HashingEmbeddings:
    """Deterministic feature-hashing embeddings (no external model).

    Uses the signed hashing trick: each token is hashed to a bucket and a
    sign, which keeps collisions roughly unbiased. Good enough for tests and
    demos; not a substitute for a real embedding model.
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _hash(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return bucket, sign

    def embed(self, texts: List[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                bucket, sign = self._hash(token)
                matrix[row, bucket] += sign
        return _normalize_rows(matrix)


class SentenceTransformerEmbeddings:
    """Dense embeddings backed by a ``sentence-transformers`` model."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "The 'sentence-transformers' backend requires the package of the "
                "same name. Install it with: pip install sentence-transformers\n"
                "Alternatively set EMBEDDING_BACKEND=hashing for a lightweight, "
                "dependency-free fallback."
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")


def get_embeddings(
    backend: str = "sentence-transformers",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    hashing_dim: int = 512,
) -> EmbeddingBackend:
    """Factory selecting an embedding backend by name."""
    backend = (backend or "").lower()
    if backend in {"hashing", "hash", "local"}:
        return HashingEmbeddings(dim=hashing_dim)
    if backend in {"sentence-transformers", "st", "sbert"}:
        return SentenceTransformerEmbeddings(model_name=model_name)
    raise ValueError(
        f"Unknown embedding backend '{backend}'. "
        "Choose 'sentence-transformers' or 'hashing'."
    )

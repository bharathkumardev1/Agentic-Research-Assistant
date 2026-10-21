"""Exact cosine search over a numpy matrix, with the passages needed to cite each hit.

A paper is a few hundred passages and a working set of papers a few thousand, so an exact
matrix product takes milliseconds; an approximate index would trade recall for speed that
isn't needed here. Saved as vectors.npy, chunks.jsonl and meta.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from ..schemas import Chunk, RetrievedChunk
from .embeddings import EmbeddingBackend

_VECTORS = "vectors.npy"
_CHUNKS = "chunks.jsonl"
_META = "meta.json"


class VectorStore:
    def __init__(self, embeddings: EmbeddingBackend) -> None:
        self.embeddings = embeddings
        self.vectors = np.zeros((0, embeddings.dim), dtype=np.float32)
        self.chunks: List[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def add(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embeddings.embed([c.text for c in chunks])
        if vectors.shape[1] != self.embeddings.dim:
            raise ValueError(f"embedding dim {vectors.shape[1]} != store dim {self.embeddings.dim}")
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.chunks.extend(chunks)

    def similarities(self, query: str) -> np.ndarray:
        """Cosine similarity of ``query`` to every passage, in passage order."""
        return self.vectors @ self.embeddings.embed([query])[0]

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        """The ``k`` passages most similar to ``query``, best first; ties go to the earlier passage."""
        if self.is_empty:
            return []
        scores = self.similarities(query)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.lexsort((top, -scores[top]))]
        return [RetrievedChunk(chunk=self.chunks[int(i)], score=float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / _VECTORS, self.vectors)
        with (directory / _CHUNKS).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")
        (directory / _META).write_text(json.dumps({"count": len(self.chunks), "dim": self.embeddings.dim,
                                                   "backend": self.embeddings.name}, indent=2), "utf-8")

    @classmethod
    def exists(cls, directory: Path) -> bool:
        return (Path(directory) / _VECTORS).exists() and (Path(directory) / _CHUNKS).exists()

    @classmethod
    def load(cls, directory: Path, embeddings: EmbeddingBackend) -> VectorStore:
        directory = Path(directory)
        if not cls.exists(directory):
            raise FileNotFoundError(f"No index in '{directory}'. Run the `ingest` command first.")
        meta = json.loads((directory / _META).read_text("utf-8")) if (directory / _META).exists() else {}
        if meta.get("backend") not in (None, embeddings.name) or meta.get("dim") not in (None, embeddings.dim):
            raise ValueError(f"index was built with {meta.get('backend')} ({meta.get('dim')} dims), "
                             f"not {embeddings.name} ({embeddings.dim}); ingest again with the same backend")
        store = cls(embeddings)
        store.vectors = np.load(directory / _VECTORS)
        store.chunks = [Chunk.model_validate_json(line) for line in
                        (directory / _CHUNKS).read_text("utf-8").splitlines() if line.strip()]
        return store

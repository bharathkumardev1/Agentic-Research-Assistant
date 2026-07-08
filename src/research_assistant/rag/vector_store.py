"""A FAISS-backed vector store with cosine similarity search and persistence.

The store keeps the FAISS index and the chunk payloads side by side and can
round-trip both to disk. Because all embeddings are L2-normalised, an
``IndexFlatIP`` (inner-product) index yields cosine similarity scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..schemas import Chunk, RetrievedChunk
from .embeddings import EmbeddingBackend

_INDEX_FILE = "index.faiss"
_CHUNKS_FILE = "chunks.jsonl"
_META_FILE = "meta.json"


def _import_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "The vector store requires 'faiss-cpu'. "
            "Install it with: pip install faiss-cpu"
        ) from exc
    return faiss


class FaissVectorStore:
    """In-memory FAISS index plus the chunk metadata needed to cite results."""

    def __init__(self, embeddings: EmbeddingBackend) -> None:
        self.embeddings = embeddings
        self._faiss = _import_faiss()
        self.index = self._faiss.IndexFlatIP(embeddings.dim)
        self.chunks: List[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def add(self, chunks: List[Chunk]) -> None:
        """Embed and index a batch of chunks."""
        if not chunks:
            return
        vectors = self.embeddings.embed([c.text for c in chunks])
        if vectors.shape[1] != self.embeddings.dim:
            raise ValueError(
                f"Embedding dim {vectors.shape[1]} != index dim {self.embeddings.dim}"
            )
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 6) -> List[RetrievedChunk]:
        """Return the ``k`` most similar chunks to ``query`` (highest score first)."""
        if self.is_empty:
            return []
        k = min(k, len(self.chunks))
        query_vec = self.embeddings.embed([query])
        scores, indices = self.index.search(query_vec, k)
        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                RetrievedChunk(chunk=self.chunks[int(idx)], score=float(score))
            )
        return results

    # --- persistence -------------------------------------------------------
    def save(self, directory: Path) -> None:
        """Persist the index, chunks and metadata to ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(directory / _INDEX_FILE))
        with (directory / _CHUNKS_FILE).open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(chunk.model_dump_json() + "\n")
        meta = {
            "count": len(self.chunks),
            "dim": self.embeddings.dim,
            "backend": getattr(self.embeddings, "name", "unknown"),
        }
        (directory / _META_FILE).write_text(json.dumps(meta, indent=2), "utf-8")

    @classmethod
    def exists(cls, directory: Path) -> bool:
        """Return True if a store previously written by :meth:`save` is at ``directory``.

        Cheap file-existence check, so callers can decide whether it's worth
        constructing an (possibly expensive to load) embedding backend before
        calling :meth:`load`.
        """
        directory = Path(directory)
        return (directory / _INDEX_FILE).exists() and (directory / _CHUNKS_FILE).exists()

    @classmethod
    def load(cls, directory: Path, embeddings: EmbeddingBackend) -> FaissVectorStore:
        """Load a store previously written by :meth:`save`."""
        directory = Path(directory)
        index_path = directory / _INDEX_FILE
        chunks_path = directory / _CHUNKS_FILE
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No index found in '{directory}'. Run the `ingest` command first."
            )

        meta = _read_meta(directory)
        if meta and meta.get("dim") not in (None, embeddings.dim):
            raise ValueError(
                f"Index was built with dim {meta['dim']} (backend "
                f"'{meta.get('backend')}') but the current embedding backend has "
                f"dim {embeddings.dim}. Re-ingest with the same backend."
            )

        store = cls.__new__(cls)
        store.embeddings = embeddings
        store._faiss = _import_faiss()
        store.index = store._faiss.read_index(str(index_path))
        store.chunks = []
        with chunks_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    store.chunks.append(Chunk.model_validate_json(line))
        return store


def _read_meta(directory: Path) -> Optional[dict]:
    meta_path = Path(directory) / _META_FILE
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None

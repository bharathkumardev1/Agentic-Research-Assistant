"""Retriever agent.

Thin, dependency-light wrapper around the vector store. It exists as its own
agent so the orchestration graph can reason about retrieval as a distinct
step (and so query refinement from the evaluator flows through one place).
"""

from __future__ import annotations

from typing import List

from ..rag.vector_store import FaissVectorStore
from ..schemas import RetrievedChunk


class RetrieverAgent:
    """Fetches the most relevant chunks for a query from the vector store."""

    def __init__(self, store: FaissVectorStore, top_k: int = 6) -> None:
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str, k: int | None = None) -> List[RetrievedChunk]:
        return self.store.search(query, k=k or self.top_k)

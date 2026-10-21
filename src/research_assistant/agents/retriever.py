"""Retriever: the top passages for a query from whatever store the run searches."""

from __future__ import annotations

from typing import List, Protocol

from ..schemas import RetrievedChunk


class Store(Protocol):
    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]: ...


class RetrieverAgent:
    def __init__(self, store: Store, top_k: int = 5) -> None:
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str, k: int | None = None) -> List[RetrievedChunk]:
        return self.store.search(query, k=k or self.top_k)

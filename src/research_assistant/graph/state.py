"""The shared state passed between graph nodes.

Defined as a ``TypedDict`` (LangGraph's default channel semantics overwrite
each key with the value a node returns), so list-valued fields are replaced
with the fully-recomputed value inside each node rather than appended to.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

from ..schemas import Evaluation, PaperSummary, RetrievedChunk, SourceRef


class ResearchState(TypedDict, total=False):
    """Everything the agents read from and write to as the graph runs."""

    question: str
    current_query: str
    queries: List[str]
    retrieved: List[RetrievedChunk]
    sources: List[SourceRef]
    summary: Optional[PaperSummary]
    evaluation: Optional[Evaluation]
    iteration: int
    max_iterations: int
    report: Optional[str]

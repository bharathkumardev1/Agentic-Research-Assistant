"""The state passed between graph nodes.

LangGraph overwrites each key with what a node returns, so list fields are rebuilt in full
by the node that changes them rather than appended to.
"""

from __future__ import annotations

from typing import List, TypedDict

from ..schemas import Answer, RetrievedChunk, Review, SourceRef
from ..trace import Trace


class ResearchState(TypedDict, total=False):
    question: str
    query: str
    queries: List[str]
    retrieved: List[RetrievedChunk]
    sources: List[SourceRef]
    answer: Answer
    reviews: List[Review]
    round: int
    max_rounds: int
    status: str
    trace: Trace

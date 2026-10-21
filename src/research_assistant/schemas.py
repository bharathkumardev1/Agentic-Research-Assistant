"""Typed data that flows between ingestion, retrieval and the agents.

Model output is validated into these at the boundary, so a malformed reply fails where it
arrives instead of three steps later.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Bibliographic metadata for one source document."""

    paper_id: str = Field(..., description="Stable, filesystem-safe identifier.")
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    source: str = Field("", description="Original path or URL of the document.")


class Chunk(BaseModel):
    """A retrievable passage plus what's needed to cite it."""

    chunk_id: str = Field(..., description="`{paper_id}::{index}`.")
    paper_id: str
    title: str
    text: str
    chunk_index: int
    source: str = ""
    section: str = ""

    def short(self, n: int = 200) -> str:
        flat = " ".join(self.text.split())
        return flat[:n] + ("..." if len(flat) > n else "")


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class SourceRef(BaseModel):
    """A numbered passage as the model saw it; ``marker`` is its [n]."""

    marker: int
    paper_id: str
    title: str
    chunk_id: str
    section: str = ""
    preview: str = ""
    text: str = ""  # the passage exactly as the model read it


class Answer(BaseModel):
    """The answerer's output: a short answer and the passages that support it."""

    answer: str
    evidence: List[int] = Field(default_factory=list, description="Markers of supporting passages.")
    unanswerable: bool = False


class Review(BaseModel):
    """The reviewer's verdict on an answer; drives the loop."""

    sufficient: bool
    grounded: bool
    missing: str = ""
    refined_query: str = ""


class ResearchResult(BaseModel):
    question: str
    answer: Answer
    cited: List[SourceRef] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)
    reviews: List[Review] = Field(default_factory=list)
    queries: List[str] = Field(default_factory=list)
    rounds: int = 0
    status: str = "answered"  # answered | round cap | deadline | budget
    trace: Dict[str, Any] = Field(default_factory=dict)
    report: str = ""

"""Typed data models shared across ingestion, retrieval and the agents.

Everything that flows between components is a Pydantic model so that LLM
output can be validated at the boundary and so the LangGraph state stays
self-describing.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Bibliographic metadata for a single source document."""

    paper_id: str = Field(..., description="Stable, filesystem-safe identifier.")
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    source: str = Field("", description="Original path or URL of the document.")


class Chunk(BaseModel):
    """A retrievable passage of text plus the metadata needed to cite it."""

    chunk_id: str = Field(..., description="`{paper_id}::{index}`.")
    paper_id: str
    title: str
    text: str
    chunk_index: int
    source: str = ""

    def short(self, n: int = 200) -> str:
        """A single-line preview, useful for references and logs."""
        flat = " ".join(self.text.split())
        return flat[:n] + ("…" if len(flat) > n else "")


class RetrievedChunk(BaseModel):
    """A chunk together with its similarity score for a given query."""

    chunk: Chunk
    score: float


class SourceRef(BaseModel):
    """A numbered citation entry shown in the final report's bibliography."""

    marker: int = Field(..., description="The [n] marker used inline in the summary.")
    paper_id: str
    title: str
    chunk_id: str
    source: str = ""
    preview: str = ""


class PaperSummary(BaseModel):
    """Structured, citation-backed synthesis returned by the summarizer agent."""

    summary: str = Field(..., description="2-4 paragraph synthesis with [n] markers.")
    methods: List[str] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    research_gaps: List[str] = Field(default_factory=list)


class Evaluation(BaseModel):
    """Critique of a summary produced by the evaluator agent.

    Drives the agentic loop: when ``sufficiency`` is ``"needs_more"`` the
    graph routes back to the retriever using ``refined_query``.
    """

    grounded: bool = Field(
        ..., description="Whether every claim is supported by the cited sources."
    )
    coverage_score: float = Field(
        ..., ge=0.0, le=1.0, description="How fully the question is answered (0-1)."
    )
    sufficiency: Literal["sufficient", "needs_more"]
    critique: str = ""
    missing_aspects: List[str] = Field(default_factory=list)
    refined_query: Optional[str] = Field(
        default=None,
        description="A better search query to run when more evidence is needed.",
    )


class ResearchResult(BaseModel):
    """The final, user-facing output of a research run."""

    question: str
    summary: PaperSummary
    sources: List[SourceRef]
    evaluation: Optional[Evaluation] = None
    queries: List[str] = Field(default_factory=list)
    iterations: int = 0
    report: str = ""

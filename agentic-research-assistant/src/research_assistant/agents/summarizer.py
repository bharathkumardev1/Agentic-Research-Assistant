"""Summarizer agent.

Given a research question and a set of retrieved passages, it asks Claude for
a structured synthesis (summary + methods + key findings + research gaps) in
which every claim carries a ``[n]`` citation marker tied to the passages.
"""

from __future__ import annotations

from typing import List, Tuple

from ..context import number_sources
from ..llm import LLMClient
from ..schemas import PaperSummary, RetrievedChunk, SourceRef

SYSTEM = (
    "You are a meticulous research synthesis assistant. You read excerpts from "
    "academic papers and produce a faithful, citation-backed synthesis. You only "
    "use information present in the provided sources and never invent citations. "
    "Every factual sentence must end with one or more bracketed markers like [1] "
    "or [2][3] that refer to the numbered sources you were given."
)

_PROMPT = """\
Research question:
{question}

You are given {n} numbered source excerpts. Synthesize an answer using ONLY
these sources. Add a bracketed citation marker (e.g. [1], [2][3]) to every
factual statement, matching the source numbers below.

Return a single JSON object with EXACTLY these keys and no extra prose:
{{
  "summary":        string  - 2-4 paragraph synthesis answering the question, with [n] markers,
  "methods":        array of strings - distinct methods/approaches used across the papers, each with [n] markers,
  "key_findings":   array of strings - the most important findings/results, each with [n] markers,
  "research_gaps":  array of strings - open problems, limitations or unexplored directions, each with [n] markers
}}

If the sources do not address part of the question, say so explicitly in the
summary rather than guessing.

=== SOURCES ===
{context}
=== END SOURCES ===
"""


class SummarizerAgent:
    """Wraps an LLM to emit a validated :class:`PaperSummary`."""

    def __init__(self, client: LLMClient, model: str, max_tokens: int = 2048) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def summarize(
        self, question: str, retrieved: List[RetrievedChunk]
    ) -> Tuple[PaperSummary, List[SourceRef]]:
        context, sources = number_sources(retrieved)
        if not sources:
            empty = PaperSummary(
                summary=(
                    "No relevant passages were retrieved for this question, so no "
                    "grounded synthesis can be produced."
                ),
                methods=[],
                key_findings=[],
                research_gaps=[],
            )
            return empty, sources

        prompt = _PROMPT.format(question=question, n=len(sources), context=context)
        data = self.client.complete_json(
            model=self.model,
            system=SYSTEM,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return PaperSummary.model_validate(data), sources

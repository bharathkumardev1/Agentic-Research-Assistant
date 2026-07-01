"""Evaluator agent.

Judges whether the current synthesis is well-grounded and adequately answers
the question. When it is not, the evaluator proposes a refined search query,
which the graph uses to loop back into retrieval — the core of the agentic
self-correction behaviour.
"""

from __future__ import annotations

import json
from typing import List

from ..llm import LLMClient
from ..schemas import Evaluation, PaperSummary, SourceRef

SYSTEM = (
    "You are a critical peer reviewer evaluating a research synthesis. You check "
    "two things: (1) grounding — is every claim supported by the cited sources, "
    "with no fabrication; and (2) coverage — does the synthesis actually answer "
    "the question. You are concise, fair and decisive."
)

_PROMPT = """\
Research question:
{question}

Candidate synthesis (JSON):
{summary_json}

Available source previews (the only evidence that exists):
{previews}

Assess the synthesis and return a single JSON object with EXACTLY these keys:
{{
  "grounded":        boolean - true if all claims are supported by the sources,
  "coverage_score":  number between 0 and 1 - how fully the question is answered,
  "sufficiency":     "sufficient" or "needs_more",
  "critique":        string - one or two sentences of specific feedback,
  "missing_aspects": array of strings - aspects of the question not yet covered (may be empty),
  "refined_query":   string or null - if "needs_more", a focused search query likely to surface the missing evidence; otherwise null
}}

Mark "needs_more" only when additional retrieval would plausibly improve the
answer and you can name a concrete, better query.
"""


class EvaluatorAgent:
    """Wraps an LLM to emit a validated :class:`Evaluation`."""

    def __init__(self, client: LLMClient, model: str, max_tokens: int = 1024) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def evaluate(
        self, question: str, summary: PaperSummary, sources: List[SourceRef]
    ) -> Evaluation:
        previews = "\n".join(f"[{s.marker}] {s.title}: {s.preview}" for s in sources)
        prompt = _PROMPT.format(
            question=question,
            summary_json=json.dumps(summary.model_dump(), indent=2),
            previews=previews or "(no sources)",
        )
        data = self.client.complete_json(
            model=self.model,
            system=SYSTEM,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return Evaluation.model_validate(data)

"""Reviewer: is the answer supported by the passages it cites, and does it answer the question?

When it doesn't, the reviewer names what's missing and a search query likely to find it,
and the loop retrieves again with that query. That loop is the "agentic" part; the QASPER
evaluation measures whether it earns its extra calls.
"""

from __future__ import annotations

import json
from typing import List

from ..llm import LLMClient, Reply, Request
from ..schemas import Answer, Review, SourceRef

SYSTEM = (
    "You review an answer to a question about a research paper. Check two things: is every "
    "part of the answer supported by the passages it cites, and does it actually answer the "
    "question. Say it's sufficient when both hold, or when the passages plainly can't answer "
    "the question and the answer says so. Otherwise name what's missing and write one search "
    "query that would find it in the paper."
)
SCHEMA = {
    "type": "object",
    "properties": {"sufficient": {"type": "boolean"}, "grounded": {"type": "boolean"},
                   "missing": {"type": "string"}, "refined_query": {"type": "string"}},
    "required": ["sufficient", "grounded", "missing", "refined_query"],
    "additionalProperties": False,
}


class EvaluatorAgent:
    def __init__(self, client: LLMClient, model: str, max_output_tokens: int = 1024,
                 reasoning_effort: str | None = "low") -> None:
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort

    def request(self, question: str, answer: Answer, sources: List[SourceRef]) -> Request:
        cited = [s for s in sources if s.marker in set(answer.evidence)]
        shown = "\n".join(f"[{s.marker}] {s.preview}" for s in cited) or "(no passages cited)"
        prompt = (f"Question: {question}\n\nAnswer: {json.dumps(answer.model_dump())}\n\n"
                  f"Cited passages:\n{shown}")
        return Request(model=self.model, system=SYSTEM, prompt=prompt,
                       max_output_tokens=self.max_output_tokens, json_schema=SCHEMA,
                       reasoning_effort=self.reasoning_effort)

    def evaluate(self, question: str, answer: Answer, sources: List[SourceRef]) -> tuple[Review, Reply]:
        reply = self.client.complete(self.request(question, answer, sources))
        return Review.model_validate(reply.json()), reply

"""Answerer: a short answer to the question from numbered passages, citing the ones it used."""

from __future__ import annotations

from typing import List, Tuple

from ..context import number_sources
from ..llm import LLMClient, Reply, Request
from ..schemas import Answer, RetrievedChunk, SourceRef

SYSTEM = (
    "You answer questions about research papers using only the numbered passages you are "
    "given. Answer in as few words as the question allows: yes or no, a number, a name or a "
    "short phrase when that answers it, a sentence only when it can't be shorter. List the "
    "numbers of the passages that support your answer. If the passages don't contain the "
    "answer, set unanswerable to true and leave the answer empty. Never use outside knowledge."
)
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"},
                   "evidence": {"type": "array", "items": {"type": "integer"}},
                   "unanswerable": {"type": "boolean"}},
    "required": ["answer", "evidence", "unanswerable"],
    "additionalProperties": False,
}


class AnswerAgent:
    def __init__(self, client: LLMClient, model: str, max_output_tokens: int = 2048,
                 reasoning_effort: str | None = "low") -> None:
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort

    def request(self, question: str, passages: str) -> Request:
        return Request(model=self.model, system=SYSTEM,
                       prompt=f"Passages:\n\n{passages}\n\nQuestion: {question}",
                       max_output_tokens=self.max_output_tokens, json_schema=SCHEMA,
                       reasoning_effort=self.reasoning_effort)

    def answer(self, question: str, retrieved: List[RetrievedChunk]
               ) -> Tuple[Answer, List[SourceRef], Reply | None]:
        passages, sources = number_sources(retrieved)
        if not sources:
            return Answer(answer="", evidence=[], unanswerable=True), sources, None
        reply = self.client.complete(self.request(question, passages))
        answer = Answer.model_validate(reply.json())
        valid = {s.marker for s in sources}
        answer.evidence = [m for m in dict.fromkeys(answer.evidence) if m in valid]  # drop invented markers
        return answer, sources, reply

"""What the answer and review calls are actually sent."""

from research_assistant.agents.answerer import AnswerAgent
from research_assistant.agents.evaluator import EvaluatorAgent
from research_assistant.context import number_sources
from research_assistant.llm import StubClient
from research_assistant.schemas import Answer, Chunk, RetrievedChunk

LONG = ("The model is trained on 12,000 annotated abstracts. " * 6
        + "On the held-out set it reaches 81.4 F1, against 74.9 for the strongest baseline.")


def retrieved():
    texts = [LONG, "An unrelated paragraph about tokenization.", "Another one, about the corpus license."]
    return [RetrievedChunk(chunk=Chunk(chunk_id=f"p::{i}", paper_id="p", title="Paper", text=t, chunk_index=i,
                                       section=f"Section {i}"), score=1.0 - i / 10)
            for i, t in enumerate(texts)]


def test_the_reviewer_reads_each_cited_passage_in_full_and_nothing_uncited():
    _, sources = number_sources(retrieved())
    answer = Answer(answer="81.4 F1", evidence=[1], unanswerable=False)
    prompt = EvaluatorAgent(StubClient(), "stub").request("What F1 does it reach?", answer, sources).prompt
    assert len(LONG) > 180 and "81.4 F1, against 74.9" in prompt  # past the 180-character preview
    assert "tokenization" not in prompt and "license" not in prompt


def test_the_reviewer_sees_the_passage_text_the_answerer_saw():
    agent = AnswerAgent(StubClient(), "stub")
    passages, sources = number_sources(retrieved())
    sent = agent.request("What F1 does it reach?", passages).prompt
    assert all(s.text in sent for s in sources)


def test_a_declined_answer_is_reviewed_against_every_passage_it_had():
    _, sources = number_sources(retrieved())
    declined = Answer(answer="", evidence=[], unanswerable=True)
    prompt = EvaluatorAgent(StubClient(), "stub").request("What F1 does it reach?", declined, sources).prompt
    assert "Passages the answer had" in prompt
    assert all(s.text in prompt for s in sources)

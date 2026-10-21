"""Each thing that can end a run ends it cleanly, with the best answer so far and the reason."""

import json

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")

from research_assistant.agents.answerer import AnswerAgent  # noqa: E402
from research_assistant.agents.evaluator import EvaluatorAgent  # noqa: E402
from research_assistant.agents.retriever import RetrieverAgent  # noqa: E402
from research_assistant.graph.workflow import ResearchPipeline  # noqa: E402
from research_assistant.llm import BudgetExceeded, Reply, Request  # noqa: E402
from research_assistant.schemas import Chunk, RetrievedChunk  # noqa: E402
from research_assistant.trace import Trace  # noqa: E402

PASSAGES = [Chunk(chunk_id=f"p::{i}", paper_id="p", title="Paper", text=f"Passage {i} says {i * 10}%.",
                  chunk_index=i) for i in range(6)]


class Store:
    """Each search brings one passage the run hasn't seen, unless ``same`` is set."""

    def __init__(self, same=False):
        self.same, self.searches = same, 0

    def search(self, query, k=5):
        start = 0 if self.same else self.searches
        self.searches += 1
        return [RetrievedChunk(chunk=c, score=1.0 - i / 10) for i, c in enumerate(PASSAGES[start:start + k])]


class Model:
    """Answers citing passage 1; the reviewer is never satisfied unless told to be."""

    def __init__(self, satisfied=False, fail_on=None, attempts=1):
        self.satisfied, self.fail_on, self.attempts, self.calls = satisfied, fail_on, attempts, []

    def complete(self, request: Request) -> Reply:
        kind = "review" if "review" in request.system.lower() else "answer"
        self.calls.append(kind)
        if self.fail_on == (kind, sum(c == kind for c in self.calls)):
            raise BudgetExceeded("cap reached")
        if kind == "review":
            out = {"sufficient": self.satisfied, "grounded": True, "missing": "the number",
                   "refined_query": f"more detail {len(self.calls)}"}
        else:
            out = {"answer": "10%", "evidence": [1], "unanswerable": False}
        return Reply(json.dumps(out), "gpt-6-luna", input_tokens=100, output_tokens=20, attempts=self.attempts)


def pipeline(model, max_rounds=3, deadline=None, store=None):
    return ResearchPipeline(RetrieverAgent(store or Store(), top_k=3), AnswerAgent(model, "gpt-6-luna"),
                            EvaluatorAgent(model, "gpt-6-luna"), max_rounds=max_rounds, deadline_seconds=deadline)


def test_the_round_cap_stops_a_reviewer_that_is_never_satisfied():
    model = Model(satisfied=False)
    result = pipeline(model, max_rounds=3).run("What share?")
    assert result.status == "round cap" and result.rounds == 3
    assert model.calls == ["answer", "review"] * 3 and result.answer.answer == "10%"


def test_a_refined_query_that_finds_nothing_new_ends_the_loop():
    model = Model(satisfied=False)
    result = pipeline(model, store=Store(same=True)).run("What share?")
    assert result.status == "no new passages" and result.rounds == 1
    assert model.calls == ["answer", "review"]  # answering again from the same passages would repeat itself


def test_a_satisfied_reviewer_ends_the_loop_early():
    result = pipeline(Model(satisfied=True)).run("What share?")
    assert result.status == "answered" and result.rounds == 1 and len(result.reviews) == 1


def test_the_deadline_is_checked_before_each_model_call():
    now = [0.0]

    def clock():
        now[0] += 25.0  # every look at the clock costs 25 seconds
        return now[0]

    model = Model(satisfied=False)
    result = pipeline(model, deadline=100).run("What share?", trace=Trace(100, clock=clock))
    assert result.status == "deadline" and result.answer.answer == "10%"
    assert len(model.calls) < 6


def test_running_out_of_budget_keeps_the_answer_already_paid_for():
    model = Model(satisfied=False, fail_on=("review", 2))
    result = pipeline(model).run("What share?")
    assert result.status == "budget" and result.rounds == 2 and result.answer.answer == "10%"


def test_the_trace_counts_retries_and_cost_per_step():
    result = pipeline(Model(satisfied=True, attempts=3)).run("What share?")
    t = result.trace
    assert t["calls"] == 2 and t["retries"] == 4
    assert t["cost_usd"] == pytest.approx(2 * (100 * 0.10 + 20 * 0.50) / 1e6)
    assert [s["node"] for s in t["steps"]] == ["retrieve", "answer", "review"]
    assert all(s["seconds"] >= 0 for s in t["steps"])


def test_a_single_pass_is_one_answer_call():
    model = Model()
    result = pipeline(model, max_rounds=1).run("What share?")
    assert model.calls == ["answer"] and result.status == "answered"
    with pytest.raises(ValueError):
        pipeline(model, max_rounds=0)

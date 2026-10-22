"""The model client: pricing, the budget, retries and the recorder."""

import json

import pytest

from research_assistant.llm import (
    Budget,
    BudgetExceeded,
    IncompleteReply,
    LLMError,
    MissingRecording,
    OpenAIClient,
    Recorder,
    Reply,
    Request,
    StubClient,
    cost,
    price,
    with_retries,
)

REQ = Request(model="gpt-6-luna", system="answer", prompt="[1] (Intro)\nThe sky is blue today.\n\nQuestion: ?",
              max_output_tokens=100)


class APITimeoutError(Exception):
    pass


class BadRequestError(Exception):
    status_code = 400


def test_cost_uses_cached_input_and_output_prices():
    r = Reply("x", "gpt-6-luna", input_tokens=1_000_000, cached_tokens=400_000, output_tokens=100_000)
    assert cost(r) == pytest.approx(0.6 * 0.10 + 0.4 * 0.01 + 0.1 * 0.50)
    assert price("gpt-6-luna-2026-05-18") == price("gpt-6-luna")
    with pytest.raises(KeyError):
        price("some-unpriced-model")


def test_budget_refuses_a_call_that_could_pass_the_cap():
    b = Budget(cap_usd=0.001)
    b.check(REQ)
    b.charge(Reply("x", "gpt-6-luna", input_tokens=2000, output_tokens=1500))
    with pytest.raises(BudgetExceeded):
        b.check(Request(model="gpt-6-luna", system="s", prompt="p" * 3000, max_output_tokens=2000))


def test_retries_transient_errors_with_backoff_and_counts_attempts():
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise APITimeoutError("timed out")
        return Reply("ok", "gpt-6-luna")

    reply = with_retries(flaky, attempts=3, backoff=0.5, sleep=waits.append)
    assert reply.attempts == 3 and waits == [0.5, 1.0]


def test_gives_up_after_the_last_attempt_and_never_retries_a_bad_request():
    with pytest.raises(APITimeoutError):
        with_retries(lambda: (_ for _ in ()).throw(APITimeoutError()), attempts=2, sleep=lambda s: None)
    calls = []

    def bad():
        calls.append(1)
        raise BadRequestError("nope")

    with pytest.raises(BadRequestError):
        with_retries(bad, attempts=5, sleep=lambda s: None)
    assert len(calls) == 1


def test_recorder_replays_by_request_and_refuses_unrecorded_calls_in_replay_mode(tmp_path):
    path = tmp_path / "calls.jsonl"
    live = Recorder(path, StubClient())
    first = live.complete(REQ, tag={"q": 1})
    assert live.made == 1 and not first.replayed
    replay = Recorder(path)
    again = replay.complete(REQ)
    assert again.replayed and again.text == first.text and replay.replayed == 1
    with pytest.raises(MissingRecording):
        replay.complete(Request(model="gpt-6-luna", system="answer", prompt="something else"))
    assert json.loads(path.read_text().splitlines()[0])["tag"] == {"q": 1}


def cut_off(model="gpt-6-luna"):
    return Reply(text='{"answer": "', model=model, input_tokens=1000, output_tokens=100, reasoning_tokens=90,
                 status="incomplete")


def test_a_cut_off_reply_is_billed_and_raises():
    client = OpenAIClient.__new__(OpenAIClient)
    client.attempts, client.budget = 1, Budget(1.0)
    client._once = lambda request: cut_off()
    with pytest.raises(IncompleteReply):
        client.complete(REQ)
    assert client.budget.spent_usd == pytest.approx(cost(cut_off())) and client.budget.spent_usd > 0


def test_a_cut_off_reply_counts_toward_the_cap_but_is_never_replayed(tmp_path):
    class CutOffOnce(StubClient):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                raise IncompleteReply(cut_off())
            return super().complete(request)

    path = tmp_path / "calls.jsonl"
    live = Recorder(path, CutOffOnce())
    with pytest.raises(IncompleteReply):
        live.complete(REQ)
    assert live.recorded_spend_usd == pytest.approx(cost(cut_off()))
    with pytest.raises(MissingRecording):
        Recorder(path).complete(REQ)  # the cut-off reply isn't something to replay
    assert Recorder(path).recorded_spend_usd == pytest.approx(cost(cut_off()))
    reply = live.complete(REQ)  # retried: the second call completes and is recorded
    assert not reply.replayed and Recorder(path).complete(REQ).replayed


def test_reply_json_must_be_an_object():
    assert Reply('{"a": 1}', "stub").json() == {"a": 1}
    with pytest.raises(LLMError):
        Reply("not json", "stub").json()
    with pytest.raises(LLMError):
        Reply("[1, 2]", "stub").json()


def test_stub_answers_from_the_passages_then_asks_for_one_more_round():
    stub = StubClient()
    answer = stub.complete(REQ).json()
    assert answer["evidence"] == [1] and "sky is blue" in answer["answer"]
    review = Request(model="stub", system="You review an answer", prompt="x")
    assert stub.complete(review).json()["sufficient"] is False
    assert stub.complete(review).json()["sufficient"] is True

"""The QASPER evaluation end to end on a synthetic paper, offline."""

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")

from research_assistant.cli import main  # noqa: E402
from research_assistant.eval import qasper as Q  # noqa: E402
from research_assistant.eval.run import agreement, failure, label_sheet, run, sample  # noqa: E402
from research_assistant.llm import BudgetExceeded, Recorder, StubClient  # noqa: E402
from research_assistant.rag.embeddings import HashingEmbeddings  # noqa: E402

MINI = Path(__file__).parent / "fixtures" / "qasper_mini.json"


def evaluate(client):
    papers = Q.load(MINI)
    return run(papers, sample(papers, None), client, HashingEmbeddings(256), "stub", "stub", "stub", top_k=2)


def test_every_question_is_scored_for_both_systems(tmp_path):
    res = evaluate(Recorder(tmp_path / "calls.jsonl", StubClient()))
    assert [(r["question_id"], r["system"]) for r in res["rows"]] == [
        (q, s) for q in ("q1", "q2", "q3") for s in ("single pass", "loop")]
    single = res["summary"]["systems"]["single pass"]
    assert single["n"] == 3 and single["calls_per_question"] == 1.0
    loop = [r for r in res["rows"] if r["system"] == "loop"]
    # The stub asks for one more round per question; the loop stops once a refined query
    # stops finding passages it hasn't seen.
    assert all(r["calls"] >= 2 and r["status"] == "no new passages" for r in loop)
    assert set(res["summary"]["loop_minus_single"]) == {"answer_f1", "evidence_f1", "judge_accuracy"}


def test_a_replay_reproduces_the_scores_without_a_client(tmp_path):
    first = evaluate(Recorder(tmp_path / "calls.jsonl", StubClient()))
    again = evaluate(Recorder(tmp_path / "calls.jsonl"))
    strip = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]  # noqa: E731
    assert strip(again["rows"]) == strip(first["rows"])


def test_running_out_of_budget_keeps_what_finished():
    class Capped(StubClient):
        def __init__(self):
            super().__init__()
            self.n = 0

        def complete(self, request):
            self.n += 1
            if self.n > 3:
                raise BudgetExceeded("cap")
            return super().complete(request)

    res = evaluate(Capped())
    assert res["stopped"] and 0 < len(res["rows"]) < 6


def test_failures_follow_the_pipeline():
    base = {"gold_unanswerable": False, "declined": False, "gold_retrieved": True, "gold_cited": True,
            "verdict": "correct"}
    assert failure(base) == "correct"
    assert failure({**base, "verdict": "incorrect"}) == "cited the evidence, answer wrong"
    assert failure({**base, "gold_cited": False}) == "evidence retrieved, not cited"
    assert failure({**base, "gold_retrieved": False, "gold_cited": False}) == "evidence not retrieved"
    assert failure({**base, "declined": True}) == "declined an answerable question"
    assert failure({**base, "gold_unanswerable": True}) == "answered an unanswerable question"


def test_label_sheet_and_agreement(tmp_path):
    res = evaluate(Recorder(tmp_path / "calls.jsonl", StubClient()))
    questions = {q.question_id: q for p in Q.load(MINI) for q in p.questions}
    sheet = label_sheet(res, questions, n=10)
    assert sheet and all(r["your_verdict"] == "" and "verdict" not in r for r in sheet)
    path = tmp_path / "labels.csv"
    sheet[0]["your_verdict"] = "incorrect"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sheet)
    check = agreement(res, str(path))
    assert check["labeled"] == 1 and check["judge_lenient"] == 1


def test_eval_command_records_and_replays(tmp_path, monkeypatch):
    data = tmp_path / "qasper"
    data.mkdir()
    (data / Q.FILES["dev"]).write_text(MINI.read_text())
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashing")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    record = tmp_path / "calls.jsonl"
    Recorder(record)  # nothing recorded yet, so a replay must fail
    assert main(["eval", "--data-dir", str(data), "--record", str(record), "--replay",
                 "--out", str(tmp_path / "qasper-dev.json")]) == 1
    rec = Recorder(record, StubClient())
    papers = Q.load(data / Q.FILES["dev"])
    from research_assistant.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    run(papers, sample(papers, 150, 0), rec, HashingEmbeddings(s.hashing_dim), s.answer_model, s.review_model,
        s.judge_model, s.top_k, s.run_deadline_seconds, s.reasoning_effort)
    assert main(["eval", "--data-dir", str(data), "--record", str(record), "--replay",
                 "--out", str(tmp_path / "qasper-dev.json")]) == 0
    res = json.loads((tmp_path / "qasper-dev.json").read_text())
    assert res["calls"]["made"] == 0 and res["questions"] == 3
    assert "## QASPER dev: 3 questions" in (tmp_path / "RESULTS.md").read_text()

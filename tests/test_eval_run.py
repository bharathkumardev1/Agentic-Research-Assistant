"""The QASPER evaluation end to end on a synthetic paper, offline."""

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph", reason="langgraph not installed")

from research_assistant.cli import main  # noqa: E402
from research_assistant.eval import qasper as Q  # noqa: E402
from research_assistant.eval.run import agreement, differences, failure, label_sheet, run, sample  # noqa: E402
from research_assistant.eval.runlog import RunLog, stop_key  # noqa: E402
from research_assistant.llm import BudgetExceeded, Recorder, StubClient  # noqa: E402
from research_assistant.rag.embeddings import HashingEmbeddings  # noqa: E402

MINI = Path(__file__).parent / "fixtures" / "qasper_mini.json"


def evaluate(client, log=None):
    papers = Q.load(MINI)
    return run(papers, sample(papers, None), client, HashingEmbeddings(256), "stub", "stub", "stub", top_k=2,
               log=log)


def stub_meta(res):
    res["meta"] = {"command": "research-assistant eval", "git_sha": "0" * 40, "date": "2026-10-21",
                   "answer_model": "stub", "review_model": "stub", "judge_model": "stub"}
    res["split"] = "dev"
    return res


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
    record, log, out = tmp_path / "calls.jsonl", tmp_path / "runs.jsonl", tmp_path / "qasper-dev.json"
    files = ["--data-dir", str(data), "--record", str(record), "--log", str(log)]
    Recorder(record)  # nothing recorded yet, so a replay must fail
    assert main(["eval", *files, "--replay", "--out", str(out)]) == 1
    papers = Q.load(data / Q.FILES["dev"])
    from research_assistant.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    run(papers, sample(papers, 150, 0), Recorder(record, StubClient()), HashingEmbeddings(s.hashing_dim),
        s.answer_model, s.review_model, s.judge_model, s.top_k, s.run_deadline_seconds, s.reasoning_effort,
        log=RunLog(log))
    logged = log.read_text()
    assert main(["eval", *files, "--replay", "--out", str(out)]) == 0
    assert log.read_text() == logged  # a replay reads the log and doesn't add to it
    res = json.loads(out.read_text())
    assert res["calls"]["made"] == 0 and res["questions"] == 3
    assert "## QASPER dev: 3 questions" in (tmp_path / "RESULTS.md").read_text()

    # check-replay reruns the results file from the recording and compares every score.
    assert main(["check-replay", str(out), *files]) == 0
    res["rows"][0]["answer_f1"] += 0.5
    out.write_text(json.dumps(res))
    assert main(["check-replay", str(out), *files]) == 1


def test_differences_ignore_wall_time_and_nothing_else(tmp_path):
    res = stub_meta(evaluate(Recorder(tmp_path / "calls.jsonl", StubClient())))
    again = json.loads(json.dumps(res))
    for r in again["rows"]:
        r["seconds"] += 3.0
    assert differences(res, again) == []
    again["rows"][2]["verdict"] = "incorrect"
    assert len(differences(res, again)) == 1
    again["meta"]["qasper_tarballs_sha256"] = {"test": "f" * 64}
    res["meta"]["qasper_tarballs_sha256"] = {"test": "0" * 64}
    assert any("tarballs" in d for d in differences(res, again))


def test_a_deadline_stop_replays_as_a_deadline(tmp_path):
    record, log = tmp_path / "calls.jsonl", RunLog(tmp_path / "runs.jsonl")
    first = evaluate(Recorder(record, StubClient()), log)
    loop = next(r for r in first["rows"] if r["system"] == "loop")
    # As if the clock had run out after the first answer, before the first review.
    log.stops[stop_key(loop["question_id"], "loop")] = {"status": "deadline", "calls": 1}
    again = evaluate(Recorder(record), log)
    replayed = next(r for r in again["rows"] if r["question_id"] == loop["question_id"] and r["system"] == "loop")
    assert replayed["status"] == "deadline" and replayed["calls"] == 1 and replayed["rounds"] == 1


def test_replay_export_rebuilds_each_round_from_the_recording(tmp_path):
    from research_assistant.eval.replay import export, outcome, pick

    record, log = tmp_path / "calls.jsonl", RunLog(tmp_path / "runs.jsonl")
    res = stub_meta(evaluate(Recorder(record, StubClient()), log))
    ids = pick(res, 2)
    assert len(ids) == 2
    out = export(Q.load(MINI), res, Recorder(record), log, HashingEmbeddings(256), ids, 2, "low")
    run_ = out["runs"][0]
    rows = {r["system"]: r for r in res["rows"] if r["question_id"] == ids[0]}
    assert run_["outcome"] == outcome(rows["single pass"], rows["loop"])
    first, single = run_["loop"]["rounds"][0], run_["single pass"]["rounds"]
    assert first["query"] == run_["question"] and first["found"] and first["answer"]["text"]
    assert "review" in first and len(single) == 1 and "review" not in single[0]
    assert set(first["found"]) <= set(run_["passages"])
    assert run_["loop"]["status"] == rows["loop"]["status"]


def test_replay_export_stops_where_the_deadline_stopped_the_run(tmp_path):
    from research_assistant.eval.replay import export

    record, log = tmp_path / "calls.jsonl", RunLog(tmp_path / "runs.jsonl")
    first = evaluate(Recorder(record, StubClient()), log)
    qid = first["rows"][0]["question_id"]
    log.stops[stop_key(qid, "loop")] = {"status": "deadline", "calls": 1}
    res = stub_meta(evaluate(Recorder(record), log))
    out = export(Q.load(MINI), res, Recorder(record), log, HashingEmbeddings(256), [qid], 2, "low")
    assert out["runs"][0]["loop"]["status"] == "deadline"
    assert len(out["runs"][0]["loop"]["rounds"]) == 1


def test_replay_export_refuses_a_results_file_the_recording_doesnt_match(tmp_path):
    from research_assistant.eval.replay import export

    record, log = tmp_path / "calls.jsonl", RunLog(tmp_path / "runs.jsonl")
    res = stub_meta(evaluate(Recorder(record, StubClient()), log))
    res["rows"][1]["evidence_ids"] = ["not-a-passage"]
    with pytest.raises(ValueError, match="replayed differently"):
        export(Q.load(MINI), res, Recorder(record), log, HashingEmbeddings(256), [res["rows"][1]["question_id"]],
               2, "low")

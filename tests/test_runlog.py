"""The run log: searches replay across machines when only float noise differs, and stops
replay where they happened."""

import pytest

from research_assistant.eval.runlog import (
    TOLERANCE,
    LoggedRetriever,
    RecordedStop,
    RetrievalChanged,
    RunLog,
)
from research_assistant.llm import Reply
from research_assistant.rag.embeddings import HashingEmbeddings
from research_assistant.rag.vector_store import VectorStore
from research_assistant.schemas import Chunk

TEXTS = [
    "dense retrieval of passages with sentence embeddings",
    "dense retrieval of passages with sentence embeddings",  # an exact tie with the first
    "retrieval of passages",
    "a recipe for bread with flour, water and salt",
]


def retriever(tmp_path, write=True):
    store = VectorStore(HashingEmbeddings(dim=256))
    store.add([Chunk(chunk_id=f"p::{i}", paper_id="p", title="P", text=t, chunk_index=i)
               for i, t in enumerate(TEXTS)])
    log = RunLog(tmp_path / "runs.jsonl", write=write)
    return LoggedRetriever(store, 2, log, "p"), log


def ids(found):
    return [r.chunk.chunk_id for r in found]


def test_a_search_is_logged_and_reloaded(tmp_path):
    r, log = retriever(tmp_path)
    assert ids(r.retrieve("dense retrieval embeddings")) == ["p::0", "p::1"]
    again = RunLog(tmp_path / "runs.jsonl")
    assert again.searches == log.searches and len(again.searches) == 1


def test_the_logged_ranking_wins_when_scores_agree_within_the_tolerance(tmp_path):
    # Another CPU broke the tie the other way: same passages, scores a hair apart.
    r, log = retriever(tmp_path)
    r.retrieve("dense retrieval embeddings")
    key = next(iter(log.searches))
    (a, score), (b, _) = log.searches[key]
    log.searches[key] = [(b, score + TOLERANCE / 2), (a, score)]
    assert ids(r.retrieve("dense retrieval embeddings")) == [b, a]


def test_a_score_that_moved_more_than_float_noise_fails(tmp_path):
    r, log = retriever(tmp_path)
    r.retrieve("dense retrieval embeddings")
    key = next(iter(log.searches))
    (a, score), (b, other) = log.searches[key]
    log.searches[key] = [(a, score + 10 * TOLERANCE), (b, other)]
    with pytest.raises(RetrievalChanged, match="moved"):
        r.retrieve("dense retrieval embeddings")


def test_a_passage_that_now_outranks_the_logged_ones_fails(tmp_path):
    r, log = retriever(tmp_path)
    r.retrieve("dense retrieval embeddings")
    key = next(iter(log.searches))
    store = r.vectors
    sims = store.similarities("dense retrieval embeddings")
    log.searches[key] = [("p::0", float(sims[0])), ("p::3", float(sims[3]))]  # p::1 scored far above p::3
    with pytest.raises(RetrievalChanged, match="now make the top 2"):
        r.retrieve("dense retrieval embeddings")


def test_a_replay_leaves_the_log_file_alone(tmp_path):
    r, _ = retriever(tmp_path, write=False)
    r.retrieve("dense retrieval embeddings")
    assert not (tmp_path / "runs.jsonl").exists()


def test_a_deadline_stop_replays_after_the_same_number_of_calls():
    clock = RecordedStop({"status": "deadline", "calls": 2})
    assert not clock.expired()
    with clock.step("answer", 1) as step:
        step.calls.append(Reply(text="{}", model="stub"))
    assert not clock.expired()
    with clock.step("review", 1) as step:
        step.calls.append(Reply(text="{}", model="stub"))
    assert clock.expired()
    finished = RecordedStop({"status": "round cap", "calls": 2})
    with finished.step("answer", 1) as step:
        step.calls.extend([Reply(text="{}", model="stub")] * 3)
    assert not finished.expired()

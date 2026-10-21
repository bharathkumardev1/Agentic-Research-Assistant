"""QASPER loading and the official metrics."""

from pathlib import Path

import pytest

from research_assistant.eval import qasper as Q

MINI = Path(__file__).parent / "fixtures" / "qasper_mini.json"


def test_papers_become_verbatim_paragraphs_with_sections():
    (paper,) = Q.load(MINI)
    assert [c.section for c in paper.paragraphs] == ["Introduction", "Introduction", "Results", "Results"]
    assert paper.paragraphs[1].text == "Our system retrieves five passages per question."
    assert [q.question_id for q in paper.questions] == ["q1", "q2", "q3"]


def test_gold_answers_follow_the_official_precedence_and_drop_figure_evidence():
    q1, q2, q3 = Q.load(MINI)[0].questions
    assert [(a.text, a.kind) for a in q1.answers] == [("five passages", "extractive"), ("5", "abstractive")]
    assert q1.answers[1].evidence == ["Our system retrieves five passages per question."]
    assert (q2.answers[0].text, q2.answers[0].kind) == ("Yes", "boolean")
    assert q3.unanswerable and q3.answers[0].text == "Unanswerable" and q3.answers[0].evidence == []


def test_token_f1_is_squad_style():
    assert Q.normalize("The Five, passages!") == "five passages"
    assert Q.token_f1("five passages", "Five passages.") == 1.0
    assert Q.token_f1("five", "five passages") == pytest.approx(2 / 3)
    assert Q.token_f1("", "") == 0.0  # the official script returns 0 when nothing overlaps


def test_paragraph_f1_edge_cases():
    assert Q.paragraph_f1([], []) == 1.0
    assert Q.paragraph_f1(["a"], []) == 0.0 and Q.paragraph_f1([], ["a"]) == 0.0
    assert Q.paragraph_f1(["a", "b"], ["a"]) == pytest.approx(2 / 3)


def test_score_takes_the_best_annotator_for_each_metric():
    q1, _, q3 = Q.load(MINI)[0].questions
    s = Q.score(q1, "5", ["Our system retrieves five passages per question."])
    assert s["answer_f1"] == 1.0 and s["answer_kind"] == "abstractive"
    assert s["evidence_f1"] == 1.0 and s["evidence_precision"] == 1.0 and s["evidence_recall"] == 1.0
    declined = Q.score(q3, Q.UNANSWERABLE, [])
    assert declined["answer_f1"] == 1.0 and declined["evidence_f1"] == 1.0

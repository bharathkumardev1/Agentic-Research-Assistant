"""QASPER: questions about NLP papers, answered from the paper, with the evidence marked.

QASPER (Dasigi et al., NAACL 2021) has 5,049 questions over 1,585 papers. Each question was
answered by one or more annotators, who also marked the paragraphs their answer rests on. An
answer is an extracted span, a yes or no, a free-form sentence, or "unanswerable".

The scoring follows the official evaluator (allenai/qasper-led-baseline, scripts/evaluator.py
at commit afd0fb9):

* answer F1: SQuAD-style token F1 after lowercasing and removing punctuation and articles,
  against each annotator's answer, keeping the best; an unanswerable gold answer is the
  string "Unanswerable" with no evidence, so a system gets credit for saying so and citing
  nothing;
* evidence F1: F1 between the set of paragraphs a system cites and each annotator's set,
  keeping the best, text paragraphs only (figures and tables are excluded, which is the
  evaluator's text-evidence-only setting; this system can't retrieve a figure).

The data comes from the release tarballs on the dataset's S3 bucket. Papers are split into
their paragraphs verbatim, so a cited passage can be matched to gold evidence exactly.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import string
import tarfile
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from ..schemas import Chunk

TARBALLS = {
    "train-dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
}
FILES = {"dev": "qasper-dev-v0.3.json", "test": "qasper-test-v0.3.json"}
UNANSWERABLE = "Unanswerable"


@dataclass
class GoldAnswer:
    text: str
    kind: str  # extractive | abstractive | boolean | none
    evidence: List[str]


@dataclass
class Question:
    question_id: str
    paper_id: str
    text: str
    answers: List[GoldAnswer]

    @property
    def unanswerable(self) -> bool:
        """Every annotator said the paper doesn't answer it."""
        return all(a.kind == "none" for a in self.answers)


@dataclass
class Paper:
    paper_id: str
    title: str
    paragraphs: List[Chunk]
    questions: List[Question] = field(default_factory=list)


def _gold(answer: dict) -> GoldAnswer:
    a = answer["answer"]
    if a.get("unanswerable"):
        return GoldAnswer(UNANSWERABLE, "none", [])
    evidence = [e for e in a.get("evidence", []) if "FLOAT SELECTED" not in e]
    if a.get("extractive_spans"):
        return GoldAnswer(", ".join(a["extractive_spans"]), "extractive", evidence)
    if a.get("free_form_answer"):
        return GoldAnswer(a["free_form_answer"], "abstractive", evidence)
    if a.get("yes_no") is not None:
        return GoldAnswer("Yes" if a["yes_no"] else "No", "boolean", evidence)
    return GoldAnswer("", "empty", evidence)  # the official evaluator raises on these; none exist


def parse(raw: Dict[str, dict]) -> List[Paper]:
    papers = []
    for paper_id, p in sorted(raw.items()):
        paragraphs: List[Chunk] = []
        for section in p.get("full_text", []):
            name = section.get("section_name") or ""
            for text in section.get("paragraphs", []):
                if text and text.strip():
                    paragraphs.append(Chunk(chunk_id=f"{paper_id}::{len(paragraphs)}", paper_id=paper_id,
                                            title=p.get("title", ""), text=text, chunk_index=len(paragraphs),
                                            section=name))
        paper = Paper(paper_id, p.get("title", ""), paragraphs)
        for qa in p.get("qas", []):
            answers = [_gold(a) for a in qa.get("answers", [])]
            answers = [a for a in answers if a.kind != "empty"]
            if answers:
                paper.questions.append(Question(qa["question_id"], paper_id, qa["question"], answers))
        papers.append(paper)
    return papers


def load(path: Path) -> List[Paper]:
    return parse(json.loads(Path(path).read_text(encoding="utf-8")))


def download(dest: Path) -> Dict[str, str]:
    """Fetch both tarballs, extract the dev and test JSON, return each tarball's SHA-256."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, url in TARBALLS.items():
        with urllib.request.urlopen(url, timeout=300) as resp:
            blob = resp.read()
        hashes[name] = hashlib.sha256(blob).hexdigest()
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            for member in tar.getmembers():
                base = Path(member.name).name
                if base in FILES.values():
                    extracted = tar.extractfile(member)
                    if extracted is not None:
                        (dest / base).write_bytes(extracted.read())
    (dest / "tarballs.sha256.json").write_text(json.dumps(hashes, indent=1))
    return hashes


def normalize(text: str) -> str:
    """SQuAD normalization, as the QASPER evaluator does it."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(gold).split()
    same = sum((Counter(pred) & Counter(ref)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)


def paragraph_f1(predicted: List[str], gold: List[str]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    p, g = set(predicted), set(gold)
    same = len(p & g)
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def score(question: Question, answer: str, evidence: List[str]) -> dict:
    """Best answer F1 and best evidence F1 over annotators, plus evidence precision and recall
    against the annotator whose evidence matched best."""
    answer_f1 = [token_f1(answer, a.text) for a in question.answers]
    best = max(range(len(question.answers)), key=lambda i: paragraph_f1(evidence, question.answers[i].evidence))
    gold = set(question.answers[best].evidence)
    cited = set(evidence)
    return {
        "answer_f1": max(answer_f1),
        "answer_kind": question.answers[max(range(len(answer_f1)), key=answer_f1.__getitem__)].kind,
        "evidence_f1": paragraph_f1(evidence, question.answers[best].evidence),
        "evidence_precision": len(cited & gold) / len(cited) if cited else float(not gold),
        "evidence_recall": len(cited & gold) / len(gold) if gold else float(not cited),
    }

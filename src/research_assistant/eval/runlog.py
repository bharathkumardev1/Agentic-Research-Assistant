"""What a replay needs besides the model's replies: the searches, and how each run stopped.

Two things decide what the model gets asked without being model calls themselves.

The passages a search returns, in order. MiniLM runs in float32, and another CPU can move a
similarity in the sixth decimal place, which is enough to swap two nearly tied passages,
change the prompt and miss the recorded reply. So each search is logged with its scores. A
later run recomputes the search and keeps the logged ranking when every score agrees to
within ``TOLERANCE``; a bigger difference means retrieval itself changed, and the run fails
instead of replaying something else.

How each run stopped. Whether the deadline cut a run short depends on how fast the API
answered that day, so a run the deadline stopped replays as stopped after the same number
of calls.

Both go to results/qasper/runs.jsonl, one JSON object per line, appended as the evaluation
runs, so an interrupted run keeps what it logged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..agents.retriever import RetrieverAgent
from ..rag.vector_store import VectorStore
from ..schemas import RetrievedChunk
from ..trace import Trace

TOLERANCE = 1e-4  # cross-CPU float32 differences are around 1e-6; a changed model or text is far above this


class RetrievalChanged(RuntimeError):
    """A search came back different from the log by more than float noise."""


def search_key(paper_id: str, query: str, k: int, embeddings: str) -> str:
    return hashlib.sha256(json.dumps([paper_id, query, k, embeddings]).encode("utf8")).hexdigest()


def stop_key(question_id: str, system: str) -> str:
    return f"{question_id}|{system}"


@dataclass
class RunLog:
    path: Path
    write: bool = True  # a replay reads the log and leaves the file alone
    searches: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    stops: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf8").splitlines():
                rec = json.loads(line)
                if rec["kind"] == "search":
                    self.searches[rec["key"]] = [(cid, score) for cid, score in rec["found"]]
                elif rec["kind"] == "stop":
                    self.stops[rec["key"]] = {"status": rec["status"], "calls": rec["calls"]}

    def _append(self, rec: Dict[str, Any]) -> None:
        if self.write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf8") as fh:
                fh.write(json.dumps(rec) + "\n")

    def add_search(self, key: str, paper_id: str, query: str, found: List[Tuple[str, float]]) -> None:
        self.searches[key] = found
        self._append({"kind": "search", "key": key, "paper_id": paper_id, "query": query,
                      "found": [[cid, score] for cid, score in found]})

    def stop(self, question_id: str, system: str) -> Optional[Dict[str, Any]]:
        return self.stops.get(stop_key(question_id, system))

    def add_stop(self, question_id: str, system: str, status: str, calls: int) -> None:
        key = stop_key(question_id, system)
        self.stops[key] = {"status": status, "calls": calls}
        self._append({"kind": "stop", "key": key, "status": status, "calls": calls})


def check(store: VectorStore, query: str, live: List[RetrievedChunk], logged: List[Tuple[str, float]]) -> None:
    """Raise unless ``live`` and ``logged`` differ only by float noise: every logged passage
    scores within the tolerance of its logged score, and no other passage now scores above
    the logged cut-off by more than the tolerance."""
    index = {c.chunk_id: i for i, c in enumerate(store.chunks)}
    gone = [cid for cid, _ in logged if cid not in index]
    if gone:
        raise RetrievalChanged(f"search {query!r}: logged passages {gone} aren't in the paper")
    sims = store.similarities(query)
    moved = max((abs(float(sims[index[cid]]) - score) for cid, score in logged), default=0.0)
    ids = {cid for cid, _ in logged}
    cutoff = min((score for _, score in logged), default=float("-inf"))
    new = [r.chunk.chunk_id for r in live if r.chunk.chunk_id not in ids and r.score > cutoff + TOLERANCE]
    if moved > TOLERANCE or new:
        raise RetrievalChanged(f"search {query!r}: scores moved by up to {moved:.2g}"
                               + (f" and {new} now make the top {len(logged)}" if new else ""))


class LoggedRetriever(RetrieverAgent):
    """Searches one paper; logs each new search and reuses the logged ranking afterwards."""

    def __init__(self, store: VectorStore, top_k: int, log: RunLog, paper_id: str) -> None:
        super().__init__(store, top_k)
        self.vectors = store
        self.log = log
        self.paper_id = paper_id

    def retrieve(self, query: str, k: int | None = None) -> List[RetrievedChunk]:
        k = k or self.top_k
        live = self.vectors.search(query, k)
        key = search_key(self.paper_id, query, k, self.vectors.embeddings.name)
        logged = self.log.searches.get(key)
        if logged is None:
            self.log.add_search(key, self.paper_id, query, [(r.chunk.chunk_id, r.score) for r in live])
            return live
        check(self.vectors, query, live, logged)
        chunks = {c.chunk_id: c for c in self.vectors.chunks}
        return [RetrievedChunk(chunk=chunks[cid], score=score) for cid, score in logged]


class RecordedStop(Trace):
    """The clock for a run that's in the log. A run checks the deadline before every model
    call, so a run the deadline stopped ended right after its last logged call; this expires
    there, and never for a run that ended any other way."""

    def __init__(self, stop: Dict[str, Any]) -> None:
        super().__init__(None)
        self.stop_after: Optional[int] = stop["calls"] if stop["status"] == "deadline" else None

    def expired(self) -> bool:
        return self.stop_after is not None and len(self.replies()) >= self.stop_after

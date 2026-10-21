"""Per-run traces: every step's wall time and every model call's tokens, cost and retries.

A run carries one :class:`Trace`. Each graph node opens a step, model replies are attached
to the step that made them, and ``to_dict`` is what results files, the API and the site's
replay demo show. The trace also owns the run's deadline, so every node checks the same
clock.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from .llm import Reply


@dataclass
class Step:
    node: str
    round: int
    started: float
    seconds: float = 0.0
    calls: List[Reply] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"node": self.node, "round": self.round, "seconds": round(self.seconds, 4),
                "calls": [{"model": r.model, "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                           "reasoning_tokens": r.reasoning_tokens, "cost_usd": round(r.cost_usd, 6),
                           "seconds": r.seconds, "attempts": r.attempts, "replayed": r.replayed}
                          for r in self.calls],
                "info": self.info}


class Trace:
    def __init__(self, deadline_seconds: Optional[float] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.started = clock()
        self.deadline_seconds = deadline_seconds
        self.steps: List[Step] = []

    def expired(self) -> bool:
        return self.deadline_seconds is not None and self.clock() - self.started >= self.deadline_seconds

    @contextmanager
    def step(self, node: str, round: int) -> Iterator[Step]:
        s = Step(node, round, self.clock())
        self.steps.append(s)
        try:
            yield s
        finally:
            s.seconds = self.clock() - s.started

    def replies(self) -> List[Reply]:
        return [r for s in self.steps for r in s.calls]

    def to_dict(self) -> Dict[str, Any]:
        replies = self.replies()
        return {"seconds": round(self.clock() - self.started, 4), "calls": len(replies),
                "input_tokens": sum(r.input_tokens for r in replies),
                "output_tokens": sum(r.output_tokens for r in replies),
                "cost_usd": round(sum(r.cost_usd for r in replies), 6),
                "retries": sum(r.attempts - 1 for r in replies),
                "steps": [s.to_dict() for s in self.steps]}

"""Model calls behind one small interface, with every call priced and recorded.

* :class:`OpenAIClient` calls the Responses API. It's the default provider; another
  provider is one more class with the same ``complete`` method.
* :class:`StubClient` answers offline from the prompt itself, for ``--dry-run`` and tests.
* :class:`Recorder` wraps any client, stores each reply under a hash of its request and
  replays it, so an evaluation reruns without a key and CI replays recorded runs.

Every reply carries the model the API says answered (which can be more specific than the
one asked for), the tokens, the cost and the wall time. Retries are done here, not inside
the SDK, so they show up in traces and the tests can count them. A :class:`Budget` refuses
a call that could take spending past its cap.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from .logging_utils import get_logger

logger = get_logger("llm")

# USD per million tokens, from developers.openai.com/api/docs/pricing on 2026-10-21.
PRICES: Dict[str, Dict[str, float]] = {
    "gpt-6-luna": {"input": 0.10, "cached": 0.01, "output": 0.50},
    "gpt-6.1-sol": {"input": 2.00, "cached": 0.10, "output": 10.00},
}
CHARS_PER_TOKEN = 3  # a deliberate overestimate of tokens for the pre-call budget check


class LLMError(RuntimeError):
    """The model couldn't be reached or returned something unusable."""


class BudgetExceeded(LLMError):
    """The next call could take spending past the cap."""


class MissingRecording(LLMError):
    """Replay-only mode met a request that was never recorded."""


@dataclass(frozen=True)
class Request:
    model: str
    system: str
    prompt: str
    max_output_tokens: int = 2048
    json_schema: Optional[Dict[str, Any]] = None  # structured output, enforced by the API
    reasoning_effort: Optional[str] = None

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode("utf8")).hexdigest()


@dataclass
class Reply:
    text: str
    model: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0  # includes reasoning tokens, which are billed as output
    reasoning_tokens: int = 0
    seconds: float = 0.0
    status: str = "completed"
    attempts: int = 1
    replayed: bool = False

    @property
    def cost_usd(self) -> float:
        return cost(self)

    def json(self) -> Dict[str, Any]:
        try:
            out = json.loads(self.text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"model returned invalid JSON: {self.text[:200]!r}") from exc
        if not isinstance(out, dict):
            raise LLMError(f"model returned JSON that isn't an object: {self.text[:200]!r}")
        return out


class LLMClient(Protocol):
    def complete(self, request: Request) -> Reply: ...


def price(model: str) -> Dict[str, float]:
    for name in sorted(PRICES, key=len, reverse=True):  # a dated snapshot bills as its model
        if model == name or model.startswith(name + "-"):
            return PRICES[name]
    if model.startswith("stub"):
        return {"input": 0.0, "cached": 0.0, "output": 0.0}
    raise KeyError(f"no price for {model!r}; add it to PRICES before spending on it")


def cost(reply: Reply) -> float:
    p = price(reply.model)
    fresh = reply.input_tokens - reply.cached_tokens
    return (fresh * p["input"] + reply.cached_tokens * p["cached"] + reply.output_tokens * p["output"]) / 1e6


@dataclass
class Budget:
    cap_usd: float
    spent_usd: float = 0.0
    calls: int = 0

    def worst_case(self, request: Request) -> float:
        p = price(request.model)
        tokens_in = (len(request.system) + len(request.prompt)) / CHARS_PER_TOKEN
        return (tokens_in * p["input"] + request.max_output_tokens * p["output"]) / 1e6

    def check(self, request: Request) -> None:
        if self.spent_usd + self.worst_case(request) > self.cap_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} of ${self.cap_usd:.2f}; the next call could pass the cap"
            )

    def charge(self, reply: Reply) -> None:
        self.spent_usd += reply.cost_usd
        self.calls += 1


def is_retryable(exc: BaseException) -> bool:
    """Timeouts, dropped connections, rate limits and 5xx are worth another try; 4xx aren't."""
    name = type(exc).__name__
    if name in {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError",
                "TimeoutError", "ConnectionError"}:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status == 429 or status >= 500)


def with_retries(call: Callable[[], Reply], attempts: int = 3, backoff: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep) -> Reply:
    """``call()`` up to ``attempts`` times on retryable errors, backing off 1, 2, 4... seconds."""
    for attempt in range(1, attempts + 1):
        try:
            reply = call()
            reply.attempts = attempt
            return reply
        except Exception as exc:  # noqa: BLE001 - classified right below
            if attempt == attempts or not is_retryable(exc):
                raise
            wait = backoff * 2 ** (attempt - 1)
            logger.warning("retryable %s on attempt %d; waiting %.1fs", type(exc).__name__, attempt, wait)
            sleep(wait)
    raise AssertionError("unreachable")


class OpenAIClient:
    """The Responses API. ``timeout`` is per attempt; retries happen in :func:`with_retries`."""

    def __init__(self, api_key: str = "", timeout: float = 60.0, attempts: int = 3,
                 budget: Optional[Budget] = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError("live runs need the 'openai' package: pip install openai") from exc
        self._client = OpenAI(api_key=api_key or None, timeout=timeout, max_retries=0)
        self.attempts = attempts
        self.budget = budget

    def _once(self, request: Request) -> Reply:
        kwargs: Dict[str, Any] = {"model": request.model, "instructions": request.system,
                                  "input": request.prompt, "max_output_tokens": request.max_output_tokens,
                                  "store": False}
        if request.reasoning_effort:
            kwargs["reasoning"] = {"effort": request.reasoning_effort}
        if request.json_schema is not None:
            kwargs["text"] = {"format": {"type": "json_schema", "name": "output",
                                         "schema": request.json_schema, "strict": True}}
        t0 = time.perf_counter()
        r = self._client.responses.create(**kwargs)
        u = r.usage
        return Reply(text=r.output_text, model=r.model, input_tokens=u.input_tokens,
                     cached_tokens=u.input_tokens_details.cached_tokens or 0,
                     output_tokens=u.output_tokens,
                     reasoning_tokens=u.output_tokens_details.reasoning_tokens or 0,
                     seconds=round(time.perf_counter() - t0, 3), status=r.status)

    def complete(self, request: Request) -> Reply:
        if self.budget is not None:
            self.budget.check(request)
        reply = with_retries(lambda: self._once(request), self.attempts)
        if reply.status != "completed":
            raise LLMError(f"response {reply.status}: the output was cut short, raise max_output_tokens")
        if self.budget is not None:
            self.budget.charge(reply)
        return reply


class StubClient:
    """Offline stand-in. Answers from the numbered passages in the prompt, cites the first
    two, and asks for one more round the first time it's asked to evaluate, so the loop runs
    end to end with no key and no network."""

    model = "stub"

    def __init__(self) -> None:
        self.evaluations = 0

    def complete(self, request: Request) -> Reply:
        system = request.system.lower()
        if "review" in system:
            self.evaluations += 1
            out: Dict[str, Any] = (
                {"sufficient": False, "grounded": True, "missing": "the exact figure",
                 "refined_query": "reported results and numbers"}
                if self.evaluations == 1 else
                {"sufficient": True, "grounded": True, "missing": "", "refined_query": ""})
        else:
            passages = re.findall(r"^\[(\d+)\][^\n]*\n(.+)$", request.prompt, flags=re.M)
            if passages:
                first = re.split(r"(?<=[.!?])\s", passages[0][1].strip())[0]
                cited = [int(n) for n, _ in passages[:2]]
                out = {"answer": f"[dry-run] {first}", "evidence": cited, "unanswerable": False}
            else:
                out = {"answer": "", "evidence": [], "unanswerable": True}
        return Reply(text=json.dumps(out), model=self.model,
                     input_tokens=len(request.prompt) // 4, output_tokens=40)


@dataclass
class Recorder:
    """Replies by request hash, appended to a JSONL file and replayed from it."""

    path: Path
    client: Optional[LLMClient] = None  # None means replay only
    calls: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    made: int = 0
    replayed: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            for line in self.path.read_text(encoding="utf8").splitlines():
                rec = json.loads(line)
                self.calls[rec["key"]] = rec

    @property
    def recorded_spend_usd(self) -> float:
        return sum(cost(Reply(**rec["reply"])) for rec in self.calls.values())

    def complete(self, request: Request, tag: Optional[Dict[str, Any]] = None) -> Reply:
        key = request.key()
        if key in self.calls:
            self.replayed += 1
            reply = Reply(**self.calls[key]["reply"])
            reply.replayed = True
            return reply
        if self.client is None:
            raise MissingRecording(f"no recorded reply for {tag or request.model}; rerun with a key to make it")
        reply = self.client.complete(request)
        rec = {"key": key, "tag": tag or {}, "reply": asdict(reply)}
        self.calls[key] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf8") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.made += 1
        return reply


def usage_summary(replies: List[Reply]) -> Dict[str, Any]:
    return {"calls": len(replies), "input_tokens": sum(r.input_tokens for r in replies),
            "output_tokens": sum(r.output_tokens for r in replies),
            "cost_usd": round(sum(r.cost_usd for r in replies), 6),
            "models": sorted({r.model for r in replies})}

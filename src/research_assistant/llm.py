"""A thin wrapper around the Anthropic Messages API.

Provides:
* :class:`ClaudeClient` — retrying ``messages.create`` calls plus a helper to
  coerce model output into JSON.
* :class:`StubClaudeClient` — an offline drop-in used by ``--dry-run`` and
  tests; it synthesises plausible structured output from the prompt so the
  full agent graph can run with no API key and no network access.

The Anthropic SDK is imported lazily so the package can be imported (and the
stub used) without the dependency installed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Protocol

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .logging_utils import get_logger

logger = get_logger("llm")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_RETRYABLE_HINTS = (
    "rate", "overloaded", "timeout", "timed out", "connection",
    "internal server", "503", "502", "529", "temporarily",
)


class LLMError(RuntimeError):
    """Raised when the model cannot be queried or returns unusable output."""


class LLMClient(Protocol):
    """Minimal interface the agents depend on."""

    def complete(self, *, model: str, system: str, prompt: str, **kw: Any) -> str: ...

    def complete_json(
        self, *, model: str, system: str, prompt: str, **kw: Any
    ) -> Dict[str, Any]: ...


def _is_retryable(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(hint in text for hint in _RETRYABLE_HINTS)


def extract_json(text: str) -> Dict[str, Any]:
    """Best-effort extraction of a single JSON object from model output."""
    candidate = text.strip()
    fenced = _JSON_FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    # Fall back to the outermost braces.
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Model did not return valid JSON. First 300 chars:\n{text[:300]}"
        ) from exc


class ClaudeClient:
    """Retrying client for the Anthropic Messages API.

    Tracks cumulative token usage and enforces a hard cap on the number of
    API calls a single instance will make (``max_calls``, default 40).
    """

    def __init__(
        self,
        api_key: str,
        default_max_tokens: int = 2048,
        max_calls: int = 40,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Calling Claude requires the 'anthropic' package. "
                "Install it with: pip install anthropic\n"
                "Or run with --dry-run to use the offline stub."
            ) from exc
        if not api_key:
            raise LLMError("An Anthropic API key is required for ClaudeClient.")
        self._client = Anthropic(api_key=api_key)
        self.default_max_tokens = default_max_tokens
        self.max_calls = max_calls
        self.call_count = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def complete(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Return the concatenated text content of a single-turn completion."""
        if self.call_count >= self.max_calls:
            raise LLMError(
                f"Hit the per-run budget of {self.max_calls} API calls. "
                "This guards against runaway spend; raise `max_calls` on "
                "ClaudeClient if a legitimate run genuinely needs more."
            )
        self.call_count += 1
        logger.info("anthropic call %d/%d model=%s", self.call_count, self.max_calls, model)
        message = self._client.messages.create(
            model=model,
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(message, "usage", None)
        if usage is not None:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        ).strip()

    def complete_json(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Run :meth:`complete` and parse the result as JSON."""
        raw = self.complete(
            model=model,
            system=system,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return extract_json(raw)


class StubClaudeClient:
    """Offline stand-in that fabricates structured output from the prompt.

    It is intentionally simple: it pulls a few sentences out of the numbered
    context block to make summaries that look grounded, and alternates the
    evaluator's verdict on the first call so the iterative loop is exercised
    before terminating.
    """

    def __init__(self) -> None:
        self._eval_calls = 0

    # The agents only ever ask the stub for JSON.
    def complete(self, *, model: str, system: str, prompt: str, **_: Any) -> str:
        return json.dumps(self.complete_json(model=model, system=system, prompt=prompt))

    def complete_json(
        self, *, model: str, system: str, prompt: str, **_: Any
    ) -> Dict[str, Any]:
        if "evaluat" in system.lower() or "critique" in system.lower():
            return self._fake_evaluation()
        return self._fake_summary(prompt)

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _context_sentences(prompt: str, limit: int = 6) -> List[str]:
        # Prefer the explicit SOURCES block; fall back to the first marker.
        body = prompt
        if "=== SOURCES ===" in prompt:
            body = prompt.split("=== SOURCES ===", 1)[1].split("=== END SOURCES ===", 1)[0]
        else:
            marker = re.search(r"\[\d+\]\s*\(source:", prompt) or re.search(r"\[1\]", prompt)
            if marker:
                body = prompt[marker.start():]
        # Drop the '[n] (source: "...")' header lines, keep the passage text.
        body = re.sub(r'\[\d+\]\s*\(source:[^)]*\)', " ", body)
        sentences = re.split(r"(?<=[.!?])\s+", body.replace("\n", " "))
        cleaned = [s.strip() for s in sentences if len(s.strip()) > 40]
        return cleaned[:limit]

    def _fake_summary(self, prompt: str) -> Dict[str, Any]:
        sents = self._context_sentences(prompt)
        if not sents:
            sents = ["The retrieved passages did not contain extractable sentences."]
        summary = " ".join(
            f"{s} [{i}]" for i, s in enumerate(sents[:3], start=1)
        )
        return {
            "summary": "[dry-run] " + summary,
            "methods": [
                "Approach synthesised from the retrieved passages [1].",
                "Evaluation protocol referenced across sources [2].",
            ],
            "key_findings": [f"{s} [{i}]" for i, s in enumerate(sents[:3], start=1)],
            "research_gaps": [
                "Generalisation beyond the studied datasets is unverified.",
                "No ablation isolates the contribution of each component.",
            ],
        }

    def _fake_evaluation(self) -> Dict[str, Any]:
        self._eval_calls += 1
        if self._eval_calls == 1:
            return {
                "grounded": True,
                "coverage_score": 0.55,
                "sufficiency": "needs_more",
                "critique": "[dry-run] Initial pass; broadening the search once.",
                "missing_aspects": ["quantitative results", "limitations"],
                "refined_query": "quantitative results and limitations",
            }
        return {
            "grounded": True,
            "coverage_score": 0.85,
            "sufficiency": "sufficient",
            "critique": "[dry-run] Synthesis now covers the question adequately.",
            "missing_aspects": [],
            "refined_query": None,
        }

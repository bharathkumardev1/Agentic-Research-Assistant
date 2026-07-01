"""Structured logging for the Agentic Research Assistant.

A single ``get_logger()`` call anywhere in the codebase returns a logger
configured consistently: level from the ``LOG_LEVEL`` environment variable
(default ``INFO``), a timestamped format, and (optionally) a per-run
correlation id so that concurrent runs in the same process don't interleave
confusingly in the logs.

This module has zero required dependencies, it only uses the standard
library, so importing it never fails even in a minimal environment.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextvars import ContextVar

_run_id: ContextVar[str] = ContextVar("run_id", default="-")
_CONFIGURED = False


class _RunIdFilter(logging.Filter):
    """Injects the current run id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get()
        return True


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    resolved = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s [%(run_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    handler.addFilter(_RunIdFilter())
    root = logging.getLogger("research_assistant")
    root.setLevel(resolved)
    root.handlers = [handler]
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring the root logger on first use."""
    configure_logging()
    return logging.getLogger(f"research_assistant.{name}")


def new_run_id() -> str:
    """Generate and activate a short correlation id for the current run.

    Call this once per pipeline invocation (e.g. at the top of
    ``ResearchPipeline.run``) so every log line emitted during that run can
    be grepped out of a shared log stream.
    """
    rid = uuid.uuid4().hex[:8]
    _run_id.set(rid)
    return rid

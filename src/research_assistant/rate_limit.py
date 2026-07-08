"""A minimal in-process rate limiter for the web service.

Sliding-window counter per client key (an authenticated caller's name, or
their IP if no API key is configured). This is intentionally in-memory: it
protects a single process from being hammered by one client, which is
enough for a single-instance deployment.

Running multiple uvicorn workers or replicas gives each its own bucket, so
the effective limit multiplies by process count. For a hard global limit
across replicas, back this with a shared store (e.g. Redis) instead.
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock


class RateLimiter:
    """Tracks request timestamps per key over a rolling 60-second window."""

    def __init__(self) -> None:
        self._hits: dict[str, deque] = {}
        self._lock = Lock()

    def allow(self, key: str, limit: int) -> bool:
        """Return True if ``key`` may make another request under ``limit`` per minute.

        A ``limit`` of 0 or less disables the check (always allowed).
        """
        if limit <= 0:
            return True
        now = time.monotonic()
        window_start = now - 60.0
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True

    def reset(self) -> None:
        """Clear all tracked state. Mainly useful for tests."""
        with self._lock:
            self._hits.clear()

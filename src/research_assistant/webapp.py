"""A minimal FastAPI web service around the research pipeline.

Run locally with:
    uvicorn research_assistant.webapp:app --reload

In production this is what the Dockerfile and render.yaml point at. On
startup it looks for a previously ``ingest``-ed index at ``INDEX_DIR``
(e.g. a mounted persistent volume) and loads that if present. Otherwise it
falls back to building a fast, offline demo index from the bundled sample
papers, so the service is demoable immediately after deploy with zero
configuration. Either way, if ``ANTHROPIC_API_KEY`` is set, questions are
answered by the real Claude API; otherwise the offline stub is used.

Endpoints
---------
GET  /health              -> liveness/readiness check, no auth required
GET  /metrics             -> Prometheus metrics, no auth required
POST /research            -> run the agentic loop for a question

Production-hardening notes
---------------------------
- Auth: set ``WEB_API_KEY`` and/or ``WEB_API_KEYS`` (see config.py) to
  require an ``X-API-Key`` header on ``/research``.
- Rate limiting: ``RATE_LIMIT_PER_MINUTE`` caps requests per caller. It's
  in-memory and per-process -- see ``rate_limit.py`` for the multi-worker
  caveat.
- Timeouts: ``REQUEST_TIMEOUT_SECONDS`` bounds how long a single request can
  hold the connection open before the caller gets a 504.
- Errors: callers never see raw exception text, only a short reference id
  that's paired with the full traceback in the server logs.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .config import get_settings
from .factory import build_client, build_index, build_pipeline, load_index
from .ingestion.loaders import load_paths
from .logging_utils import get_logger
from .rag.vector_store import FaissVectorStore
from .rate_limit import RateLimiter

logger = get_logger("webapp")
_state: dict = {"pipeline": None, "mode": "unknown"}
_rate_limiter = RateLimiter()
# A small pool just to enforce request timeouts; the actual work is CPU/IO
# bound Claude + FAISS calls, not thread-heavy, so a handful is plenty.
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="research")

REQUEST_COUNT = Counter(
    "research_requests_total", "Total /research requests by outcome", ["status"]
)
REQUEST_LATENCY = Histogram(
    "research_request_duration_seconds", "Wall-clock time for /research calls"
)


def _sample_dir():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent.parent / "examples" / "sample_papers"


def _load_default_pipeline() -> None:
    settings = get_settings()
    live_answers = bool(settings.anthropic_api_key)

    # Only bother constructing a (possibly slow-to-load, model-downloading)
    # real embedding backend if there's actually a persisted index on disk;
    # otherwise fall straight through to the fast offline demo path below.
    store = None
    if FaissVectorStore.exists(settings.index_dir):
        try:
            store = load_index(settings, dry_run=False)
            logger.info(
                "loaded persisted index from %s: chunks=%d", settings.index_dir, len(store)
            )
        except (FileNotFoundError, ValueError, ImportError) as exc:
            logger.warning(
                "failed to load persisted index at %s (%s); using bundled demo papers",
                settings.index_dir,
                exc,
            )

    if store is None:
        papers = load_paths([_sample_dir()])
        store = build_index(papers, settings, dry_run=True)

    client = build_client(settings, dry_run=not live_answers)
    _state["pipeline"] = build_pipeline(store, client, settings)
    _state["mode"] = "live" if live_answers else "dry_run"
    logger.info("startup complete: mode=%s chunks=%d", _state["mode"], len(store))


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _load_default_pipeline()
    yield


app = FastAPI(
    title="Agentic Research Assistant",
    description="Multi-agent RAG over research papers, with citation-backed answers.",
    version="0.3.0",
    lifespan=_lifespan,
)

_cors_origins = get_settings().cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors_origins.strip() == "*" else [o.strip() for o in _cors_origins.split(",") if o.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_api_keys(settings) -> dict:
    """Return {key: client_name}, merging WEB_API_KEY and WEB_API_KEYS."""
    mapping: dict = {}
    if settings.web_api_key:
        mapping[settings.web_api_key] = "default"
    for pair in settings.web_api_keys.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, key = pair.partition(":")
        key = (key or name).strip()
        name = name.strip() or "unnamed"
        if key:
            mapping[key] = name
    return mapping


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _require_api_key(
    request: Request, x_api_key: Optional[str] = Header(default=None)
) -> str:
    """Authenticate the request and return a client identity.

    The identity is used to bucket rate limits and tag logs per caller. If
    no API key is configured, auth is open by configuration choice and the
    caller's IP is used as its identity instead.
    """
    settings = get_settings()
    keys = _parse_api_keys(settings)
    if not keys:
        return _client_ip(request)
    if x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
    return keys[x_api_key]


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

@app.get("/")
def root() -> RedirectResponse:
    """Redirect the bare URL to the interactive API docs, purely cosmetic."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe. No auth required so load balancers can call it freely."""
    return {"status": "ok", "mode": _state.get("mode", "unknown")}


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint. No auth; put it behind a private network/route if needed."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/research")
def research(payload: ResearchRequest, client_id: str = Depends(_require_api_key)) -> dict:
    """Run the agentic research loop for a question and return the structured result."""
    settings = get_settings()
    if not _rate_limiter.allow(client_id, settings.rate_limit_per_minute):
        REQUEST_COUNT.labels(status="rate_limited").inc()
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again shortly.",
            headers={"Retry-After": "60"},
        )

    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service is still starting up.")

    ref = uuid.uuid4().hex[:8]
    started = time.monotonic()
    try:
        future = _executor.submit(pipeline.run, payload.question)
        result = future.result(timeout=settings.request_timeout_seconds)
    except FutureTimeoutError:
        REQUEST_COUNT.labels(status="timeout").inc()
        logger.warning(
            "research request timed out: client=%s ref=%s limit=%.0fs",
            client_id,
            ref,
            settings.request_timeout_seconds,
        )
        raise HTTPException(
            status_code=504, detail=f"Request timed out. Reference: {ref}"
        ) from None
    except Exception as exc:  # noqa: BLE001 - never leak internals to the caller
        REQUEST_COUNT.labels(status="error").inc()
        logger.exception("research run failed: client=%s ref=%s", client_id, ref)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error processing the request. Reference: {ref}",
        ) from exc
    else:
        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(time.monotonic() - started)
        return result.model_dump()

"""The research pipeline behind a small HTTP API; the Docker image runs this.

    uvicorn research_assistant.webapp:app --reload

On startup it loads the index at ``INDEX_DIR`` if one was built with ``ingest``; otherwise
it indexes the three sample papers with hashing embeddings. Answers come from the model when
``OPENAI_API_KEY`` is set and from the offline stub when it isn't; ``/health`` says which.

GET  /health     liveness, no auth
GET  /metrics    Prometheus counters and latency histogram, no auth
POST /research   one question through the loop; returns the answer, citations and trace

Set ``WEB_API_KEY`` or ``WEB_API_KEYS`` to require an ``X-API-Key`` header on /research.
``RATE_LIMIT_PER_MINUTE`` limits each caller, in memory and per process (see rate_limit.py
for what that means with several workers). ``REQUEST_TIMEOUT_SECONDS`` returns a 504 when a
run takes too long. A failed run returns a short reference id, and the traceback goes to the
log under the same id, never to the caller.
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
from .rag.vector_store import VectorStore
from .rate_limit import RateLimiter

logger = get_logger("webapp")
_state: dict = {"pipeline": None, "mode": "unknown"}
_rate_limiter = RateLimiter()
# A small pool just to enforce request timeouts; the work is mostly waiting on model
# calls, so a handful of threads is plenty.
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
    live_answers = bool(settings.openai_api_key)

    # Only bother constructing a (possibly slow-to-load, model-downloading)
    # real embedding backend if there's actually a persisted index on disk;
    # otherwise fall straight through to the fast offline demo path below.
    store = None
    if VectorStore.exists(settings.index_dir):
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
    description="Answers questions about research papers, citing the passages it used.",
    version="0.4.0",
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
    """The bare URL goes to the interactive API docs."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health() -> dict:
    """No auth, so a load balancer can poll it."""
    return {"status": "ok", "mode": _state.get("mode", "unknown")}


@app.get("/metrics")
def metrics() -> Response:
    """No auth; keep it on a private network if that matters."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/research")
def research(payload: ResearchRequest, client_id: str = Depends(_require_api_key)) -> dict:
    """One question through the loop: answer, cited passages, reviews and trace."""
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

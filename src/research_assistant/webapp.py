"""A minimal FastAPI web service around the research pipeline.

Run locally with:
    uvicorn research_assistant.webapp:app --reload

In production this is what the Dockerfile and render.yaml point at. On
startup it builds an index from the bundled sample papers (offline, no
network) unless ``ANTHROPIC_API_KEY`` is set, in which case it still starts
in offline mode for the index (to avoid a slow cold start downloading an
embedding model on every deploy) but answers questions with the real Claude
API. Point ``INDEX_DIR`` at a persistent volume and run ``ingest`` yourself
for a real corpus; this default is meant to make the service demoable
immediately after deploy with zero configuration.

Endpoints
---------
GET  /health              -> liveness/readiness check, no auth required
POST /research            -> run the agentic loop for a question
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .factory import build_client, build_index, build_pipeline
from .ingestion.loaders import load_paths
from .logging_utils import get_logger

logger = get_logger("webapp")
_state: dict = {"pipeline": None, "mode": "unknown"}


def _sample_dir():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent.parent / "examples" / "sample_papers"


def _load_default_pipeline() -> None:
    settings = get_settings()
    dry_run_index = True  # always build the demo index offline; fast, no downloads
    live_answers = bool(settings.anthropic_api_key)

    papers = load_paths([_sample_dir()])
    store = build_index(papers, settings, dry_run=dry_run_index)
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
    version="0.2.0",
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


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.web_api_key:
        return  # auth disabled; open by configuration choice
    if x_api_key != settings.web_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


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


@app.post("/research", dependencies=[Depends(_require_api_key)])
def research(payload: ResearchRequest) -> dict:
    """Run the agentic research loop for a question and return the structured result."""
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service is still starting up.")
    try:
        result = pipeline.run(payload.question)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the caller
        logger.exception("research run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result.model_dump()


"""Application configuration, loaded from environment variables / ``.env``.

All knobs live here so the agents and pipeline stay free of magic numbers.
Environment variables are matched case-insensitively, e.g. ``TOP_K`` sets
``Settings.top_k``. The OpenAI key uses the conventional ``OPENAI_API_KEY`` name.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the research assistant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model calls. Model IDs are recorded with every reply, so results say what answered.
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    answer_model: str = Field(default="gpt-6-luna")
    review_model: str = Field(default="gpt-6-luna")
    judge_model: str = Field(default="gpt-6.1-sol", description="Grades answers in the evaluation.")
    reasoning_effort: str = Field(default="low")
    max_output_tokens: int = Field(default=2048, ge=256)
    call_timeout_seconds: float = Field(default=60.0, gt=0, description="Per attempt of one model call.")
    call_attempts: int = Field(default=3, ge=1, description="Tries per call on timeouts, 429s and 5xx.")
    run_deadline_seconds: float = Field(default=120.0, gt=0, description="Checked before every model call.")
    run_budget_usd: float = Field(default=0.50, gt=0, description="Spending cap for one question.")
    web_api_key: str = Field(
        default="",
        alias="WEB_API_KEY",
        description="If set, the web service requires this key in the X-API-Key header.",
    )
    web_api_keys: str = Field(
        default="",
        alias="WEB_API_KEYS",
        description=(
            "Comma-separated 'name:key' pairs for scoped API access, e.g. "
            "'alice:sk-alice,bob:sk-bob'. Adds to (does not replace) WEB_API_KEY. "
            "Each name is used to bucket rate limits and tag logs per caller."
        ),
    )
    cors_origins: str = Field(
        default="*",
        alias="CORS_ORIGINS",
        description="Comma-separated list of allowed browser origins for the web service, or '*' for any.",
    )
    rate_limit_per_minute: int = Field(
        default=20,
        ge=0,
        alias="RATE_LIMIT_PER_MINUTE",
        description="Max /research requests per minute per client (by API key or IP). 0 disables the limit.",
    )
    request_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        alias="REQUEST_TIMEOUT_SECONDS",
        description="Hard wall-clock cap on a single /research call; the caller gets a 504 if it's exceeded.",
    )

    # Embeddings and retrieval.
    # "minilm" (all-MiniLM-L6-v2 through ONNX Runtime, downloaded on first use) or "hashing"
    # (dependency-free, deterministic; used for offline tests / dry runs).
    embedding_backend: str = Field(default="minilm")
    embedding_model_dir: Path = Field(default=Path("models/all-MiniLM-L6-v2"))
    hashing_dim: int = Field(default=512, ge=64, description="Dim of hashing fallback.")

    chunk_size: int = Field(default=1100, ge=200, description="Max chars per chunk.")
    chunk_overlap: int = Field(default=150, ge=0, description="Overlap chars.")
    top_k: int = Field(default=5, ge=1, description="Chunks retrieved per query.")

    # The loop: retrieve, answer, review, and again with a refined query, at most this often.
    max_rounds: int = Field(default=3, ge=1, description="1 is the single-pass baseline.")

    # Storage.
    index_dir: Path = Field(default=Path("data/index"))
    raw_dir: Path = Field(default=Path("data/raw"))

    def require_api_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to the environment or .env, "
                               "or run with --dry-run to use the offline stub.")
        return self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()

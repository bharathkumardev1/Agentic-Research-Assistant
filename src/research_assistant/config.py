
"""Application configuration, loaded from environment variables / ``.env``.

All knobs live here so the agents and pipeline stay free of magic numbers.
Environment variables are matched case-insensitively, e.g. ``TOP_K`` sets
``Settings.top_k``. The Anthropic key uses the conventional
``ANTHROPIC_API_KEY`` name.
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

    # --- Anthropic / Claude ------------------------------------------------
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    summarizer_model: str = Field(default="claude-sonnet-4-6")
    evaluator_model: str = Field(default="claude-opus-4-8")
    max_tokens: int = Field(default=2048, ge=256)
    max_api_calls: int = Field(
        default=40, ge=1, description="Hard cap on Claude calls per pipeline run."
    )
    web_api_key: str = Field(
        default="",
        alias="WEB_API_KEY",
        description="If set, the web service requires this key in the X-API-Key header.",
    )
    cors_origins: str = Field(
        default="*",
        alias="CORS_ORIGINS",
        description="Comma-separated list of allowed browser origins for the web service, or '*' for any.",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)

    # --- Embeddings / RAG --------------------------------------------------
    # "sentence-transformers" (default, high quality) or "hashing"
    # (dependency-free, deterministic; used for offline tests / dry runs).
    embedding_backend: str = Field(default="sentence-transformers")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    hashing_dim: int = Field(default=512, ge=64, description="Dim of hashing fallback.")

    chunk_size: int = Field(default=1100, ge=200, description="Max chars per chunk.")
    chunk_overlap: int = Field(default=150, ge=0, description="Overlap chars.")
    top_k: int = Field(default=6, ge=1, description="Chunks retrieved per query.")

    # --- Agentic loop ------------------------------------------------------
    max_iterations: int = Field(
        default=3, ge=1, description="Max retrieve→summarize→evaluate cycles."
    )

    # --- Storage -----------------------------------------------------------
    index_dir: Path = Field(default=Path("data/index"))
    raw_dir: Path = Field(default=Path("data/raw"))

    def require_api_key(self) -> str:
        """Return the API key or raise a friendly error if it is missing."""
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your environment or a "
                ".env file (see .env.example). For an offline demonstration that "
                "does not call the API, run with --dry-run."
            )
        return self.anthropic_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()

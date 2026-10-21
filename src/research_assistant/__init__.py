"""Agentic Research Assistant.

Answers questions about research papers with cited passages: retrieve, answer, have a
reviewer check the answer against the passages it cites, and retrieve again with a refined
query when it falls short. The QASPER evaluation in ``research_assistant.eval`` measures
that loop against a single pass.

The top-level package imports nothing heavy, so submodules such as
``research_assistant.ingestion.chunking`` load without ``langgraph`` or the embedding model.
"""

from __future__ import annotations

__version__ = "0.4.0"

__all__ = ["__version__"]

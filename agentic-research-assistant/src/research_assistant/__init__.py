"""Agentic Research Assistant.

A multi-agent system (Claude + LangGraph) that autonomously reads and
synthesizes research papers over a FAISS-backed RAG pipeline, producing
citation-backed summaries with extracted methods, key findings and
research gaps.

The top-level package is intentionally light on imports so that individual
submodules (e.g. ``research_assistant.ingestion.chunking``) can be imported
without pulling in heavy optional dependencies such as ``langgraph``,
``faiss`` or ``sentence-transformers``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]

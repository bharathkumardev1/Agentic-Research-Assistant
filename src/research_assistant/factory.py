"""Composition root: build the pipeline's pieces from :class:`Settings`.

Centralising construction keeps the CLI thin and makes the wiring easy to
reuse in tests. The ``dry_run`` flag swaps in the offline embedding/LLM
backends so the whole system runs with no API key or model download.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .agents.answerer import AnswerAgent
from .agents.evaluator import EvaluatorAgent
from .agents.retriever import RetrieverAgent
from .config import Settings
from .graph.workflow import EventHook, ResearchPipeline
from .ingestion.chunking import chunk_paper
from .ingestion.loaders import LoadedPaper
from .llm import Budget, LLMClient, OpenAIClient, StubClient
from .rag.embeddings import EmbeddingBackend, get_embeddings
from .rag.vector_store import FaissVectorStore
from .schemas import Chunk


def build_embeddings(settings: Settings, dry_run: bool = False) -> EmbeddingBackend:
    if dry_run:
        return get_embeddings("hashing", hashing_dim=settings.hashing_dim)
    return get_embeddings(
        settings.embedding_backend,
        model_name=settings.embedding_model,
        hashing_dim=settings.hashing_dim,
    )


def build_client(settings: Settings, dry_run: bool = False, budget: Budget | None = None) -> LLMClient:
    if dry_run:
        return StubClient()
    return OpenAIClient(api_key=settings.require_api_key(), timeout=settings.call_timeout_seconds,
                        attempts=settings.call_attempts, budget=budget)


def chunks_from_papers(papers: List[LoadedPaper], settings: Settings) -> List[Chunk]:
    chunks: List[Chunk] = []
    for meta, text in papers:
        chunks.extend(
            chunk_paper(
                meta,
                text,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
        )
    return chunks


def build_index(
    papers: List[LoadedPaper],
    settings: Settings,
    dry_run: bool = False,
) -> FaissVectorStore:
    """Chunk papers, embed them and return a populated vector store."""
    embeddings = build_embeddings(settings, dry_run=dry_run)
    store = FaissVectorStore(embeddings)
    store.add(chunks_from_papers(papers, settings))
    return store


def load_index(settings: Settings, dry_run: bool = False) -> FaissVectorStore:
    embeddings = build_embeddings(settings, dry_run=dry_run)
    return FaissVectorStore.load(Path(settings.index_dir), embeddings)


def build_pipeline(
    store: FaissVectorStore,
    client: LLMClient,
    settings: Settings,
    on_event: EventHook = None,
) -> ResearchPipeline:
    effort = settings.reasoning_effort or None
    return ResearchPipeline(
        RetrieverAgent(store, top_k=settings.top_k),
        AnswerAgent(client, settings.answer_model, settings.max_output_tokens, effort),
        EvaluatorAgent(client, settings.review_model, min(settings.max_output_tokens, 1024), effort),
        max_rounds=settings.max_rounds,
        deadline_seconds=settings.run_deadline_seconds,
        on_event=on_event,
    )

"""Composition root: build the pipeline's pieces from :class:`Settings`.

Centralising construction keeps the CLI thin and makes the wiring easy to
reuse in tests. The ``dry_run`` flag swaps in the offline embedding/LLM
backends so the whole system runs with no API key or model download.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .agents.evaluator import EvaluatorAgent
from .agents.retriever import RetrieverAgent
from .agents.summarizer import SummarizerAgent
from .config import Settings
from .graph.workflow import EventHook, ResearchPipeline
from .ingestion.chunking import chunk_paper
from .ingestion.loaders import LoadedPaper
from .llm import ClaudeClient, LLMClient, StubClaudeClient
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


def build_client(settings: Settings, dry_run: bool = False) -> LLMClient:
    if dry_run:
        return StubClaudeClient()
    return ClaudeClient(
        api_key=settings.require_api_key(),
        default_max_tokens=settings.max_tokens,
        max_calls=settings.max_api_calls,
    )


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
    retriever = RetrieverAgent(store, top_k=settings.top_k)
    summarizer = SummarizerAgent(
        client, model=settings.summarizer_model, max_tokens=settings.max_tokens
    )
    evaluator = EvaluatorAgent(
        client, model=settings.evaluator_model, max_tokens=min(settings.max_tokens, 1024)
    )
    return ResearchPipeline(
        retriever,
        summarizer,
        evaluator,
        max_iterations=settings.max_iterations,
        on_event=on_event,
    )

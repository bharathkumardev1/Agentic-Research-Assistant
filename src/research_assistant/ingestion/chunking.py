"""Boundary-aware text chunking.

Splits text into overlapping chunks that respect paragraph and sentence
boundaries where possible. Pure standard library so it can be unit-tested
without any heavy dependency.

Guarantees:
    * every returned chunk has ``len(chunk) <= chunk_size``;
    * consecutive chunks share up to ``overlap`` characters of context;
    * no empty/whitespace-only chunks are returned.
"""

from __future__ import annotations

import re
from typing import List

from ..schemas import Chunk, PaperMetadata

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"[ \t]+")


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(_WS_RE.sub(" ", line).rstrip() for line in text.split("\n"))
    return text.strip()


def _hard_split(token: str, size: int) -> List[str]:
    """Split an over-long token (a word with no spaces) by characters."""
    return [token[i : i + size] for i in range(0, len(token), size)]


def _atomic_units(text: str, size: int) -> List[str]:
    """Break ``text`` into units each no longer than ``size`` characters.

    Prefers paragraph, then sentence, then word, then character boundaries.
    """
    units: List[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= size:
            units.append(paragraph)
            continue
        # Paragraph too big -> split into sentences.
        for sentence in _SENTENCE_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= size:
                units.append(sentence)
                continue
            # Sentence too big -> pack words.
            current = ""
            for word in sentence.split(" "):
                if len(word) > size:
                    if current:
                        units.append(current)
                        current = ""
                    units.extend(_hard_split(word, size))
                    continue
                candidate = f"{current} {word}".strip()
                if len(candidate) <= size:
                    current = candidate
                else:
                    if current:
                        units.append(current)
                    current = word
            if current:
                units.append(current)
    return units


def _overlap_prefix(chunk: str, overlap: int) -> str:
    """Return a trailing window of ``chunk`` (<= ``overlap`` chars), word-aligned."""
    if overlap <= 0 or not chunk:
        return ""
    tail = chunk[-overlap:]
    # Avoid starting mid-word.
    space = tail.find(" ")
    if space != -1:
        tail = tail[space + 1 :]
    return tail.strip()


def chunk_text(text: str, chunk_size: int = 1100, overlap: int = 150) -> List[str]:
    """Split ``text`` into overlapping, boundary-aware chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = _normalize(text)
    if not text:
        return []

    units = _atomic_units(text, chunk_size)
    chunks: List[str] = []
    body = ""
    prefix = ""

    for unit in units:
        projected = len(prefix) + (1 if prefix else 0) + len(body) + 1 + len(unit)
        if body and projected > chunk_size:
            chunk = f"{prefix} {body}".strip() if prefix else body
            chunks.append(chunk)
            prefix = _overlap_prefix(chunk, overlap)
            body = ""
            # If the overlap prefix leaves no room for this unit, drop it.
            if len(prefix) + 1 + len(unit) > chunk_size:
                prefix = ""
        body = f"{body} {unit}".strip() if body else unit

    if body:
        chunks.append(f"{prefix} {body}".strip() if prefix else body)

    return [c for c in chunks if c.strip()]


def chunk_paper(
    metadata: PaperMetadata,
    text: str,
    chunk_size: int = 1100,
    overlap: int = 150,
) -> List[Chunk]:
    """Chunk a paper's full text into :class:`Chunk` objects with metadata."""
    pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return [
        Chunk(
            chunk_id=f"{metadata.paper_id}::{i}",
            paper_id=metadata.paper_id,
            title=metadata.title,
            text=piece,
            chunk_index=i,
            source=metadata.source,
        )
        for i, piece in enumerate(pieces)
    ]

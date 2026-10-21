"""Turn retrieved chunks into the numbered passages the model reads, and their citations.

Keeping this in one place guarantees that the [n] the answerer cites is the passage the
report shows as [n].
"""

from __future__ import annotations

from typing import List, Tuple

from .schemas import RetrievedChunk, SourceRef


def number_sources(retrieved: List[RetrievedChunk]) -> Tuple[str, List[SourceRef]]:
    """``(passages, sources)``: the numbered text for the prompt and one SourceRef per number."""
    lines: List[str] = []
    sources: List[SourceRef] = []
    for marker, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        where = chunk.section or chunk.title
        lines.append(f"[{marker}] ({where})\n{' '.join(chunk.text.split())}")
        sources.append(SourceRef(marker=marker, paper_id=chunk.paper_id, title=chunk.title,
                                 chunk_id=chunk.chunk_id, section=chunk.section,
                                 preview=chunk.short(180)))
    return "\n\n".join(lines), sources

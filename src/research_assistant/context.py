"""Turn retrieved chunks into a numbered context block and citation list.

Keeping this in one place guarantees the ``[n]`` markers the summarizer is
asked to cite line up exactly with the bibliography rendered in the report.
"""

from __future__ import annotations

from typing import List, Tuple

from .schemas import RetrievedChunk, SourceRef


def number_sources(retrieved: List[RetrievedChunk]) -> Tuple[str, List[SourceRef]]:
    """Return a ``(context_block, sources)`` pair.

    ``context_block`` is the numbered text passed to the model; ``sources`` is
    the matching list of :class:`SourceRef` entries (one per marker).
    """
    lines: List[str] = []
    sources: List[SourceRef] = []
    for marker, item in enumerate(retrieved, start=1):
        chunk = item.chunk
        lines.append(f'[{marker}] (source: "{chunk.title}")\n{chunk.text}')
        sources.append(
            SourceRef(
                marker=marker,
                paper_id=chunk.paper_id,
                title=chunk.title,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                preview=chunk.short(180),
            )
        )
    return "\n\n".join(lines), sources

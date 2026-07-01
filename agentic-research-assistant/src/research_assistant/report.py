"""Render a completed research run as a Markdown report."""

from __future__ import annotations

from typing import List, Optional

from .schemas import Evaluation, PaperSummary, SourceRef


def _bullets(items: List[str], empty: str) -> str:
    if not items:
        return f"_{empty}_"
    return "\n".join(f"- {item}" for item in items)


def render_report(
    question: str,
    summary: PaperSummary,
    sources: List[SourceRef],
    evaluation: Optional[Evaluation] = None,
    queries: Optional[List[str]] = None,
) -> str:
    """Produce a self-contained Markdown report for the run."""
    queries = queries or []

    references = (
        "\n".join(
            f"{s.marker}. **{s.title}**"
            + (f" — `{s.source}`" if s.source else "")
            + f"\n   > {s.preview}"
            for s in sources
        )
        or "_No sources were cited._"
    )

    eval_block = ""
    if evaluation is not None:
        eval_block = (
            "\n## Evaluation\n\n"
            f"- **Sufficiency:** {evaluation.sufficiency}\n"
            f"- **Coverage score:** {evaluation.coverage_score:.2f}\n"
            f"- **Grounded:** {'yes' if evaluation.grounded else 'no'}\n"
            f"- **Reviewer note:** {evaluation.critique or '—'}\n"
        )
        if evaluation.missing_aspects:
            eval_block += "- **Still missing:** " + ", ".join(evaluation.missing_aspects) + "\n"

    trail = ""
    if queries:
        trail = "\n## Search trail\n\n" + "\n".join(
            f"{i}. `{q}`" for i, q in enumerate(queries, start=1)
        ) + "\n"

    return f"""# Research synthesis

**Question:** {question}

## Summary

{summary.summary}

## Methods

{_bullets(summary.methods, "No methods were extracted.")}

## Key findings

{_bullets(summary.key_findings, "No key findings were extracted.")}

## Research gaps

{_bullets(summary.research_gaps, "No research gaps were identified.")}
{eval_block}{trail}
## References

{references}
"""

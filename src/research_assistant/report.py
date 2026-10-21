"""A finished run as Markdown: the answer, the passages it cites, and how it got there."""

from __future__ import annotations

from .schemas import ResearchResult


def render_report(result: ResearchResult) -> str:
    a = result.answer
    answer = "The passages retrieved don't answer this." if a.unanswerable else a.answer
    if a.evidence and not a.unanswerable:
        answer += " " + "".join(f"[{m}]" for m in a.evidence)
    cited = "\n".join(f"{s.marker}. {s.title}" + (f", {s.section}" if s.section else "") + f"\n   > {s.preview}"
                      for s in result.cited) or "No passages cited."
    trail = "\n".join(f"{i}. `{q}`" for i, q in enumerate(result.queries, start=1))
    reviews = "\n".join(f"- round {i}: {'sufficient' if r.sufficient else 'needs more'}"
                        + (f" ({r.missing})" if r.missing else "") for i, r in enumerate(result.reviews, 1))
    t = result.trace
    cost = (f"{t.get('calls', 0)} model calls, {t.get('input_tokens', 0):,} tokens in, "
            f"{t.get('output_tokens', 0):,} out, ${t.get('cost_usd', 0):.4f}, {t.get('seconds', 0):.1f} s")
    return (f"# {result.question}\n\n{answer}\n\n## Cited passages\n\n{cited}\n\n"
            f"## How it got there\n\nStopped after {result.rounds} round(s): {result.status}. {cost}.\n\n"
            f"Searches:\n\n{trail}\n" + (f"\nReviews:\n\n{reviews}\n" if reviews else ""))

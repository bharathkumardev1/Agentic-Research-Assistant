# Contributing

Thanks for considering a contribution. This is a small solo project, so the
process is lightweight, but a few conventions keep it consistent.

## Setup

```bash
git clone https://github.com/bharathkumardev1/Agentic-Research-Assistant.git
cd Agentic-Research-Assistant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

Run the checks CI will run:

```bash
ruff check src tests
pytest
```

Both must pass. If you're adding a feature, add a test for it in `tests/`,
following the existing style (see `tests/test_chunking.py` for a simple
example, or `tests/test_pipeline_dry_run.py` for something that exercises
the full graph).

Tests that depend on `faiss-cpu` or `langgraph` should skip gracefully with
`pytest.importorskip(...)` when those aren't installed, matching the
existing tests, so a minimal environment can still run the fast suite.

## Style

- Keep functions small and typed. This codebase leans on Pydantic models at
  every boundary where LLM output enters the system; new agent output should
  follow the same pattern rather than passing raw dicts around.
- Heavy imports (`anthropic`, `faiss`, `langgraph`, `sentence_transformers`)
  are imported lazily inside functions, not at module level, so the package
  stays importable in minimal environments. Follow that pattern for any new
  optional dependency.
- Log through `research_assistant.logging_utils.get_logger(__name__)`
  rather than `print`, for anything inside `src/`. `print` is fine in
  `cli.py` for direct user-facing output.

## Reporting issues

Open a GitHub issue with what you ran, what you expected, and what happened.
If it's related to a live Claude run, include whether `--dry-run` reproduces
the same problem, that narrows down whether it's a pipeline bug or a
model-response issue.

## Commit messages

Short, present-tense, one logical change per commit. No strict format
enforced, but "Fix X" reads better than "fixed X" or "fixes for X".

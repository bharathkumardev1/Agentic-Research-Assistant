# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-07-01

### Added
- Structured logging (`research_assistant.logging_utils`), configurable via
  the `LOG_LEVEL` environment variable, with a per-run correlation id so
  concurrent runs don't interleave in shared logs.
- A hard per-run cap on live Claude API calls (`MAX_API_CALLS`, default 40)
  as a cost/runaway guard, independent of `MAX_ITERATIONS`.
- Token usage tracking on `ClaudeClient` (`call_count`, `input_tokens`,
  `output_tokens`), surfaced as a summary line after live CLI runs.
- `Dockerfile` and `.dockerignore` for containerized deployment.
- This changelog.

### Fixed
- CI badge URL now matches the repository's actual casing.

## [0.1.0] - 2026-06-30

### Added
- Initial release: three-agent LangGraph loop (retriever, summarizer,
  evaluator) over a FAISS-backed RAG pipeline.
- Citation-backed summaries with methods, key findings, and research gaps
  extracted into a structured, validated report.
- Offline `--dry-run` mode with deterministic hashing embeddings and a stub
  Claude client, so the full pipeline runs with no API key.
- CLI (`research-assistant`) with `ingest`, `research`, and `demo`
  subcommands.
- Test suite covering chunking, schemas, citation alignment, embeddings,
  the stub client, the FAISS store, and a full offline end-to-end run.
- GitHub Actions CI running the suite on Python 3.9, 3.11, and 3.12.

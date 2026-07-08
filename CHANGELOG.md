# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-07-08

### Added
- Per-caller rate limiting on `/research` (`RATE_LIMIT_PER_MINUTE`), in-memory
  and bucketed by API key name or IP (`research_assistant.rate_limit`).
- Enforced request timeout (`REQUEST_TIMEOUT_SECONDS`): a stuck run now
  returns `504` instead of holding the connection open indefinitely.
- Named/scoped API keys via `WEB_API_KEYS` (comma-separated `name:key`
  pairs), alongside the existing single `WEB_API_KEY`.
- `GET /metrics`: Prometheus request-count and latency metrics.
- The web service now loads a previously `ingest`-ed index from `INDEX_DIR`
  on startup if one exists (e.g. a mounted persistent volume), instead of
  always rebuilding the bundled demo index from the sample papers.
- `WEB_CONCURRENCY` support in the Dockerfile/`render.yaml` to run multiple
  uvicorn workers.
- Scheduled `keepalive` GitHub Actions workflow pinging `/health` to reduce
  free-tier idle-sleep cold starts.
- Frontend: a Vite + React demo UI (see `frontend/`) with an offline demo
  mode and a live mode against the deployed API.
- CORS support (`CORS_ORIGINS`) so the frontend can call the API from a
  browser.

### Changed
- `/research` no longer returns raw exception text on failure; errors are a
  generic message plus a short reference id, with the full detail logged
  server-side against that id.
- `requirements.txt` now pins exact dependency versions for a reproducible
  Docker build (`pyproject.toml` keeps loose ranges for library installs).

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

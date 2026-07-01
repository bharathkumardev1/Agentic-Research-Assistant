# Agentic Research Assistant
#
# Build:        docker build -t research-assistant .
# Run the API:  docker run --rm -p 8000:8000 research-assistant
#               then: curl http://localhost:8000/health
# Run the CLI:  docker run --rm research-assistant research-assistant demo
# Live API:     docker run --rm -p 8000:8000 --env-file .env research-assistant
#
# The image ships with the offline demo data pre-baked, so the web service
# answers questions immediately after `docker run`, no volumes or env vars
# required. Set ANTHROPIC_API_KEY (via --env-file .env) for real Claude
# answers; the service still starts fine without it, in dry-run mode.

FROM python:3.11-slim AS base

# faiss-cpu and numpy need a C toolchain on some platforms; keep the image
# lean by removing build tools after install in a single layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# Run as a non-root user.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    INDEX_DIR=/app/data/index \
    PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn research_assistant.webapp:app --host 0.0.0.0 --port ${PORT}"]

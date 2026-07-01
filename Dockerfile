# Agentic Research Assistant
#
# Build:  docker build -t research-assistant .
# Demo:   docker run --rm research-assistant demo
# Live:   docker run --rm --env-file .env research-assistant research "your question" --index-dir /data/index
#
# The image ships with the offline demo data pre-baked, so `docker run
# research-assistant demo` works immediately with no volumes or env vars.

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
    INDEX_DIR=/app/data/index

ENTRYPOINT ["research-assistant"]
CMD ["demo"]

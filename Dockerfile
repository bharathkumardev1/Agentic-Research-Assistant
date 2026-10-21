# The research assistant's API. Model weights and indexes are mounted at run time, not
# baked in.
#
#   docker build -t research-assistant .
#
# With nothing set, it answers questions about the three sample papers using hashing
# embeddings and the offline stub, which is what CI's smoke test runs:
#
#   docker run --rm -p 8000:8000 research-assistant
#
# With a key, MiniLM weights and an index built by `research-assistant ingest`:
#
#   docker run --rm -p 8000:8000 --env-file .env \
#     -v "$PWD/models:/app/models:ro" -v "$PWD/data/index:/app/data/index:ro" research-assistant
#
# GET /health says whether answers come from the model or the stub.
FROM python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src \
    OMP_NUM_THREADS=1

WORKDIR /app
COPY requirements.lock .
RUN pip install --require-hashes -r requirements.lock
COPY src ./src
COPY examples ./examples

RUN useradd --create-home --uid 1000 app && mkdir data && chown app data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "research_assistant.webapp:app", "--host", "0.0.0.0", "--port", "8000"]

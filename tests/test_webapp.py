"""Tests for the FastAPI web service."""

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402

from research_assistant import webapp  # noqa: E402
from research_assistant.webapp import app  # noqa: E402


@pytest.fixture()
def client():
    webapp._rate_limiter.reset()
    with TestClient(app) as c:
        yield c
    webapp._rate_limiter.reset()


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mode"] in {"dry_run", "live"}


def test_research_runs_the_pipeline(client):
    response = client.post("/research", json={"question": "What methods are used?"})
    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert "sources" in body
    assert body["iterations"] >= 1


def test_research_rejects_empty_question(client):
    response = client.post("/research", json={"question": ""})
    assert response.status_code == 422


def test_health_needs_no_api_key_even_when_configured(client, monkeypatch):
    from research_assistant.config import get_settings

    get_settings().web_api_key = "secret"
    try:
        response = client.get("/health")
        assert response.status_code == 200
    finally:
        get_settings().web_api_key = ""


def test_research_enforces_api_key_when_configured(client):
    from research_assistant.config import get_settings

    get_settings().web_api_key = "secret"
    try:
        denied = client.post("/research", json={"question": "test"})
        assert denied.status_code == 401

        allowed = client.post(
            "/research",
            json={"question": "test"},
            headers={"X-API-Key": "secret"},
        )
        assert allowed.status_code == 200
    finally:
        get_settings().web_api_key = ""


def test_research_accepts_named_keys_from_web_api_keys(client):
    from research_assistant.config import get_settings

    get_settings().web_api_keys = "alice:alice-key,bob:bob-key"
    try:
        denied = client.post(
            "/research", json={"question": "test"}, headers={"X-API-Key": "wrong"}
        )
        assert denied.status_code == 401

        allowed = client.post(
            "/research",
            json={"question": "test"},
            headers={"X-API-Key": "bob-key"},
        )
        assert allowed.status_code == 200
    finally:
        get_settings().web_api_keys = ""


def test_research_rate_limits_repeated_calls(client):
    from research_assistant.config import get_settings

    get_settings().rate_limit_per_minute = 2
    try:
        for _ in range(2):
            ok = client.post("/research", json={"question": "test"})
            assert ok.status_code == 200

        limited = client.post("/research", json={"question": "test"})
        assert limited.status_code == 429
        assert "Retry-After" in limited.headers
    finally:
        get_settings().rate_limit_per_minute = 20


def test_research_error_response_does_not_leak_internals(client, monkeypatch):
    pipeline = webapp._state["pipeline"]

    def boom(question):
        raise RuntimeError("super secret internal detail")

    monkeypatch.setattr(pipeline, "run", boom)
    response = client.post("/research", json={"question": "test"})
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "super secret internal detail" not in detail
    assert "Reference:" in detail


def test_research_times_out_returns_504(client, monkeypatch):
    import time

    from research_assistant.config import get_settings

    pipeline = webapp._state["pipeline"]
    get_settings().request_timeout_seconds = 0.01
    monkeypatch.setattr(pipeline, "run", lambda question: time.sleep(1))
    try:
        response = client.post("/research", json={"question": "test"})
        assert response.status_code == 504
        assert "Reference:" in response.json()["detail"]
    finally:
        get_settings().request_timeout_seconds = 120.0


def test_metrics_endpoint_exposes_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "research_requests_total" in response.text


class _FakeStore:
    def __len__(self):
        return 3


def test_load_default_pipeline_prefers_a_persisted_index(monkeypatch):
    calls = {"load_index": False, "build_index": False}

    monkeypatch.setattr(webapp.FaissVectorStore, "exists", staticmethod(lambda directory: True))
    monkeypatch.setattr(
        webapp,
        "load_index",
        lambda settings, dry_run=False: calls.__setitem__("load_index", True) or _FakeStore(),
    )
    monkeypatch.setattr(
        webapp,
        "build_index",
        lambda papers, settings, dry_run=True: calls.__setitem__("build_index", True) or _FakeStore(),
    )
    monkeypatch.setattr(webapp, "build_client", lambda settings, dry_run: object())
    monkeypatch.setattr(webapp, "build_pipeline", lambda store, client, settings: object())

    webapp._load_default_pipeline()

    assert calls == {"load_index": True, "build_index": False}


def test_load_default_pipeline_falls_back_to_demo_papers_when_no_index_exists(monkeypatch):
    calls = {"load_index": False, "build_index": False}

    monkeypatch.setattr(webapp.FaissVectorStore, "exists", staticmethod(lambda directory: False))
    monkeypatch.setattr(
        webapp,
        "load_index",
        lambda settings, dry_run=False: calls.__setitem__("load_index", True) or _FakeStore(),
    )
    monkeypatch.setattr(
        webapp,
        "build_index",
        lambda papers, settings, dry_run=True: calls.__setitem__("build_index", True) or _FakeStore(),
    )
    monkeypatch.setattr(webapp, "build_client", lambda settings, dry_run: object())
    monkeypatch.setattr(webapp, "build_pipeline", lambda store, client, settings: object())

    webapp._load_default_pipeline()

    assert calls == {"load_index": False, "build_index": True}

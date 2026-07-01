"""Tests for the FastAPI web service."""

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402

from research_assistant.webapp import app  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


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

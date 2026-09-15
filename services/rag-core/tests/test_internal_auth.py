from __future__ import annotations

import pytest

from app.config import get_settings


def test_ping_requires_secret(client):
    assert client.get("/internal/ping").status_code == 401


def test_ping_rejects_wrong_secret(client):
    r = client.get("/internal/ping", headers={"X-Internal-Secret": "nope"})
    assert r.status_code == 401


def test_ping_accepts_correct_secret(client):
    r = client.get("/internal/ping", headers={"X-Internal-Secret": "test-secret"})
    assert r.status_code == 200
    assert r.json()["authenticated"] is True


def test_ping_fails_closed_without_configured_secret(monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.delenv("RAG_CORE_SHARED_SECRET", raising=False)
    get_settings.cache_clear()
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        # no secret configured -> reject everything on the internal boundary
        assert c.get("/internal/ping", headers={"X-Internal-Secret": "anything"}).status_code == 503
    get_settings.cache_clear()

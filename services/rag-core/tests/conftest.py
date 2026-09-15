from __future__ import annotations

import pytest

from app.config import get_settings


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """TestClient with a known shared secret configured."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("RAG_CORE_SHARED_SECRET", "test-secret")
    get_settings.cache_clear()
    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    get_settings.cache_clear()

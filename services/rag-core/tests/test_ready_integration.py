from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

_RUN = os.getenv("RUN_INTEGRATION") == "1"


@pytest.mark.skipif(not _RUN, reason="set RUN_INTEGRATION=1 with Qdrant reachable")
def test_ready_reports_qdrant_reachable(client):
    r = client.get("/ready")
    assert r.status_code == 200
    qdrant = r.json()["qdrant"]
    assert qdrant["reachable"] is True
    # Stage 0 must NOT have created the collection.
    assert qdrant["target_collection_exists"] is False


@pytest.mark.skipif(not _RUN, reason="set RUN_INTEGRATION=1 with Qdrant reachable")
def test_no_ai_provider_env_required(client):
    # The service is up and serving without any AI provider key present.
    r = client.get("/health")
    assert r.status_code == 200

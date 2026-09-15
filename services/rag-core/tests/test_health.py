from __future__ import annotations


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "rag-core"


def test_ready_route_exists(client):
    # Without Qdrant reachable this returns 503; with it, 200. Either proves the
    # route is wired and never raises.
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    assert "qdrant" in r.json()

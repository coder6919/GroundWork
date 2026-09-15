"""Route-level tests for POST /internal/query/stream (Stage 9)."""

from __future__ import annotations

from app import main


def _parse_sse(body: str) -> list[tuple[str, str]]:
    events = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        event = next(line[len("event: ") :] for line in lines if line.startswith("event: "))
        data = next(line[len("data: ") :] for line in lines if line.startswith("data: "))
        events.append((event, data))
    return events


def _install_fakes(monkeypatch, *, stream_events):
    monkeypatch.setattr(main, "search", lambda *a, **k: [object(), object()])
    monkeypatch.setattr(main, "get_embedder", lambda: object())
    monkeypatch.setattr(main, "get_reranker", lambda: object())
    monkeypatch.setattr(main, "get_anthropic_client", lambda: object())
    monkeypatch.setattr(main, "generate_answer_stream", lambda *a, **k: iter(stream_events))


def test_streams_meta_token_and_done_events(client, monkeypatch):
    _install_fakes(
        monkeypatch,
        stream_events=[
            ("token", {"text": "Hello "}),
            ("token", {"text": "world."}),
            ("done", {"text": "Hello world.", "citations": [], "model": "claude-sonnet-5", "stop_reason": "end_turn"}),
        ],
    )

    r = client.post(
        "/internal/query/stream",
        headers={"X-Internal-Secret": "test-secret"},
        json={"query": "hi"},
    )

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    names = [e[0] for e in events]
    assert names == ["meta", "token", "token", "done"]
    assert '"retrieved_count": 2' in events[0][1] or "retrieved_count" in events[0][1]


def test_score_floor_short_circuit_yields_meta_and_done_only(client, monkeypatch):
    _install_fakes(
        monkeypatch,
        stream_events=[
            ("done", {"text": "I don't have a confident answer to that in the provided documents.", "citations": [], "model": "claude-sonnet-5", "stop_reason": "low_confidence"}),
        ],
    )

    r = client.post(
        "/internal/query/stream",
        headers={"X-Internal-Secret": "test-secret"},
        json={"query": "what is the capital of France?"},
    )

    events = _parse_sse(r.text)
    assert [e[0] for e in events] == ["meta", "done"]
    assert "low_confidence" in events[1][1]


def test_requires_the_shared_secret(client, monkeypatch):
    _install_fakes(monkeypatch, stream_events=[])
    r = client.post("/internal/query/stream", json={"query": "hi"})
    assert r.status_code == 401


def test_rejects_empty_query(client, monkeypatch):
    _install_fakes(monkeypatch, stream_events=[])
    r = client.post(
        "/internal/query/stream", headers={"X-Internal-Secret": "test-secret"}, json={"query": ""}
    )
    assert r.status_code == 422

"""Route-level tests for Stage 10's naive/improved toggle: `pipeline: "naive"`
on POST /internal/query and /internal/query/stream routes to app/naive's
naive_search + naive_generate instead of the hybrid+rerank+citations
pipeline. Monkeypatches naive_search/naive_generate directly - both are
already unit-tested in isolation (test_naive_pipeline.py,
test_naive_generation.py)."""

from __future__ import annotations

import anthropic

from app import main


def _install_naive_fakes(monkeypatch, *, chunks, answer_text="a naive answer"):
    search_calls = []
    generate_calls = []

    def fake_naive_search(query, **kwargs):
        search_calls.append(query)
        return chunks

    def fake_naive_generate(query, retrieved_chunks, **kwargs):
        generate_calls.append((query, retrieved_chunks))
        return answer_text

    monkeypatch.setattr(main, "naive_search", fake_naive_search)
    monkeypatch.setattr(main, "naive_generate", fake_naive_generate)
    monkeypatch.setattr(main, "get_embedder", lambda: object())
    monkeypatch.setattr(main, "get_anthropic_client", lambda: object())
    # improved-pipeline collaborators must never be touched when pipeline=naive
    monkeypatch.setattr(main, "search", lambda *a, **k: (_ for _ in ()).throw(AssertionError("hybrid search called")))
    monkeypatch.setattr(
        main, "generate_answer", lambda *a, **k: (_ for _ in ()).throw(AssertionError("improved generate called"))
    )

    return search_calls, generate_calls


def test_blocking_query_routes_to_naive_pipeline(client, monkeypatch):
    search_calls, generate_calls = _install_naive_fakes(monkeypatch, chunks=["chunk one", "chunk two"])

    r = client.post(
        "/internal/query",
        headers={"X-Internal-Secret": "test-secret"},
        json={"query": "What is the refund window?", "pipeline": "naive"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "a naive answer"
    assert body["citations"] == []
    assert body["stop_reason"] == "naive_pipeline"
    assert body["retrieved_count"] == 2
    assert search_calls == ["What is the refund window?"]
    assert generate_calls == [("What is the refund window?", ["chunk one", "chunk two"])]


def test_blocking_query_naive_with_no_hits_skips_generation(client, monkeypatch):
    _search_calls, generate_calls = _install_naive_fakes(monkeypatch, chunks=[])

    r = client.post(
        "/internal/query",
        headers={"X-Internal-Secret": "test-secret"},
        json={"query": "something absent from the corpus", "pipeline": "naive"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["stop_reason"] == "naive_pipeline"
    assert "don't have information" in body["answer"]
    assert generate_calls == []  # no Claude call when nothing retrieved


def test_default_pipeline_is_improved(client, monkeypatch):
    search_calls = []
    monkeypatch.setattr(main, "search", lambda query, **k: (search_calls.append(query), [])[1])
    monkeypatch.setattr(main, "get_embedder", lambda: object())
    monkeypatch.setattr(main, "get_reranker", lambda: object())
    monkeypatch.setattr(main, "get_anthropic_client", lambda: object())

    r = client.post(
        "/internal/query",
        headers={"X-Internal-Secret": "test-secret"},
        json={"query": "What's the refund policy?"},
    )

    assert r.status_code == 200
    assert search_calls == ["What's the refund policy?"]  # hybrid search ran, not naive_search


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


def test_streaming_query_routes_to_naive_pipeline(client, monkeypatch):
    _install_naive_fakes(monkeypatch, chunks=["chunk one"], answer_text="streamed naive answer")

    r = client.post(
        "/internal/query/stream",
        headers={"X-Internal-Secret": "test-secret"},
        json={"query": "hi", "pipeline": "naive"},
    )

    assert r.status_code == 200
    events = _parse_sse(r.text)
    names = [e[0] for e in events]
    assert names == ["meta", "token", "done"]
    assert "streamed naive answer" in events[1][1]
    assert "naive_pipeline" in events[2][1]


def test_streaming_naive_query_ends_gracefully_on_an_anthropic_failure(monkeypatch, client):
    """An Anthropic API failure (e.g. an exhausted credit balance, observed
    live) must end the SSE stream with a "done" event, not an uncaught
    exception - the route has already sent a 200 + streaming headers by the
    time naive_generate runs, so raising here aborts the connection
    mid-stream instead of becoming a clean HTTP error (this is exactly what
    crashed the Node api service's proxy live)."""
    monkeypatch.setattr(main, "naive_search", lambda *a, **k: ["chunk one"])
    monkeypatch.setattr(main, "get_embedder", lambda: object())
    monkeypatch.setattr(main, "get_anthropic_client", lambda: object())

    def failing_naive_generate(*a, **k):
        raise anthropic.APIConnectionError(message="credit balance too low", request=None)

    monkeypatch.setattr(main, "naive_generate", failing_naive_generate)

    r = client.post(
        "/internal/query/stream",
        headers={"X-Internal-Secret": "test-secret"},
        json={"query": "hi", "pipeline": "naive"},
    )

    assert r.status_code == 200
    events = _parse_sse(r.text)
    names = [e[0] for e in events]
    assert names == ["meta", "done"]  # no token event - generation never produced text
    assert "generation_error" in events[1][1]

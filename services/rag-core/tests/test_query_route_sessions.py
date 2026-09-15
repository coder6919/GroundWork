"""Route-level test for Stage 7's session wiring in POST /internal/query.

Monkeypatches the module-level collaborators `internal_query` calls (search,
generate_answer, rewrite_query, and the lazy singleton factories) so this
runs with no real network - the collaborators themselves are already unit-
tested in isolation (test_retrieval.py, test_generation.py,
test_query_rewrite.py). This test exists to prove the wiring: does a second
turn in the same session actually see the first turn's history and get its
query rewritten, and does the turn get persisted?
"""

from __future__ import annotations

from app import main
from app.generation.claude import Answer


def _install_fakes(monkeypatch, *, rewritten: str | None = None):
    search_calls = []
    rewrite_calls = []

    def fake_search(query, **kwargs):
        search_calls.append(query)
        return []

    def fake_generate_answer(query, chunks, **kwargs):
        return Answer(text=f"answer to: {query}", citations=[], model="claude-sonnet-5", stop_reason="end_turn")

    def fake_rewrite_query(history, latest_query, **kwargs):
        rewrite_calls.append((list(history), latest_query))
        return rewritten if rewritten is not None else latest_query

    monkeypatch.setattr(main, "search", fake_search)
    monkeypatch.setattr(main, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(main, "rewrite_query", fake_rewrite_query)
    monkeypatch.setattr(main, "get_embedder", lambda: object())
    monkeypatch.setattr(main, "get_reranker", lambda: object())
    monkeypatch.setattr(main, "get_anthropic_client", lambda: object())

    return search_calls, rewrite_calls


def test_first_turn_in_a_session_has_no_history_to_rewrite_against(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "get_session_store", lambda: main.SQLiteSessionStore(str(tmp_path / "s.db")))
    search_calls, rewrite_calls = _install_fakes(monkeypatch)

    r = client.post(
        "/internal/query",
        headers={"X-Internal-Secret": "test-secret"},
        json={"query": "What's the US shipping cost?", "session_id": "sess-1"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "sess-1"
    assert body["rewritten_query"] is None
    assert rewrite_calls == [([], "What's the US shipping cost?")]
    assert search_calls == ["What's the US shipping cost?"]


def test_second_turn_sees_the_first_turns_history_and_uses_the_rewritten_query(client, monkeypatch, tmp_path):
    store = main.SQLiteSessionStore(str(tmp_path / "s.db"))
    monkeypatch.setattr(main, "get_session_store", lambda: store)
    search_calls, rewrite_calls = _install_fakes(monkeypatch, rewritten="What is the EU shipping cost?")

    client.post(
        "/internal/query",
        headers={"X-Internal-Secret": "test-secret"},
        json={"query": "What's the US shipping cost?", "session_id": "sess-1"},
    )
    r = client.post(
        "/internal/query",
        headers={"X-Internal-Secret": "test-secret"},
        json={"query": "what about the EU?", "session_id": "sess-1"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["rewritten_query"] == "What is the EU shipping cost?"
    # second call's history includes the first turn
    history, latest = rewrite_calls[1]
    assert len(history) == 1
    assert history[0].user_query == "What's the US shipping cost?"
    assert latest == "what about the EU?"
    # search ran on the rewritten query, not the raw follow-up
    assert search_calls[1] == "What is the EU shipping cost?"

    persisted = store.get_turns("sess-1")
    assert len(persisted) == 2
    assert persisted[1].user_query == "what about the EU?"
    assert persisted[1].rewritten_query == "What is the EU shipping cost?"


def test_omitting_session_id_keeps_the_original_stateless_behavior(client, monkeypatch):
    _search_calls, rewrite_calls = _install_fakes(monkeypatch)

    r = client.post(
        "/internal/query",
        headers={"X-Internal-Secret": "test-secret"},
        json={"query": "What's the refund policy?"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] is None
    assert body["rewritten_query"] is None
    assert rewrite_calls == []  # never called at all without a session_id

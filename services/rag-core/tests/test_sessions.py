from __future__ import annotations

from pathlib import Path

import pytest

from app.db.models import Turn
from app.db.sessions import SQLiteSessionStore


@pytest.fixture
def store(tmp_path: Path) -> SQLiteSessionStore:
    return SQLiteSessionStore(str(tmp_path / "sub" / "sessions.db"))


def _turn(**overrides) -> Turn:
    defaults = {
        "session_id": "sess-1",
        "user_query": "What's the refund policy?",
        "rewritten_query": "What's the refund policy?",
        "answer_text": "Refunds take 30 days.",
    }
    defaults.update(overrides)
    return Turn(**defaults)


def test_creates_parent_directory_for_db_file(tmp_path: Path):
    db_path = tmp_path / "nested" / "dir" / "sessions.db"
    SQLiteSessionStore(str(db_path))
    assert db_path.exists()


def test_unknown_session_returns_empty_list(store: SQLiteSessionStore):
    assert store.get_turns("nope") == []


def test_add_turn_then_get_turns_roundtrips(store: SQLiteSessionStore):
    store.add_turn(_turn())
    turns = store.get_turns("sess-1")
    assert len(turns) == 1
    assert turns[0].user_query == "What's the refund policy?"
    assert turns[0].answer_text == "Refunds take 30 days."


def test_get_turns_returns_oldest_first(store: SQLiteSessionStore):
    store.add_turn(_turn(user_query="first"))
    store.add_turn(_turn(user_query="second"))
    store.add_turn(_turn(user_query="third"))

    queries = [t.user_query for t in store.get_turns("sess-1")]
    assert queries == ["first", "second", "third"]


def test_turns_are_isolated_per_session(store: SQLiteSessionStore):
    store.add_turn(_turn(session_id="sess-a", user_query="a"))
    store.add_turn(_turn(session_id="sess-b", user_query="b"))

    assert [t.user_query for t in store.get_turns("sess-a")] == ["a"]
    assert [t.user_query for t in store.get_turns("sess-b")] == ["b"]


def test_rewritten_query_can_differ_from_user_query(store: SQLiteSessionStore):
    store.add_turn(_turn(user_query="what about the EU?", rewritten_query="What is the EU shipping cost?"))
    turn = store.get_turns("sess-1")[0]
    assert turn.user_query == "what about the EU?"
    assert turn.rewritten_query == "What is the EU shipping cost?"

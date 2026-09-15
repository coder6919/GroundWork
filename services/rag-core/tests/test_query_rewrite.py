from __future__ import annotations

from types import SimpleNamespace

from app.db.models import Turn
from app.generation.query_rewrite import rewrite_query


class _FakeMessages:
    def __init__(self, response=None, exc: Exception | None = None):
        self._response = response
        self._exc = exc
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._exc:
            raise self._exc
        return self._response


class _FakeAnthropic:
    def __init__(self, response=None, exc: Exception | None = None):
        self.messages = _FakeMessages(response, exc)


def _text_response(text: str):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _turn(**overrides) -> Turn:
    defaults = {
        "session_id": "sess-1",
        "user_query": "What's the US shipping cost?",
        "rewritten_query": "What's the US shipping cost?",
        "answer_text": "Standard is $5.00, Express is $15.00.",
    }
    defaults.update(overrides)
    return Turn(**defaults)


def _fake_settings():
    from app.config import Settings

    return Settings(anthropic_api_key="test-key", query_rewrite_model="claude-haiku-4-5")


def test_no_history_returns_raw_query_without_calling_the_model():
    client = _FakeAnthropic()
    result = rewrite_query([], "what about the EU?", settings=_fake_settings(), client=client)
    assert result == "what about the EU?"
    assert client.messages.last_kwargs is None


def test_with_history_calls_the_model_and_returns_rewritten_text():
    client = _FakeAnthropic(_text_response("What is the EU shipping cost?"))
    result = rewrite_query([_turn()], "what about the EU?", settings=_fake_settings(), client=client)
    assert result == "What is the EU shipping cost?"
    assert client.messages.last_kwargs["model"] == "claude-haiku-4-5"


def test_history_and_latest_query_are_included_in_the_prompt():
    client = _FakeAnthropic(_text_response("rewritten"))
    rewrite_query([_turn()], "what about the EU?", settings=_fake_settings(), client=client)

    sent_content = client.messages.last_kwargs["messages"][0]["content"]
    assert "What's the US shipping cost?" in sent_content
    assert "Standard is $5.00" in sent_content
    assert "what about the EU?" in sent_content


def test_empty_model_response_falls_back_to_raw_query():
    client = _FakeAnthropic(_text_response("   "))
    result = rewrite_query([_turn()], "what about the EU?", settings=_fake_settings(), client=client)
    assert result == "what about the EU?"


def test_model_call_failure_falls_back_to_raw_query_not_raising():
    client = _FakeAnthropic(exc=RuntimeError("rate limited"))
    result = rewrite_query([_turn()], "what about the EU?", settings=_fake_settings(), client=client)
    assert result == "what about the EU?"

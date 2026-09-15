from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.naive.generation import naive_generate


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeAnthropic:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def _fake_settings() -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        generation_model="claude-sonnet-5",
        generation_max_tokens=1500,
        generation_effort="low",
    )


def _response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        stop_reason="end_turn",
    )


def test_naive_generate_sends_a_single_flat_text_prompt_not_document_blocks():
    client = _FakeAnthropic(_response("some answer"))

    naive_generate(
        "What is the refund policy?",
        ["chunk one text", "chunk two text"],
        settings=_fake_settings(),
        client=client,
    )

    sent = client.messages.last_kwargs
    content = sent["messages"][0]["content"]
    assert isinstance(content, str)  # plain string, not a list of document blocks
    assert "chunk one text" in content
    assert "chunk two text" in content
    assert "What is the refund policy?" in content


def test_naive_generate_has_no_system_prompt_or_citations_config():
    client = _FakeAnthropic(_response("some answer"))

    naive_generate("q", ["ctx"], settings=_fake_settings(), client=client)

    sent = client.messages.last_kwargs
    assert "system" not in sent  # no context-only / refusal instruction, unlike app/generation
    assert "\"citations\"" not in str(sent)


def test_naive_generate_still_applies_cost_controls():
    client = _FakeAnthropic(_response("some answer"))

    naive_generate("q", ["ctx"], settings=_fake_settings(), client=client)

    sent = client.messages.last_kwargs
    assert sent["thinking"] == {"type": "disabled"}
    assert sent["output_config"] == {"effort": "low"}
    assert sent["max_tokens"] == 1500


def test_naive_generate_returns_plain_string_not_a_structured_answer():
    client = _FakeAnthropic(_response("The refund takes 30 days."))

    result = naive_generate("q", ["ctx"], settings=_fake_settings(), client=client)

    assert result == "The refund takes 30 days."
    assert isinstance(result, str)

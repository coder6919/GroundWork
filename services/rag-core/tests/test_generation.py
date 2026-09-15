from __future__ import annotations

from types import SimpleNamespace

import anthropic

from app.generation.claude import generate_answer
from app.retrieval.search import RetrievedChunk


def _chunk(**overrides) -> RetrievedChunk:
    defaults = {
        "chunk_id": "chunk-1",
        "doc_id": "doc-1",
        "source_filename": "refund-policy.md",
        "heading_path": "Refund Policy > International Orders",
        "chunk_text": "International orders are refunded within 30 days.",
        "content_type": "prose",
        "score": 0.9,
    }
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


def test_no_chunks_returns_refusal_without_calling_claude():
    answer = generate_answer("What is the refund policy?", [], settings=_fake_settings())
    assert "don't have information" in answer.text
    assert answer.citations == []
    assert answer.stop_reason == "no_context"


class _FakeMessages:
    def __init__(self, response, *, stream_chunks=None, stream_final=None):
        self._response = response
        self._stream_chunks = stream_chunks
        self._stream_final = stream_final
        self.last_kwargs = None
        self.stream_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response

    def stream(self, **kwargs):
        self.stream_kwargs = kwargs
        return _FakeStream(self._stream_chunks or [], self._stream_final)


class _FakeStream:
    def __init__(self, text_chunks, final_message):
        self.text_stream = text_chunks
        self._final_message = final_message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final_message


class _FakeAnthropic:
    def __init__(self, response=None, *, stream_chunks=None, stream_final=None):
        self.messages = _FakeMessages(response, stream_chunks=stream_chunks, stream_final=stream_final)


def _text_block(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations)


def _citation(document_index, cited_text, ctype="char_location"):
    return SimpleNamespace(type=ctype, document_index=document_index, cited_text=cited_text)


def _fake_settings():
    from app.config import Settings

    return Settings(
        anthropic_api_key="test-key",
        generation_model="claude-sonnet-5",
        generation_max_tokens=1500,
        generation_effort="low",
    )


def test_generate_answer_maps_citations_back_to_source_chunks():
    chunk = _chunk()
    response = SimpleNamespace(
        content=[
            _text_block(
                "International orders are refunded within 30 days.",
                citations=[_citation(0, "International orders are refunded within 30 days.")],
            )
        ],
        usage=SimpleNamespace(input_tokens=42, output_tokens=8),
        stop_reason="end_turn",
    )
    client = _FakeAnthropic(response)

    answer = generate_answer(
        "How long do international refunds take?",
        [chunk],
        settings=_fake_settings(),
        client=client,
    )

    assert answer.text == "International orders are refunded within 30 days."
    assert len(answer.citations) == 1
    assert answer.citations[0].chunk_id == "chunk-1"
    assert answer.citations[0].doc_id == "doc-1"
    assert answer.citations[0].source_filename == "refund-policy.md"
    assert answer.citations[0].cited_text == "International orders are refunded within 30 days."
    assert answer.stop_reason == "end_turn"


def test_generate_answer_ignores_out_of_range_document_index():
    chunk = _chunk()
    response = SimpleNamespace(
        content=[_text_block("Some answer.", citations=[_citation(7, "bogus")])],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        stop_reason="end_turn",
    )
    client = _FakeAnthropic(response)

    answer = generate_answer("q", [chunk], settings=_fake_settings(), client=client)
    assert answer.citations == []  # out-of-range index dropped, not crashed


def test_generate_answer_builds_one_document_block_per_chunk():
    chunks = [_chunk(chunk_id="a", chunk_text="Text A"), _chunk(chunk_id="b", chunk_text="Text B")]
    response = SimpleNamespace(
        content=[_text_block("answer")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    client = _FakeAnthropic(response)

    generate_answer("q", chunks, settings=_fake_settings(), client=client)

    sent = client.messages.last_kwargs
    doc_blocks = [b for b in sent["messages"][0]["content"] if b["type"] == "document"]
    assert len(doc_blocks) == 2
    assert doc_blocks[0]["source"]["data"] == "Text A"
    assert doc_blocks[1]["source"]["data"] == "Text B"
    assert all(b["citations"]["enabled"] is True for b in doc_blocks)
    # cost controls actually sent on the wire
    assert sent["thinking"] == {"type": "disabled"}
    assert sent["output_config"] == {"effort": "low"}
    assert sent["max_tokens"] == 1500


def test_generate_answer_requires_api_key_when_no_client_given():
    import pytest

    from app.config import Settings

    settings = Settings(anthropic_api_key="")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        generate_answer("q", [_chunk()], settings=settings)


# ---- Stage 6: score floor, clarification ----


def _settings_with_floor(floor: float):
    from app.config import Settings

    return Settings(
        anthropic_api_key="test-key",
        generation_model="claude-sonnet-5",
        generation_max_tokens=1500,
        generation_effort="low",
        retrieval_score_floor=floor,
    )


def test_below_score_floor_skips_the_claude_call():
    chunk = _chunk(rerank_score=0.1)
    client = _FakeAnthropic(response=None)  # would blow up if ever called

    answer = generate_answer("q", [chunk], settings=_settings_with_floor(0.35), client=client)

    assert answer.stop_reason == "low_confidence"
    assert answer.citations == []
    assert client.messages.last_kwargs is None


def test_at_or_above_score_floor_calls_claude_normally():
    chunk = _chunk(rerank_score=0.35)
    response = SimpleNamespace(
        content=[_text_block("answer")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    client = _FakeAnthropic(response)

    answer = generate_answer("q", [chunk], settings=_settings_with_floor(0.35), client=client)

    assert answer.stop_reason == "end_turn"
    assert client.messages.last_kwargs is not None


def test_missing_rerank_score_skips_the_floor_check_entirely():
    # No reranker was used (e.g. the naive-comparison script) - rerank_score
    # is None, so the raw fused score (however small) must not be gated on.
    chunk = _chunk(score=0.001, rerank_score=None)
    response = SimpleNamespace(
        content=[_text_block("answer")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    client = _FakeAnthropic(response)

    answer = generate_answer("q", [chunk], settings=_settings_with_floor(0.35), client=client)

    assert answer.stop_reason == "end_turn"
    assert client.messages.last_kwargs is not None


def test_clarification_marker_is_stripped_and_sets_stop_reason():
    from app.generation.prompts import CLARIFICATION_MARKER

    chunk = _chunk(rerank_score=0.9)
    response = SimpleNamespace(
        content=[_text_block(f"{CLARIFICATION_MARKER} Did you mean the US or EU policy?")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    client = _FakeAnthropic(response)

    answer = generate_answer("what's the policy?", [chunk], settings=_settings_with_floor(0.35), client=client)

    assert answer.stop_reason == "clarification_needed"
    assert answer.text == "Did you mean the US or EU policy?"


def test_answer_without_the_marker_is_not_treated_as_a_clarification():
    chunk = _chunk(rerank_score=0.9)
    response = SimpleNamespace(
        content=[_text_block("A normal answer.")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    client = _FakeAnthropic(response)

    answer = generate_answer("q", [chunk], settings=_settings_with_floor(0.35), client=client)

    assert answer.stop_reason == "end_turn"
    assert answer.text == "A normal answer."


# ---- Stage 9: generate_answer_stream ----


def test_stream_no_chunks_yields_single_done_event_without_calling_claude():
    from app.generation.claude import generate_answer_stream

    client = _FakeAnthropic()
    events = list(generate_answer_stream("q", [], settings=_fake_settings(), client=client))

    assert len(events) == 1
    name, payload = events[0]
    assert name == "done"
    assert payload["stop_reason"] == "no_context"
    assert client.messages.stream_kwargs is None


def test_stream_below_score_floor_yields_single_done_event_without_calling_claude():
    from app.generation.claude import generate_answer_stream

    chunk = _chunk(rerank_score=0.1)
    client = _FakeAnthropic()

    events = list(
        generate_answer_stream("q", [chunk], settings=_settings_with_floor(0.35), client=client)
    )

    assert len(events) == 1
    assert events[0] == ("done", events[0][1])
    assert events[0][1]["stop_reason"] == "low_confidence"
    assert client.messages.stream_kwargs is None


def test_stream_yields_token_events_then_a_done_event_with_citations():
    from app.generation.claude import generate_answer_stream

    chunk = _chunk(rerank_score=0.9)
    final_message = SimpleNamespace(
        content=[
            _text_block(
                "International orders are refunded within 30 days.",
                citations=[_citation(0, "International orders are refunded within 30 days.")],
            )
        ],
        usage=SimpleNamespace(input_tokens=42, output_tokens=8),
        stop_reason="end_turn",
    )
    client = _FakeAnthropic(stream_chunks=["International ", "orders are refunded within 30 days."], stream_final=final_message)

    events = list(
        generate_answer_stream("q", [chunk], settings=_settings_with_floor(0.35), client=client)
    )

    token_events = [e for e in events if e[0] == "token"]
    done_events = [e for e in events if e[0] == "done"]
    assert [e[1]["text"] for e in token_events] == ["International ", "orders are refunded within 30 days."]
    assert len(done_events) == 1
    done = done_events[0][1]
    assert done["text"] == "International orders are refunded within 30 days."
    assert done["stop_reason"] == "end_turn"
    assert len(done["citations"]) == 1
    assert done["citations"][0]["chunk_id"] == "chunk-1"


def test_stream_detects_clarification_marker_in_the_final_message():
    from app.generation.claude import generate_answer_stream
    from app.generation.prompts import CLARIFICATION_MARKER

    chunk = _chunk(rerank_score=0.9)
    final_message = SimpleNamespace(
        content=[_text_block(f"{CLARIFICATION_MARKER} Did you mean the US or EU policy?")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    client = _FakeAnthropic(stream_chunks=[], stream_final=final_message)

    events = list(
        generate_answer_stream("q", [chunk], settings=_settings_with_floor(0.35), client=client)
    )

    done = dict(events)["done"]
    assert done["stop_reason"] == "clarification_needed"
    assert done["text"] == "Did you mean the US or EU policy?"


# ---- generate_answer_stream: Anthropic failures must not raise into the SSE
# route (the route has already sent a 200 + streaming headers by the time
# this generator runs - an uncaught exception here aborts the connection
# mid-stream instead of becoming a clean HTTP error, which is what crashed
# the Node api service's proxy live - see app/generation/claude.py) ----


class _FailingOnOpenMessages:
    """Simulates client.messages.stream(...) itself raising, e.g. a 400 from
    an exhausted credit balance - the failure mode observed live."""

    def __init__(self, exc):
        self._exc = exc
        self.stream_kwargs = None

    def stream(self, **kwargs):
        self.stream_kwargs = kwargs
        raise self._exc


class _FailingMidStream:
    """Simulates the connection dying partway through iterating text_stream."""

    def __init__(self, chunks_before_failure, exc):
        self.text_stream = self._iter(chunks_before_failure, exc)

    @staticmethod
    def _iter(chunks, exc):
        yield from chunks
        raise exc

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def get_final_message(self):
        raise AssertionError("should never be reached - the stream failed first")


def test_stream_yields_a_done_event_when_anthropic_fails_to_open_the_stream():
    from app.generation.claude import generate_answer_stream

    chunk = _chunk(rerank_score=0.9)
    client = SimpleNamespace(
        messages=_FailingOnOpenMessages(anthropic.APIConnectionError(message="credit balance too low", request=None))
    )

    events = list(generate_answer_stream("q", [chunk], settings=_settings_with_floor(0.35), client=client))

    assert len(events) == 1
    name, payload = events[0]
    assert name == "done"
    assert payload["stop_reason"] == "generation_error"
    assert payload["citations"] == []


def test_stream_yields_a_done_event_when_anthropic_fails_mid_stream():
    from app.generation.claude import generate_answer_stream

    chunk = _chunk(rerank_score=0.9)

    class _Messages:
        def stream(self, **kwargs):
            return _FailingMidStream(["Partial ", "answer"], anthropic.APIConnectionError(message="dropped", request=None))

    client = SimpleNamespace(messages=_Messages())

    events = list(generate_answer_stream("q", [chunk], settings=_settings_with_floor(0.35), client=client))

    token_events = [e for e in events if e[0] == "token"]
    done_events = [e for e in events if e[0] == "done"]
    # tokens that streamed before the failure are preserved, not swallowed
    assert [e[1]["text"] for e in token_events] == ["Partial ", "answer"]
    assert len(done_events) == 1
    assert done_events[0][1]["stop_reason"] == "generation_error"

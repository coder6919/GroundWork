"""Claude generation with native citations (Stage 2: happy-path grounded QA).

Each retrieved chunk becomes its own `document` content block with citations
enabled, so Claude's response carries structured citations mapped straight back
to chunk_id/doc_id/heading_path - no manual JSON schema needed (citations and
output_config.format are mutually exclusive on the API, and citations already
gives us structured data).

Cost-minimal by design: thinking disabled and effort "low" - this is grounded
QA, not multi-step reasoning, so neither buys quality here.

Stage 6 adds a hard score-floor gate (`retrieval_score_floor`, checked against
the top chunk's Cohere `rerank_score` - a calibrated 0-1 relevance score, unlike
the raw fused RRF score from Stage 5's hybrid retrieval, which isn't on a
comparable scale). Skips the Claude call below the floor, same cost-saving
pattern as the pre-existing zero-chunks case. When no reranker was used
(`rerank_score` is None on every chunk - e.g. the naive-comparison script),
there's no calibrated confidence signal, so the floor check is skipped and
generation proceeds as before Stage 6.

Stage 9 adds `generate_answer_stream()` for the public API's SSE endpoint,
sharing every rule above with `generate_answer()` (score floor, citation
extraction, clarification-marker handling) via `_extract_answer()`. Citations
are not available incrementally while streaming - the Anthropic streaming API
has no documented per-token citations delta - so a streamed answer's citations
arrive in one final event once `stream.get_final_message()` returns, using the
exact same extraction path as the non-streaming call.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import anthropic

from ..config import Settings
from ..logging import get_logger, log_external_call
from ..retrieval.search import RetrievedChunk
from .prompts import CLARIFICATION_MARKER, SYSTEM_PROMPT

log = get_logger("generation.claude")


@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    source_filename: str
    heading_path: str
    cited_text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    model: str
    stop_reason: str


def make_anthropic_client(api_key: str) -> anthropic.Anthropic:
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required to use generation")
    return anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=30.0)


def _document_blocks(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "type": "document",
            "title": chunk.source_filename or chunk.doc_id,
            "source": {"type": "text", "media_type": "text/plain", "data": chunk.chunk_text},
            "citations": {"enabled": True},
        }
        for chunk in chunks
    ]


def _below_score_floor(chunks: list[RetrievedChunk], floor: float) -> bool:
    """True only when the best chunk has a calibrated Cohere `rerank_score`
    below the floor. A raw fused RRF score (rerank_score is None) isn't
    comparable to the configured floor, so it's never gated on."""
    top_rerank_score = chunks[0].rerank_score
    return top_rerank_score is not None and top_rerank_score < floor


def _no_context_answer(settings: Settings) -> Answer:
    return Answer(
        text="I don't have information on that in the provided documents.",
        citations=[],
        model=settings.generation_model,
        stop_reason="no_context",
    )


def _low_confidence_answer(settings: Settings) -> Answer:
    return Answer(
        text="I don't have a confident answer to that in the provided documents.",
        citations=[],
        model=settings.generation_model,
        stop_reason="low_confidence",
    )


def _generation_error_answer(settings: Settings) -> Answer:
    return Answer(
        text="Something went wrong generating an answer. Please try again in a moment.",
        citations=[],
        model=settings.generation_model,
        stop_reason="generation_error",
    )


def _request_kwargs(query: str, chunks: list[RetrievedChunk], settings: Settings) -> dict:
    return {
        "model": settings.generation_model,
        "max_tokens": settings.generation_max_tokens,
        "system": SYSTEM_PROMPT,
        "thinking": {"type": "disabled"},
        "output_config": {"effort": settings.generation_effort},
        "messages": [
            {"role": "user", "content": [*_document_blocks(chunks), {"type": "text", "text": query}]}
        ],
    }


def _extract_answer(response, chunks: list[RetrievedChunk], settings: Settings) -> Answer:
    """Shared by generate_answer() and generate_answer_stream(): turns a
    completed (streamed or not) Message into an Answer, applying the
    clarification-marker rule identically either way."""
    text_parts: list[str] = []
    citations: list[Citation] = []
    for block in response.content:
        if block.type != "text":
            continue
        text_parts.append(block.text)
        for citation in block.citations or []:
            if citation.type != "char_location":
                continue
            idx = citation.document_index
            if not (0 <= idx < len(chunks)):
                continue
            source = chunks[idx]
            citations.append(
                Citation(
                    chunk_id=source.chunk_id,
                    doc_id=source.doc_id,
                    source_filename=source.source_filename,
                    heading_path=source.heading_path,
                    cited_text=citation.cited_text,
                )
            )

    text = "".join(text_parts)
    stop_reason = response.stop_reason or "end_turn"
    if text.startswith(CLARIFICATION_MARKER):
        # Prompt-driven, not hard-enforced: Claude decided the question is
        # ambiguous against the retrieved documents and asked for
        # clarification instead of guessing (see prompts.py rule 6).
        text = text[len(CLARIFICATION_MARKER) :].strip()
        stop_reason = "clarification_needed"

    return Answer(text=text, citations=citations, model=settings.generation_model, stop_reason=stop_reason)


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    settings: Settings,
    client: anthropic.Anthropic | None = None,
) -> Answer:
    if not chunks:
        return _no_context_answer(settings)
    if _below_score_floor(chunks, settings.retrieval_score_floor):
        return _low_confidence_answer(settings)

    client = client or make_anthropic_client(settings.anthropic_api_key)
    response = client.messages.create(**_request_kwargs(query, chunks, settings))

    log_external_call(
        "anthropic",
        "generate",
        model=settings.generation_model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        stop_reason=response.stop_reason,
    )

    return _extract_answer(response, chunks, settings)


def generate_answer_stream(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    settings: Settings,
    client: anthropic.Anthropic | None = None,
) -> Iterator[tuple[str, dict]]:
    """Same score-floor/no-context/citation/clarification rules as
    `generate_answer()`, yielding `(event_name, payload)` pairs for an SSE
    route to serialize: `("token", {"text": ...})` for each piece of text as
    it streams in, then exactly one `("done", {...full Answer fields...})`.
    The zero-chunks and below-floor cases still yield a single "done" event
    with no Claude call at all - same cost-saving skip as the blocking path.
    """
    if not chunks:
        yield "done", _answer_dict(_no_context_answer(settings))
        return
    if _below_score_floor(chunks, settings.retrieval_score_floor):
        yield "done", _answer_dict(_low_confidence_answer(settings))
        return

    client = client or make_anthropic_client(settings.anthropic_api_key)

    # The SSE route has already sent a 200 + streaming headers by the time
    # this generator runs (see internal_query_stream in main.py) - an
    # uncaught exception here doesn't become a clean HTTP error response, it
    # aborts the connection mid-stream at the ASGI level. That abrupt close
    # is what crashed the Node api service's proxy (an unhandled 'error'
    # event on the piped stream - see services/api/src/routes/query.ts) the
    # first time a request landed here with the Anthropic account out of
    # credit. Any Anthropic API failure - auth, rate limit, bad request,
    # connection drop - must end the SSE stream with a normal "done" event
    # instead of propagating.
    try:
        with client.messages.stream(**_request_kwargs(query, chunks, settings)) as stream:
            for text in stream.text_stream:
                yield "token", {"text": text}
            response = stream.get_final_message()
    except anthropic.APIError as exc:
        log.error("generation_stream_failed", error=str(exc))
        yield "done", _answer_dict(_generation_error_answer(settings))
        return

    log_external_call(
        "anthropic",
        "generate_stream",
        model=settings.generation_model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        stop_reason=response.stop_reason,
    )

    answer = _extract_answer(response, chunks, settings)
    yield "done", _answer_dict(answer)


def _answer_dict(answer: Answer) -> dict:
    return {
        "text": answer.text,
        "citations": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source_filename": c.source_filename,
                "heading_path": c.heading_path,
                "cited_text": c.cited_text,
            }
            for c in answer.citations
        ],
        "model": answer.model,
        "stop_reason": answer.stop_reason,
    }

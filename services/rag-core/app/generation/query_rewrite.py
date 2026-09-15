"""Conversation-aware query rewriting (Stage 7), using a small/cheap model
(`claude-haiku-4-5`) to turn a multi-turn follow-up into a standalone
question the existing single-turn retrieve-then-generate pipeline can search
and answer directly. Only the query text carries conversation context
forward - `search()` and `generate_answer()` remain stateless per call
(no growing transcript replayed into the generation model), keeping cost
bounded and the architecture the same classic RAG pipeline as every earlier
stage.

Skipped entirely on a session's first turn - there is no history to resolve
against, so the call would be pure cost with no effect. Falls back to the
raw query (logged as a warning, not raised) if the rewrite call itself
fails - a real system-boundary call, and this project has already hit real
rate limits on other providers more than once; degrading gracefully here is
worth more than a hard failure over a non-essential rewrite step.
"""

from __future__ import annotations

import anthropic

from ..config import Settings
from ..db.models import Turn
from ..logging import get_logger, log_external_call

log = get_logger("generation.query_rewrite")

QUERY_REWRITE_SYSTEM_PROMPT = """\
You rewrite the latest message in a conversation into a fully self-contained, \
standalone question for a document search engine that has no memory of the \
conversation.

Rules:
1. Resolve pronouns, references, and implicit context (e.g. "that", "the other \
one", "what about the EU?") using the conversation history, so the rewritten \
question makes complete sense on its own.
2. Do not answer the question. Do not add information, assumptions, or facts that \
are not already present in the conversation history.
3. If the latest message is already a standalone question, return it unchanged \
(minor grammar cleanup only - never reword its intent).
4. Output ONLY the rewritten question - no preamble, no quotation marks, no \
explanation."""


def _format_history(history: list[Turn]) -> str:
    return "\n".join(f"User: {t.user_query}\nAssistant: {t.answer_text}" for t in history)


def rewrite_query(
    history: list[Turn],
    latest_query: str,
    *,
    settings: Settings,
    client: anthropic.Anthropic,
) -> str:
    if not history:
        return latest_query

    try:
        # claude-haiku-4-5 predates the adaptive-thinking/effort model family:
        # it has no `thinking` mode to disable and errors on `output_config.effort`
        # ("errors on Sonnet 4.5 / Haiku 4.5" per the claude-api skill) - omitting
        # both entirely is the correct cost-minimal shape for this model, not an
        # oversight relative to claude.py's Sonnet call.
        response = client.messages.create(
            model=settings.query_rewrite_model,
            max_tokens=settings.query_rewrite_max_tokens,
            system=QUERY_REWRITE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{_format_history(history)}\n\n"
                        f"Latest message: {latest_query}"
                    ),
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 - a failed rewrite must not fail the whole turn
        log.warning("query_rewrite_failed_using_raw_query", error=str(exc))
        return latest_query

    log_external_call(
        "anthropic",
        "rewrite_query",
        model=settings.query_rewrite_model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )

    rewritten = "".join(block.text for block in response.content if block.type == "text").strip()
    return rewritten or latest_query

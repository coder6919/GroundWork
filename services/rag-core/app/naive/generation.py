"""Naive generation: chunks flattened into one text blob, a generic prompt, no
context-only constraint, no refusal instruction, no citations. Contrast with
app/generation/claude.py, which has all four.

Cost controls (thinking disabled, low effort) are kept even here - that's a
standing budget constraint on every call in this project, not something being
tested by the naive/improved comparison.
"""

from __future__ import annotations

import anthropic

from ..config import Settings
from ..logging import get_logger, log_external_call

log = get_logger("naive.generation")

PROMPT_TEMPLATE = "Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}"


def naive_generate(
    query: str,
    chunks: list[str],
    *,
    settings: Settings,
    client: anthropic.Anthropic | None = None,
) -> str:
    if client is None:
        from ..generation.claude import make_anthropic_client

        client = make_anthropic_client(settings.anthropic_api_key)

    prompt = PROMPT_TEMPLATE.format(context="\n\n".join(chunks), query=query)

    response = client.messages.create(
        model=settings.generation_model,
        max_tokens=settings.generation_max_tokens,
        thinking={"type": "disabled"},
        output_config={"effort": settings.generation_effort},
        messages=[{"role": "user", "content": prompt}],
    )

    log_external_call(
        "anthropic",
        "naive_generate",
        model=settings.generation_model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        stop_reason=response.stop_reason,
    )

    return "".join(block.text for block in response.content if block.type == "text")

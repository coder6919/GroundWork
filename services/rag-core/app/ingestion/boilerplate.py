"""Strip repeated headers/footers/page-number lines from per-page PDF text
before chunking, so running boilerplate doesn't pollute every chunk.

Operates on prose text only - never on table content, since a short repeated
cell value (e.g. a column header) isn't boilerplate.

Real footers commonly mix static text with a changing page number on one line
(e.g. "Confidential - Page 3 of 12"), so repeated lines are detected after
normalizing digit runs to a placeholder - otherwise every page's footer looks
like a distinct, one-off line. Known tradeoff: a genuinely repeated heading
that happens to contain a number (rare, since headings are excluded from
tables/prose is all this sees) could also be stripped; acceptable for v1.
"""

from __future__ import annotations

import re
from collections import Counter

_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s+)?\d+\s*(of|/)?\s*\d*\s*$", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")


def _normalize(line: str) -> str:
    return _DIGITS_RE.sub("#", line.strip())


def strip_boilerplate(page_texts: list[str], min_repeat_ratio: float = 0.5) -> list[str]:
    """Remove lines that repeat (after digit-normalization) across a majority
    of pages, plus lines that are nothing but a page number, from each page."""
    if len(page_texts) < 3:
        # too few pages for cross-page frequency analysis to mean anything
        return [_strip_page_numbers(t) for t in page_texts]

    per_page_lines = [t.splitlines() for t in page_texts]
    normalized_counts: Counter[str] = Counter()
    for lines in per_page_lines:
        for normalized in {_normalize(line) for line in lines if line.strip()}:
            normalized_counts[normalized] += 1

    threshold = max(2, int(len(page_texts) * min_repeat_ratio))
    boilerplate = {norm for norm, count in normalized_counts.items() if count >= threshold}

    return [
        "\n".join(
            line
            for line in lines
            if _normalize(line) not in boilerplate and not _PAGE_NUMBER_RE.match(line.strip())
        ).strip()
        for lines in per_page_lines
    ]


def _strip_page_numbers(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not _PAGE_NUMBER_RE.match(line.strip())
    ).strip()

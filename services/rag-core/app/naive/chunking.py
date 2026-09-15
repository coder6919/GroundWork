"""Naive fixed-length chunking - the "stuff docs into fixed windows" baseline.

No heading awareness, no atomic protection for tables or code fences. This
will cut mid-sentence, mid-table, mid-code-block - that's the point; contrast
with app/ingestion/chunking.py's structure-aware, atomic-block-preserving splitter.
"""

from __future__ import annotations

CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 100


def naive_chunk(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Fixed-width character windows with fixed overlap. No paragraph, heading,
    table, or code-fence awareness whatsoever."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks

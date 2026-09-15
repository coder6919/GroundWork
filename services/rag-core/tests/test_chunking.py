from __future__ import annotations

from itertools import pairwise

from app.ingestion.chunking import chunk_text, count_tokens

MARKDOWN = """# Refund Policy

Intro text about refunds in general.

## International Orders

International refunds take 30 days to process after we receive the item.

### Exceptions

Custom items are excluded from this policy unless defective.

## Domestic Orders

Domestic refunds take 5 days.
"""

CODE_DOC = """# API Reference

Call the endpoint below to check status.

```json
{
  "status": "pending",
  "id": 42
}
```

That's it.
"""

TABLE_DOC = """# Shipping

| Region | Cost |
|--------|------|
| US     | $5   |
| EU     | $12  |
"""


def test_heading_path_is_built_from_ancestors_plus_own_heading():
    chunks = chunk_text(MARKDOWN, target_tokens=500, overlap_ratio=0.15)
    paths = {c.heading_path for c in chunks}
    assert "Refund Policy" in paths
    assert "Refund Policy > International Orders" in paths
    assert "Refund Policy > International Orders > Exceptions" in paths
    assert "Refund Policy > Domestic Orders" in paths


def test_heading_path_is_prepended_to_chunk_text():
    chunks = chunk_text(MARKDOWN, target_tokens=500, overlap_ratio=0.15)
    international = next(c for c in chunks if c.heading_path == "Refund Policy > International Orders")
    assert international.text.startswith("Refund Policy > International Orders")
    assert "30 days" in international.text


def test_plain_text_with_no_headings_gets_no_path_prefix():
    plain = "Just some plain prose.\n\nA second paragraph with no structure at all."
    chunks = chunk_text(plain, target_tokens=500, overlap_ratio=0.15)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ""
    assert chunks[0].text == plain.strip()


def test_code_fence_is_never_split_and_is_classified_as_code():
    chunks = chunk_text(CODE_DOC, target_tokens=500, overlap_ratio=0.15)
    code_chunks = [c for c in chunks if "```json" in c.text]
    assert len(code_chunks) == 1
    assert '"status": "pending"' in code_chunks[0].text
    assert '"id": 42' in code_chunks[0].text  # whole fence intact, not truncated


def test_oversized_code_fence_still_kept_atomic():
    # Force a tiny token target so the fence alone exceeds it - it must still
    # come out as a single, uncut chunk rather than being sliced mid-block.
    chunks = chunk_text(CODE_DOC, target_tokens=5, overlap_ratio=0.0)
    code_chunks = [c for c in chunks if "```json" in c.text]
    assert len(code_chunks) == 1
    assert code_chunks[0].content_type == "code"
    assert count_tokens(code_chunks[0].text) > 5  # oversized, and that's allowed


def test_table_is_never_split_and_is_classified_as_table():
    chunks = chunk_text(TABLE_DOC, target_tokens=500, overlap_ratio=0.15)
    table_chunks = [c for c in chunks if c.content_type == "table"]
    assert len(table_chunks) == 1
    assert "| US     | $5   |" in table_chunks[0].text
    assert "| EU     | $12  |" in table_chunks[0].text


def test_overlap_carries_tail_of_previous_chunk_into_next():
    long_doc = "# Section\n\n" + "\n\n".join(f"Paragraph number {i} with some content." for i in range(60))
    chunks = chunk_text(long_doc, target_tokens=40, overlap_ratio=0.25)
    assert len(chunks) > 1
    # the boundary text from the end of chunk N should reappear at the start of chunk N+1
    for prev, cur in pairwise(chunks):
        prev_tail_words = prev.text.split()[-3:]
        assert " ".join(prev_tail_words) in cur.text


def test_empty_document_yields_no_chunks():
    assert chunk_text("   \n\n  ", target_tokens=500, overlap_ratio=0.15) == []


def test_short_document_not_artificially_split():
    short = "# FAQ\n\nA two-line answer."
    chunks = chunk_text(short, target_tokens=500, overlap_ratio=0.15)
    assert len(chunks) == 1

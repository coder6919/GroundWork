from __future__ import annotations

import pytest

from app.naive.chunking import naive_chunk


def test_empty_text_yields_no_chunks():
    assert naive_chunk("   \n\n  ") == []


def test_short_text_is_a_single_chunk():
    assert naive_chunk("hello world", chunk_size=800, overlap=100) == ["hello world"]


def test_splits_at_a_fixed_character_width_regardless_of_structure():
    text = "a" * 500 + "|CUT HERE|" + "b" * 500
    chunks = naive_chunk(text, chunk_size=500, overlap=0)
    # the marker straddles the fixed boundary and is cut apart - no structure awareness
    assert "|CUT HERE|" not in chunks[0]
    assert chunks[0] == "a" * 500


def test_consecutive_chunks_overlap_by_the_configured_amount():
    text = "x" * 1000
    chunks = naive_chunk(text, chunk_size=400, overlap=100)
    # chunk 2 should start 300 chars in (step = chunk_size - overlap)
    assert chunks[0] == "x" * 400
    assert chunks[1] == "x" * 400  # still all x's, so just confirms length/stepping
    assert len(chunks) == 4  # steps of 300 over 1000 chars: 0,300,600,900


def test_does_not_respect_table_or_code_fence_boundaries():
    markdown_table = "| A | B |\n|---|---|\n| 1 | 2 |"
    text = "prefix " * 100 + markdown_table + " suffix " * 100
    chunks = naive_chunk(text, chunk_size=len("prefix " * 100) + 10, overlap=0)
    # the table is not guaranteed to appear intact in any single chunk
    assert not any(markdown_table in c for c in chunks)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="overlap"):
        naive_chunk("some text", chunk_size=100, overlap=100)

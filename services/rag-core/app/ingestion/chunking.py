"""Structure-aware chunking for Markdown and plain text.

Strategy (per the build spec):
  1. Split on headings first (via LlamaIndex's MarkdownNodeParser), tracking the
     full heading path for each section.
  2. Sub-split oversized sections by token count with overlap.
  3. Never split inside a fenced code block or a Markdown table - such a block
     stays atomic even if it alone exceeds the token target.
  4. Prepend the heading path to each chunk's text before it is embedded.

Plain text (no headings) collapses to step 2 alone: MarkdownNodeParser returns
the whole document as a single, path-less node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise

import tiktoken
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

_ENCODING = tiktoken.get_encoding("cl100k_base")
_HEADING_LINE_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_TABLE_BLOCK_RE = re.compile(r"(?:^\|.*\|[ \t]*$\n?)+", re.MULTILINE)


@dataclass
class Chunk:
    text: str  # final text, heading-path prefixed, ready to embed
    heading_path: str
    content_type: str  # "prose" | "table" | "code"
    token_count: int
    page_number: int | None = None  # set by the PDF path only; None for text/Markdown


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def chunk_text(
    raw_text: str,
    *,
    target_tokens: int = 500,
    overlap_ratio: float = 0.15,
) -> list[Chunk]:
    """Chunk Markdown or plain text. Works for plain text too - MarkdownNodeParser
    just returns it as a single, heading-less node."""
    if not raw_text.strip():
        return []

    nodes = MarkdownNodeParser().get_nodes_from_documents([Document(text=raw_text)])
    chunks: list[Chunk] = []
    for node in nodes:
        heading_path = _full_heading_path(node.metadata.get("header_path", "/"), node.text)
        bodies = _split_no_overlap(node.text, target_tokens)
        bodies = _apply_overlap(bodies, target_tokens, overlap_ratio)
        for body in bodies:
            prefixed = f"{heading_path}\n\n{body}" if heading_path else body
            chunks.append(
                Chunk(
                    text=prefixed,
                    heading_path=heading_path,
                    content_type=_classify(body),
                    token_count=count_tokens(prefixed),
                )
            )
    return chunks


def _full_heading_path(parent_path: str, node_text: str) -> str:
    """Combine the parent header_path (e.g. "/Refund Policy/") with this node's
    own heading line (its first line, if it is one) into "Refund Policy > X"."""
    parts = [p for p in parent_path.strip("/").split("/") if p]
    first_line = node_text.split("\n", 1)[0]
    m = _HEADING_LINE_RE.match(first_line)
    if m:
        parts.append(m.group(2))
    return " > ".join(parts)


def _classify(body: str) -> str:
    """A chunk may combine a short heading with the atomic block that follows
    it, so classify by whether the block appears anywhere in the chunk, not
    just at the very start."""
    if _CODE_FENCE_RE.search(body):
        return "code"
    if _TABLE_BLOCK_RE.search(body):
        return "table"
    return "prose"


def _atomic_spans(text: str) -> list[tuple[int, int]]:
    """Byte ranges that must never be sub-split: fenced code blocks and tables."""
    spans = [m.span() for m in _CODE_FENCE_RE.finditer(text)]
    for m in _TABLE_BLOCK_RE.finditer(text):
        s, e = m.span()
        if not any(cs <= s < ce for cs, ce in spans):
            spans.append((s, e))
    return sorted(spans)


def _pieces(text: str) -> list[str]:
    """Break text into whole-preserving pieces: atomic blocks stay intact; prose
    is split on paragraph (blank-line) boundaries."""
    spans = _atomic_spans(text)
    pieces: list[str] = []
    pos = 0
    for s, e in spans:
        if s > pos:
            pieces.extend(p.strip() for p in re.split(r"\n\s*\n", text[pos:s]) if p.strip())
        pieces.append(text[s:e])
        pos = e
    if pos < len(text):
        pieces.extend(p.strip() for p in re.split(r"\n\s*\n", text[pos:]) if p.strip())
    return pieces


def _split_no_overlap(text: str, target_tokens: int) -> list[str]:
    pieces = _pieces(text)
    if not pieces:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for piece in pieces:
        piece_tokens = count_tokens(piece)
        if current and current_tokens + piece_tokens > target_tokens:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(piece)
        current_tokens += piece_tokens
        if current_tokens > target_tokens and len(current) == 1:
            # A single piece (atomic block or one huge paragraph) already exceeds
            # the target - emit it alone rather than corrupting it.
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _apply_overlap(bodies: list[str], target_tokens: int, overlap_ratio: float) -> list[str]:
    if len(bodies) <= 1 or overlap_ratio <= 0:
        return bodies
    overlap_tokens = int(target_tokens * overlap_ratio)
    out = [bodies[0]]
    for prev, cur in pairwise(bodies):
        tail = _tail_tokens(prev, overlap_tokens)
        out.append(f"{tail}\n\n{cur}" if tail else cur)
    return out


def _tail_tokens(text: str, n_tokens: int) -> str:
    if n_tokens <= 0:
        return ""
    ids = _ENCODING.encode(text)
    return _ENCODING.decode(ids[-n_tokens:]) if ids else ""

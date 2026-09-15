"""File-type detection and text extraction for plain text, Markdown, and PDF.

HTML loading is not yet added.
"""

from __future__ import annotations

from pathlib import Path

import charset_normalizer

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


class UnsupportedFileType(Exception):
    pass


def detect_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".md", ".markdown"):
        return "text/markdown"
    if ext == ".txt":
        return "text/plain"
    if ext == ".pdf":
        return "application/pdf"
    raise UnsupportedFileType(f"unsupported extension: {ext or '(none)'}")


def load_text(path: Path) -> str:
    """Read a text/Markdown file, normalizing encoding. Raises on files that
    cannot be decoded as text at all (e.g. binary data with this extension)."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    match = charset_normalizer.from_bytes(raw).best()
    if match is None:
        raise ValueError(f"could not detect a text encoding for {path.name}")
    return str(match)

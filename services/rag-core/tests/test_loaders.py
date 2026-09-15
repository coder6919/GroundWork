from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.loaders import UnsupportedFileType, detect_mime_type, load_text


def test_detect_mime_type_markdown(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("hi")
    assert detect_mime_type(p) == "text/markdown"


def test_detect_mime_type_txt(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("hi")
    assert detect_mime_type(p) == "text/plain"


def test_detect_mime_type_pdf(tmp_path: Path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"%PDF-1.4")
    assert detect_mime_type(p) == "application/pdf"


def test_detect_mime_type_rejects_unsupported(tmp_path: Path):
    p = tmp_path / "a.docx"
    p.write_bytes(b"not really a docx")
    with pytest.raises(UnsupportedFileType):
        detect_mime_type(p)


def test_load_text_utf8(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("héllo wörld", encoding="utf-8")
    assert load_text(p) == "héllo wörld"


def test_load_text_normalizes_non_utf8_encoding(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_bytes("café latin-1".encode("latin-1"))
    text = load_text(p)
    assert "caf" in text

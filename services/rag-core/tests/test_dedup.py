from __future__ import annotations

from pathlib import Path

from app.ingestion.dedup import content_hash, content_hash_file


def test_content_hash_file_matches_content_hash(tmp_path: Path):
    path = tmp_path / "a.txt"
    data = b"some file content for hashing"
    path.write_bytes(data)
    assert content_hash_file(path) == content_hash(data)


def test_content_hash_file_works_across_multiple_read_chunks(tmp_path: Path):
    # force several internal 1 MiB reads to prove the streaming loop is correct
    path = tmp_path / "big.bin"
    data = b"x" * (3 * 1024 * 1024 + 17)
    path.write_bytes(data)
    assert content_hash_file(path) == content_hash(data)


def test_same_bytes_same_hash():
    assert content_hash(b"hello world") == content_hash(b"hello world")


def test_different_bytes_different_hash():
    assert content_hash(b"hello world") != content_hash(b"hello there")


def test_hash_is_hex_sha256_length():
    assert len(content_hash(b"anything")) == 64
    assert all(c in "0123456789abcdef" for c in content_hash(b"anything"))

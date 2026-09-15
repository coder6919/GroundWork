# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "pyyaml"]
# ///
"""Fetch the demo corpus described in corpus/manifest.yaml into corpus/files/.

corpus/files/ is gitignored - these are third-party documents (see NOTICE for
their licenses), not something this repo redistributes. Re-running is safe and
idempotent; it just re-downloads and overwrites.

Usage:
    uv run scripts/fetch_corpus.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "corpus" / "manifest.yaml"
OUTPUT_DIR = ROOT / "corpus" / "files"


def main() -> int:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    sources = manifest.get("sources") or []
    if not sources:
        print("No sources listed in corpus/manifest.yaml - nothing to fetch.")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for src in sources:
            ext = {"markdown": ".md", "pdf": ".pdf"}.get(src.get("type"), "")
            dest = OUTPUT_DIR / f"{src['id']}{ext}"
            try:
                resp = client.get(src["source_url"])
                resp.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(resp.content)
                digest = hashlib.sha256(resp.content).hexdigest()
                print(f"OK    {src['id']:40s} {len(resp.content):>7} bytes  sha256={digest[:12]}...")
            except Exception as exc:  # noqa: BLE001 - report and continue, one bad source shouldn't block the rest
                failures += 1
                print(f"FAIL  {src['id']:40s} {exc}")

    print(f"\n{len(sources) - failures}/{len(sources)} fetched into {OUTPUT_DIR}")
    if failures:
        print("Some sources failed - see above. Re-run to retry just by running the script again.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

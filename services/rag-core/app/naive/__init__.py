"""The naive RAG baseline (Stage 3).

Everything under this package is deliberately unsophisticated - fixed-length
chunking with no structure awareness, a flat-context prompt with no
context-only constraint, no refusal instruction, no citations. It exists only
to give `scripts/compare_naive_vs_improved.py` something honest to contrast
against app/ingestion, app/retrieval, and app/generation. Do not "improve" this
package - if it stops being naive, the comparison stops meaning anything.
"""

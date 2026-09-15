from __future__ import annotations

from app.ingestion.boilerplate import strip_boilerplate


def test_exact_repeated_header_is_stripped():
    pages = [
        "Company Handbook\nSection one content.",
        "Company Handbook\nSection two content.",
        "Company Handbook\nSection three content.",
    ]
    cleaned = strip_boilerplate(pages)
    assert all("Company Handbook" not in p for p in cleaned)
    assert "Section one content." in cleaned[0]
    assert "Section two content." in cleaned[1]
    assert "Section three content." in cleaned[2]


def test_footer_with_varying_page_number_is_stripped_via_normalization():
    pages = [
        "Intro text.\nConfidential - Page 1 of 3",
        "Middle text.\nConfidential - Page 2 of 3",
        "Closing text.\nConfidential - Page 3 of 3",
    ]
    cleaned = strip_boilerplate(pages)
    assert all("Confidential" not in p for p in cleaned)
    assert "Intro text." in cleaned[0]
    assert "Middle text." in cleaned[1]
    assert "Closing text." in cleaned[2]


def test_bare_page_number_lines_are_always_stripped():
    pages = ["Body one.\n1", "Body two.\n2", "Body three.\n3"]
    cleaned = strip_boilerplate(pages)
    assert cleaned == ["Body one.", "Body two.", "Body three."]


def test_non_repeated_content_is_preserved():
    pages = [
        "Refund Policy\nInternational orders take 30 days.",
        "Refund Policy\nDomestic orders take 5 days.",
        "Refund Policy\nExceptions apply to custom items.",
    ]
    cleaned = strip_boilerplate(pages)
    # "Refund Policy" repeats on every page - correctly treated as a running header
    assert "International orders take 30 days." in cleaned[0]
    assert "Domestic orders take 5 days." in cleaned[1]
    assert "Exceptions apply to custom items." in cleaned[2]


def test_fewer_than_three_pages_only_strips_page_numbers_not_frequency():
    pages = ["Header\nBody one.\n1", "Header\nBody two.\n2"]
    cleaned = strip_boilerplate(pages)
    # too few pages to trust frequency analysis - "Header" survives even though
    # it repeats on both pages
    assert cleaned == ["Header\nBody one.", "Header\nBody two."]


def test_minority_repetition_is_not_treated_as_boilerplate():
    # a line repeating on only 1 of 4 pages should never be stripped
    pages = ["Shared once.\nUnique A", "Unique B", "Unique C", "Unique D"]
    cleaned = strip_boilerplate(pages)
    assert "Shared once." in cleaned[0]

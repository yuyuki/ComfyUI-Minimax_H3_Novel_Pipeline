"""Regression coverage for the pipeline text normalization policy."""
from minimax_h3_novel_pipeline.core.source_text import normalize


def test_normalization_preserves_content_and_paragraphs():
    original = "« L’e\u0301te\u0301\u00a0! »\r\n\r\n‘Oui’\u202f: A  B\rFin."
    expected = "\" L'été ! \"\n\n'Oui' : A  B\nFin."
    assert normalize(original) == expected
    assert normalize(expected) == expected

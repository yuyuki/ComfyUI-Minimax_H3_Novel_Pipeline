"""Nonblank schema strings remain valid for LM Studio grammar conversion."""
import re
import sys

import pytest
from jsonschema import Draft202012Validator

from minimax_h3_novel_pipeline import editable_schemas, reference_links


def pattern_fields(value):
    if isinstance(value, dict):
        if "pattern" in value:
            yield value
        for child in value.values():
            yield from pattern_fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from pattern_fields(child)


@pytest.mark.parametrize("schema", [
    editable_schemas.LINK_SCHEMA,
    editable_schemas.DESIGN_SCHEMA,
    reference_links.RESPONSE_SCHEMA["schema"],
])
def test_nonblank_patterns_use_grammar_supported_syntax_and_preserve_validation(schema):
    fields = list(pattern_fields(schema))
    assert fields
    whitespace = "".join(chr(code) for code in range(sys.maxunicode + 1) if chr(code).isspace())
    for field in fields:
        pattern = field["pattern"]
        assert pattern.startswith("^") and pattern.endswith("$")
        # The grammar converter supports explicit classes, Unicode escapes and \n,
        # but not regex shorthand classes such as \S or lookaround/inline flags.
        assert "(?" not in pattern
        assert not re.search(r"\\(?!u[0-9a-fA-F]{4}|n|r|t)", pattern)
        validator = Draft202012Validator({"type": "string", "pattern": pattern})
        for text in ["", whitespace, *whitespace]:
            assert not validator.is_valid(text), repr(text)
        for text in ["a", "la gardienne", "é", "山", "😀", '"', "\\", "\ntexte\n",
                     "première ligne\nseconde ligne", whitespace + "é" + whitespace]:
            assert validator.is_valid(text), repr(text)

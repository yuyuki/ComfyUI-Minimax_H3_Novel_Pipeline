"""Shared typography policy across extraction and validation."""
from copy import deepcopy
import hashlib
import json
from unittest.mock import Mock

import pytest

from tests.passages_helpers import with_passage_ids

from minimax_h3_novel_pipeline.core import source_text as st
from minimax_h3_novel_pipeline.core import character_evidence as evidence
from minimax_h3_novel_pipeline.core import character_profiles as profiles
from minimax_h3_novel_pipeline.core import lmstudio_json, path_access, util
from tests.test_character_evidence import observation
from tests.test_character_profiles import document
from tests.test_character_evidence import settings
from tests.test_structured_json import Stream, client_for, SCHEMA


@pytest.mark.parametrize("original,quote", [
    ("L’homme s’arrête.", "L'homme s'arrête."),
    ('« Elle dit “oui”. »', '\" Elle dit \"oui\". \"'),
    ("E\u0301lise\u00a0avait\r\n  les yeux verts.", "Élise avait les yeux verts."),
    ("\u1100\u1161\u11a8", "각"),
    ("A\u2028B\u2029C", "A B C"),
])
def test_equivalent_quotes_use_only_normalized_offsets(original, quote):
    text = st.normalize("Before.\r\n" + original + "\nAfter.")
    start, end = st.source_span(text, quote)
    assert text[start:end] == st.normalize(original)
    assert start == len("Before.\n")
    assert st.contains(text, quote)
    assert st.comparison_text(original) == st.comparison_text(quote)
    assert st.normalize(st.normalize(original)) == st.normalize(original)


@pytest.mark.parametrize("quote", [
    "L'homme n'est pas là.", "l'homme est là.", "L'homme est la.",
    "L'homme était là.", "L'homme … là.", "", "  ",
])
def test_source_comparisons_reject_changes_and_empty_quotes(quote):
    source = "L’homme est là."
    assert not st.contains(source, quote)
    assert st.source_span(source, quote) is None


def test_typography_equivalence_does_not_hide_ambiguous_locations():
    source = "L’homme attend. L'homme attend."
    assert st.source_span(source, "L'homme attend.") is None
    assert len(st.source_spans(source, "L'homme attend.")) == 2
    assert st.source_span("aaaa", "aaa") is None


def test_source_membership_cannot_split_a_combining_character():
    assert st.source_spans("e\u0301", "e") == []
    assert not st.contains("e\u0301", "e")


def test_evidence_and_profiles_share_typography_policy_and_keep_provenance():
    original = "L’homme avait les yeux verts."
    data = document(observation(original))
    item = data["chapters"][0]["evidence"][0]
    item["quotation"] = original.replace("’", "'")
    snapshot = deepcopy(data)
    checked = evidence.validate_result({"evidence": [observation(item["quotation"])]}, original)[0]
    assert not checked.get("warnings")
    record = next(iter(profiles.prepare_evidence(data).values()))
    assert record["validation_confidence"] == "high"
    assert record["provenance"] == item["provenance"]
    assert data == snapshot


def test_both_readers_model_prompts_and_saved_locations_use_the_same_source(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    raw = '\ufeffPréface.\r\n\r\nE\u0301lise avait l’œil vert. « Elle s’arrête. »\r\n' + "Fin. " * 20
    path = tmp_path / "chapter.md"
    path.write_bytes(raw.encode("utf-8"))
    expected = st.normalize(raw.removeprefix("\ufeff"))
    assert evidence.read_source(path) == (expected, "normalized_utf8_text")
    assert util.read_chapter(path) == expected
    quote = "Élise avait l'œil vert."
    calls = []

    def chat(client, model, system, prompt, schema, *args):
        payload = json.loads(prompt)
        calls.append(payload)
        assert all(part["text"] in expected for part in payload["source_passages"])
        assert "\r" not in prompt and "’" not in prompt
        return {"evidence": [observation(quote)]}

    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", with_passage_ids(chat))
    monkeypatch.setattr(evidence, "audit_observations", lambda rows, *args: rows)
    chapter = evidence.extract_chapter(path, tmp_path / "output", None, "qwen", settings(), 1)
    assert "source_text" not in chapter
    assert evidence.read_verified_source(chapter) == expected
    assert chapter["source_text_sha256"] == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    assert chapter["location_basis"] == "normalized_utf8_text"
    assert calls
    item = chapter["evidence"][0]
    for loc in item["provenance"]["locations"]:
        assert expected[loc["start_char"]:loc["end_char"]] == item["quotation"] == quote
        assert loc["start_line"] == 3
    records = profiles.prepare_evidence({"format_version": evidence.VERSION,
        "location_convention": "Normalized offsets", "chapters": [chapter]})
    profile = next(iter(records.values()))
    assert profile["validation_confidence"] == "high"
    assert path.read_bytes() == raw.encode("utf-8")


def test_pdf_and_text_use_the_identical_normalized_basis(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    raw = "E\u0301lise s’arrête.\r\n"
    path = tmp_path / "chapter.pdf"
    path.touch()
    monkeypatch.setattr("pypdf.PdfReader", Mock(return_value=Mock(pages=[Mock(extract_text=lambda: raw)])))
    assert st.read_source(path) == (st.normalize(raw), "normalized_pdf_extracted_text")


@pytest.mark.parametrize("change,reason", [
    (lambda d: d.update(format_version=1), "Regenerate evidence"),
    (lambda d: d["chapters"][0].update(source_text="L’homme."), "unexpected chapter fields"),
    (lambda d: d["chapters"][0].update(location_basis="utf8_decoded_text"), "normalized location_basis"),
    (lambda d: d["chapters"][0].update(source_text_sha256="invalid"), "source_text_sha256"),
])
def test_old_or_inconsistent_source_bases_are_rejected_without_rebasing(change, reason):
    data = document(observation("L'homme."))
    change(data)
    with pytest.raises(ValueError, match=reason):
        evidence.validate_document(data)


def test_shared_transport_normalizes_all_model_text_before_returning_it():
    raw = 'E\u0301lise\u00a0s’arrête.\r\n« Oui. »'
    client, _ = client_for(Stream([json.dumps({"value": raw})]), passage_fixtures=True)
    result = lmstudio_json.chat_json(client, "qwen", "Return data.", "Source.", SCHEMA, 0.1, 1024)
    assert result == {"value": st.normalize(raw)}

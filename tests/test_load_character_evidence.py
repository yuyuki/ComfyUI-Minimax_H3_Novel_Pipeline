"""Saved evidence interchangeability, fidelity, validation and disk changes."""
import json
import math
import os
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from minimax_h3_novel_pipeline.core import character_evidence, path_access
from minimax_h3_novel_pipeline.nodes import LoadCharacterEvidenceNode as Loader
from minimax_h3_novel_pipeline.nodes import CharacterEvidenceExtractorNode as Extractor
from tests.test_character_evidence import observation, settings


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path / kind)
    source = tmp_path / "input/chapter.txt"
    source.parent.mkdir()
    source.write_bytes("Élise avait les yeux verts.  \r\n".encode())
    chat = Mock(return_value={"evidence": [observation(), None, {}]})
    monkeypatch.setattr(character_evidence.lmstudio_json, "chat_json", chat)
    output = tmp_path / "output"
    chapter = character_evidence.extract_chapter(source, output, None, "qwen3.5", settings(), 1)
    data = {"format_version": character_evidence.VERSION, "location_convention": "Original offsets",
            "chapters": [chapter]}
    path = output / "character_evidence.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    # Neither extraction, model setup nor LLM validation may happen during loading.
    for module, name in ((character_evidence, "extract_chapter"), (character_evidence, "validate_result"),
                         (character_evidence.lmstudio_pipeline, "make_client_and_model"),
                         (character_evidence.lmstudio_json, "chat_json")):
        monkeypatch.setattr(module, name, Mock(side_effect=AssertionError("Unexpected extraction work")))
    return path, data, source


def test_registered_contract_matches_extractor():
    assert NODE_CLASS_MAPPINGS["LoadCharacterEvidenceNode"] is Loader
    assert NODE_DISPLAY_NAME_MAPPINGS["LoadCharacterEvidenceNode"] == "Load Character Evidence"
    assert Loader.RETURN_TYPES == Extractor.RETURN_TYPES
    assert Loader.RETURN_NAMES == Extractor.RETURN_NAMES
    assert set(Loader.INPUT_TYPES()["required"]) == {"evidence_path"}


def test_loads_actual_extractor_data_without_sources_or_writes(saved):
    path, data, source = saved
    source.unlink()
    before = path.read_bytes(), path.stat().st_mtime_ns
    for name in (str(path), path.name):
        assert Loader().run(name) == (data, str(path))
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert sorted(p.name for p in path.parent.iterdir()) == [".cache", path.name]


def test_manual_edits_metadata_quotes_and_provenance_are_unchanged(saved):
    path, data, _ = saved
    data["review"] = {"author": "éditeur", "notes": ["keep all values"]}
    chapter = data["chapters"][0]
    item = chapter["evidence"][0]
    item.update(value="vert sombre", confidence=0.4, classification="inferred", notes="manual")
    item["provenance"]["review"] = "verified by reader"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8-sig")
    renamed = path.with_name("manually_corrected.json")
    path.rename(renamed)
    before = renamed.read_bytes()
    assert Loader().run(str(renamed)) == (data, str(renamed))
    assert renamed.read_bytes() == before


@pytest.mark.parametrize("value", ["", "missing.json", "../outside.json", None])
def test_invalid_paths(saved, value):
    with pytest.raises(ValueError):
        Loader().run(value)
    assert math.isnan(Loader.IS_CHANGED(value))


def test_directory_and_external_paths_rejected(saved):
    path, _, _ = saved
    for value in (str(path.parent), str(path.parent.parent / "outside.json")):
        with pytest.raises(ValueError):
            Loader().run(value)


@pytest.mark.parametrize("contents, message", [
    ('{\n broken', "invalid JSON at line 2, column"),
    ('[]', "document must be an object"),
    ('{}', "missing required field 'format_version'"),
])
def test_invalid_json(saved, contents, message):
    path, _, _ = saved
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        Loader().run(str(path))


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(format_version=999), "Unsupported.*format_version.*expected 1"),
    (lambda d: d.update(format_version=True), "format_version must be int"),
    (lambda d: d.update(chapters={}), "chapters must be list"),
    (lambda d: d["chapters"][0].pop("source_text"), "missing required field 'source_text'"),
    (lambda d: d["chapters"][0]["evidence"][0].pop("provenance"), "missing required field 'provenance'"),
    (lambda d: d["chapters"][0]["evidence"][0]["provenance"].update(locations={}), "locations must be list"),
    (lambda d: d["chapters"][0]["evidence"][0]["provenance"]["locations"][0].pop("start_char"),
     "missing required field 'start_char'"),
])
def test_invalid_document_structure(saved, change, message):
    path, data, _ = saved
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        Loader().run(str(path))


def test_utf8_error_is_actionable(saved):
    path, _, _ = saved
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        Loader().run(str(path))


def test_content_hash_and_fresh_load_with_same_size_and_timestamp(saved):
    path, data, _ = saved
    before = Loader.IS_CHANGED(str(path))
    assert Loader.IS_CHANGED(str(path)) == before
    stat = path.stat()
    path.write_bytes(path.read_bytes().replace(b"verts", b"bleus"))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert Loader.IS_CHANGED(str(path)) != before
    assert Loader().run(str(path))[0] != data
    path.unlink()
    assert math.isnan(Loader.IS_CHANGED(str(path)))
    with pytest.raises(ValueError, match="does not exist"):
        Loader().run(str(path))

"""Saved visual profiles remain interchangeable, editable and independent of models."""
import json
import math
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from minimax_h3_novel_pipeline.core import character_visual_profiles as vp, path_access, util
from minimax_h3_novel_pipeline.nodes import CharacterVisualProfileGeneratorNode as Generator
from minimax_h3_novel_pipeline.nodes import LoadCharacterVisualProfilesNode as Loader
from tests.test_character_visual_profiles import consolidated


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path / kind)
    data = consolidated()
    result, filename = vp.generate(data, ['chapter1.txt'], tmp_path / "output")
    for module, name in ((vp, "generate"), (vp, "prepare_profiles"),
                         (vp.source, "build_profiles"), (util, "save_json")):
        monkeypatch.setattr(module, name, Mock(side_effect=AssertionError("Unexpected generation or write")))
    return Path(filename), result


def test_registered_contract_matches_generator():
    assert NODE_CLASS_MAPPINGS["LoadCharacterVisualProfilesNode"] is Loader
    assert NODE_DISPLAY_NAME_MAPPINGS["LoadCharacterVisualProfilesNode"] == "Load Character Visual Profiles"
    assert Loader.RETURN_TYPES == Generator.RETURN_TYPES
    assert Loader.RETURN_NAMES == Generator.RETURN_NAMES
    assert Loader.CATEGORY == Generator.CATEGORY
    assert set(Loader.INPUT_TYPES()["required"]) == {"profiles_path"}
    from minimax_h3_novel_pipeline.nodes import NODE_CLASS_MAPPINGS as node_mappings
    assert node_mappings["LoadCharacterVisualProfilesNode"] is Loader


def test_loads_generator_output_without_writes_or_sources(saved):
    path, data = saved
    before = path.read_bytes(), path.stat().st_mtime_ns
    files = sorted(path.parent.rglob("*"))
    # The embedded source path does not exist; all needed evidence is retained in JSON.
    for name in (str(path), path.name, f"  {path.name}  "):
        assert Loader().run(name) == (data, str(path))
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert sorted(path.parent.rglob("*")) == files


def test_preserves_all_manual_edits_and_metadata(saved):
    path, data = saved
    data.update(review={"author": "éditeur", "notes": ["retain everything"]}, source_digest="manual-digest")
    profile = data["profiles"][0]
    profile.update(local_id="my-character", canonical_name="Élise, reviewed", review={"approved": True})
    data["visual_designs"]["entities"][0]["added_details"]["hair"] = "Manual hair."
    path = path.with_name("reviewed_copy.json")
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8-sig")
    before = path.read_bytes()
    assert Loader().run(str(path)) == (data, str(path))
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", ["", "missing.json", "../outside.json", None])
def test_invalid_paths(saved, value):
    with pytest.raises(ValueError):
        Loader().run(value)
    assert math.isnan(Loader.IS_CHANGED(value))


def test_directory_and_external_paths_rejected(saved):
    path, _ = saved
    for value in (str(path.parent), str(path.parent.parent / "outside.json")):
        with pytest.raises(ValueError):
            Loader().run(value)
        assert math.isnan(Loader.IS_CHANGED(value))


@pytest.mark.parametrize("contents, message", [
    ('{\n broken', "invalid JSON at line 2, column"),
    ('[]', "document must be an object"),
    ('{}', "missing required field 'format_version'"),
])
def test_invalid_json(saved, contents, message):
    path, _ = saved
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        Loader().run(str(path))


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(format_version=999), "unsupported format_version.*expected 4"),
    (lambda d: d.update(format_version=True), "format_version must be int"),
    (lambda d: d.update(profiles={}), "profiles must be list"),
    (lambda d: d.update(source_document={}), "missing required field 'format_version'"),
    (lambda d: d["profiles"].append(None), "must be an object"),
    (lambda d: d["profiles"][0].pop("local_id"), "missing required field 'local_id'"),
    (lambda d: d["profiles"][0].update(supporting_evidence=[{}]), "missing required field 'evidence_id'"),
])
def test_incompatible_structure(saved, change, message):
    path, data = saved
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        Loader().run(str(path))


def test_utf8_and_file_access_errors(saved, monkeypatch):
    path, _ = saved
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        Loader().run(str(path))
    monkeypatch.setattr(Path, "read_text", Mock(side_effect=PermissionError))
    with pytest.raises(ValueError, match="Check file access permissions"):
        Loader().run(str(path))


def test_same_size_and_timestamp_edits_are_detected_and_reloaded(saved):
    path, data = saved
    before = Loader.IS_CHANGED(str(path))
    assert Loader.IS_CHANGED(str(path)) == before
    stat = path.stat()
    path.write_bytes(path.read_bytes().replace(b"cinematic photographic", b"realistic photographic"))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert Loader.IS_CHANGED(str(path)) != before
    assert Loader().run(str(path))[0] != data
    path.unlink()
    assert math.isnan(Loader.IS_CHANGED(str(path)))
    with pytest.raises(ValueError, match="does not exist"):
        Loader().run(str(path))


def test_empty_generated_document(saved):
    path, data = saved
    data["profiles"] = []
    data["source_document"]["profiles"] = []
    data["source_document"]["source_document"]["chapters"] = []
    path.write_text(json.dumps(data), encoding="utf-8")
    assert Loader().run(str(path)) == (data, str(path))

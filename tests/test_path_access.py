"""Confinement regressions with temporary server roots and no network work."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline import lmstudio_pipeline, path_access, util
from minimax_h3_novel_pipeline.consolidate_references import ConsolidateReferencesNode
from minimax_h3_novel_pipeline.extract_chapter_references import ExtractChapterReferencesNode
from minimax_h3_novel_pipeline.generate_h3_prompts import GenerateH3PromptsNode
from minimax_h3_novel_pipeline.load_chapter_catalogs import LoadChapterCatalogsNode
from minimax_h3_novel_pipeline.load_consolidated_references import LoadConsolidatedReferencesNode


@pytest.fixture
def roots(tmp_path, monkeypatch):
    source = tmp_path / "input"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(
        get_input_directory=lambda: str(source),
        get_output_directory=lambda: str(output),
    ))
    return source, output / "minimax_h3_novel"


def link_to(link, target):
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:
        if os.name == "nt" and target.is_dir():
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            pytest.skip(f"Symlink creation unavailable: {exc}")


@pytest.mark.parametrize("value", [
    "../escape", "safe/../../escape", "safe/../inside", "file.txt:secret",
    "NUL.txt", "con", "COM1.txt", "LPT².txt", "trailing.", "trailing ",
    "bad\x00name", "bad?.txt", "C:relative.txt", "\\rooted.txt", "",
])
def test_invalid_paths_rejected_for_both_roots(roots, value):
    for resolve in (path_access.input_path, path_access.output_path):
        with pytest.raises(ValueError):
            resolve(value)


def test_external_absolute_paths_rejected_before_resolution(roots, tmp_path, monkeypatch):
    outside = tmp_path / "private.txt"
    original = Path.resolve

    def resolve(path, *args, **kwargs):
        assert path != outside, "External path must not be probed"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="inside"):
        util.discover_inputs([outside])
    with pytest.raises(ValueError, match="inside"):
        util.save_json(outside, {"bad": True})
    assert not outside.exists()


def test_input_discovery_uses_server_root_and_natural_order(roots, monkeypatch, tmp_path):
    source, _ = roots
    chapters = source / "minimax_h3_novel"
    chapters.mkdir()
    for name in ("chapter10.txt", "chapter2.md"):
        (chapters / name).write_text("Chapter text. " * 20, encoding="utf-8")
    (chapters / "ignore.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    files = util.discover_inputs([Path("minimax_h3_novel"), chapters / "chapter2.md"])
    assert [p.name for p in files] == ["chapter2.md", "chapter10.txt"]
    assert util.read_chapter(files[0]).startswith("Chapter text.")


@pytest.mark.parametrize("node,kwargs", [
    (ExtractChapterReferencesNode, {"chapter_paths": "", "saved_chapter": ""}),
    (ConsolidateReferencesNode, {"chapter_catalogs": [{"chapter_id": "one"}]}),
    (GenerateH3PromptsNode, {"consolidated_references": {}, "chapter_paths": "", "saved_chapter": ""}),
])
def test_all_stages_reject_output_escape_before_loading_pipeline(roots, monkeypatch, tmp_path, node, kwargs):
    load = Mock(side_effect=AssertionError("Must reject before pipeline/network work"))
    monkeypatch.setattr(lmstudio_pipeline, "load", load)
    for out_dir in ("../escape", str(tmp_path / "escape")):
        with pytest.raises(ValueError):
            node().run(lmstudio_config={}, out_dir=out_dir, **kwargs)
    load.assert_not_called()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("node,extra", [
    (ExtractChapterReferencesNode, {}),
    (GenerateH3PromptsNode, {"consolidated_references": {}}),
])
def test_chapter_fields_reject_external_reads(roots, monkeypatch, tmp_path, node, extra):
    load = Mock(side_effect=AssertionError("Must reject before pipeline/network work"))
    monkeypatch.setattr(lmstudio_pipeline, "load", load)
    outside = tmp_path / "private.txt"
    outside.write_text("Private text. " * 20, encoding="utf-8")
    for fields in ({"chapter_paths": str(outside), "saved_chapter": ""},
                   {"chapter_paths": "", "saved_chapter": str(outside)}):
        with pytest.raises(ValueError):
            node().run(lmstudio_config={}, out_dir="safe", **fields, **extra)
    load.assert_not_called()


def test_loaders_resume_from_confined_outputs(roots, tmp_path):
    _, output = roots
    catalog = {"schema_version": util.CHAPTER_SCHEMA, "chapter_id": "one", "characters": []}
    util.save_json(Path("chapter_catalogs/one_references.json"), catalog)
    assert LoadChapterCatalogsNode().run("chapter_catalogs")[0] == [catalog]
    registry = {"schema_version": util.REGISTRY_SCHEMA, "entities": [], "chapter_entity_map": {}, "entity_asset_index": {},
                "picture_assets": [], "audio_assets": []}
    util.save_json(Path("references/consolidated_references.json"), registry)
    assert LoadConsolidatedReferencesNode().run("references")[0] == registry
    assert LoadConsolidatedReferencesNode().run(str(output / "references/consolidated_references.json"))[0] == registry
    for loader in (LoadChapterCatalogsNode(), LoadConsolidatedReferencesNode()):
        with pytest.raises(ValueError):
            loader.run(str(tmp_path))


def test_directory_links_cannot_escape_input_output_or_loader_roots(roots, tmp_path):
    source, output = roots
    output.mkdir()
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "chapter.txt").write_text("Private text. " * 20, encoding="utf-8")
    link_to(source / "linked", outside)
    link_to(output / "linked", outside)
    for operation in (
        lambda: util.discover_inputs([Path("linked")]),
        lambda: util.read_chapter(source / "linked/chapter.txt"),
        lambda: util.save_json(Path("linked/new.json"), {}),
        lambda: LoadChapterCatalogsNode().run("linked"),
        lambda: LoadConsolidatedReferencesNode().run("linked"),
    ):
        with pytest.raises(ValueError, match="inside"):
            operation()
    assert not (outside / "new.json").exists()


def test_output_plugin_root_cannot_itself_escape(roots, tmp_path):
    _, output = roots
    link_to(output, tmp_path)
    with pytest.raises(ValueError, match="inside"):
        util.output_path("references")


def test_discovery_rejects_external_file_links(roots, tmp_path):
    source, _ = roots
    outside = tmp_path / "private.txt"
    outside.write_text("Private text. " * 20, encoding="utf-8")
    link_to(source / "linked.txt", outside)
    with pytest.raises(ValueError, match="inside"):
        util.discover_inputs([source])


def test_extraction_existing_cache_still_loads(roots):
    source, output = roots
    chapter = source / "chapter.txt"
    chapter.write_text("Chapter text. " * 20, encoding="utf-8")
    step = lmstudio_pipeline.load("extract")
    payload = {"schema_version": step.SCHEMA_VERSION, "source": {"sha256": step.sha256_file(chapter)}}
    util.save_json(output / "chapter_references.json", payload)
    saved = step.process_chapter(chapter, output, None, "mock", SimpleNamespace(force=False))
    assert util.load_json(saved) == payload


@pytest.mark.parametrize("step_name,linked_dir", [
    ("extract", ".cache"), ("extract", ".cache/chapter"),
    ("generate", "chapter"), ("generate", "chapter/.cache"),
])
def test_bundled_steps_reject_nested_directory_escape(roots, tmp_path, step_name, linked_dir):
    source, output = roots
    output.mkdir()
    chapter = source / "chapter.txt"
    chapter.write_text("Chapter text. " * 20, encoding="utf-8")
    outside = tmp_path / "private"
    outside.mkdir()
    (output / linked_dir).parent.mkdir(parents=True, exist_ok=True)
    link_to(output / linked_dir, outside)
    step = lmstudio_pipeline.load(step_name)
    args = SimpleNamespace(force=True, chunk_chars=5500, overlap_paragraphs=0, out_dir=output)
    with pytest.raises(ValueError, match="inside"):
        if step_name == "extract":
            step.process_chapter(chapter, output, None, "mock", args)
        else:
            step.process_chapter(chapter, {"chapter_entity_map": {}}, None, "mock", args)
    assert list(outside.iterdir()) == []


def test_generated_scene_files_and_manifest_stay_in_output(roots, monkeypatch):
    source, output = roots
    output.mkdir()
    chapter = source / "chapter.txt"
    chapter.write_text("Chapter text. " * 20, encoding="utf-8")
    step = lmstudio_pipeline.load("generate")
    monkeypatch.setattr(step, "plan_scenes", lambda *args: [])
    args = SimpleNamespace(force=True, chunk_chars=5500, overlap_paragraphs=0,
                           out_dir=output, duration=8, delay=0, max_scenes=0)
    manifest = step.process_chapter(chapter, {"chapter_entity_map": {}}, None, "mock", args)
    assert json.loads((output / "chapter/manifest.json").read_text()) == manifest
    scene = SimpleNamespace(title="A scene", visual_event="An event", adaptation_notes="", source_excerpt="Text")
    bindings = {"picture_input_order": [], "subjects": [], "audio": []}
    validation = SimpleNamespace(ok=True, errors=[], word_count=1)
    entry = step.save_scene(output / "chapter", 1, scene, bindings, "Prompt", validation)
    assert (output / "chapter" / entry["prompt_file"]).read_text().strip() == "Prompt"

"""Offline execution records, settings coverage and credential exclusion."""
from contextlib import nullcontext
import json

import pytest

from minimax_h3_novel_pipeline import (
    configuration_snapshot as snapshots, lmstudio_pipeline, path_access, run_output, util,
)
from minimax_h3_novel_pipeline import extract_chapter_references as extract
from minimax_h3_novel_pipeline import consolidate_references as consolidate
from minimax_h3_novel_pipeline import generate_h3_prompts as generate
from minimax_h3_novel_pipeline.lmstudio_config import LMStudioConfigurationNode


ADAPTED = [{"chapter_name": "chapter", "sequences": [{"sequence": 1, "source": "Text",
            "adaptation": {"initialState": "A character.", "event": "", "endingState": ""}}]}]

def defaults(cls):
    return {key: spec[1]["default"]
            for fields in cls.INPUT_TYPES().values() for key, spec in fields.items()
            if len(spec) > 1 and "default" in spec[1]}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: tmp_path)
    config = {**defaults(LMStudioConfigurationNode), "run_folder": "20260916120000",
              "api_key": "secret-must-not-be-saved", "unexpected": "secret-must-not-be-saved"}
    chapter = tmp_path / "chapter.txt"
    chapter.write_text("A chapter with a character and a location.\n" * 10, encoding="utf-8")
    for module in (generate,):
        monkeypatch.setattr(module, "selected_chapter_paths", lambda selection: [str(chapter)])
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "qwen3.5-test"))
    return tmp_path, config, chapter


@pytest.mark.parametrize("stage,module,cls", [
    ("extract", extract, extract.ExtractChapterReferencesNode),
    ("consolidate", consolidate, consolidate.ConsolidateReferencesNode),
    ("generate", generate, generate.GenerateH3PromptsNode),
])
def test_each_node_records_all_controls_and_result_hashes(setup, monkeypatch, stage, module, cls):
    root, config, chapter = setup
    pipeline = lmstudio_pipeline.load(stage)
    catalog = {"schema_version": util.CHAPTER_SCHEMA, "chapter_id": "chapter", "chapter_name": "chapter",
               "source": {}, "timeline_version": "cinematic-reference-timeline.v1",
               "characters": [], "locations": [], "objects": [],
               "sequences": [{"sequence": 1, "source": "text", "adaptation": {
                   "initialState": "Aster waits.", "event": "Aster moves.", "endingState": "Aster stops."}}]}
    registry = {"schema_version": util.REGISTRY_SCHEMA, "entities": [], "picture_assets": [], "audio_assets": []}
    if stage == "extract":
        def process(path, output, *args):
            saved = output / "chapter_references.json"
            util.save_json(saved, catalog)
            return saved
        monkeypatch.setattr(extract.cinematic_references, "process_chapter", process)
        inputs = {"cinematic_chapters": ADAPTED}
    elif stage == "consolidate":
        monkeypatch.setattr(pipeline, "reconcile_chapter", lambda *args: [])
        monkeypatch.setattr(pipeline, "audit_registry", lambda *args: [])
        monkeypatch.setattr(module, "prepare_designs", lambda *args: {"entities": []})
        inputs = {"chapter_catalogs": [catalog]}
    else:
        def process(path, registry, client, model, args):
            manifest = {"chapter_id": "chapter", "outputs": []}
            util.save_json(args.out_dir / "chapter/manifest.json", manifest)
            return manifest
        monkeypatch.setattr(pipeline, "process_chapter", process)
        inputs = {"chapter_selection": {}, "consolidated_references": registry}
    params = defaults(cls)
    # Optional defaults must be recorded even when old workflows omit them.
    if stage == "consolidate":
        params.pop("image_style")
        params.pop("image_asset_scope")
    cls().run(lmstudio_config=config, **inputs, **params)
    output = root / config["run_folder"] / params["out_dir"]
    path = output / f"{stage}_configuration.json"
    text = path.read_text(encoding="utf-8")
    record = json.loads(text)
    assert "secret-must-not-be-saved" not in text
    assert record["lmstudio_config"] == defaults(LMStudioConfigurationNode)
    assert record["node_settings"] == defaults(cls)
    assert record["resolved_model"] == "qwen3.5-test"
    assert record["status"] == "completed"
    assert record["completed_at"] >= record["started_at"]
    assert record["outputs"]
    assert record["inputs"]
    for artifact in record["outputs"]:
        assert snapshots.file_digest(output / artifact["file"]) == artifact["sha256"]
    if stage == "extract":
        assert record["inputs"]["chapters"][0]["sha256"] == extract.cinematic_references.chapter_digest(ADAPTED[0])
    assert record["model_controls"]["top_p"] == 0.8


@pytest.mark.parametrize("toggle,anchors,expected", [
    (None, None, None),
    (False, None, None),
    (True, None, {}),
    (False, {}, {}),
    (False, {"tablet.wall": "right wall"}, {"tablet.wall": "right wall"}),
    (True, {"tablet.wall": "right wall"}, {"tablet.wall": "right wall"}),
])
def test_generate_continuity_toggle_and_anchor_connection(setup, monkeypatch, toggle, anchors, expected):
    root, config, _ = setup
    pipeline = lmstudio_pipeline.load("generate")
    captured = []

    def process(path, registry, client, model, args):
        captured.append(getattr(args, "spatial_anchors", None))
        return {"chapter_id": "chapter", "outputs": []}

    monkeypatch.setattr(pipeline, "process_chapter", process)
    params = defaults(generate.GenerateH3PromptsNode)
    if toggle is None:
        params.pop("enable_spatial_continuity")  # Existing workflows omit the new widget.
    else:
        params["enable_spatial_continuity"] = toggle
    if anchors is not None:
        params["spatial_continuity"] = {"schema_version": "minimax-spatial-continuity.v1", "anchors": anchors}
    registry = {"schema_version": util.REGISTRY_SCHEMA, "entities": [], "picture_assets": [], "audio_assets": []}
    generate.GenerateH3PromptsNode().run(registry, config, {}, **params)
    assert captured == [expected]
    record = util.load_json(root / config["run_folder"] / params["out_dir"] / "generate_configuration.json")
    assert record["node_settings"]["enable_spatial_continuity"] is (expected is not None)


def test_failed_execution_does_not_claim_completed_results(setup, monkeypatch):
    root, config, _ = setup
    def fail(*args):
        raise RuntimeError("secret-must-not-be-saved")
    monkeypatch.setattr(extract.cinematic_references, "process_chapter", fail)
    with pytest.raises(RuntimeError):
        extract.ExtractChapterReferencesNode().run(config, ADAPTED, **defaults(extract.ExtractChapterReferencesNode))
    text = (root / config["run_folder"] / "chapter_catalogs/extract_configuration.json").read_text()
    assert "secret-must-not-be-saved" not in text
    assert json.loads(text)["status"] == "started"
    assert json.loads(text)["outputs"] == []


def test_mistral_records_ignored_qwen_controls_and_normalized_node_settings(setup, monkeypatch):
    root, config, _ = setup
    config.update(model_family="Mistral", thinking=True, qwen35_top_k=90)
    def process(path, output, *args):
        saved = output / "chapter_references.json"
        util.save_json(saved, {"chapter_id": "chapter"})
        return saved
    monkeypatch.setattr(extract.cinematic_references, "process_chapter", process)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mistral-test"))
    params = defaults(extract.ExtractChapterReferencesNode)
    extract.ExtractChapterReferencesNode().run(config, ADAPTED, **params)
    record = util.load_json(root / config["run_folder"] / "chapter_catalogs/extract_configuration.json")
    assert record["lmstudio_config"]["thinking"] is True
    assert record["model_controls"]["settings"] == {"thinking": False}
    assert record["model_controls"]["request_extra_body"] == {}
    assert record["model_controls"]["chatml_fallback_allowed"] is False
    assert "merge_batch_size" not in record["node_settings"]


def test_generate_has_no_obsolete_narrative_socket():
    inputs = generate.GenerateH3PromptsNode.INPUT_TYPES()
    assert set(inputs["optional"]) == {"spatial_continuity", "camera_direction"}

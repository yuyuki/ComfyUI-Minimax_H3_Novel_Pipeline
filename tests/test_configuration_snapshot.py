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
    for module in (extract, generate):
        monkeypatch.setattr(module, "selected_chapter_paths", lambda selection: str(chapter))
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
    catalog = {"schema_version": util.CHAPTER_SCHEMA, "chapter_id": "chapter", "source": {}}
    registry = {"schema_version": util.REGISTRY_SCHEMA, "entities": [], "picture_assets": [], "audio_assets": []}
    if stage == "extract":
        def process(path, output, *args):
            saved = output / "chapter_references.json"
            util.save_json(saved, catalog)
            return saved
        monkeypatch.setattr(pipeline, "process_chapter", process)
        inputs = {"chapter_selection": {}}
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
        assert record["inputs"]["chapters"][0]["sha256"] == snapshots.file_digest(chapter)
    assert record["model_controls"]["top_p"] == 0.8


def test_failed_execution_does_not_claim_completed_results(setup, monkeypatch):
    root, config, _ = setup
    pipeline = lmstudio_pipeline.load("extract")
    def fail(*args):
        raise RuntimeError("secret-must-not-be-saved")
    monkeypatch.setattr(pipeline, "process_chapter", fail)
    with pytest.raises(RuntimeError):
        extract.ExtractChapterReferencesNode().run(config, {}, **defaults(extract.ExtractChapterReferencesNode))
    text = (root / config["run_folder"] / "chapter_catalogs/extract_configuration.json").read_text()
    assert "secret-must-not-be-saved" not in text
    assert json.loads(text)["status"] == "started"
    assert json.loads(text)["outputs"] == []


def test_mistral_records_ignored_qwen_controls_and_normalized_node_settings(setup, monkeypatch):
    root, config, _ = setup
    config.update(model_family="Mistral", thinking=True, qwen35_top_k=90)
    pipeline = lmstudio_pipeline.load("extract")
    def process(path, output, *args):
        saved = output / "chapter_references.json"
        util.save_json(saved, {"chapter_id": "chapter"})
        return saved
    monkeypatch.setattr(pipeline, "process_chapter", process)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mistral-test"))
    params = {**defaults(extract.ExtractChapterReferencesNode), "merge_batch_size": 1}
    extract.ExtractChapterReferencesNode().run(config, {}, **params)
    record = util.load_json(root / config["run_folder"] / "chapter_catalogs/extract_configuration.json")
    assert record["lmstudio_config"]["thinking"] is True
    assert record["model_controls"]["settings"] == {"thinking": False}
    assert record["model_controls"]["request_extra_body"] == {}
    assert record["model_controls"]["chatml_fallback_allowed"] is False
    assert record["node_settings"]["merge_batch_size"] == 2

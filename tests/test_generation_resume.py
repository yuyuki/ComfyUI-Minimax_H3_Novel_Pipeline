"""Resume planning and prompt generation without changing the previous run."""
from contextlib import nullcontext
from copy import deepcopy
from argparse import Namespace
import json
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import lmstudio_pipeline, path_access, util
from minimax_h3_novel_pipeline.nodes import generate_h3_prompts as node
from tests.test_configuration_snapshot import defaults


@pytest.fixture
def setup(tmp_path, monkeypatch, cinematic_registry):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    source = tmp_path / "chapter.txt"
    source.write_text("A character waits.", encoding="utf-8")
    monkeypatch.setattr(node, "selected_chapter_paths", lambda selection: [str(source)])
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock"))
    refs = cinematic_registry(source, schema_version=util.REGISTRY_SCHEMA,
                              entities=[], picture_assets=[], audio_assets=[])
    return tmp_path, refs


@pytest.mark.parametrize("change", [None, "force", "model", "temperature", "timeline", "bad_plan",
                                  "bad_prompt", "invalid_prompt", "prompt_key", "current_cache"])
def test_resume_plans_and_prompts(setup, monkeypatch, change):
    root, refs = setup
    step = lmstudio_pipeline.load("generate")
    plan = Mock(side_effect=lambda *a: [step.Scene("Wait", "Wait", "Wait", "", [], [], [], False, "")])
    generate = Mock(return_value="valid prompt")
    monkeypatch.setattr(step, "plan_scenes", plan)
    monkeypatch.setattr(step, "generate_prompt", generate)
    monkeypatch.setattr(step, "build_bindings", lambda *a: {"subjects": [{}], "audio": []})
    monkeypatch.setattr(step, "validate_prompt", lambda *a: step.Validation(True, [], 400))
    monkeypatch.setattr(step, "save_scene", lambda *a: {
        "subject_count": 1, "picture_count": 1, "audio_count": 0})
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261101000000"}
    params = defaults(node.GenerateH3PromptsNode)
    params.update(refine_camera=False)
    node.GenerateH3PromptsNode().run(refs, config, {}, **params)
    previous = root / config["run_folder"]
    cache = previous / params["out_dir"] / ".cache/chapter"
    if change in {"bad_plan", "bad_prompt"}:
        (cache / ("plan_001.json" if change == "bad_plan" else "prompt_001.json")).write_text("{broken")
    if change in {"invalid_prompt", "prompt_key"}:
        record = util.load_json(cache / "prompt_001.json")
        record["valid" if change == "invalid_prompt" else "cache_key"] = False
        util.save_json(cache / "prompt_001.json", record)
    before = {p.relative_to(previous): p.read_bytes() for p in previous.rglob("*") if p.is_file()}
    config["run_folder"] = "20261101000001"
    params["resume_run"] = " 20261101000000 "
    if change == "force":
        params["force"] = True
    if change == "temperature":
        params["temperature"] = 0.5
    if change == "model":
        monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "other"))
    if change == "timeline":
        refs = deepcopy(refs)
        refs["chapter_timelines"]["chapter"]["sequences"][0]["adaptation"]["event"] = "A changed event."
    current = root / config["run_folder"] / params["out_dir"]
    if change == "current_cache":
        util.save_json(current / ".cache/chapter/plan_001.json", {"cache_key": "stale"})
        util.save_json(current / ".cache/chapter/prompt_001.json", {"cache_key": "stale"})
    node.GenerateH3PromptsNode().run(refs, config, {}, **params)
    assert plan.call_count == (2 if change in {"force", "model", "timeline", "bad_plan"} else 1)
    assert generate.call_count == (2 if change in {
        "force", "model", "temperature", "timeline"} else 1)
    assert before == {p.relative_to(previous): p.read_bytes() for p in previous.rglob("*") if p.is_file()}
    assert (current / ".cache/chapter/plan_001.json").is_file()
    assert (current / ".cache/chapter/prompt_001.json").is_file()
    metrics = json.loads((current / ".cache/chapter/request_metrics.json").read_text())
    reasons = metrics["cache_reasons"]
    if change == "bad_plan":
        assert reasons["planning"]["unreadable_record"] == 1
    elif change == "bad_prompt":
        assert reasons["final_prompt"]["unreadable_record"] == 1
    elif change == "invalid_prompt":
        assert reasons["final_prompt"]["validation_failed"] == 1
    elif change == "prompt_key":
        assert reasons["final_prompt"]["dependencies_changed"] >= 1
    elif change == "force":
        assert all(reasons[stage]["force"] == 1 for stage in ("evidence", "planning", "final_prompt"))
    elif change is None:
        assert all(reasons[stage]["hit"] == 1 for stage in ("evidence", "planning", "final_prompt"))
    record = json.loads((current / "generate_configuration.json").read_text())
    assert record["status"] == "completed"
    assert record["node_settings"]["resume_run"] == previous.name
    assert "resume_cache_dir" not in record["node_settings"]
    # Every reused entry is saved locally, allowing a chain of resumed runs.
    plan.reset_mock()
    generate.reset_mock()
    config["run_folder"] = "20261101000002"
    params.update(resume_run="20261101000001", force=False)
    node.GenerateH3PromptsNode().run(refs, config, {}, **params)
    plan.assert_not_called()
    generate.assert_not_called()


@pytest.mark.parametrize("resume,match", [(123, "timestamped"), ("../outside", "Invalid run folder"),
    ("20261101000000", "current run"), ("20261101000001", "cache does not exist")])
def test_bad_resume_rejected_before_client_creation(setup, monkeypatch, resume, match):
    _, refs = setup
    client = Mock()
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", client)
    params = defaults(node.GenerateH3PromptsNode)
    params["resume_run"] = resume
    with pytest.raises(ValueError, match=match):
        node.GenerateH3PromptsNode().run(refs, {"run_folder": "20261101000000"}, {}, **params)
    client.assert_not_called()


def test_completed_scene_files_resume_and_incomplete_scene_does_not(tmp_path, monkeypatch):
    step = lmstudio_pipeline.load("generate")
    old, current = tmp_path / "old", tmp_path / "new"
    chapter, cache = old / "chapter", old / ".cache/chapter"
    chapter.mkdir(parents=True)
    cache.mkdir(parents=True)
    target = current / "chapter"
    target.mkdir(parents=True)
    (current / ".cache/chapter").mkdir(parents=True)
    scene = step.Scene("Wait", "Wait", "Wait", "", [], [], [], False, "")
    bindings = {"subjects": [], "audio": [], "picture_input_order": []}
    validation = step.Validation(True, [], 400)
    monkeypatch.setattr(step, "validate_prompt", lambda *args: validation)
    entry = step.save_scene(chapter, 1, scene, bindings, "prompt", validation)
    saved_scene = step.scene_to_dict(scene)
    record = {"contract": 2, "valid": True, "prompt": "prompt", "scene": saved_scene,
              "bindings": bindings, "camera_warnings": []}
    record["result_digest"] = step.generation_requests.digest(
        [record["prompt"], saved_scene, bindings, []])
    (cache / "prompt_001.json").write_text(json.dumps(record), encoding="utf-8")
    args = Namespace(duration=8, max_shots=1)
    assert step.resume_scene_files(chapter, target, cache, 1, scene, bindings, args) is not None
    for key in ("assets_file", "prompt_file", "source_file"):
        assert (target / entry[key]).read_bytes() == (chapter / entry[key]).read_bytes()
    assert (current / ".cache/chapter/prompt_001.json").is_file()
    (chapter / entry["source_file"]).unlink()
    assert step.resume_scene_files(chapter, target, cache, 1, scene, bindings, args) is None

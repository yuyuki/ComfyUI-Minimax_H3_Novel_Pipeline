"""Resume planning and prompt generation without changing the previous run."""
from contextlib import nullcontext
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
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock"))
    refs = cinematic_registry(source, schema_version=util.REGISTRY_SCHEMA,
                              entities=[], picture_assets=[], audio_assets=[])
    return tmp_path, refs




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
    record = {"contract": 3, "valid": True, "prompt": "prompt", "scene": saved_scene,
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

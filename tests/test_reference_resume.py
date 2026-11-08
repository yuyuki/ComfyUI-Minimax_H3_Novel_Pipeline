"""Resume interrupted reference consolidation without touching previous outputs."""
from contextlib import nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import lmstudio_pipeline, path_access, reference_requests, util
from minimax_h3_novel_pipeline.nodes import consolidate_references as node
from tests.test_configuration_snapshot import defaults


CONTRACT = {"name": "resume_test", "strict": True, "schema": {
    "type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["ids"], "additionalProperties": False}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    return tmp_path


def request(chat, args, prompt="chapter", model="mock"):
    def check(result):
        if result["ids"] != ["A", "B"]:
            raise ValueError("Missing independent observations")
    return reference_requests.validated_request(chat, None, model, "system", prompt, CONTRACT, args, check)


@pytest.mark.parametrize("change", [None, "prompt", "model", "temperature", "force", "malformed", "semantic", "schema"])
def test_request_cache_revalidates_and_invalidates(setup, change, capsys):
    old = setup / "20261101000000/references/.cache"
    current = setup / "20261101000001/references/.cache"
    args = SimpleNamespace(cache_dir=old, temperature=0.1, max_tokens=100, max_attempts=2, force=False)
    chat = Mock(return_value={"ids": ["A", "B"]})
    request(chat, args)
    saved, = old.glob("*.json")
    if change == "malformed":
        saved.write_text("{broken", encoding="utf-8")
    if change == "semantic":
        util.save_json(saved, {"ids": ["A"]})
    if change == "schema":
        util.save_json(saved, {"ids": ["A", "B"], "unexpected": True})
    before = saved.read_bytes()
    args.cache_dir, args.resume_cache_dir = current, old
    if change == "temperature":
        args.temperature = 0.2
    args.force = change == "force"
    assert request(chat, args, prompt="changed" if change == "prompt" else "chapter",
                   model="other" if change == "model" else "mock") == {"ids": ["A", "B"]}
    assert chat.call_count == (1 if change is None else 2)
    assert saved.read_bytes() == before
    assert len(list(current.glob("*.json"))) == 1
    log = capsys.readouterr().out
    if change is None:
        assert "cache hit (previous run)" in log
    elif change == "force":
        assert "resume cache bypassed (force=True)" in log
    elif change in {"malformed", "semantic", "schema"}:
        assert "cache rejected" in log
    else:
        assert "resume cache miss (no matching request)" in log


def test_interrupted_node_resumes_each_stage_and_writes_current_run(setup, monkeypatch):
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261101000000"}
    params = defaults(node.ConsolidateReferencesNode)
    chapter = {"schema_version": util.CHAPTER_SCHEMA, "chapter_id": "chapter", "chapter_name": "chapter",
               "timeline_version": "cinematic-reference-timeline.v1", "source": {},
               "characters": [], "locations": [], "objects": [],
               "sequences": [{"sequence": 1, "source": "Text", "adaptation": {
                   "initialState": "Opening", "event": "Action", "endingState": "Ending"}}]}
    pipeline = lmstudio_pipeline.load("consolidate")
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock"))
    calls = []
    interrupted = True

    def perform(stage, args):
        def chat(*values):
            calls.append(stage)
            assert values[4] == CONTRACT
            if stage == "audio" and interrupted:
                raise RuntimeError("interrupted")
            return {"ids": ["A", "B"]}
        request(chat, args, stage)
        return []

    monkeypatch.setattr(pipeline, "reconcile_chapter", lambda c, m, chapter, registry, args: perform("reconcile", args))
    monkeypatch.setattr(pipeline, "audit_registry", lambda c, m, registry, args: perform("audit", args))
    monkeypatch.setattr(node, "prepare_designs", lambda chat, c, m, registry, args, path:
                        {"entities": perform("designs", args)})
    monkeypatch.setattr(pipeline, "generate_picture_assets", lambda c, m, specs, args: perform("pictures", args))
    monkeypatch.setattr(pipeline, "generate_audio_assets", lambda c, m, specs, args: perform("audio", args))
    with pytest.raises(RuntimeError, match="interrupted"):
        node.ConsolidateReferencesNode().run([chapter], config, **params)
    previous = setup / config["run_folder"]
    before = {p.relative_to(previous): p.read_bytes() for p in previous.rglob("*") if p.is_file()}
    interrupted = False
    config["run_folder"] = "20261101000001"
    params["resume_run"] = previous.name
    node.ConsolidateReferencesNode().run([chapter], config, **params)
    assert calls == ["reconcile", "audit", "designs", "pictures", "audio", "audio"]
    assert before == {p.relative_to(previous): p.read_bytes() for p in previous.rglob("*") if p.is_file()}
    output = setup / config["run_folder"] / params["out_dir"]
    assert (output / ".cache/chapter").is_dir()
    assert (output / "consolidated_references.json").is_file()
    record = json.loads((output / "consolidate_configuration.json").read_text())
    assert record["status"] == "completed"
    assert record["node_settings"]["resume_run"] == previous.name
    assert "resume_cache_dir" not in record["node_settings"]


@pytest.mark.parametrize("resume,match", [(123, "timestamped"), ("../outside", "Invalid run folder"),
    ("20261101000000", "current run"), ("20261101000001", "cache does not exist")])
def test_bad_resume_rejected_before_client_creation(setup, monkeypatch, resume, match):
    from tests.test_reference_timeline import catalog
    client = Mock()
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", client)
    params = defaults(node.ConsolidateReferencesNode)
    params["resume_run"] = resume
    with pytest.raises(ValueError, match=match):
        node.ConsolidateReferencesNode().run([deepcopy(catalog())],
            {"run_folder": "20261101000000"}, **params)
    client.assert_not_called()

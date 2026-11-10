"""Checkpoint recovery must preserve exact requests and never skip evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import generation_requests as requests
from minimax_h3_novel_pipeline.core import lmstudio_json, lmstudio_model_gemma, lmstudio_pipeline, path_access
from tests.test_structured_json import SCHEMA, Stream, client_for


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    client = SimpleNamespace(base_url="http://127.0.0.1:1234/v1", api_key="never-save-this-key",
                             _minimax_h3_profile=lmstudio_model_gemma,
                             _minimax_h3_settings={"thinking": True})
    args = SimpleNamespace(out_dir=tmp_path / "first" / "h3", force=False)
    return client, args


def call(client, **changes):
    values = dict(model="gemma", system="instructions", user="all evidence", schema=deepcopy(SCHEMA),
                  temperature=0.1, max_tokens=8192)
    values.update(changes)
    return requests.chat_json(client, **values)


def test_carry_forward_immutable_checkpoints_and_revalidate_schema(setup, monkeypatch):
    client, args = setup
    network = Mock(side_effect=lambda *a: {"value": "draft"})
    monkeypatch.setattr(lmstudio_json, "chat_json", network)
    with requests.chapter("chapter", args), requests.task("generation"):
        result = call(client)
        result["value"] = "caller mutation"
        assert call(client) == {"value": "draft"}
    old = args.out_dir / ".cache"
    before = {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    args.out_dir = args.out_dir.parent.parent / "second" / "h3"
    args.resume_cache_dir = old
    with requests.chapter("chapter", args), requests.task("generation"):
        assert call(client) == {"value": "draft"}
    assert network.call_count == 1
    assert before == {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    files = list((args.out_dir / ".cache/chapter/requests").glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    record["result"] = {"unexpected": "field"}
    record["result_digest"] = requests.digest(record["result"])
    files[0].write_text(json.dumps(record))
    del args.resume_cache_dir
    with requests.chapter("chapter", args):
        assert call(client) == {"value": "draft"}
    assert network.call_count == 2
    assert "never-save-this-key" not in "".join(p.read_text() for p in args.out_dir.rglob("*.json"))


@pytest.mark.parametrize("change", ["user", "system", "schema", "temperature", "max_tokens", "model",
                                  "thinking", "endpoint", "force", "version"])
def test_exact_request_dependencies_invalidate(setup, monkeypatch, change):
    client, args = setup
    network = Mock(return_value={"value": "draft"})
    monkeypatch.setattr(lmstudio_json, "chat_json", network)
    with requests.chapter("chapter", args):
        call(client)
    values = {}
    if change in {"user", "system", "model"}:
        values[change] = "changed"
    elif change == "schema":
        values[change] = {**SCHEMA, "name": "new_policy"}
    elif change == "temperature":
        values[change] = 0.2
    elif change == "max_tokens":
        values[change] = 16384
    elif change == "thinking":
        client._minimax_h3_settings["thinking"] = False
    elif change == "endpoint":
        client.base_url = "http://127.0.0.1:9999/v1"
    elif change == "force":
        args.force = True
    elif change == "version":
        monkeypatch.setattr(requests, "VERSION", requests.VERSION + 1)
    with requests.chapter("chapter", args):
        call(client, **values)
    assert network.call_count == 2


def test_review_override_retries_with_schema_and_preserves_generation_thinking(setup):
    _, args = setup
    client, network = client_for(Stream(['{']), Stream(['{"value":"review"}']), Stream(['{"value":"generation"}']))
    client._minimax_h3_profile = lmstudio_model_gemma
    client._minimax_h3_settings = {"thinking": True}
    with requests.chapter("chapter", args):
        with requests.task("continuity_review", False):
            assert call(client)["value"] == "review"
            assert call(client)["value"] == "review"
        with requests.task("generation"):
            assert call(client)["value"] == "generation"
    assert network.call_count == 3
    assert [c.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"]
            for c in network.call_args_list] == [False, False, True]
    assert all(c.kwargs["response_format"] == {"type": "json_schema", "json_schema": SCHEMA}
               for c in network.call_args_list)
    assert client._minimax_h3_settings == {"thinking": True}
    assert lmstudio_json.REQUEST_THINKING.get() is None
    metrics = json.loads((args.out_dir / ".cache/chapter/request_metrics.json").read_text())
    assert metrics["tasks"]["continuity_review"]["llm_requests"] == 2
    assert metrics["tasks"]["continuity_review"]["cache_hits"] == 1


@pytest.mark.parametrize("failure", [RuntimeError("failure"), lmstudio_json.StructuredOutputLengthError("length")])
def test_failed_response_is_not_cached_and_overrides_restore(setup, monkeypatch, failure):
    client, args = setup
    network = Mock(side_effect=[failure, {"value": "recovered"}])
    monkeypatch.setattr(lmstudio_json, "chat_json", network)
    with pytest.raises(type(failure)), requests.chapter("chapter", args), requests.task("camera", False):
        call(client)
    assert lmstudio_json.REQUEST_THINKING.get() is None
    assert lmstudio_json.REQUEST_METRICS.get() is None
    assert not list(args.out_dir.rglob("requests/*.json"))
    with requests.chapter("chapter", args), requests.task("camera", False):
        assert call(client)["value"] == "recovered"
    assert network.call_count == 2


def test_stop_is_checked_on_cache_hit(setup, monkeypatch):
    client, args = setup
    network = Mock(return_value={"value": "draft"})
    monkeypatch.setattr(lmstudio_json, "chat_json", network)
    with requests.chapter("chapter", args):
        call(client)
    class InterruptProcessingException(Exception):
        pass
    monkeypatch.setattr(lmstudio_pipeline, "comfy_interrupt_check", Mock(side_effect=InterruptProcessingException))
    with pytest.raises(InterruptProcessingException), requests.chapter("chapter", args):
        call(client)
    assert network.call_count == 1


def test_inflight_metrics_and_scoped_details_do_not_change_cache_identity(setup, monkeypatch):
    client, args = setup
    path = args.out_dir / ".cache/chapter/request_metrics.json"
    def network(*a):
        metrics = json.loads(path.read_text())
        assert metrics["last_status"] == "requesting"
        assert metrics["detail"] == {"review_round": 2, "batch": 3}
        assert metrics["request_started_at"] > 0
        assert metrics["request_bytes"] > 0
        return {"value": "review"}
    monkeypatch.setattr(lmstudio_json, "chat_json", network)
    with requests.chapter("chapter", args), requests.task("continuity_review"):
        with requests.detail(review_round=2), requests.detail(batch=3):
            call(client)
        call(client)
    metrics = json.loads(path.read_text())
    assert metrics["last_status"] == "cached"
    assert metrics["detail"] == {}
    assert metrics["tasks"]["continuity_review"]["cache_misses"] == 1
    assert metrics["tasks"]["continuity_review"]["cache_hits"] == 1
    assert "never-save-this-key" not in path.read_text()


def test_review_failure_resumes_generation_and_each_completed_evidence_batch(setup, monkeypatch, cinematic_registry):
    from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step, continuity_context
    client, args = setup
    args.__dict__.update(max_scenes=0, duration=8, max_shots=1, max_tokens=8192, repair_attempts=0,
                         refine_camera=True, review_thinking=False, camera_thinking=False,
                         spatial_plan={"chapters": {"chapter": {"sequences": [{"sequence": 1}]}}})
    path = args.out_dir.parent / "chapter.txt"
    refs = cinematic_registry(path)
    scenes = [step.Scene(str(i), "evidence " * (300 + i), f"event {i}", "", [], [], [], False, "") for i in range(4)]
    monkeypatch.setattr(continuity_context, "REQUEST_BYTES", 11000)
    monkeypatch.setattr(step, "plan_scenes", lambda *a: deepcopy(scenes))
    monkeypatch.setattr(step, "build_bindings", lambda *a: {"subjects": [{"global_id": "hero"}], "audio": []})
    monkeypatch.setattr(step, "generate_prompt", lambda c, m, *a: step.chat_json(c, m, "generate", "input", step.PROMPT_SCHEMA, .1, 8192)["prompt_text"])
    def camera(c, m, prompt, *a):
        step.chat_json(c, m, "camera", prompt, step.CAMERA_SCHEMA, .1, 1400)
        return prompt + " camera", []
    monkeypatch.setattr(step, "refine_camera_prompt", camera)
    monkeypatch.setattr(step, "validate_prompt", lambda *a: step.Validation(True, [], 400))
    monkeypatch.setattr(step, "save_scene", lambda *a: {"subject_count": 1, "picture_count": 1, "audio_count": 0})
    seen, reviews = [], []
    fail = True
    def network(c, m, system, user, schema, temperature, max_tokens):
        nonlocal fail
        seen.append(schema["name"])
        if schema == step.PROMPT_SCHEMA:
            assert lmstudio_json.model_settings(c, m)[1]["thinking"] is True
            return {"prompt_text": "draft"}
        assert lmstudio_json.model_settings(c, m)[1]["thinking"] is False
        if schema == step.CAMERA_SCHEMA:
            return {"shot_cameras": ["Fixed framing."]}
        reviews.append(json.loads(user))
        if len(reviews) == 2 and fail:
            fail = False
            raise lmstudio_json.StructuredOutputLengthError("reasoning exhausted")
        return {"errors": []}
    monkeypatch.setattr(lmstudio_json, "chat_json", network)
    with pytest.raises(lmstudio_json.StructuredOutputLengthError):
        step.process_chapter(path, refs, client, "gemma", args)
    assert len(reviews) == 2
    assert not (args.out_dir / ".cache/chapter/prompt_001.json").exists()
    args.resume_cache_dir = args.out_dir / ".cache"
    args.out_dir = args.out_dir.parent.parent / "second" / "h3"
    result = step.process_chapter(path, refs, client, "gemma", args)
    assert len(result["outputs"]) == 4
    assert seen.count(step.PROMPT_SCHEMA["name"]) == 1
    assert seen.count(step.CAMERA_SCHEMA["name"]) == 1
    # Batch one was recovered, batch two was retried with exactly the same evidence.
    assert reviews[1] == reviews[2]
    assert reviews.count(reviews[0]) == 1
    before = len(seen)
    step.process_chapter(path, refs, client, "gemma", args)
    assert len(seen) == before  # Complete cached prompts still validate without LLM calls.
    # Changing future evidence invalidates the final prompt and affected review requests.
    plan_path = args.out_dir / ".cache/chapter/plan_001.json"
    plan_record = json.loads(plan_path.read_text())
    plan_record["scenes"][-1]["source_excerpt"] = "Changed future evidence."
    plan_path.write_text(json.dumps(plan_record))
    step.process_chapter(path, refs, client, "gemma", args)
    assert len(seen) > before
    assert any(item["source_excerpt"] == "Changed future evidence."
               for review in reviews for item in review["future_requirements"])

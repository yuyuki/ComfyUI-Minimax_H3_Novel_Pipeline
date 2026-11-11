"""Stage reuse never substitutes for final format or complete evidence checks."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import generation_artifacts as artifacts, generation_requests as requests
from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step, path_access, lmstudio_json, continuity_context


@pytest.fixture
def scope(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    return SimpleNamespace(out_dir=tmp_path / "first/h3", force=False)


def test_artifact_resume_is_detached_integrity_checked_and_forceable(scope, tmp_path):
    compute = Mock(return_value={"prompt": "draft"})
    def run():
        with requests.chapter("chapter", scope):
            return artifacts.cached("draft", {"input": "source"}, compute, lambda r: r == {"prompt": "draft"})
    first = run()
    first["prompt"] = "mutation"
    old = scope.out_dir / ".cache"
    before = {p: p.read_bytes() for p in old.rglob("artifacts/**/*.json")}
    scope.resume_cache_dir = old
    scope.out_dir = tmp_path / "second/h3"
    assert run() == {"prompt": "draft"}
    assert compute.call_count == 1
    assert all(path.read_bytes() == data for path, data in before.items())
    del scope.resume_cache_dir
    record_path = next(scope.out_dir.rglob("artifacts/draft/*.json"))
    record = json.loads(record_path.read_text())
    record["result"]["prompt"] = "corrupt"
    record_path.write_text(json.dumps(record))
    assert run() == {"prompt": "draft"}
    assert compute.call_count == 2
    scope.force = True
    run()
    assert compute.call_count == 3


def test_failure_or_invalid_result_never_becomes_an_artifact(scope):
    with requests.chapter("chapter", scope):
        with pytest.raises(RuntimeError):
            artifacts.cached("draft", {}, Mock(side_effect=RuntimeError()), lambda r: True)
        assert artifacts.cached("format", {}, lambda: "invalid", lambda r: False) == "invalid"
    assert not list(scope.out_dir.rglob("artifacts/**/*.json"))


def test_source_fingerprint_tracks_transitive_helpers_not_other_stages(tmp_path):
    file = tmp_path / "sample.py"
    file.write_text('VALUE = 1\ndef helper(): return VALUE\ndef draft(): return helper()\ndef camera(): return 1\n')
    before = artifacts.implementation(file, ["draft"])
    file.write_text(file.read_text().replace('def camera(): return 1', 'def camera(): return 2'))
    assert artifacts.implementation(file, ["draft"]) == before
    file.write_text(file.read_text().replace('VALUE = 1', 'VALUE = 2'))
    assert artifacts.implementation(file, ["draft"]) != before


@pytest.mark.parametrize("change,drafts,cameras", [
    ("review_thinking", 1, 1), ("camera_thinking", 1, 2),
    ("temperature", 2, 1), ("duration", 2, 2), ("force", 2, 2),
])
def test_settings_recompute_only_affected_stages(scope, monkeypatch, cinematic_registry, change, drafts, cameras):
    chapter = scope.out_dir.parent / "chapter.txt"
    refs = cinematic_registry(chapter)
    scope.__dict__.update(duration=8, max_shots=1, max_scenes=0, max_tokens=8192,
                          repair_attempts=1, refine_camera=True, camera_thinking=False,
                          review_thinking=False, temperature=.38,
                          spatial_plan={"chapters": {"chapter": {"sequences": [{"sequence": 1}]}}})
    monkeypatch.setattr(step, "plan_scenes", lambda *a: [step.Scene("Wait", "Source", "Wait", "", [], [], [], False, "")])
    monkeypatch.setattr(step, "build_bindings", lambda *a: {"subjects": [{"global_id": "hero"}], "audio": []})
    generation, camera = Mock(return_value="draft"), Mock(return_value=("camera wording", []))
    monkeypatch.setattr(step, "generate_prompt", generation)
    monkeypatch.setattr(step, "refine_camera_prompt", camera)
    monkeypatch.setattr(step, "validate_prompt", lambda *a: step.Validation(True, [], 400))
    monkeypatch.setattr(step, "save_scene", lambda *a: {"subject_count": 1, "picture_count": 1, "audio_count": 0})
    network = Mock(return_value={"errors": []})
    monkeypatch.setattr(lmstudio_json, "chat_json", network)
    step.process_chapter(chapter, refs, None, "gemma", scope)
    setattr(scope, change, {"temperature": .7, "duration": 12}.get(change, True))
    result = step.process_chapter(chapter, refs, None, "gemma", scope)
    assert generation.call_count == drafts
    assert camera.call_count == cameras
    assert result["outputs"][0]["continuity_review"]["valid"]
    if change == "review_thinking":
        assert network.call_count == 2


def test_missing_or_reordered_evidence_cannot_have_complete_coverage(scope):
    groups = {"future": [{"id": 1}, {"id": 1}, {"id": 2}]}
    with requests.chapter("chapter", scope):
        for rows in ([{"id": 1}, {"id": 2}], [{"id": 2}, {"id": 1}, {"id": 1}]):
            with pytest.raises(ValueError, match="every evidence occurrence"):
                artifacts.ReviewCoverage({}, {}, groups, [{"future": rows}])
        coverage = artifacts.ReviewCoverage({}, {}, groups, [groups])
        with pytest.raises(ValueError, match="Incomplete"):
            coverage.finish()
        coverage.completed(0, {"errors": ["conflict"]}, ["conflict"])
        coverage.finish()
        assert coverage.record["status"] == "completed" and not coverage.record["approved"]


def test_review_resume_and_changed_evidence_reuse_only_exact_batches(scope, monkeypatch):
    monkeypatch.setattr(continuity_context, "REQUEST_BYTES", 14000)
    current = step.Scene("Current", "Source", "Wait", "", [], [], [], False, "")
    future = [step.Scene(str(i), str(i) * 6000, str(i), "", [], [], [], False, "") for i in range(7)]
    calls = []
    fail = True
    def network(c, m, system, user, *unused):
        nonlocal fail
        calls.append(json.loads(user))
        if len(calls) == 2 and fail:
            fail = False
            raise RuntimeError("interrupted")
        return {"errors": []}
    monkeypatch.setattr(lmstudio_json, "chat_json", network)
    def review(prompt="wording"):
        with requests.chapter("chapter", scope), requests.task("continuity_review", False):
            return step.check_prompt_continuity(None, "gemma", current, prompt, {}, {}, [], future, 8192)
    with pytest.raises(RuntimeError):
        review()
    paths = list(scope.out_dir.rglob("reviews/*.json"))
    failed = json.loads(paths[0].read_text())
    assert failed["status"] == "failed" and not failed["approved"]
    assert [b["status"] for b in failed["batches"]].count("completed") == 1
    assert review() == []
    complete = json.loads(paths[0].read_text())
    assert complete["approved"] and complete["status"] == "completed"
    assert complete["expected"]["future_requirements"] == [requests.digest(step.scene_to_dict(s)) for s in future]
    count = len(calls)
    assert review() == [] and len(calls) == count
    future[-1].source_excerpt = "z" * 6000
    assert review() == [] and len(calls) == count + 1
    count = len(calls)
    review("changed wording")
    assert len(calls) == count + len(complete["batches"])
    metrics = json.loads((scope.out_dir / ".cache/chapter/request_metrics.json").read_text())
    assert metrics["review_coverage"]["approved"]
    assert metrics["review_coverage"]["completed_batches"] == metrics["review_coverage"]["required_batches"]


def test_corrupt_manifest_never_bypasses_review(scope, monkeypatch):
    current = step.Scene("Current", "Source", "Wait", "", [], [], [], False, "")
    network = Mock(return_value={"errors": []})
    monkeypatch.setattr(lmstudio_json, "chat_json", network)
    with requests.chapter("chapter", scope):
        step.check_prompt_continuity(None, "gemma", current, "wording", {}, {}, [], [], 8192)
    path = next(scope.out_dir.rglob("reviews/*.json"))
    original = deepcopy(json.loads(path.read_text()))
    path.write_text('{"approved": true, "batches": []}')
    with requests.chapter("chapter", scope):
        step.check_prompt_continuity(None, "gemma", current, "wording", {}, {}, [], [], 8192)
    assert json.loads(path.read_text()) == original
    assert network.call_count == 1  # Exact response reused, evidence checked again.

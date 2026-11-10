"""Quality gates must avoid work without accepting unreviewed candidates."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step
from tests.test_h3_prompt_validation import BINDINGS, prompt_with_shots


@pytest.mark.parametrize("field,value,invalidates", [
    ("camera_thinking", True, False), ("review_thinking", True, False),
    ("camera_direction", "Track left", False), ("refine_camera", True, False),
    ("temperature", .8, False), ("max_tokens", 32768, False),
    ("max_scenes", 3, False), ("duration", 12, True),
    ("max_shots", 2, True), ("scenes_per_chunk", 2, True),
    ("repair_attempts", 3, True),
])
def test_planning_cache_depends_only_on_consumed_settings(field, value, invalidates):
    args = SimpleNamespace()
    implementation = step.planning_implementation()
    def key():
        return step.planning_key(None, "gemma", args, "chapter", "exact source", 1, 1, [], implementation)
    before = key()
    setattr(args, field, value)
    assert (key() != before) is invalidates


def test_planning_cache_tracks_source_catalog_and_implementation():
    args = SimpleNamespace()
    inputs = [None, "gemma", args, "chapter", "source", 1, 1, [], {"planner": "v1"}]
    before = step.planning_key(*inputs)
    for index, replacement in [(1, "other-model"), (4, "changed source"),
                               (7, [{"global_id": "new"}]), (8, {"planner": "v2"})]:
        changed = list(inputs)
        changed[index] = replacement
        assert step.planning_key(*changed) != before


@pytest.mark.parametrize("tasks,audio,changed", [
    ("reference generation + keyframe completion", [], True),
    ("reference generation + audio reference", [{}], True),
    ("reference generation + audio reference", [], False),
    ("reference generation", [{}], False),
    ("reference generation of a quiet scene", [], False),
    ("reference generation + unknown", [], False),
])
def test_only_unambiguous_binding_consistent_prefix_is_normalized(tasks, audio, changed):
    original = prompt_with_shots("[Shot 1] A figure pauses.").replace(
        "[reference generation] A quiet scene.", tasks)
    result = step.normalize_prompt(original, {"audio": audio})
    assert result == (original.replace(tasks, f"[{tasks}]") if changed else original)
    assert step.normalize_prompt(result, {"audio": audio}) == result


def test_malformed_draft_is_repaired_before_any_semantic_review(monkeypatch):
    valid = prompt_with_shots("[Shot 1] A figure pauses.")
    scene = step.Scene("Wait", "Wait", "Wait", "", [], [], [], False, "")
    events = []
    def repair(*args):
        events.append("repair")
        return valid
    def review(*args):
        assert args[3] == valid
        events.append("review")
        return []
    monkeypatch.setattr(step, "repair_prompt", repair)
    monkeypatch.setattr(step, "check_prompt_continuity", review)
    result = step.review_and_repair_continuity(None, "gemma", scene, "malformed", BINDINGS,
                                              {}, [], [], SimpleNamespace(repair_attempts=1, duration=8, max_tokens=8192))
    assert events == ["repair", "review"]
    assert result[1].ok


def test_failed_semantic_repair_cannot_trigger_review_or_hide_original_errors(monkeypatch):
    valid = prompt_with_shots("[Shot 1] A figure pauses.")
    scene = step.Scene("Wait", "Wait", "Wait", "", [], [], [], False, "")
    review = Mock(return_value=["Wrong eyeline"])
    monkeypatch.setattr(step, "check_prompt_continuity", review)
    monkeypatch.setattr(step, "repair_prompt", Mock(return_value="malformed"))
    prompt, validation, errors, _ = step.review_and_repair_continuity(
        None, "gemma", scene, valid, BINDINGS, {}, [], [],
        SimpleNamespace(repair_attempts=1, duration=8, max_tokens=8192))
    assert review.call_count == 1
    assert prompt == valid
    assert not validation.ok
    assert errors == ["Wrong eyeline"]


def test_unchanged_repair_stops_without_approving(monkeypatch):
    valid = prompt_with_shots("[Shot 1] A figure pauses.")
    scene = step.Scene("Wait", "Wait", "Wait", "", [], [], [], False, "")
    review = Mock(return_value=["Wrong eyeline"])
    repair = Mock(return_value=valid)
    monkeypatch.setattr(step, "check_prompt_continuity", review)
    monkeypatch.setattr(step, "repair_prompt", repair)
    _, validation, _, attempts = step.review_and_repair_continuity(
        None, "gemma", scene, valid, BINDINGS, {}, [], [],
        SimpleNamespace(repair_attempts=10, duration=8, max_tokens=8192))
    assert review.call_count == repair.call_count == attempts == 1
    assert not validation.ok


def test_chapter_does_not_review_exhausted_format_failures(tmp_path, monkeypatch, cinematic_registry):
    from minimax_h3_novel_pipeline.core import path_access
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    chapter = tmp_path / "chapter.txt"
    args = SimpleNamespace(out_dir=tmp_path / "output", force=False, max_scenes=0,
                           duration=8, max_tokens=8192, repair_attempts=1, refine_camera=True,
                           spatial_plan={"chapters": {"chapter": {"sequences": [{"sequence": 1}]}}})
    monkeypatch.setattr(step, "plan_scenes", lambda *a: [step.Scene("Wait", "Wait", "Wait", "", [], [], [], False, "")])
    monkeypatch.setattr(step, "build_bindings", lambda *a: {"subjects": [{"global_id": "hero"}], "audio": []})
    monkeypatch.setattr(step, "generate_prompt", lambda *a: "malformed")
    repair = Mock(return_value="still malformed")
    monkeypatch.setattr(step, "repair_prompt", repair)
    monkeypatch.setattr(step, "validate_prompt", lambda *a: step.Validation(False, ["Missing sections"], 0))
    review, camera = Mock(), Mock()
    monkeypatch.setattr(step, "review_and_repair_continuity", review)
    monkeypatch.setattr(step, "refine_camera_prompt", camera)
    monkeypatch.setattr(step, "save_scene", lambda *a: {})
    result = step.process_chapter(chapter, cinematic_registry(chapter), None, "gemma", args)
    assert repair.call_count == 1
    review.assert_not_called()
    camera.assert_not_called()
    assert result["outputs"][0]["continuity_review"]["status"] == "blocked_by_format"
    assert not result["outputs"][0]["continuity_review"]["valid"]

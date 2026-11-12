from tests.cinematic_generation_helpers import chapter_input
"""One authoritative cinematic sequence per generated H3 scene."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step, continuity_issues, path_access
from minimax_h3_novel_pipeline.nodes.generate_h3_prompts import GenerateH3PromptsNode


def metadata():
    return dict(dialogue_present=True, visible_entity_ids=["hero"], speaking_entity_ids=["voice"],
                location_global_id="", reference_view_requests=[])


def catalog():
    return [{"global_id": gid, "entity_type": "character", "available_picture_views": []}
            for gid in ("hero", "voice")]


def test_socket_has_only_cinematic_contract():
    inputs = GenerateH3PromptsNode.INPUT_TYPES()["required"]
    assert inputs["cinematic_chapter"][0] == "MINIMAX_CINEMATIC_CHAPTERS"
    assert "chapter_selection" not in inputs and "scenes_per_chunk" not in inputs
    assert "duration" not in inputs
    assert not hasattr(step, "plan_scenes") and not hasattr(step, "dedupe_scenes")


@pytest.mark.parametrize("duration", [None, True, 3, 16, float("nan")])
def test_invalid_duration_fails_before_model(tmp_path, monkeypatch, cinematic_registry, duration):
    from minimax_h3_novel_pipeline.nodes import generate_h3_prompts as node
    from minimax_h3_novel_pipeline.core import util
    refs = cinematic_registry(tmp_path / "chapter.txt", schema_version=util.REGISTRY_SCHEMA)
    chapter = chapter_input(refs)
    chapter["duration"] = duration
    monkeypatch.setattr(node, "stage_output", lambda *a: tmp_path)
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model",
                        lambda *a: pytest.fail("Invalid duration must not call LM Studio"))
    with pytest.raises(ValueError, match="duration"):
        GenerateH3PromptsNode().run(refs, {}, [chapter], "h3")


def test_metadata_keeps_phases_source_and_offscreen_dialogue(monkeypatch):
    sequence = {"sequence": 7, "source": "Original exact source.", "adaptation": {
        "initialState": "Indy hangs.", "event": "Doriane calls offscreen: Jones !", "endingState": "Indy looks up."}}
    row = {"sequence": sequence, "catalog": catalog(), "character_context": None}
    original = deepcopy(row)
    calls = []
    def request(client, model, system, user, schema, temperature, max_tokens):
        calls.append(user)
        assert schema["strict"] and max_tokens == 16384
        assert "Original exact source." in user and "Jones !" in user
        result = metadata()
        if len(calls) == 1:
            result["dialogue_present"] = False
        Draft202012Validator(schema["schema"]).validate(result)
        return result
    monkeypatch.setattr(step, "chat_json", request)
    scene = step.prepare_cinematic_scene(None, "model", row, "1922", SimpleNamespace(max_tokens=16384, repair_attempts=1))
    assert len(calls) == 2
    assert scene.sequence == 7 and scene.source_excerpt == sequence["source"]
    assert scene.visual_event == sequence["adaptation"]["event"]
    assert all(text in scene.adaptation_notes for text in sequence["adaptation"].values())
    assert scene.dialogue_present and scene.speaking_entity_ids == ["voice"]
    assert scene.visible_entity_ids == ["hero"] and row == original


def test_schema_only_offers_resolvable_pointers_and_preserves_all_container_evidence():
    payload = {"current_scene": {"adaptation_notes": "Exact notes", "dialogue_present": False},
               "current_prompt_sections": {"detailed_description": "Exact prompt"},
               "future_requirements": [{"adaptation_notes": "Future A"}, {"adaptation_notes": "Future B"}],
               "bindings": {"subjects": [{"global_id": "hero"}]}}
    schema = continuity_issues.review_schema(payload)
    paths = schema["schema"]["properties"]["errors"]["items"]["properties"]["evidence_paths"]["items"]["enum"]
    for path in paths:
        continuity_issues.resolve(payload, path)
    assert "/current_scene/adaptation_notes" in paths
    assert "/current_prompt_sections/adaptation_notes" not in paths
    result = {"errors": [{"target": "scene_plan", "message": "Dialogue exists", "evidence_paths": ["/current_scene/adaptation_notes"]}]}
    assert Draft202012Validator(schema["schema"]).is_valid(result)
    result["errors"][0]["evidence_paths"] = ["/current_prompt_sections/adaptation_notes"]
    assert not Draft202012Validator(schema["schema"]).is_valid(result)
    with pytest.raises(continuity_issues.UnresolvedEvidenceError):
        continuity_issues.validate(result, payload)


def test_one_scene_per_sequence_retains_repetitions_and_future_when_capped(tmp_path, monkeypatch, cinematic_registry):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    refs = cinematic_registry(tmp_path / "never_read.txt")
    first = refs["chapter_timelines"]["chapter"]["sequences"][0]
    refs["chapter_timelines"]["chapter"]["sequences"] = [{**deepcopy(first), "sequence": i} for i in (1, 3, 8)]
    monkeypatch.setattr(step, "sequence_catalog", lambda *a: catalog())
    monkeypatch.setattr(step, "chat_json", lambda *a: metadata())
    monkeypatch.setattr(step, "build_bindings", lambda *a: {"subjects": [{"global_id": "hero"}], "audio": []})
    monkeypatch.setattr(step, "generate_prompt", lambda *a: "prompt")
    monkeypatch.setattr(step, "validate_prompt", lambda *a: step.Validation(True, [], 400))
    monkeypatch.setattr(step, "save_scene", lambda *a: {"subject_count": 1, "picture_count": 0, "audio_count": 0})
    reviewed = []
    def review(client, model, scene, prompt, bindings, context, previous, future, *a):
        reviewed.append((scene.sequence, [s.sequence for s in future]))
        return []
    monkeypatch.setattr(step, "check_prompt_continuity", review)
    spatial = {"chapters": {"chapter": {"sequences": [{"sequence": i} for i in (1, 3, 8)]}}}
    args = SimpleNamespace(out_dir=tmp_path / "first/h3", force=False, max_scenes=1, duration=8,
                           max_tokens=8192, repair_attempts=1, temperature=.38, spatial_plan=spatial)
    result = step.process_chapter(chapter_input(refs), refs, None, "model", args)
    assert result["scene_count"] == 1 and reviewed == [(1, [3, 8])]
    args.max_scenes = 0
    result = step.process_chapter(chapter_input(refs), refs, None, "model", args)
    assert [r["sequence"] for r in result["outputs"]] == [1, 3, 8]
    assert not list(args.out_dir.rglob("plan_*.json"))
    assert not (tmp_path / "never_read.txt").exists()


@pytest.mark.parametrize("field", ["source", "adaptation", "sequence"])
def test_changed_cinematic_input_rejects_stale_registry(tmp_path, cinematic_registry, field):
    refs = cinematic_registry(tmp_path / "chapter.txt")
    chapter = chapter_input(refs)
    chapter["sequences"][0][field] = {"source": "Changed", "sequence": 2,
        "adaptation": {"initialState": "Before", "event": "Changed", "endingState": "After"}}[field]
    with pytest.raises(ValueError, match="differs from consolidated"):
        step.selected_timeline(refs, chapter)

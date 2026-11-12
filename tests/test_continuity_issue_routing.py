from tests.cinematic_generation_helpers import chapter_input, metadata_scene
"""Evidence-linked repairs preserve beats, approved inputs and cache validity."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import continuity_issues as issues
from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step, path_access, lmstudio_json, lmstudio_model_gemma
from tests.h3_review_helpers import issue
from tests.test_structured_json import Stream, client_for


def scene():
    return step.Scene("Call", "He shouts for help.", "He calls for help", "", ["hero"], [], [], False, "Only the call.", 1)


def row():
    return {"catalog": [{"global_id": "hero", "entity_type": "character", "available_picture_views": [{"view_type": "face_front"}]}],
            "character_context": None, "sequence": {"sequence": 1, "source": "He shouts for help."}}


def metadata():
    return {"dialogue_present": True, "visible_entity_ids": ["hero"], "speaking_entity_ids": ["hero"],
            "location_global_id": "", "reference_view_requests": []}


def test_review_captures_exact_cited_evidence():
    payload = {"current_scene": {"visual_event": "Call"}, "future_requirements": [{"visual_event": "Fall"}]}
    result = issues.validate({"errors": [issue("Do not fall yet", path="/future_requirements/0/visual_event")]}, payload)
    payload["future_requirements"][0]["visual_event"] = "mutated"
    assert result[0]["evidence"] == [{"path": "/future_requirements/0/visual_event", "value": "Fall"}]


@pytest.mark.parametrize("bad", [
    "old string error", {**issue(), "target": "arbitrary"}, {**issue(), "message": " "},
    {**issue(), "evidence_paths": []}, issue(path="/missing"), issue(path="/list/-1"),
    issue(path="/list/9"), issue(path="relative"),
])
def test_invalid_review_or_unresolved_evidence_is_not_success(bad):
    with pytest.raises(ValueError, match="invalid scene continuity review"):
        issues.validate({"errors": [bad]}, {"current_prompt": "draft", "list": ["fact"]})


def test_evidence_pointer_cannot_continue_through_a_scalar():
    payload = {"current_scene": {"visual_event": "He calls for help."}}
    with pytest.raises(issues.UnresolvedEvidenceError, match="unresolved evidence path") as error:
        issues.validate(
            {"errors": [issue(path="/current_scene/visual_event/description")]}, payload)
    assert error.value.paths == ("/current_scene/visual_event/description",)


def test_prompt_section_citation_is_rejected_without_mutating_response():
    payload = {"current_prompt": "Full prompt", "current_prompt_sections": {
        "detailed_description": "The rope is visible."}}
    response = {"errors": [issue("Include the rope", path="/current_prompt/detailed_description")]}
    original = deepcopy(response)
    with pytest.raises(issues.UnresolvedEvidenceError):
        issues.validate(response, payload)
    assert response == original


@pytest.mark.parametrize("path", [
    "/current_prompt/missing", "/current_prompt/detailed_description/extra",
    "/other_prompt/detailed_description",
])
def test_prompt_section_correction_does_not_guess_other_paths(path):
    payload = {"current_prompt": "Full prompt", "current_prompt_sections": {
        "detailed_description": "The rope is visible."}}
    with pytest.raises(issues.UnresolvedEvidenceError) as error:
        issues.validate({"errors": [issue(path=path)]}, payload)
    assert error.value.paths == (path,)


def test_metadata_repairs_use_strict_schema_on_initial_request_and_transport_retry():
    response = metadata()
    client, network = client_for(Stream(['{']), Stream([json.dumps(response)]))
    client._minimax_h3_profile = lmstudio_model_gemma
    client._minimax_h3_settings = {"thinking": True}
    original = step.scene_to_dict(scene())
    corrected = issues.repair_metadata(client, "gemma", original, row(),
                                       [issue("Dialogue is present", "scene_plan")],
                                       SimpleNamespace(max_tokens=8192), lmstudio_json.chat_json)
    assert corrected["dialogue_present"]
    for key in ("title", "source_excerpt", "visual_event", "adaptation_notes", "sequence"):
        assert corrected[key] == original[key]
    assert original["dialogue_present"] is False
    assert network.call_count == 2
    for call in network.call_args_list:
        assert call.kwargs["response_format"]["type"] == "json_schema"
        assert call.kwargs["response_format"]["json_schema"] == issues.metadata_schema(row()["catalog"])


@pytest.mark.parametrize("change", ["unknown_id", "unavailable_view", "offscreen", "new_beat", "invalid_speaker"])
def test_metadata_corrections_cannot_invent_ids_views_visibility_or_story(change):
    answer, evidence = metadata(), row()
    if change == "unknown_id":
        answer["visible_entity_ids"] = ["invented"]
    elif change == "unavailable_view":
        answer["reference_view_requests"] = [{"global_id": "hero", "view_types": ["invented"],
                                               "prominence": "primary", "reason": "Wrong"}]
    elif change == "offscreen":
        evidence["character_context"] = {"registry_ids": {"local": "hero"}, "sequence": {
            "characters": [{"local_id": "local", "presence": "offscreen"}]}}
    elif change == "new_beat":
        answer["visual_event"] = "Fall now"
    else:
        answer["dialogue_present"] = False
    with pytest.raises(ValueError):
        issues.repair_metadata(None, "gemma", step.scene_to_dict(scene()), evidence, [],
                               SimpleNamespace(max_tokens=8192), Mock(return_value=answer))


def test_binding_repair_rebuilds_without_llm_or_metadata_mutation(monkeypatch):
    current = scene()
    original = step.scene_to_dict(current)
    request = Mock(side_effect=AssertionError("Binding repair must be deterministic"))
    monkeypatch.setattr(step, "chat_json", request)
    correct = {"subjects": [{"global_id": "hero"}], "audio": []}
    monkeypatch.setattr(step, "build_bindings", Mock(return_value=correct))
    fixed, bindings = step.repair_scene_inputs(None, "gemma", current, {}, row(), {}, "chapter",
                                               [issue("Wrong table", "binding")], SimpleNamespace())
    assert bindings == correct
    assert step.scene_to_dict(fixed) == original == step.scene_to_dict(current)
    request.assert_not_called()


@pytest.mark.parametrize("target", ["upstream_conflict", "scene_plan", "binding"])
def test_non_prompt_issue_never_goes_to_prompt_only_repair_without_a_valid_route(monkeypatch, target):
    monkeypatch.setattr(step, "validate_prompt", lambda *a: step.Validation(True, [], 400))
    monkeypatch.setattr(step, "check_prompt_continuity", lambda *a: [issue("Unresolved", target)])
    repair = Mock()
    monkeypatch.setattr(step, "repair_prompt", repair)
    result = step.review_and_repair_continuity(None, "gemma", scene(), "draft", {}, {}, [], [],
                                             SimpleNamespace(duration=8, max_tokens=8192, repair_attempts=3))
    assert not result[1].ok
    repair.assert_not_called()


def test_changed_metadata_rechecks_prompt_and_keeps_inputs_after_resume(tmp_path, monkeypatch, cinematic_registry):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    chapter = tmp_path / "chapter.txt"
    refs = cinematic_registry(chapter)
    args = SimpleNamespace(out_dir=tmp_path / "first/h3", force=False, max_scenes=0, duration=8,
                           max_tokens=8192, repair_attempts=1, refine_camera=False,
                           spatial_plan={"chapters": {"chapter": {"sequences": [{"sequence": 1}]}}})
    monkeypatch.setattr(step, "sequence_catalog", lambda *a: row()["catalog"])
    monkeypatch.setattr(step, "prepare_cinematic_scene", metadata_scene)
    def bindings(refs, candidate, *a):
        return {"subjects": [{"global_id": "hero"}], "audio": ["voice"] if candidate.dialogue_present else []}
    monkeypatch.setattr(step, "build_bindings", bindings)
    generation = Mock(return_value="draft")
    monkeypatch.setattr(step, "generate_prompt", generation)
    monkeypatch.setattr(step, "validate_prompt", lambda *a: step.Validation(True, [], 400))
    request = Mock(return_value=metadata())
    monkeypatch.setattr(step, "chat_json", request)
    reviewed = []
    def review(client, model, current, prompt, table, *a):
        reviewed.append((current.dialogue_present, prompt, deepcopy(table)))
        return [] if current.dialogue_present else [issue("Dialogue is present", "scene_plan")]
    monkeypatch.setattr(step, "check_prompt_continuity", review)
    repair = Mock(return_value="corrected dialogue")
    monkeypatch.setattr(step, "repair_prompt", repair)
    monkeypatch.setattr(step, "save_scene", lambda *a: {"subject_count": 1, "picture_count": 1, "audio_count": 1})
    first = step.process_chapter(chapter_input(refs), refs, None, "gemma", args)
    assert first["outputs"][0]["continuity_review"]["valid"]
    assert [entry[0] for entry in reviewed] == [False, True]
    assert reviewed[-1][2]["audio"] == ["voice"]
    record = json.loads((args.out_dir / ".cache/chapter/prompt_001.json").read_text())
    assert record["scene"]["dialogue_present"] is True
    args.resume_cache_dir = args.out_dir / ".cache"
    args.out_dir = tmp_path / "second/h3"
    second = step.process_chapter(chapter_input(refs), refs, None, "gemma", args)
    assert second["outputs"][0]["continuity_review"]["valid"]
    assert reviewed[-1][0] is True
    assert generation.call_count == request.call_count == repair.call_count == 1


def test_rejected_metadata_repair_is_saved_as_unresolved_not_rewritten(monkeypatch):
    monkeypatch.setattr(step, "validate_prompt", lambda *a: step.Validation(True, [], 400))
    monkeypatch.setattr(step, "check_prompt_continuity", lambda *a: [issue("Wrong visibility", "scene_plan")])
    repair = Mock()
    monkeypatch.setattr(step, "repair_prompt", repair)
    result = step.review_and_repair_continuity(None, "gemma", scene(), "draft", {}, {}, [], [],
                                             SimpleNamespace(duration=8, max_tokens=8192, repair_attempts=3),
                                             repair_inputs=Mock(side_effect=ValueError("Invalid presence")))
    assert not result[1].ok
    assert "failed local validation" in result[1].errors[-1]
    repair.assert_not_called()

"""Section patches are transactional and never replace unrelated valid wording."""
import copy
import json
from types import SimpleNamespace

from jsonschema import Draft202012Validator
import pytest

from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step
from tests.test_h3_prompt_validation import BINDINGS, prompt_with_shots
from tests.h3_review_helpers import issue


def scene():
    return step.Scene("Waiting", "He waits. Later he calls.", "He waits.", "", [], [], [], False,
                      "Only waiting; the call belongs to the next scene.")


def settings(attempts=2):
    return SimpleNamespace(duration=8, max_shots=1, repair_attempts=attempts, temperature=.38, max_tokens=8192)


def test_two_error_groups_use_separate_schemas_and_preserve_other_sections(monkeypatch):
    good = prompt_with_shots("[Shot 1] He waits.")
    bad = step.replace_prompt_sections(good, {"summary": "reference generation + video continuation. He waits.",
                                             "detailed_description": "[Shot 1] He waits."})
    calls = []
    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        sections = payload["sections_to_repair"]
        calls.append(payload)
        assert schema["strict"] and schema["schema"]["additionalProperties"] is False
        assert schema["schema"]["required"] == sections
        assert payload["expected_summary_prefix"] == "[reference generation]"
        assert payload["scene"]["source_excerpt"] == scene().source_excerpt
        assert "neighbouring beats" in system
        result = {name: step.section_body(good, name) for name in sections}
        Draft202012Validator(schema["schema"]).validate(result)
        return result
    monkeypatch.setattr(step, "chat_json", chat)
    result = step.format_prompt(None, "model", bad, scene(), BINDINGS, settings(), {})
    assert [c["sections_to_repair"] for c in calls] == [["summary"], ["detailed_description"]]
    assert result["prompt"] == good
    assert calls[1]["current_sections"]["summary"] == step.section_body(good, "summary")


@pytest.mark.parametrize("rejected", [
    {"summary": "[reference generation] He waits.", "detailed_description": "Overwrite!"},
    {"summary": "summary:\n[reference generation] He waits."},
    {"summary": "[reference generation] <Subject 99> waits."},
    {"summary": "reference generation. He waits."},
])
def test_rejected_patch_keeps_original_and_retries_with_feedback(monkeypatch, rejected):
    good = prompt_with_shots("[Shot 1] He waits.")
    bad = step.replace_prompt_sections(good, {"summary": "reference generation. He waits."})
    calls = []
    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        calls.append(payload)
        assert payload["current_sections"]["summary"] == step.section_body(bad, "summary")
        assert schema["schema"]["required"] == ["summary"]
        if len(calls) == 1:
            return rejected
        assert payload["previous_repair_feedback"]
        return {"summary": step.section_body(good, "summary")}
    monkeypatch.setattr(step, "chat_json", chat)
    result = step.format_prompt(None, "model", bad, scene(), BINDINGS, settings(), {})
    assert len(calls) == 2
    assert result["prompt"] == good
    assert step.validate_prompt(result["prompt"], BINDINGS, 8).ok


def test_failed_group_does_not_discard_successful_other_group(monkeypatch):
    good = prompt_with_shots("[Shot 1] He waits.")
    bad = step.replace_prompt_sections(good, {"summary": "Wrong prefix", "detailed_description": "[Shot 1] Wait."})
    def chat(client, model, system, user, schema, *args):
        names = json.loads(user)["sections_to_repair"]
        return {name: step.section_body(good, name) if name == "summary" else "[Shot 1] Still too short."
                for name in names}
    monkeypatch.setattr(step, "chat_json", chat)
    result = step.format_prompt(None, "model", bad, scene(), BINDINGS, settings(), {})
    assert step.section_body(result["prompt"], "summary") == step.section_body(good, "summary")
    assert step.section_body(result["prompt"], "detailed_description") == "[Shot 1] Wait."
    assert not step.validate_prompt(result["prompt"], BINDINGS, 8).ok


def test_scene_one_asset_ids_are_fixed_locally_and_summary_uses_one_patch(monkeypatch):
    bindings = {"subjects": [{"h3_subject_label": "<Subject 1>", "pictures": [
        {"asset_id": f"PIC_LOC_{i}", "h3_picture_label": f"<Picture {i}>"} for i in (1, 2, 3)]}], "audio": []}
    original = copy.deepcopy(bindings)
    good = prompt_with_shots("[Shot 1] He waits.")
    bad = step.replace_prompt_sections(good, {
        "subject_definitions": "<Subject 1> Cave: PIC_LOC_1 wide, PIC_LOC_2 side, PIC_LOC_3 texture.",
        "summary": "reference generation + video continuation. A quiet scene."})
    calls = []
    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        calls.append(payload)
        assert payload["sections_to_repair"] == ["summary"]
        assert "PIC_LOC_1" not in payload["current_sections"]["subject_definitions"]
        return {"summary": step.section_body(good, "summary")}
    monkeypatch.setattr(step, "chat_json", chat)
    result = step.format_prompt(None, "model", bad, scene(), bindings, settings(), {})
    assert len(calls) == 1
    assert step.validate_prompt(result["prompt"], bindings, 8).ok
    assert step.section_body(result["prompt"], "detailed_description") == step.section_body(bad, "detailed_description")
    assert bindings == original


def test_semantic_section_patch_requires_fresh_review_before_approval(monkeypatch):
    bad = prompt_with_shots("[Shot 1] He calls early.")
    good = bad.replace("He calls early.", "He waits silently.")
    reviewed = []
    def review(client, model, scene, prompt, *args):
        reviewed.append(prompt)
        return [issue("The call belongs to the next scene.", path="/current_prompt_sections/detailed_description")] if prompt == bad else []
    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        assert payload["sections_to_repair"] == ["detailed_description"]
        assert payload["continuity_issues"][0]["message"] == "The call belongs to the next scene."
        return {"detailed_description": step.section_body(good, "detailed_description")}
    monkeypatch.setattr(step, "check_prompt_continuity", review)
    monkeypatch.setattr(step, "chat_json", chat)
    prompt, validation, errors, _ = step.review_and_repair_continuity(
        None, "model", scene(), bad, BINDINGS, {}, [], [], settings())
    assert reviewed == [bad, good]
    assert prompt == good and validation.ok and not errors


def test_zero_budget_makes_no_repair_requests(monkeypatch):
    monkeypatch.setattr(step, "chat_json", lambda *a: pytest.fail("No repair budget"))
    bad = prompt_with_shots("[Shot 1] Wait.").replace("[reference generation]", "Wrong prefix")
    assert step.format_prompt(None, "model", bad, scene(), BINDINGS, settings(0), {})["prompt"] == bad


def test_missing_section_requires_atomic_structural_patch(monkeypatch):
    good = prompt_with_shots("[Shot 1] Wait.")
    bad = good.replace("subject_definitions:\nN/A\n", "")
    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        assert payload["current_prompt"] == bad
        assert schema["schema"]["required"] == step.SECTIONS
        return {name: step.section_body(good, name) for name in step.SECTIONS}
    monkeypatch.setattr(step, "chat_json", chat)
    result = step.format_prompt(None, "model", bad, scene(), BINDINGS, settings(), {})
    assert step.validate_prompt(result["prompt"], BINDINGS, 8).ok


def test_partial_patch_preserves_indented_section_headings():
    original = prompt_with_shots("[Shot 1] Wait.").replace("summary:\n", "  summary :\n")
    updated = step.replace_prompt_sections(original, {"summary": "[reference generation] Changed summary."})
    assert updated == original.replace("[reference generation] A quiet scene.", "[reference generation] Changed summary.")


@pytest.mark.parametrize("preamble", ["", "Unparsed context before headings.\n"])
def test_metadata_repair_preserves_unparsed_text_without_duplicating_complete_prompt(monkeypatch, preamble):
    good = prompt_with_shots("[Shot 1] He waits.")
    original = preamble + good
    validation = step.Validation(False, ["Scene metadata and bindings were corrected."], 0)
    calls = []

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        calls.append(payload)
        assert schema["strict"] and schema["schema"]["required"] == step.SECTIONS
        assert ("current_prompt" in payload) == bool(preamble)
        if preamble:
            assert payload["current_prompt"] == original
        assert payload["current_sections"] == {name: step.section_body(original, name) for name in step.SECTIONS}
        return {name: step.section_body(good, name) for name in step.SECTIONS}

    monkeypatch.setattr(step, "chat_json", chat)
    step.repair_prompt(None, "model", original, scene(), BINDINGS, validation, 8, settings(), {})
    assert len(calls) == 1

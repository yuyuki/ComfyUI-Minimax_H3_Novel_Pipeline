"""Regression coverage for chapter-sized H3 continuity requests."""
import copy
import json

import pytest

from minimax_h3_novel_pipeline.core import continuity_context as context
from minimax_h3_novel_pipeline.core import pipeline_step3_generate as generate
from tests.h3_review_helpers import issue


def expanded(payload):
    """Independent reconstruction for asserting full evidence coverage."""
    excerpts = payload.get("shared_source_excerpts", {})
    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: visit(item) for key, item in value.items()}
        if "source_excerpt_ref" in result:
            result["source_excerpt"] = excerpts[result.pop("source_excerpt_ref")]
        return result
    result = visit(payload)
    result.pop("shared_source_excerpts", None)
    return result


def scene(index, size=2500):
    return generate.Scene(str(index), "é" * size, f"event {index}", "room",
                          [], [], [], False, f"local {index}" + context.EVIDENCE_MARKER
                          + '{"phase":"shared source"}\nExact reference observations by phase: ["tablet"]')


def test_packing_preserves_exact_evidence_without_mutating_scenes():
    payload = {"scenes": [generate.scene_to_dict(scene(i)) for i in range(4)]}
    original = copy.deepcopy(payload)
    packed = expanded(json.loads(context.pack(payload)))
    assert payload == original
    assert len(packed["shared_sequence_evidence"]) == 1
    for old, new in zip(original["scenes"], packed["scenes"]):
        ref = new.pop("sequence_evidence_ref")
        new["adaptation_notes"] += packed["shared_sequence_evidence"][ref]
        assert new == old


def test_review_batches_cover_all_history_future_and_distant_conflicts(monkeypatch):
    history = [{"scene": generate.scene_to_dict(scene(i, 2500 + i)), "prompt": f"prompt {i}",
                "bindings": {"global_id": f"entity {i}"}} for i in range(30)]
    future = [scene(i, 2500 + i) for i in range(30, 60)]
    calls = []

    def chat(client, model, system, user, schema, temperature, max_tokens):
        assert len(system.encode()) + len(user.encode()) + len(json.dumps(schema).encode()) <= context.REQUEST_BYTES
        payload = expanded(json.loads(user))
        assert schema["name"] == "scene_continuity_review_v3" and schema["strict"]
        assert max_tokens == 9000
        assert payload["current_prompt"] == "current wording"
        calls.append(payload)
        return {"errors": [issue("Common correction"), issue(f"Correction {len(calls)}")]}

    monkeypatch.setattr(generate, "chat_json", chat)
    errors = generate.check_prompt_continuity(None, "gemma", scene(-1), "current wording",
                                               {}, {}, history, future, 9000)
    assert len(calls) > 2
    assert [p["prompt"] for batch in calls for p in batch["previous_final_prompts"]] == [p["prompt"] for p in history]
    assert [s["title"] for batch in calls for s in batch["future_requirements"]] == [s.title for s in future]
    assert [error["message"] for error in errors] == ["Common correction", *[f"Correction {i}" for i in range(1, len(calls) + 1)]]


def test_review_exposes_exact_prompt_sections_as_evidence(monkeypatch):
    scene_data = scene(0)
    prompt = "\n".join(
        f"{name}:\nExact {name} text."
        for name in generate.SECTIONS
    )
    calls = []

    def chat(client, model, system, user, schema, temperature, max_tokens):
        payload = json.loads(user)
        calls.append((system, payload))
        assert "current_prompt" not in payload
        assert payload["current_prompt_sections"]["detailed_description"] == (
            "Exact detailed_description text."
        )
        return {"errors": [issue(
            "Correct the detailed description.",
            path="/current_prompt_sections/detailed_description",
        )]}

    monkeypatch.setattr(generate, "chat_json", chat)
    errors = generate.check_prompt_continuity(
        None, "gemma", scene_data, prompt, {}, {}, [], [], 8000)

    assert len(calls) == 1
    assert "current_prompt_sections/<section_name>" in calls[0][0]
    assert errors[0]["evidence"] == [{
        "path": "/current_prompt_sections/detailed_description",
        "value": "Exact detailed_description text.",
    }]


@pytest.mark.parametrize("oversized_current", [True, False])
def test_single_oversized_record_fails_explicitly_without_network(monkeypatch, oversized_current):
    def unexpected(*args):
        pytest.fail("Oversized request reached the model")
    monkeypatch.setattr(generate, "chat_json", unexpected)
    with pytest.raises(ValueError, match="no evidence was discarded"):
        generate.check_prompt_continuity(None, "gemma", scene(0, 50000 if oversized_current else 1),
                                         "prompt", {}, {}, [], [] if oversized_current else [scene(1, 50000)], 8000)


def test_section_only_review_reduces_batches_preserving_all_evidence(monkeypatch):
    sections = {name: f"Exact {name}: " + ("évidence " * 180).strip() for name in generate.SECTIONS}
    prompt = "\n\n".join(f"{name}:\n{body}" for name, body in sections.items())
    current = scene(-1)
    history = [{"prompt": f"Earlier prompt {i}", "scene": generate.scene_to_dict(scene(i))} for i in range(4)]
    future = [scene(i, 2500 + i) for i in range(4, 42)]
    calls = []

    def chat(client, model, system, user, schema, temperature, max_tokens):
        payload = expanded(json.loads(user))
        assert "current_prompt" not in payload
        assert payload["current_prompt_sections"] == sections
        assert schema["strict"] and schema["name"] == "scene_continuity_review_v3"
        assert len(system.encode()) + len(user.encode()) + len(json.dumps(schema).encode()) <= context.REQUEST_BYTES
        calls.append((system, payload, schema))
        return {"errors": []}

    monkeypatch.setattr(generate, "chat_json", chat)
    assert generate.check_prompt_continuity(None, "model", current, prompt, {}, {}, history, future, 8192) == []
    assert [item["prompt"] for _, payload, _ in calls for item in payload["previous_final_prompts"]] == [h["prompt"] for h in history]
    assert [item["title"] for _, payload, _ in calls for item in payload["future_requirements"]] == [s.title for s in future]
    old_base = {"current_scene": generate.scene_to_dict(current), "current_prompt": prompt,
                "current_prompt_sections": sections, "bindings": {}, "continuity_plan": {}, "camera_direction": ""}
    old_batches = list(context.batches(calls[0][0], old_base, {
        "previous_final_prompts": history, "future_requirements": [generate.scene_to_dict(s) for s in future]}, calls[0][2]))
    assert len(calls) < len(old_batches)


@pytest.mark.parametrize("damage", ["preamble", "missing", "duplicate", "reordered"])
def test_review_keeps_raw_text_when_sections_cannot_preserve_it(monkeypatch, damage):
    names = list(generate.SECTIONS)
    if damage == "missing":
        names.pop(0)
    elif damage == "duplicate":
        names.insert(0, names[0])
    elif damage == "reordered":
        names.reverse()
    prompt = "\n".join(f"{name}:\nExact {index} text." for index, name in enumerate(names))
    if damage == "preamble":
        prompt = "Unparsed important detail.\n" + prompt

    def chat(client, model, system, user, schema, *args):
        assert json.loads(user)["current_prompt"] == prompt
        return {"errors": [issue("Preserve raw evidence", path="/current_prompt")]}

    monkeypatch.setattr(generate, "chat_json", chat)
    errors = generate.check_prompt_continuity(None, "model", scene(0), prompt, {}, {}, [], [], 8192)
    assert errors[0]["evidence"][0]["value"] == prompt


def test_later_batch_failure_cannot_be_reported_as_success(monkeypatch):
    calls = []

    def chat(*args):
        calls.append(args)
        return {"errors": []} if len(calls) == 1 else {"errors": "malformed"}

    monkeypatch.setattr(generate, "chat_json", chat)
    with pytest.raises(ValueError, match="invalid scene continuity review"):
        generate.check_prompt_continuity(None, "gemma", scene(-1), "prompt", {}, {}, [],
                                         [scene(i, 2500 + i) for i in range(30)], 8000)
    assert len(calls) == 2


def test_repair_refuses_oversized_corrections_before_network(monkeypatch):
    from types import SimpleNamespace

    def unexpected(*args):
        pytest.fail("Oversized repair reached the model")

    monkeypatch.setattr(generate, "chat_json", unexpected)
    with pytest.raises(ValueError, match="repair exceeds the safe request budget") as failure:
        generate.repair_prompt(None, "gemma", "prompt", scene(0, 1), {},
                               generate.Validation(False, ["Scene continuity: " + "correction " * 6000], 0), 8,
                               SimpleNamespace(temperature=0.1, max_tokens=8000), {})
    message = str(failure.value)
    for field in ("scene_sequence=", "sections=", "request_bytes=", "limit_bytes=48000",
                  "system_bytes=", "payload_bytes=", "schema_bytes=", "largest_field_value_bytes="):
        assert field in message
    assert "correction correction" not in message


def test_cited_snapshots_are_shared_losslessly_without_merging_issues():
    value = {"observations": ["évidence " * 100, "distinct observation"], "present": True}
    payload = {"continuity_issues": [
        {"message": f"issue {index}", "evidence": [{"path": f"/old/{index}", "value": value}]}
        for index in range(6)]}
    payload["continuity_issues"].append({"message": "changed snapshot", "evidence": [
        {"path": "/old/0", "value": {**value, "present": False}}]})
    original = copy.deepcopy(payload)
    packed_text = context.pack(payload)
    packed = json.loads(packed_text)
    assert payload == original
    assert len(packed["shared_cited_values"]) == 1
    for finding in packed["continuity_issues"]:
        for citation in finding["evidence"]:
            if "value_ref" in citation:
                citation["value"] = packed["shared_cited_values"][citation.pop("value_ref")]
    del packed["shared_cited_values"]
    assert packed == original
    assert len(packed_text.encode()) < len(json.dumps(original, ensure_ascii=False).encode())


def test_repeated_source_prose_reduces_batches_without_merging_beats():
    scenes = [generate.scene_to_dict(scene(i, 6000)) for i in range(12)]
    original = {"future_requirements": scenes}
    packed = json.loads(context.pack(original))
    assert len(packed["shared_source_excerpts"]) == 1
    restored = expanded(packed)
    for old, new in zip(scenes, restored["future_requirements"]):
        new["adaptation_notes"] += restored["shared_sequence_evidence"][new.pop("sequence_evidence_ref")]
        assert new == old
    assert len(list(context.batches("system", {}, original, {}))) == 1
    assert len(json.dumps(original).encode()) > context.REQUEST_BYTES
    assert len(context.pack(original).encode()) < context.REQUEST_BYTES

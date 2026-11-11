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
        assert schema == generate.CONTINUITY_REVIEW_SCHEMA and schema["strict"]
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
        assert payload["current_prompt"] == prompt
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
    with pytest.raises(ValueError, match="repair exceeds the safe request budget"):
        generate.repair_prompt(None, "gemma", "prompt", scene(0, 1), {},
                               generate.Validation(False, ["Scene continuity: " + "correction " * 6000], 0), 8,
                               SimpleNamespace(temperature=0.1, max_tokens=8000), {})


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

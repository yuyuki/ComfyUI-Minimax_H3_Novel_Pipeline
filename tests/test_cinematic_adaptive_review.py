"""Length recovery changes the task without losing source coverage or approval checks."""
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca, narrative_validation as nv


def scene(source):
    return {"source": source, "initial_state": "Avant.", "events": [source], "final_state": "Après."}


@pytest.mark.parametrize("failed_focus", [None, "source_fidelity", "cinematic_pacing", "local_transition"])
def test_length_recovery_checks_every_aspect_and_never_approves_partial_work(monkeypatch, failed_focus):
    scenes = [scene("Il attend."), scene(" Il part.")]
    calls, snapshots = [], []

    def chat(client, model, system, user, schema, temperature, tokens):
        payload = json.loads(user)
        if schema != nv.SCHEMA:
            return {"sequences": deepcopy(scenes)}
        assert schema["strict"]
        assert tokens == 8192 and ca.lmstudio_json.REQUEST_OUTPUT_LIMIT.get() == 8192
        assert len(payload["sequences"]) == 1 and "next_sequence" not in payload
        if payload["sequences"][0]["source"] == scenes[0]["source"]:
            return {"issues": []}
        focus = payload.get("review_focus")
        calls.append(focus)
        if focus is None or focus == failed_focus:
            raise ca.lmstudio_json.StructuredOutputLengthError("length")
        assert payload["sequences"] == scenes[1:]
        assert len(system) < len(nv.SYSTEM)
        if focus == "local_transition":
            assert payload["previous_sequence"]["source"] == scenes[0]["source"]
            assert payload["previous_sequence"]["final_state"] == scenes[0]["final_state"]
        else:
            assert "previous_sequence" not in payload and "previous_final_state" not in payload
        return {"issues": []}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    kwargs = dict(chunk_chars=4000, temperature=0.15, max_tokens=8192, correction_attempts=1,
                  on_checkpoint=lambda state: snapshots.append(deepcopy(state)))
    result = ca.adapt_chapter(None, "mock", "Il attend. Il part.", **kwargs)
    assert calls == [None, "source_fidelity", "cinematic_pacing", "local_transition"]
    assert "".join(s["source"] for s in result["sequences"]) == "Il attend. Il part."
    if failed_focus:
        assert max(len(state["approved_reviews"]) for state in snapshots) == 1
        committed = [state for state in snapshots if state["stage"] == "scene_committed"]
        assert len(committed) == 2
        assert committed[0]["sequences"][0]["source"] == scenes[0]["source"]
        assert committed[0]["pending_sequences"] == scenes[1:]
        assert result["sequences"][1]["review"]["status"] == "incomplete"
        assert result["narrative_review"]["warnings"][-1]["check"] == failed_focus
    else:
        assert len(snapshots[-1]["approved_reviews"]) == 2
        assert result["narrative_review"]["warnings"][0]["status"] == "recovered_by_focused_checks"
    assert ca.lmstudio_json.REQUEST_OUTPUT_LIMIT.get() is None


def test_focused_defects_are_combined_and_sent_to_local_repair(monkeypatch):
    reviews, repairs = [], []

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema == nv.SCHEMA:
            focus = payload.get("review_focus")
            reviews.append(focus)
            if repairs:
                return {"issues": []}
            if focus is None:
                raise ca.lmstudio_json.StructuredOutputLengthError("length")
            return {"issues": [{"target": 1, "message": focus}]}
        if "rejected_adaptation" in payload:
            repairs.append(payload)
        return {"sequences": [scene(payload["current_passage"])]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    ca.adapt_chapter(None, "mock", "Il attend.", chunk_chars=4000, temperature=0.15,
                     max_tokens=8192, correction_attempts=1)
    assert len(repairs) == 1
    assert repairs[0]["defects_to_correct"] == list(nv.ADAPTATION_REVIEW_FOCUSES)
    assert reviews == [None, *nv.ADAPTATION_REVIEW_FOCUSES, None]


def test_non_length_errors_do_not_trigger_adaptive_review(monkeypatch):
    calls = []

    def chat(client, model, system, user, schema, *args):
        calls.append(schema["name"])
        if schema == nv.SCHEMA:
            raise RuntimeError("connection failed")
        return {"sequences": [scene("Il attend.")]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = ca.adapt_chapter(None, "mock", "Il attend.", chunk_chars=4000, temperature=0.15,
                              max_tokens=8192, correction_attempts=1)
    assert result["sequences"][0]["review"]["status"] == "incomplete"
    assert result["narrative_review"]["warnings"][0]["reason"] == "request_failed"
    assert calls == [ca.SCHEMA["name"], nv.SCHEMA["name"]]
    assert ca.lmstudio_json.REQUEST_OUTPUT_LIMIT.get() is None

"""Advisory failures remain visible without approval, lost source, or blocked successors."""
from copy import deepcopy
import json
import logging

import pytest

pytestmark = pytest.mark.usefixtures("mock_scene_planning")
from jsonschema import Draft202012Validator

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca, cinematic_references as cr
from minimax_h3_novel_pipeline.core import narrative_validation as nv


def scene(source):
    return {"source": source, "initial_state": "Before.", "events": [source], "final_state": "After."}


def run(**kwargs):
    return ca.adapt_chapter(None, "mock", "A. B. C.", chapter_name="Chapter", chunk_chars=4000,
                            temperature=0.1, max_tokens=16384, correction_attempts=0, **kwargs)


@pytest.mark.parametrize("missing_planning", [False, True])
def test_failed_focus_keeps_successful_doubts_and_runs_all_checks_and_successors(monkeypatch, caplog, missing_planning):
    calls, snapshots = [], []
    assess = ca.scene_planning.assess

    def planning(*args, **kwargs):
        if missing_planning and args[3]["source"] == " B.":
            raise RuntimeError("secret-backend-text")
        return assess(*args, **kwargs)

    monkeypatch.setattr(ca.scene_planning, "assess", planning)

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema != nv.SCHEMA:
            return {"sequences": [scene("A."), scene(" B."), scene(" C.")]}
        source = payload["sequences"][0]["source"]
        focus = payload.get("review_focus")
        calls.append((source, focus))
        if source != " B.":
            return {"issues": []}
        assert (payload["scene_plan"] is None) == missing_planning
        if focus is None or focus == "source_fidelity":
            raise ca.lmstudio_json.StructuredOutputLengthError("secret-backend-text")
        return {"issues": [{"target": 1, "message": "Pacing concern."}] if focus == "cinematic_pacing" else []}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with caplog.at_level(logging.WARNING):
        result = run(on_checkpoint=lambda state: snapshots.append(deepcopy(state)))
    assert calls[-1] == (" C.", None)
    assert [focus for source, focus in calls if source == " B."] == [None, *nv.ADAPTATION_REVIEW_FOCUSES]
    review = result["sequences"][1]["review"]
    assert review["status"] == "incomplete" and review["issues"] == ["Pacing concern."]
    assert review["failed_checks"][-1] == {"stage": "adaptation", "check": "source_fidelity", "reason": "output_limit"}
    assert len(snapshots[-1]["approved_reviews"]) == 2
    assert "secret-backend-text" not in json.dumps(result) + caplog.text
    assert all(record.levelno == logging.WARNING for record in caplog.records)
    assert "source_fidelity" in caplog.text
    assert "".join(s["source"] for s in result["sequences"]) == "A. B. C."
    assert cr.parse_chapters([result])[0] == result
    catalog = {"schema_version": "minimax-h3-novel-refs.chapter.v4", "timeline_version": cr.TEMPORAL_VERSION,
               "cache_key": "test", "chapter_id": "chapter", "chapter_name": "Chapter",
               "source": {"file": "chapter.txt", "sha256": "test"}, "sequences": result["sequences"],
               "narrative_review": {**result["narrative_review"], "extraction_issues": []},
               "characters": [], "locations": [], "objects": []}
    Draft202012Validator(cr.CATALOG_SCHEMA).validate(catalog)


def test_selected_proposal_with_failed_planning_still_receives_semantic_review(monkeypatch):
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: {"sequences": [scene("A. B. C.")]})
    assess = ca.scene_planning.assess
    reviews = []

    def planning(*args, **kwargs):
        if args[3].get("chosen_event"):
            raise RuntimeError("private backend text")
        return assess(*args, **kwargs)

    def review(chat, client, model, payload, *args):
        reviews.append(deepcopy(payload))
        return {1: ["Check source-supported posture."]}

    def propose(*args):
        return [{"id": name, "selected": name == "A", "reason": "Source interpretation.", "adaptation": {
            "initialState": "Before.", "event": "Wait.", "endingState": "After.",
        }} for name in ("A", "B")]

    monkeypatch.setattr(ca.scene_planning, "assess", planning)
    monkeypatch.setattr(ca.narrative_validation, "review", review)
    monkeypatch.setattr(ca.narrative_choices, "propose", propose)
    result = run()
    assert len(reviews) == 2 and reviews[-1]["scene_plan"] is None
    assert "defects_to_recheck" in reviews[-1]
    selected = result["narrative_review"]["issues"][0]["proposals"][0]
    assert selected["review"] == {"status": "incomplete", "issues": ["Check source-supported posture."]}
    assert result["sequences"][0]["review"]["issues"] == selected["review"]["issues"]
    assert "private backend text" not in json.dumps(result)


def test_partial_resume_preserves_warnings_and_does_not_approve_failed_review(monkeypatch):
    snapshots = []
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: {
        "sequences": [scene("A."), scene(" B."), scene(" C.")]})

    def review(chat, client, model, payload, *args):
        if payload["sequences"][0]["source"] == "A.":
            raise RuntimeError("unavailable")
        return {}

    def checkpoint(state):
        snapshots.append(deepcopy(state))
        if state["stage"] == "scene_committed" and len(state["sequences"]) == 2:
            raise KeyboardInterrupt()

    monkeypatch.setattr(ca.narrative_validation, "review", review)
    with pytest.raises(KeyboardInterrupt):
        run(on_checkpoint=checkpoint)
    saved = snapshots[-1]
    assert len(saved["approved_reviews"]) == 1
    assert saved["sequences"][0]["review"]["status"] == "incomplete"
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: pytest.fail("Do not regenerate"))
    reviews = []
    monkeypatch.setattr(ca.narrative_validation, "review", lambda *args: reviews.append(args[3]) or {})
    result = run(resume_state=saved)
    assert [payload["sequences"][0]["source"] for payload in reviews] == [" C."]
    assert result["narrative_review"]["warnings"] == saved["narrative_review"]["warnings"]
    assert result["sequences"][0] == saved["sequences"][0]


@pytest.mark.parametrize("error", [KeyboardInterrupt(), PermissionError("storage unavailable"),
                                 type("InterruptProcessingException", (RuntimeError,), {})("Stop")])
def test_review_cancellation_and_storage_errors_still_propagate(monkeypatch, error):
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: {"sequences": [scene("A. B. C.")]})

    def review(*args):
        raise error

    monkeypatch.setattr(ca.narrative_validation, "review", review)
    with pytest.raises(type(error)):
        run()


@pytest.mark.parametrize("failed_stage", ["proposals", "proposal_review"])
def test_proposal_failure_keeps_valid_scene_and_continues(monkeypatch, failed_stage):
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: {"sequences": [scene("A. B. C.")]})

    def review(chat, client, model, payload, *args):
        if "defects_to_recheck" in payload:
            raise RuntimeError("review backend unavailable")
        return {1: ["Check posture."]}

    def propose(*args):
        if failed_stage == "proposals":
            raise ValueError("malformed proposals")
        return [{"id": letter, "selected": letter == "A", "reason": "Option.", "adaptation": {
            "initialState": "Before.", "event": "Wait.", "endingState": "After.",
        }} for letter in ("A", "B")]

    monkeypatch.setattr(ca.narrative_validation, "review", review)
    monkeypatch.setattr(ca.narrative_choices, "propose", propose)
    result = run()
    assert result["sequences"][0]["review"]["status"] == "incomplete"
    warning = result["narrative_review"]["warnings"][0]
    assert warning["stage"] == ("correction" if failed_stage == "proposals" else "proposal_review")
    if failed_stage == "proposal_review":
        assert result["narrative_review"]["issues"][0]["proposals"][0]["review"]["status"] == "incomplete"
    cr.parse_chapters([result])

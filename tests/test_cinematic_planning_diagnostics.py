"""Text-free planning diagnostics survive retry, recovery, interruption and resume."""
from copy import deepcopy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca, cinematic_scene_planning as planning, cinematic_references as cr
from minimax_h3_novel_pipeline.core import narrative_validation as nv


def scene():
    return {"source": "A person acts.", "initial_state": "Before.",
            "events": ["Sensitive original event"], "final_state": "After."}


def plan(item, text=None):
    return {"timing": {"beats": [{"events": [{"event_index": 1, "text": text or item["events"][0]}],
                                  "seconds": 2, "reason": "Natural activity."}],
                       "pause_seconds": 0, "estimated_seconds": 2, "reason": "Natural activity."},
            "continuity": {"reset": False, "reset_evidence": "", "updates": []}}


def run(**kwargs):
    return ca.adapt_chapter(None, "mock", scene()["source"], chapter_name="Chapter", chunk_chars=4000, temperature=0.15,
                            max_tokens=16384, correction_attempts=2, **kwargs)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def test_mismatch_metrics_identify_original_event_separately_from_fourth_beat():
    item = {**scene(), "events": ["Prepare", "Move the item then wait"]}
    value = plan(item)
    refs = [(1, "Prepare"), (2, "Move"), (2, " the item"), (2, " then stop")]
    value["timing"]["beats"] = [{"events": [{"event_index": i, "text": text}],
                                    "seconds": 1, "reason": "Activity."} for i, text in refs]
    with pytest.raises(planning._PlanningValidationError) as caught:
        planning.validate(value, item)
    event = item["events"][1]
    assert caught.value.field_path == "/timing/beats/3/events/0/text"
    assert caught.value.metrics == {
        "event_index": 2, "beat_index": 3, "reference_index": 0,
        "validated_prefix_length": event.index("then"), "matched_chars": len("then "),
        "expected_remaining_length": len("then wait"), "received_length": len("then stop"),
        "mismatch_offset": event.index("wait"),
    }


@pytest.mark.parametrize("defect", ["punctuation", "missing_tail", "duplicate"])
def test_metrics_distinguish_extra_punctuation_missing_tail_and_duplicate_reference(defect):
    item = scene()
    event = item["events"][0]
    value = plan(item, event + "." if defect == "punctuation" else event)
    if defect == "missing_tail":
        value["timing"]["beats"][0]["events"][0]["text"] = "Sensitive"
    elif defect == "duplicate":
        value["timing"]["beats"].append(deepcopy(value["timing"]["beats"][0]))
    with pytest.raises(planning._PlanningValidationError) as caught:
        planning.validate(value, item)
    metrics = caught.value.metrics
    if defect == "punctuation":
        assert metrics["received_length"] == len(event) + 1
        assert metrics["expected_remaining_length"] == metrics["matched_chars"] == metrics["mismatch_offset"] == len(event)
    elif defect == "missing_tail":
        assert caught.value.reason == "event_text_incomplete"
        assert metrics["validated_prefix_length"] == metrics["mismatch_offset"] == len("Sensitive")
        assert metrics["received_length"] == metrics["matched_chars"] == 0
    else:
        assert metrics["expected_remaining_length"] == 0
        assert metrics["mismatch_offset"] == len(event)


def test_every_validation_attempt_is_recorded_even_when_retry_recovers(monkeypatch):
    item = scene()
    snapshots, actual_hashes = [], []

    def chat(client, model, system, user, schema, temperature, tokens):
        if schema == planning.SCHEMA:
            assert system.count("Maximum scene duration:") == 1 and tokens == 16384
            actual_hashes.append((digest(system), digest(schema), digest({"system": system, "user": user})))
            if len(actual_hashes) == 1:
                return plan(item, item["events"][0] + ".")
            if len(actual_hashes) == 2:
                return plan(item, "private rejected model body")
            return plan(item)
        if schema == nv.SCHEMA:
            return {"issues": []}
        return {"sequences": [deepcopy(item)]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run(on_checkpoint=lambda state: snapshots.append(deepcopy(state)))
    records = result["narrative_review"]["planning_diagnostics"]
    assert [d["attempt"] for d in records] == [1, 2, 3]
    assert [d["status"] for d in records] == ["failed", "failed", "completed"]
    assert [d["assessment_id"] for d in records] == [1, 1, 1]
    for record, expected in zip(records, actual_hashes):
        assert record["sequence"] == 1 and record["stage"] == "adaptation"
        assert (record["system_prompt_sha256"], record["schema_sha256"], record["request_prompt_sha256"]) == expected
    assert records[0]["request_prompt_sha256"] != records[1]["request_prompt_sha256"]
    attempts = [s for s in snapshots if s["stage"] == "planning_attempt"]
    assert [len(s["narrative_review"]["planning_diagnostics"]) for s in attempts] == [1, 2, 3]
    assert "warnings" not in result["narrative_review"] and "review" not in result["sequences"][0]
    serialized = json.dumps(records)
    assert item["events"][0] not in serialized and "private rejected model body" not in serialized
    assert cr.parse_chapters([result])[0] == result
    catalog = {"schema_version": "minimax-h3-novel-refs.chapter.v4", "timeline_version": cr.TEMPORAL_VERSION,
               "cache_key": "test", "chapter_id": "chapter", "chapter_name": "Chapter",
               "source": {"file": "chapter.txt", "sha256": "test"}, "sequences": result["sequences"],
               "narrative_review": {**result["narrative_review"], "extraction_issues": []},
               "characters": [], "locations": [], "objects": []}
    Draft202012Validator(cr.CATALOG_SCHEMA).validate(catalog)
    saved = snapshots[-1]
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: pytest.fail("Do not regenerate completed work"))
    assert run(resume_state=saved) == result


@pytest.mark.parametrize("failure,reason", [(RuntimeError("private backend response"), "request_failed"),
                                           (ca.lmstudio_json.StructuredOutputLengthError("private response"), "output_limit")])
def test_request_failure_records_hashes_without_backend_messages(monkeypatch, failure, reason, caplog):
    def chat(client, model, system, user, schema, *args):
        if schema == planning.SCHEMA:
            raise failure
        return {"issues": []} if schema == nv.SCHEMA else {"sequences": [scene()]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run()
    records = result["narrative_review"]["planning_diagnostics"]
    assert len(records) == 1 and records[0]["reason"] == reason
    assert "private" not in json.dumps(records) + caplog.text
    assert result["sequences"][0]["review"]["status"] == "incomplete"


def test_interruption_after_failed_attempt_preserves_metadata_on_resume(monkeypatch):
    snapshots = []

    def chat(client, model, system, user, schema, *args):
        if schema == planning.SCHEMA:
            return plan(scene(), "private rejected text")
        return {"issues": []} if schema == nv.SCHEMA else {"sequences": [scene()]}

    def checkpoint(state):
        snapshots.append(deepcopy(state))
        if state["stage"] == "planning_attempt":
            raise KeyboardInterrupt()

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(KeyboardInterrupt):
        run(on_checkpoint=checkpoint)
    saved = snapshots[-1]
    assert saved["approved_reviews"] == []
    first_record = saved["narrative_review"]["planning_diagnostics"][0]
    assert first_record["status"] == "failed"

    def resumed_chat(client, model, system, user, schema, *args):
        assert schema in (planning.SCHEMA, nv.SCHEMA)
        return plan(scene()) if schema == planning.SCHEMA else {"issues": []}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", resumed_chat)
    result = run(resume_state=saved)
    records = result["narrative_review"]["planning_diagnostics"]
    assert records[0] == first_record and records[1]["assessment_id"] == 2
    assert records[1]["attempt"] == 1 and records[1]["status"] == "completed"
    assert "review" not in result["sequences"][0]


@pytest.mark.parametrize("failure", [PermissionError("storage"), OSError("filesystem"), RuntimeError("checkpoint"),
                                    ValueError("checkpoint integrity"), KeyboardInterrupt()])
def test_attempt_checkpoint_failures_propagate_unchanged(monkeypatch, failure):
    def chat(client, model, system, user, schema, *args):
        return plan(scene()) if schema == planning.SCHEMA else {"sequences": [scene()]}

    def checkpoint(state):
        if state["stage"] == "planning_attempt":
            raise failure

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(type(failure)) as caught:
        run(on_checkpoint=checkpoint)
    assert caught.value is failure


@pytest.mark.parametrize("change", ["system", "schema", "duration"])
def test_fingerprints_track_actual_planning_instructions_and_schema(monkeypatch, change):
    records = []

    def chat(*args):
        return plan(scene())

    def observe(record):
        records.append(record)

    planning.assess(chat, None, "mock", scene(), [], 8, 16384, 0, on_diagnostic=observe)
    if change == "system":
        monkeypatch.setattr(planning, "SYSTEM", planning.SYSTEM + "\nGeneric additional instruction.")
    elif change == "schema":
        monkeypatch.setattr(planning, "SCHEMA", {**planning.SCHEMA, "name": "new_planning_contract"})
    planning.assess(chat, None, "mock", scene(), [], 9 if change == "duration" else 8, 16384, 0, on_diagnostic=observe)
    field = "schema_sha256" if change == "schema" else "system_prompt_sha256"
    assert records[0][field] != records[1][field]

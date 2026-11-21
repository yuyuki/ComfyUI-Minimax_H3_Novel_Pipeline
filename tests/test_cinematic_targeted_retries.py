"""Retry feedback changes strategy, preserves constraints and explains validation defects."""
from copy import deepcopy
from contextlib import nullcontext
import json

import pytest

from tests.passages_helpers import with_passage_ids

from minimax_h3_novel_pipeline.core import cinematic_scene_planning as planning
from minimax_h3_novel_pipeline.core import cinematic_diagnostics as diagnostics


def scene():
    return {"source": "La corde se resserre sous ses bras.", "initial_state": "Avant.",
            "events": ["la corde se resserre sous ses bras"], "final_state": "Après."}


def plan():
    return {"timing": {"beats": [{"events": [{"event_index": 1, "text": None}],
                                  "seconds": 2, "reason": "Action."}],
                       "pause_seconds": 0, "estimated_seconds": 2, "reason": "Action."},
            "continuity": {"reset": False, "reset_evidence": "", "updates": []}}


def test_repetition_budget_remains_bounded_and_does_not_approve():
    records = []

    def chat(*args):
        value = plan()
        value["timing"]["beats"][0]["events"][0]["text"] = scene()["events"][0] + "."
        return value

    with pytest.raises(planning._PlanningValidationError):
        planning.assess(with_passage_ids(chat), None, "mock", scene(), [], 8, 1000, 2, on_diagnostic=records.append)
    assert len(records) == 3 and all(r["status"] == "failed" for r in records)


def test_export_separates_diagnostics_but_preserves_editable_choices():
    value = {"chapter_name": "test", "sequences": [{"sequence": 1, "source": "Source.",
             "adaptation": {"initialState": "Avant.", "event": "1. Action.", "endingState": "Après."},
             "review": {"status": "incomplete", "failed_checks": []}}],
             "narrative_review": {"issues": [{"sequence": 1, "proposals": [
                 {"selected": True, "adaptation": {"event": "Choix."}, "review": {"status": "incomplete"}}]}],
                 "warnings": [{"sequence": 1, "status": "incomplete", "reason": "request_failed"}],
                 "planning_diagnostics": []}}
    original = deepcopy(value)
    exported = diagnostics.downstream_chapter(value)
    assert value == original
    assert "review" not in exported["sequences"][0]
    assert set(exported["narrative_review"]) == {"issues"}
    assert exported["narrative_review"]["issues"][0]["proposals"][0] == {
        "selected": True, "adaptation": {"event": "Choix."}}
    report = diagnostics.report(value, chapter_name="test", source_file="test.md", maximum_seconds=8, completed=True)
    assert report["generation_status"] == "completed" and report["status"] == "incomplete"
    assert report["sequences"][0]["timing"]["estimated_seconds"] is None
    assert report["narrative_choices"][0]["proposals"][0]["review"]["status"] == "incomplete"


def test_sidecar_distinguishes_available_timing_from_allowance_overrun():
    value = plan()
    value["timing"]["beats"] *= 2
    value["timing"]["estimated_seconds"] = 7
    state = {"sequences": [{"sequence": 1, "source": scene()["source"], "adaptation": {"event": "Action."},
                            "review": {"status": "incomplete"}}],
             "scene_plans": [{"plan": value, "events": scene()["events"]}]}
    timing = diagnostics.report(state, chapter_name="test", source_file="test.md", maximum_seconds=8)["sequences"][0]["timing"]
    assert timing["status"] == "exceeded" and timing["seconds_with_allowance"] == 8.75


@pytest.mark.parametrize("interrupted", [False, True])
def test_node_persists_sidecar_during_retry_and_exports_only_chapter_data(tmp_path, monkeypatch, interrupted):
    from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca, narrative_validation as nv, path_access
    from minimax_h3_novel_pipeline.nodes import cinematic_chapter_adapter as adapter

    source_path = tmp_path / "test.md"
    source = " ".join([scene()["source"]] * 5)
    source_path.write_text(source, encoding="utf-8")
    output = tmp_path / "output"
    sidecar_path = output / "001_test.cinematic.diagnostics.json"
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(adapter, "stage_output", lambda *args: output)
    monkeypatch.setattr(adapter.lmstudio_pipeline, "make_client_and_model", lambda *args: (nullcontext(), "mock"))
    calls = 0

    def chat(client, model, system, user, schema, *args):
        nonlocal calls
        if schema["name"] == planning.SCHEMA["name"]:
            calls += 1
            value = plan()
            if calls == 1:
                value["timing"]["beats"][0]["events"][0]["text"] = scene()["events"][0] + "."
            else:
                saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
                assert saved["summary"]["failed_planning_attempts"] == 1
                assert saved["planning_attempts"][0]["field_path"].startswith("/timing")
                assert not list(output.glob("*.cinematic.json"))
                if interrupted:
                    raise KeyboardInterrupt()
            return value
        return {"issues": []} if schema["name"] == nv.SCHEMA["name"] else {"sequences": [{**scene(), "source": source}]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))

    def run():
        return adapter.CinematicChapterAdapterNode().run(
            {"api_url": "unused", "run_folder": "test"}, {"chapter_paths": [str(source_path)]})

    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            run()
        saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert saved["status"] == "incomplete" and saved["generation_status"] == "in_progress"
    else:
        response = run()
        saved = json.loads(sidecar_path.read_text(encoding="utf-8"))
        chapter_path = output / "001_test.cinematic.json"
        chapter = json.loads(chapter_path.read_text(encoding="utf-8"))
        assert chapter["narrative_review"] == {"issues": []}
        assert "review" not in chapter["sequences"][0]
        assert "planning_diagnostics" not in json.dumps(response)
        assert response["result"][1] == str(chapter_path)
        assert saved["planning_attempts"][0]["resolved_by_attempt"] == 2
        assert saved["status"] == saved["generation_status"] == "completed"
        assert saved["summary"]["planning_attempts"] == 2
        assert saved["sequences"][0]["timing"]["estimated_seconds"] == 2
        assert saved["sequences"][0]["timing"]["beats"][0]["events"][0]["text"] == scene()["events"][0]
    assert not list(output.rglob("*.tmp"))

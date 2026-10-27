"""Generation consumes consolidated cinematic provenance without reading prose."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step


def test_selection_uses_registry_id_and_rejects_missing_or_ambiguous_timeline(tmp_path, cinematic_registry):
    path = tmp_path / "different_source_name.txt"
    refs = cinematic_registry(path)
    assert step.selected_timeline(refs, path)[0] == "chapter"
    with pytest.raises(ValueError, match="Expected one"):
        step.selected_timeline(refs, tmp_path / "other.txt")
    refs["chapters"].append(deepcopy(refs["chapters"][0]))
    with pytest.raises(ValueError, match="Expected one"):
        step.selected_timeline(refs, path)
    refs["chapters"].pop()
    del refs["chapter_timelines"]
    with pytest.raises(ValueError, match="re-extract"):
        step.selected_timeline(refs, path)


def test_catalog_preserves_conflicts_without_future_or_carried_states():
    entity = {"global_id": "CHAR_001", "entity_type": "character", "canonical_name": "Aster",
              "chapters_seen": ["chapter"], "timeline": {"chapter": {
                  "1": {"event": [{"state": "holding rope"}, {"state": "empty hands"}]},
                  "2": {"endingState": [{"state": "injured"}]}}}}
    refs = {"entities": [entity]}
    catalog = step.sequence_catalog(refs, "chapter", 1)
    assert catalog[0]["state_by_phase"] == {
        "initialState": [], "event": [{"state": "holding rope"}, {"state": "empty hands"}], "endingState": []}
    assert "injured" not in json.dumps(catalog)
    assert step.sequence_catalog(refs, "chapter", 3) == []
    catalog[0]["state_by_phase"]["event"].clear()
    assert len(entity["timeline"]["chapter"]["1"]["event"]) == 2


def test_sequence_order_cache_invalidation_and_repeated_beats(tmp_path, monkeypatch, cinematic_registry):
    path = tmp_path / "unread_source.txt"
    refs = cinematic_registry(path)
    sequences = refs["chapter_timelines"]["chapter"]["sequences"]
    sequences.append({**deepcopy(sequences[0]), "sequence": 3})
    calls = []

    def plan(client, model, chapter, chunk, index, total, catalog, args):
        calls.append(json.loads(chunk))
        return [step.Scene("Wait", "A character waits.", "Wait", "", [], [], [], False, "")]

    monkeypatch.setattr(step, "plan_scenes", plan)
    monkeypatch.setattr(step, "build_bindings", lambda *a: {"subjects": [], "audio": []})
    args = SimpleNamespace(out_dir=tmp_path / "generated", force=False, max_scenes=0, duration=8)
    manifest = step.process_chapter(path, refs, None, "mock", args)
    assert [entry["sequence"] for entry in manifest["outputs"]] == [1, 3]
    assert len(calls) == 2
    step.process_chapter(path, refs, None, "mock", args)
    assert len(calls) == 2
    sequences[1]["adaptation"]["endingState"] = "A changed ending."
    step.process_chapter(path, refs, None, "mock", args)
    assert len(calls) == 3
    assert calls[-1]["sequence"] == 3
    cached = json.loads((args.out_dir / ".cache/chapter/plan_002.json").read_text())
    assert cached["scenes"][0]["sequence"] == 3
    assert "A changed ending." in cached["scenes"][0]["adaptation_notes"]
    assert not path.exists()



def test_temporal_context_reaches_planning_and_prompt_notes(tmp_path, monkeypatch, cinematic_registry):
    path = tmp_path / "chapter.txt"
    refs = cinematic_registry(path)
    refs["chapter_timelines"]["chapter"]["default_temporal_context"] = "1925"
    calls = []
    def plan(client, model, chapter, chunk, index, total, catalog, args):
        calls.append(json.loads(chunk))
        return [step.Scene("Wait", "Wait", "Wait", "", [], [], [], False, "")]
    monkeypatch.setattr(step, "plan_scenes", plan)
    monkeypatch.setattr(step, "build_bindings", lambda *a: {"subjects": [], "audio": []})
    args = SimpleNamespace(out_dir=tmp_path / "generated", force=False, max_scenes=0, duration=8)
    step.process_chapter(path, refs, None, "mock", args)
    assert "1925" in calls[0]["temporal_context"]
    assert "flashbacks" in calls[0]["temporal_context"]
    cached = json.loads((args.out_dir / ".cache/chapter/plan_001.json").read_text())
    assert "1925" in cached["scenes"][0]["adaptation_notes"]
    refs["chapter_timelines"]["chapter"]["default_temporal_context"] = "2040"
    step.process_chapter(path, refs, None, "mock", args)
    assert len(calls) == 2
    assert "2040" in calls[-1]["temporal_context"]

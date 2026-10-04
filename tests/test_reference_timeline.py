"""Offline chronology regressions through real reconciliation and duplicate merges."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline import pipeline_step2_consolidate as step, util
from minimax_h3_novel_pipeline.reference_timeline import (
    merge_timeline, refresh_first_occurrence, state_at, validate_catalogs,
)


def catalog(cid="chapter", kind="characters", name="Indy", alias=None):
    state = "chapter_appearance" if kind == "characters" else "chapter_state"
    observations = {
        "3": {"event": [{state: "holds the torch", "evidence": ["torch held"]}]},
        "1": {"endingState": [{state: "eyes closed", "evidence": ["closes eyes"]}],
              "event": [{state: "looks upward", "evidence": ["looks"]}],
              "initialState": [{state: "has no torch", "evidence": ["empty hands"]}]},
        "2": {"endingState": [{state: "holds the torch"}],
              "event": [{state: "receives a torch"}],
              "initialState": [{state: "has no torch"}]},
    }
    return {"schema_version": util.CHAPTER_SCHEMA, "timeline_version": "cinematic-reference-timeline.v1",
            "chapter_id": cid, "chapter_name": cid, "source": {},
            "sequences": [{"sequence": n, "source": "Evidence", "adaptation": {
                "initialState": "Opening", "event": "Action", "endingState": "Ending"}} for n in (1, 2, 3)],
            **{k: [] for k in ("characters", "locations", "objects")},
            kind: [{"local_id": "LOCAL_1", "canonical_name": name, "aliases": alias or [],
                    "stable_visual_description": "Brown hair" if kind == "characters" else "Hemp fibers",
                    "distinguishing_features": ["Permanent mark"], "voice_description": "Low voice",
                    "first_sequence": 3, "first_phase": "event", "state_by_sequence": observations}]}


@pytest.fixture
def mocked_reconcile(monkeypatch):
    calls = []
    def chat(client, model, system, user, schema, *args):
        if schema["name"] == "reference_link_proposals":
            data = json.loads(user)
            return {"entities": [{"id": key, "classification": "entity"} for key in data["current_entity_ids"]],
                    "links": []}
        incoming = json.loads(user.split("INCOMING:\n", 1)[1].split("\n\nCANDIDATES:", 1)[0])
        candidates = json.loads(user.split("CANDIDATES:\n", 1)[1])
        calls.append(user)
        return {"resolutions": [{**item, "match_global_id": candidates[item["local_id"]][0]["global_id"]
                                if candidates[item["local_id"]] else "NEW"} for item in incoming]}
    monkeypatch.setattr(step, "chat_json", chat)
    args = SimpleNamespace(candidate_count=12, include_all_below=35, temperature=0.1, max_tokens=8000)
    def run(chapter, registry=None):
        return step.reconcile_chapter(None, "mock", chapter, registry if registry is not None else [], args)
    return run, calls


@pytest.mark.parametrize("kind", ["characters", "locations", "objects"])
def test_identity_and_exact_states_never_flatten(mocked_reconcile, kind):
    run, calls = mocked_reconcile
    source = catalog(kind=kind)
    original = deepcopy(source)
    entity, = run(source)
    state = "chapter_appearance" if kind == "characters" else "chapter_state"
    assert source == original
    assert list(entity["timeline"]["chapter"]) == ["1", "2", "3"]
    assert list(entity["timeline"]["chapter"]["1"]) == ["initialState", "event", "endingState"]
    assert entity["first_sequence"] == 1 and entity["first_phase"] == "initialState"
    assert state_at(entity, "chapter", 1, "initialState") == [{state: "has no torch", "evidence": ["empty hands"]}]
    assert state_at(entity, "chapter", 1, "endingState")[0][state] == "eyes closed"
    assert state_at(entity, "chapter", 2, "initialState")[0][state] == "has no torch"
    assert state_at(entity, "chapter", 2, "event")[0][state] == "receives a torch"
    assert state_at(entity, "chapter", 3, "event")[0][state] == "holds the torch"
    assert state_at(entity, "chapter", 3, "initialState") == []
    assert state_at(entity, "missing", 1, "initialState") == []
    assert entity["stable_visual_description"] == source[kind][0]["stable_visual_description"]
    assert entity["distinguishing_features"] == ["Permanent mark"]
    assert all(word not in calls[0] for word in ("torch", "eyes closed", "empty hands", "looks upward"))
    assert "chapter_variations" not in entity
    result = state_at(entity, "chapter", 1, "event")
    result.clear()
    assert state_at(entity, "chapter", 1, "event")


def test_aliases_across_chapters_and_stable_observations(mocked_reconcile):
    run, calls = mocked_reconcile
    first = catalog("chapter-Z")
    second = catalog("chapter-A", name="Henry Jones", alias=["Indy"])
    second["characters"][0]["state_by_sequence"]["1"]["initialState"][0]["chapter_appearance"] = "wears a disguise"
    second["characters"][0]["state_by_sequence"]["3"]["event"][0]["voice_description"] = "Low voice"
    registry = run(second, run(first))
    entity, = registry
    assert {"Henry Jones", "Indy"} <= set(entity["aliases"])
    assert list(entity["timeline"]) == ["chapter-Z", "chapter-A"]
    assert entity["first_chapter_id"] == "chapter-Z"
    assert state_at(entity, "chapter-A", 1, "initialState")[0]["chapter_appearance"] == "wears a disguise"
    assert state_at(entity, "chapter-Z", 1, "initialState")[0]["chapter_appearance"] == "has no torch"
    assert "Low voice" in calls[1] and "wears a disguise" not in calls[1]
    assert step.build_chapter_map(registry) == {"chapter-Z": {"LOCAL_1": "CHAR_001"}, "chapter-A": {"LOCAL_1": "CHAR_001"}}


def test_object_breaks_later_and_audit_preserves_duplicate_observations(mocked_reconcile):
    run, _ = mocked_reconcile
    chapter = catalog(kind="objects", name="rope")
    states = chapter["objects"][0]["state_by_sequence"]
    states["1"] = {"initialState": [{"chapter_state": "supports Indy", "evidence": ["taut rope"]}]}
    states["3"] = {"event": [{"chapter_state": "breaks", "evidence": ["rope snaps"]}]}
    first, = run(chapter)
    second = deepcopy(first)
    second["global_id"] = "OBJ_002"
    second["source_entities"][0]["local_id"] = "LOCAL_2"
    second["timeline"]["chapter"]["1"]["initialState"].append({"chapter_state": "swings"})
    removed = step._apply_audit_result([first, second], {"merge_groups": [{
        "keep_global_id": "OBJ_001", "merge_global_ids": ["OBJ_002"]}]})
    assert removed == {"OBJ_002"}
    assert len(state_at(first, "chapter", 1, "initialState")) == 3  # Even identical observations remain.
    assert all("breaks" not in json.dumps(o) for o in state_at(first, "chapter", 1, "initialState"))
    assert len(state_at(first, "chapter", 3, "event")) == 2
    assert step.build_chapter_map([first])["chapter"]["LOCAL_2"] == "OBJ_001"


def test_first_occurrence_uses_numeric_order_and_phase_rank():
    entity = {}
    merge_timeline(entity, {"B": {"10": {"initialState": [{}]}, "2": {"endingState": [{}], "event": [{}]}},
                            "A": {"1": {"endingState": [{}]}}})
    assert entity["first_sequence"] == 2 and entity["first_phase"] == "event"
    refresh_first_occurrence(entity, ["A", "B"])
    assert entity["first_chapter_id"] == "A"
    assert entity["first_occurrence_by_chapter"]["B"] == {"first_sequence": 2, "first_phase": "event"}


@pytest.mark.parametrize("mutate", [
    lambda c: c.pop("timeline_version"),
    lambda c: c.update(schema_version="minimax-h3-novel-refs.chapter.v3"),
    lambda c: c["characters"][0].pop("state_by_sequence"),
    lambda c: c["characters"][0]["state_by_sequence"].update({"4": {"event": [{}]}}),
    lambda c: c["characters"][0]["state_by_sequence"]["1"].update(future=[{}]),
    lambda c: c["characters"][0]["state_by_sequence"]["1"].update(event=[]),
])
def test_reject_legacy_and_malformed_timeline(mutate):
    chapter = catalog()
    mutate(chapter)
    with pytest.raises(ValueError):
        validate_catalogs([chapter])


def test_duplicate_chapter_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        validate_catalogs([catalog(), catalog()])


def test_node_persistence_and_loader_preserve_canonical_timeline(mocked_reconcile, tmp_path, monkeypatch):
    from contextlib import nullcontext
    from minimax_h3_novel_pipeline import consolidate_references as node, lmstudio_pipeline, path_access, run_output
    from minimax_h3_novel_pipeline.load_consolidated_references import LoadConsolidatedReferencesNode

    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *args: (nullcontext(), "mock"))
    monkeypatch.setattr(node, "prepare_designs", lambda *args: {"entities": []})
    monkeypatch.setattr(step, "generate_picture_assets", lambda *args: [])
    monkeypatch.setattr(step, "generate_audio_assets", lambda *args: [])
    options = {key: value[1]["default"] for key, value in node.ConsolidateReferencesNode.INPUT_TYPES()["required"].items()
               if len(value) > 1 and "default" in value[1]}
    chapters = [catalog("Z"), catalog("A", name="Henry Jones", alias=["Indy"])]
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261003120000", "thinking": False}
    registry, _ = node.ConsolidateReferencesNode().run(chapters, config, **options)["result"]
    path = tmp_path / config["run_folder"] / "references/consolidated_references.json"
    loaded, = LoadConsolidatedReferencesNode().run(str(path))
    assert loaded == registry
    entity, = loaded["entities"]
    assert list(entity["timeline"]) == ["Z", "A"]
    assert entity["timeline"]["Z"] == chapters[0]["characters"][0]["state_by_sequence"]
    assert entity["timeline"]["A"] == chapters[1]["characters"][0]["state_by_sequence"]
    assert loaded["chapter_timelines"]["Z"]["sequences"] == chapters[0]["sequences"]
    assert state_at(entity, "Z", 1, "initialState")[0]["chapter_appearance"] == "has no torch"


@pytest.mark.parametrize("mode", ["missing", "duplicate", "wrong_type", "unknown_match"])
def test_invalid_identity_resolutions_retry_without_losing_observations(monkeypatch, mode):
    from minimax_h3_novel_pipeline import lmstudio_json
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    calls = []
    def chat(*args):
        calls.append(args)
        resolution = {"local_id": "LOCAL_1", "entity_type": "character", "match_global_id": "NEW"}
        if len(calls) > 1:
            return {"resolutions": [resolution]}
        if mode == "missing":
            return {"resolutions": []}
        if mode == "duplicate":
            return {"resolutions": [resolution, resolution]}
        resolution["entity_type" if mode == "wrong_type" else "match_global_id"] = "location" if mode == "wrong_type" else "UNKNOWN"
        return {"resolutions": [resolution]}
    monkeypatch.setattr(step, "chat_json", chat)
    args = SimpleNamespace(candidate_count=12, include_all_below=35, temperature=0.1, max_tokens=8000)
    entity, = step.reconcile_chapter(None, "mock", catalog(), [], args)
    assert len(calls) == 2
    assert len(state_at(entity, "chapter", 1, "event")) == 1

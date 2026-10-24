"""Offline chronology regressions through real reconciliation and duplicate merges."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline.core import pipeline_step2_consolidate as step, util
from minimax_h3_novel_pipeline.core.reference_timeline import (
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
            kind: [{"local_id": "LOCAL_1", "link_to_local_id": None, "canonical_name": name, "aliases": alias or [],
                    "stable_visual_description": "Brown hair" if kind == "characters" else "Hemp fibers",
                    "distinguishing_features": ["Permanent mark"], "voice_description": "Low voice",
                    "first_sequence": 3, "first_phase": "event", "state_by_sequence": observations}]}


@pytest.fixture
def mocked_reconcile(monkeypatch):
    calls = []
    def chat(client, model, system, user, schema, *args):
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


def test_node_persistence_and_loader_preserve_canonical_timeline(mocked_reconcile, tmp_path, monkeypatch, narrative_review):
    from contextlib import nullcontext
    from minimax_h3_novel_pipeline.nodes import consolidate_references as node
    from minimax_h3_novel_pipeline.core import lmstudio_pipeline, path_access, run_output
    from minimax_h3_novel_pipeline.nodes.load_consolidated_references import LoadConsolidatedReferencesNode

    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *args: (nullcontext(), "mock"))
    def prepare_designs(chat, client, model, entities, args, path):
        assert args.temporal_context_by_chapter["Z"]["default_temporal_context"] == "1925"
        assert args.temporal_context_by_chapter["A"]["default_temporal_context"] == "2040"
        assert args.temporal_context_by_chapter["Z"]["sequences"][0]["adaptation"]["event"] == "B chosen event"
        return {"entities": []}
    monkeypatch.setattr(node, "prepare_designs", prepare_designs)
    monkeypatch.setattr(step, "generate_picture_assets", lambda *args: [])
    monkeypatch.setattr(step, "generate_audio_assets", lambda *args: [])
    options = {key: value[1]["default"] for key, value in node.ConsolidateReferencesNode.INPUT_TYPES()["required"].items()
               if len(value) > 1 and "default" in value[1]}
    chapters = [catalog("Z"), catalog("A", name="Henry Jones", alias=["Indy"])]
    chapters[0]["narrative_review"] = narrative_review
    chapters[0]["default_temporal_context"] = "1925"
    chapters[1]["default_temporal_context"] = "2040"
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261003120000", "thinking": False}
    registry, _ = node.ConsolidateReferencesNode().run(chapters, config, **options)["result"]
    path = tmp_path / config["run_folder"] / "references/consolidated_references.json"
    loaded, = LoadConsolidatedReferencesNode().run(str(path))
    assert loaded == registry
    entity, = loaded["entities"]
    assert list(entity["timeline"]) == ["Z", "A"]
    assert entity["timeline"]["Z"] == chapters[0]["characters"][0]["state_by_sequence"]
    assert entity["timeline"]["A"] == chapters[1]["characters"][0]["state_by_sequence"]
    timeline = loaded["chapter_timelines"]["Z"]
    assert timeline["default_temporal_context"] == "1925"
    assert loaded["chapter_timelines"]["A"]["default_temporal_context"] == "2040"
    assert timeline["narrative_review"] == narrative_review
    assert timeline["sequences"][0]["adaptation"]["event"] == "B chosen event"
    assert chapters[0]["sequences"][0]["adaptation"]["event"] == "Action"
    assert state_at(entity, "Z", 1, "initialState")[0]["chapter_appearance"] == "has no torch"


@pytest.mark.parametrize("mode", ["missing", "duplicate", "wrong_type", "unknown_match"])
def test_invalid_identity_resolutions_retry_without_losing_observations(monkeypatch, mode):
    from minimax_h3_novel_pipeline.core import lmstudio_json
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    calls = []
    def chat(*args):
        calls.append(args)
        resolution = {"local_id": "LOCAL_1", "link_to_local_id": None, "entity_type": "character", "match_global_id": "NEW"}
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


@pytest.mark.parametrize("kind", ["characters", "locations", "objects"])
@pytest.mark.parametrize("linked_first", [True, False])
def test_local_links_group_before_model_and_preserve_all_states(mocked_reconcile, kind, linked_first):
    from minimax_h3_novel_pipeline.core import pipeline_step3_generate as generate
    run, calls = mocked_reconcile
    chapter = catalog(kind=kind)
    main = chapter[kind][0]
    other = deepcopy(main)
    other.update(local_id="LOCAL_2", link_to_local_id="LOCAL_1", canonical_name="Secondary mention")
    state = "chapter_appearance" if kind == "characters" else "chapter_state"
    other["state_by_sequence"] = {"1": {"event": [{state: "contradictory observation", "evidence": ["source"]}]}}
    chapter[kind] = [other, main] if linked_first else [main, other]
    original = deepcopy(chapter)
    registry = run(chapter)
    entity, = registry
    incoming = json.loads(calls[0].split("INCOMING:\n", 1)[1].split("\n\nCANDIDATES:", 1)[0])
    assert [item["local_id"] for item in incoming] == ["LOCAL_1"]
    assert "Secondary mention" in incoming[0]["aliases"]
    assert "contradictory observation" not in calls[0]
    assert chapter == original
    assert state_at(entity, "chapter", 1, "event") == main["state_by_sequence"]["1"]["event"] + other["state_by_sequence"]["1"]["event"]
    assert state_at(entity, "chapter", 1, "initialState") == main["state_by_sequence"]["1"]["initialState"]
    mapping = step.build_chapter_map(registry)
    assert mapping["chapter"] == {"LOCAL_1": entity["global_id"], "LOCAL_2": entity["global_id"]}
    refs = {"entities": registry, "chapter_entity_map": mapping, "picture_assets": []}
    assert len(generate.chapter_catalog(refs, "chapter")) == 1
    # The entire group can still match an identity from another chapter.
    registry = run(catalog("next", kind=kind), registry)
    assert len(registry) == 1
    assert step.build_chapter_map(registry)["next"]["LOCAL_1"] == entity["global_id"]


@pytest.mark.parametrize("mode", ["missing_field", "unknown", "self", "wrong_type", "chain", "cycle", "invalid_type"])
def test_invalid_local_links_fail_before_model(mocked_reconcile, mode):
    run, calls = mocked_reconcile
    chapter = catalog()
    main = chapter["characters"][0]
    other = deepcopy(main)
    other.update(local_id="LOCAL_2", link_to_local_id="LOCAL_1")
    chapter["characters"].append(other)
    if mode == "missing_field":
        other.pop("link_to_local_id")
    elif mode == "unknown":
        other["link_to_local_id"] = "MISSING"
    elif mode == "self":
        other["link_to_local_id"] = "LOCAL_2"
    elif mode == "wrong_type":
        chapter["objects"] = [chapter["characters"].pop()]
    elif mode == "chain":
        third = deepcopy(main)
        third.update(local_id="LOCAL_3", link_to_local_id="LOCAL_2")
        chapter["characters"].append(third)
    elif mode == "cycle":
        main["link_to_local_id"] = "LOCAL_2"
    else:
        other["link_to_local_id"] = []
    with pytest.raises(ValueError, match="chapter/.*link_to_local_id"):
        run(chapter)
    assert calls == []


def test_catalog_edits_invalidate_loader_cache(tmp_path, monkeypatch):
    from minimax_h3_novel_pipeline.core import path_access
    from minimax_h3_novel_pipeline.nodes.load_chapter_catalogs import LoadChapterCatalogsNode
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    path = tmp_path / "chapter_references.json"
    chapter = catalog()
    util.save_json(path, chapter)
    before = LoadChapterCatalogsNode.IS_CHANGED(str(tmp_path))
    chapter["characters"][0]["canonical_name"] = "Corrected identity"
    util.save_json(path, chapter)
    assert LoadChapterCatalogsNode.IS_CHANGED(str(tmp_path)) != before
    loaded, _ = LoadChapterCatalogsNode().run(str(path))
    assert loaded == [chapter]


def test_node_global_designs_only_import_selected_catalog_characters(mocked_reconcile, tmp_path, monkeypatch):
    from contextlib import nullcontext
    from minimax_h3_novel_pipeline.nodes import consolidate_references as node
    from minimax_h3_novel_pipeline.core import lmstudio_pipeline, path_access, run_output, visual_designs
    from tests.test_image_reference_outputs import design_checks

    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *args: (nullcontext(), "mock"))
    reconcile_chat = step.chat_json
    calls = []

    def chat(client, model, system, user, schema, *args):
        calls.append(schema["name"])
        if schema is visual_designs.CONFLICT_SCHEMA:
            return design_checks(user)
        return reconcile_chat(client, model, system, user, schema, *args)

    monkeypatch.setattr(step, "chat_json", chat)
    monkeypatch.setattr(step, "generate_picture_assets", lambda *args: [])
    monkeypatch.setattr(step, "generate_audio_assets", lambda *args: [])
    current = catalog("selected-chapter", name="Indy")
    path = tmp_path / "global/visual_designs.json"
    util.save_json(path, {"schema_version": visual_designs.DESIGN_SCHEMA_VERSION,
                         "image_style": "cinematic photographic", "entities": [
        {"global_id": "char_henry", "entity_type": "character", "canonical_name": "Henry Jones",
         "aliases": ["Indy"], "source_facts": {"stable_visual_description": "Unrelated chapter disguise",
                                                "distinguishing_features": []},
         "added_details": {"eyes": "Edited brown eyes."}},
        {"global_id": "CHAR_001", "entity_type": "character", "canonical_name": "Absent villain",
         "source_facts": {"stable_visual_description": "", "distinguishing_features": []},
         "added_details": {"default_outfit": "Other chapter uniform."}},
    ]})
    options = {key: value[1]["default"] for key, value in node.ConsolidateReferencesNode.INPUT_TYPES()["required"].items()
               if len(value) > 1 and "default" in value[1]}
    options["no_audit"] = True
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261003120000", "thinking": False}
    result, _ = node.ConsolidateReferencesNode().run([current], config, visual_designs_path=str(path), **options)["result"]
    character, = result["entities"]
    assert character["canonical_name"] == "Indy"
    assert character["timeline"]["selected-chapter"] == current["characters"][0]["state_by_sequence"]
    imported, = result["visual_designs"]["entities"]
    assert imported["added_details"] == {"eyes": "Edited brown eyes."}
    assert imported["source_facts"]["stable_visual_description"] == "Brown hair"
    assert len(calls) == 2  # Existing reconciliation and conflict check only.
    assert len(util.load_json(path)["entities"]) == 2

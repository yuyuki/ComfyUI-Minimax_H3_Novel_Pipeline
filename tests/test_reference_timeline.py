from tests.neutral_reference_helpers import catalog as neutral_catalog
"""Offline chronology regressions through real reconciliation and duplicate merges."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline.core import pipeline_step2_consolidate as step, util
from minimax_h3_novel_pipeline.core.reference_timeline import (
    merge_timeline, refresh_first_occurrence, source_observations_at, validate_catalogs,
)


def catalog(cid="chapter", kind="characters", name="Indy", alias=None):
    observations = {
        "3": {"source": [{"evidence": ["torch held"]}]},
        "1": {"source": [{"evidence": ["empty hands"]}, {"evidence": ["looks"]}, {"evidence": ["closes eyes"]}]},
        "2": {"source": [{"evidence": ["receives a torch"]}, {"evidence": ["holds the torch"]}]},
    }
    return neutral_catalog({"schema_version": util.CHAPTER_SCHEMA, "timeline_version": util.REFERENCE_TIMELINE_VERSION,
            "chapter_id": cid, "chapter_name": cid, "source": {},
            "sequences": [{"sequence": n, "source": "Evidence", "adaptation": {
                "initialState": "Opening", "event": "Action", "endingState": "Ending"}} for n in (1, 2, 3)],
            **{k: [] for k in ("characters", "locations", "objects")},
            kind: [{"local_id": "LOCAL_1", "canonical_name": name, "aliases": alias or [],
                    "stable_visual_description": "Brown hair" if kind == "characters" else "Hemp fibers",
                    "distinguishing_features": ["Permanent mark"], "voice_description": "Low voice",
                    "first_sequence": 3, "state_by_sequence": observations}]})


@pytest.fixture
def mocked_reconcile(monkeypatch):
    calls = []
    def chat(client, model, system, user, schema, *args):
        data = json.JSONDecoder().raw_decode(user)[0]
        if schema == step.IDENTITY_REVIEW_SCHEMA:
            return {"decisions": [{"decision_number": row["decision_number"], "verdict": "supported"}
                                  for row in data["decisions"]]}
        incoming, candidates = data["incoming"], data["candidates"]
        calls.append(data)
        resolutions = []
        fields = set(step.reconciliation_item_schema()["properties"])
        for item in incoming:
            lid = item["local_id"]
            match = candidates[lid][0]["global_id"] if candidates[lid] else "NEW"
            selected = [row["id"] for row in data["source_passages"] if row["entity"] in {lid, match}] if match != "NEW" else []
            resolutions.append({**{key: value for key, value in item.items() if key in fields},
                                "match_global_id": match, "passage_ids": selected, "reason": "", "confidence": 1})
        return {"resolutions": resolutions}
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
    assert source == original
    assert list(entity["timeline"]["chapter"]) == ["1", "2", "3"]
    assert list(entity["timeline"]["chapter"]["1"]) == ["source"]
    assert entity["first_sequence"] == 1 and "first_phase" not in entity
    assert source_observations_at(entity, "chapter", 1) == source[kind][0]["state_by_sequence"]["1"]["source"]
    assert source_observations_at(entity, "chapter", 2)[0]["evidence"] == ["receives a torch"]
    assert source_observations_at(entity, "chapter", 3)[0]["evidence"] == ["torch held"]
    assert source_observations_at(entity, "missing", 1) == []
    assert entity["stable_visual_description"] == source[kind][0]["stable_visual_description"]
    assert entity["distinguishing_features"] == ["Permanent mark"]
    assert any(row["text"] == "empty hands" for row in calls[0]["source_passages"])
    assert all(word not in json.dumps(calls[0]["incoming"]) for word in ("torch", "eyes closed", "empty hands", "looks upward"))
    assert "chapter_variations" not in entity
    result = source_observations_at(entity, "chapter", 1)
    result.clear()
    assert source_observations_at(entity, "chapter", 1)


def test_aliases_across_chapters_and_stable_observations(mocked_reconcile):
    run, calls = mocked_reconcile
    first = catalog("chapter-Z")
    second = catalog("chapter-A", name="Henry Jones", alias=["Indy"])
    second["characters"][0]["state_by_sequence"]["1"]["source"][0]["evidence"] = ["wears a disguise"]
    second["characters"][0]["state_by_sequence"]["3"]["source"][0]["voice_description"] = "Low voice"
    registry = run(second, run(first))
    entity, = registry
    assert {"Henry Jones", "Indy"} <= set(entity["aliases"])
    assert list(entity["timeline"]) == ["chapter-Z", "chapter-A"]
    assert entity["first_chapter_id"] == "chapter-Z"
    assert source_observations_at(entity, "chapter-A", 1)[0]["evidence"] == ["wears a disguise"]
    assert source_observations_at(entity, "chapter-Z", 1)[0]["evidence"] == ["empty hands"]
    assert "Low voice" in json.dumps(calls[1]["incoming"])
    assert any(row["text"] == "wears a disguise" and row["chapter"] == "chapter-A"
               for row in calls[1]["source_passages"])
    assert step.build_chapter_map(registry) == {"chapter-Z": {"LOCAL_1": "CHAR_001"}, "chapter-A": {"LOCAL_1": "CHAR_001"}}


def test_object_breaks_later_and_audit_preserves_duplicate_observations(mocked_reconcile):
    run, _ = mocked_reconcile
    chapter = catalog(kind="objects", name="rope")
    states = chapter["objects"][0]["state_by_sequence"]
    states["1"] = {"source": [{"evidence": ["taut rope"]}]}
    states["3"] = {"source": [{"evidence": ["rope snaps"]}]}
    first, = run(neutral_catalog(chapter))
    second = deepcopy(first)
    second["global_id"] = "OBJ_002"
    second["source_entities"][0]["local_id"] = "LOCAL_2"
    second["timeline"]["chapter"]["1"]["source"].append({"evidence": ["swings"]})
    removed = step._apply_audit_result([first, second], {"merge_groups": [{
        "keep_global_id": "OBJ_001", "merge_global_ids": ["OBJ_002"]}]})
    assert removed == {"OBJ_002"}
    assert len(source_observations_at(first, "chapter", 1)) == 3  # Even identical observations remain.
    assert all("rope snaps" not in json.dumps(o) for o in source_observations_at(first, "chapter", 1))
    assert len(source_observations_at(first, "chapter", 3)) == 2
    assert step.build_chapter_map([first])["chapter"]["LOCAL_2"] == "OBJ_001"


def test_first_occurrence_uses_numeric_and_chapter_order():
    entity = {}
    merge_timeline(entity, {"B": {"10": {"source": [{}]}, "2": {"source": [{}]}},
                            "A": {"1": {"source": [{}]}}})
    assert entity["first_sequence"] == 2 and "first_phase" not in entity
    refresh_first_occurrence(entity, ["A", "B"])
    assert entity["first_chapter_id"] == "A"
    assert entity["first_occurrence_by_chapter"]["B"] == {"first_sequence": 2}


@pytest.mark.parametrize("mutate", [
    lambda c: c.pop("timeline_version"),
    lambda c: c.update(timeline_version="cinematic-reference-timeline.v1"),
    lambda c: c["characters"][0].update(link_to_local_id=None),
    lambda c: c["characters"][0].update(first_phase="source"),
    lambda c: c.update(schema_version="minimax-h3-novel-refs.chapter.v3"),
    lambda c: c["characters"][0].pop("state_by_sequence"),
    lambda c: c["characters"][0]["state_by_sequence"].update({"4": {"event": [{}]}}),
    lambda c: c["characters"][0]["state_by_sequence"]["1"].update(future=[{}]),
    lambda c: c["characters"][0]["state_by_sequence"]["1"].update(source=[]),
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
    assert source_observations_at(entity, "Z", 1)[0]["evidence"] == ["empty hands"]


@pytest.mark.parametrize("mode", ["missing", "duplicate", "wrong_type", "unknown_match"])
def test_invalid_identity_resolutions_retry_without_losing_observations(monkeypatch, mode):
    from minimax_h3_novel_pipeline.core import lmstudio_json
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    calls = []
    def chat(*args):
        calls.append(args)
        item = json.JSONDecoder().raw_decode(args[3])[0]["incoming"][0]
        resolution = {**{key: value for key, value in item.items() if key in step.reconciliation_item_schema()["properties"]},
                      "match_global_id": "NEW", "passage_ids": [], "reason": "", "confidence": 1}
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
    assert len(source_observations_at(entity, "chapter", 1)) == 3


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
        if schema is visual_designs.DESIGN_SCHEMA:
            return {"added_details": [{"trait": "default_outfit", "description": "A brown wool jacket."}]}
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
    assert imported["added_details"] == {"eyes": "Edited brown eyes.", "default_outfit": "A brown wool jacket."}
    assert imported["source_facts"]["stable_visual_description"] == ""
    assert calls.count("visual_design_additions") == 1
    assert len(util.load_json(path)["entities"]) == 2


def test_source_observations_survive_reconciliation_audit_and_generation(mocked_reconcile):
    from minimax_h3_novel_pipeline.core.pipeline_step3_generate import sequence_catalog
    run, _ = mocked_reconcile
    chapter = catalog()
    chapter["characters"][0].update(first_sequence=1,
        state_by_sequence={"1": {"source": [{"evidence": ["rope"]}]}})
    other = deepcopy(chapter["characters"][0])
    other.update(local_id="LOCAL_2", canonical_name="Henry", aliases=["Indy"],
                 state_by_sequence={"2": {"source": [{"evidence": ["waits"]}]}})
    chapter["characters"].append(other)
    chapter = neutral_catalog(chapter)
    registry = run(chapter)
    assert len(registry) == 2
    target, duplicate = [e["global_id"] for e in registry]
    # Exercise the real audit merge contract directly; the identity decision is mocked.
    result = {"merge_groups": [{"keep_global_id": target, "merge_global_ids": [duplicate]}]}
    removed = step._apply_audit_result(registry, result)
    assert removed == {duplicate}
    entity, = [e for e in registry if e["global_id"] not in removed]
    assert set(entity["timeline"]["chapter"]) == {"1", "2"}
    assert "first_phase" not in entity
    from minimax_h3_novel_pipeline.core import reference_visuals
    entity["visual_reference"] = reference_visuals.compile_reference(entity["timeline"], [], entity["global_id"])
    projected = sequence_catalog({"entities": [entity]}, "chapter", 1)
    assert projected[0]["source_observations"] == entity["timeline"]["chapter"]["1"]["source"]
    assert "state_by_phase" not in projected[0]
    assert sequence_catalog({"entities": [entity]}, "chapter", 3) == []

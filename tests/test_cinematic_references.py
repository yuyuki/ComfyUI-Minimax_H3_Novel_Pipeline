"""Timeline boundaries are enforced before mocked model extraction."""
from copy import deepcopy
from contextlib import nullcontext
import argparse
import json

import pytest

from minimax_h3_novel_pipeline import cinematic_references as cr, lmstudio_pipeline, path_access, run_output, util
from minimax_h3_novel_pipeline.extract_chapter_references import ExtractChapterReferencesNode
from minimax_h3_novel_pipeline.load_cinematic_chapters import LoadCinematicChaptersNode


def chapter(name="chapter"):
    return {"chapter_name": name, "source_file": "/not/read.txt", "sequences": [
        {"sequence": 1, "source": "Original supporting text mentions a torch later.", "adaptation": {
            "initialState": "Indy hangs from a rope.", "event": "Indy waits.", "endingState": "Indy hangs."}},
        {"sequence": 2, "source": "Throwing.", "adaptation": {
            "initialState": "Indy hangs.", "event": "Doriane throws a torch toward Indy.",
            "endingState": "The torch is in flight."}},
        {"sequence": 3, "source": "Catching.", "adaptation": {
            "initialState": "The torch is in flight.", "event": "Indy catches the torch.",
            "endingState": "Indy holds the torch."}},
    ]}


@pytest.fixture
def execution(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock"))
    calls = []
    def extract(client, model, system, user, schema, temperature, max_tokens):
        data = json.loads(user)
        if schema["name"] == "narrative_semantic_review":
            return {"issues": []}
        if "sequence" not in data:
            return {"links": [{"local_id": entity["local_id"], "link_to_local_id": None}
                              for entities in data.values() for entity in entities]}
        calls.append(data)
        result = {"chunk_summary": "", "characters": [], "locations": [], "objects": []}
        for name, kind in (("Indy", "characters"), ("Doriane", "characters"), ("torch", "objects")):
            if name in data["text"]:
                state = "chapter_appearance" if kind == "characters" else "chapter_state"
                result[kind].append({"canonical_name": name, "aliases": [], state: data["text"]})
        return result
    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", extract)
    params = {k: spec[1]["default"] for k, spec in ExtractChapterReferencesNode.INPUT_TYPES()["required"].items()
              if len(spec) > 1 and "default" in spec[1]}
    return calls, params, {"api_url": "http://localhost:1234/v1", "run_folder": "20261003120000"}


def test_multiple_chapters_ordered_phases_and_identity(execution):
    calls, params, config = execution
    chapters = [chapter(), chapter("second")]
    original = deepcopy(chapters)
    response = ExtractChapterReferencesNode().run(config, chapters, **params)
    result, summary = response["result"]
    assert response["ui"] == {"text": [summary]}
    assert chapters == original
    assert len(result) == 2
    assert [c["chapter_id"] for c in result] == ["chapter", "second"]
    assert [(c["sequence"], c["phase"]) for c in calls[:9]] == [
        (i, p) for i in (1, 2, 3) for p in cr.PHASES]
    assert all("torch" not in c["text"] for c in calls if c["sequence"] == 1)
    for catalog in result:
        assert "chapter_summary" not in catalog
        assert catalog["sequences"] == chapters[0]["sequences"]
        torch = catalog["objects"][0]
        assert torch["first_sequence"] == 2
        assert torch["first_phase"] == "event"
        states = torch["state_by_sequence"]
        assert "1" not in states and "initialState" not in states["2"]
        assert "throws" in states["2"]["event"][0]["chapter_state"]
        assert "in flight" in states["3"]["initialState"][0]["chapter_state"]
        assert "catches" in states["3"]["event"][0]["chapter_state"]
        assert "holds" in states["3"]["endingState"][0]["chapter_state"]
        assert torch["chapter_state"] == ""
        assert torch["link_to_local_id"] is None
        indy = [e for e in catalog["characters"] if e["canonical_name"] == "Indy"]
        assert len(indy) == 1
        assert list(indy[0]["state_by_sequence"]) == ["1", "2", "3"]
    count = len(calls)
    cached = ExtractChapterReferencesNode().run(config, chapters, **params)
    assert cached == response
    assert len(calls) == count
    chapters[0]["sequences"][0]["adaptation"]["event"] = "Indy shouts."
    ExtractChapterReferencesNode().run(config, chapters, **params)
    assert len(calls) > count


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(sequences=[]),
    lambda c: c["sequences"][1].update(sequence=1),
    lambda c: c["sequences"].reverse(),
    lambda c: c["sequences"][0].update(sequence=True),
    lambda c: c["sequences"][0]["adaptation"].pop("event"),
    lambda c: c["sequences"][0]["adaptation"].update(event=[]),
])
def test_invalid_timeline(mutate):
    value = chapter()
    mutate(value)
    with pytest.raises(ValueError):
        cr.parse_chapters([value])


def test_serialized_input_and_socket():
    assert cr.parse_chapters(json.dumps([chapter()])) == [chapter()]
    inputs = ExtractChapterReferencesNode.INPUT_TYPES()["required"]
    assert "chapter_selection" not in inputs
    assert inputs["cinematic_chapters"][0] == "MINIMAX_CINEMATIC_CHAPTERS"


def test_duplicate_chapter_names_have_separate_files(execution):
    _, params, config = execution
    results, _ = ExtractChapterReferencesNode().run(config, [chapter(), chapter()], **params)["result"]
    assert [c["chapter_id"] for c in results] == ["chapter_001", "chapter_002"]
    assert all(c["schema_version"] == util.CHAPTER_SCHEMA for c in results)


def test_extraction_resumes_from_saved_cinematic_chapter(execution, tmp_path):
    calls, params, config = execution
    value = chapter()
    value.pop("source_file")
    util.save_json(tmp_path / "previous/001_chapter.cinematic.json", value)
    loaded, _ = LoadCinematicChaptersNode().run("previous")
    results, _ = ExtractChapterReferencesNode().run(config, loaded, **params)["result"]
    assert results[0]["sequences"] == value["sequences"]
    assert [(c["sequence"], c["phase"]) for c in calls] == [
        (i, phase) for i in (1, 2, 3) for phase in cr.PHASES]


def test_old_completed_catalog_cache_is_regenerated(execution, tmp_path):
    _, params, config = execution
    node = ExtractChapterReferencesNode()
    result, _ = node.run(config, [chapter()], **params)["result"]
    saved, = list(tmp_path.rglob("chapter_references.json"))
    old = {**result[0], "schema_version": "minimax-h3-novel-refs.chapter.v3"}
    util.save_json(saved, old)
    regenerated, _ = node.run(config, [chapter()], **params)["result"]
    assert regenerated[0]["schema_version"] == util.CHAPTER_SCHEMA
    assert regenerated[0]["characters"] == result[0]["characters"]


def test_catalog_without_summary_reuses_phase_cache(execution, tmp_path):
    calls, params, config = execution
    node = ExtractChapterReferencesNode()
    result, _ = node.run(config, [chapter()], **params)["result"]
    saved, = list(tmp_path.rglob("chapter_references.json"))
    step = lmstudio_pipeline.load("extract")
    args = argparse.Namespace(
        chunk_chars=params["chunk_chars"], temperature=params["temperature"],
        max_tokens=params["max_tokens"], force=params["force"], base_url=config["api_url"],
    )
    old_key = cr.fingerprint(
        "mock", args, cr.TEMPORAL_VERSION, step.EXTRACT_SYSTEM + "\n" + cr.PHASE_SYSTEM,
        step.CHUNK_SCHEMA, chapter(), client=None,
    )
    util.save_json(saved, {**result[0], "cache_key": old_key, "chapter_summary": ""})
    count = len(calls)

    regenerated, _ = node.run(config, [chapter()], **params)["result"]

    assert regenerated == result
    assert "chapter_summary" not in util.load_json(saved)
    assert regenerated[0]["cache_key"] != old_key
    assert len(calls) == count


def test_catalog_schema_is_copied_and_validates_local_links(execution, tmp_path):
    _, params, config = execution
    results, _ = ExtractChapterReferencesNode().run(config, [chapter()], **params)["result"]
    schema_path, = list(tmp_path.rglob("chapter_references.schema.json"))
    assert schema_path.is_file()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    cr.validate_document(results[0], schema, "chapter_references.json")


def test_local_link_review_connects_descriptive_reference(monkeypatch):
    catalog = {
        "characters": [
            {"local_id": "CHAR_001", "canonical_name": "Doriane", "aliases": [],
             "stable_visual_description": "", "evidence": ["Doriane cries."], "state_by_sequence": {"1": {}}},
            {"local_id": "CHAR_002", "canonical_name": "someone screaming", "aliases": [],
             "stable_visual_description": "", "evidence": ["Someone screams; it is Doriane."], "state_by_sequence": {"1": {}}},
        ],
        "locations": [],
        "objects": [],
    }
    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", lambda *args: {"links": [
        {"local_id": "CHAR_001", "link_to_local_id": None},
        {"local_id": "CHAR_002", "link_to_local_id": "CHAR_001"},
    ]})

    cr.assign_local_links(catalog, None, "mock", argparse.Namespace(temperature=0.18, max_tokens=8192))

    assert catalog["characters"][0]["link_to_local_id"] is None
    assert catalog["characters"][1]["link_to_local_id"] == "CHAR_001"


def test_local_link_review_defaults_omitted_entities_to_null(monkeypatch):
    catalog = {
        "characters": [
            {"local_id": "CHAR_001", "canonical_name": "Doriane", "aliases": [],
             "stable_visual_description": "", "evidence": [], "state_by_sequence": {"1": {}}},
            {"local_id": "CHAR_002", "canonical_name": "someone screaming", "aliases": [],
             "stable_visual_description": "", "evidence": [], "state_by_sequence": {"1": {}}},
        ],
        "locations": [],
        "objects": [],
    }
    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", lambda *args: {"links": [
        {"local_id": "CHAR_002", "link_to_local_id": "CHAR_001"},
    ]})

    cr.assign_local_links(catalog, None, "mock", argparse.Namespace(temperature=0.18, max_tokens=8192))

    assert catalog["characters"][0]["link_to_local_id"] is None
    assert catalog["characters"][1]["link_to_local_id"] == "CHAR_001"


def test_identity_context_contains_only_prior_observations(execution):
    calls, params, config = execution
    ExtractChapterReferencesNode().run(config, [chapter()], **params)
    assert all(not entities for entities in calls[0]["known_references"].values())
    for call in calls:
        context = call["known_references"]
        if call["sequence"] == 1:
            assert "torch" not in json.dumps(context)
        if call["sequence"] == 2 and call["phase"] == "event":
            assert context["objects"] == []
            assert context["characters"][0]["canonical_name"] == "Indy"
        if call["sequence"] == 3 and call["phase"] == "event":
            assert context["objects"][0]["canonical_name"] == "torch"
            assert "catches" not in json.dumps(context)


@pytest.mark.parametrize("period", ["", "  1925  ", "France, fin du XIXe siècle"])
def test_default_temporal_context_persistence_and_cache(execution, tmp_path, period):
    calls, params, config = execution
    node = ExtractChapterReferencesNode()
    source = chapter()
    source["sequences"][0]["adaptation"]["initialState"] = "In 1810, Indy hangs from a rope."
    original = deepcopy(source)
    results, _ = node.run(config, [source], default_temporal_context=period, **params)["result"]
    assert results[0]["default_temporal_context"] == period.strip()
    assert source == original
    assert all(c["default_temporal_context"] == period.strip() for c in calls)
    assert "1810" in calls[0]["text"]
    saved, = tmp_path.rglob("chapter_references.json")
    assert util.load_json(saved) == results[0]
    count = len(calls)
    node.run(config, [source], default_temporal_context=period, **params)
    assert len(calls) == count
    changed, _ = node.run(config, [source], default_temporal_context="2040", **params)["result"]
    assert len(calls) > count
    assert changed[0]["cache_key"] != results[0]["cache_key"]
    assert changed[0]["source"] == results[0]["source"]


def test_invalid_temporal_context_rejected_before_model_work(execution):
    calls, params, config = execution
    with pytest.raises(ValueError, match="must be a string"):
        ExtractChapterReferencesNode().run(config, [chapter()], default_temporal_context=1925, **params)
    assert calls == []

"""Regression coverage for ordered cinematic reference extraction."""
from contextlib import nullcontext
from copy import deepcopy
import json

import pytest

from tests.neutral_reference_helpers import candidates as neutral_candidates
from tests.passages_helpers import with_passage_ids

from minimax_h3_novel_pipeline.core import cinematic_references as cr, lmstudio_pipeline, path_access, run_output, util
from minimax_h3_novel_pipeline.nodes.extract_chapter_references import ExtractChapterReferencesNode
from minimax_h3_novel_pipeline.nodes.load_cinematic_chapters import LoadCinematicChaptersNode


def test_summary_scales_to_passage_within_shared_output_budget():
    from minimax_h3_novel_pipeline.core.editable_schemas import validate_document

    step = lmstudio_pipeline.load("extract")
    summary = "Une corde apparaît. " * 30
    assert len(summary) > 450
    result = {"chunk_summary": summary, "characters": [], "locations": [], "objects": []}
    validate_document(result, step.CHUNK_SCHEMA["schema"], step.CHUNK_SCHEMA["name"])
    summary_schema = step.CHUNK_SCHEMA["schema"]["properties"]["chunk_summary"]
    assert "maxLength" not in summary_schema
    assert "proportional to the current local passage" in summary_schema["description"]
    assert "Scale its length to the current local passage" in step.EXTRACT_SYSTEM
    assert "output token budget is shared" in step.EXTRACT_SYSTEM
    assert "450" not in step.EXTRACT_SYSTEM + summary_schema["description"]
    assert "Shorten by rephrasing" in step.EXTRACT_SYSTEM


def chapter(name="chapter"):
    return {"chapter_name": name, "source_file": "/not/read.txt", "sequences": [
        {"sequence": 1, "source": "Indy hangs from a rope.", "adaptation": {
            "initialState": "Indy hangs from a rope.", "event": "Indy waits.", "endingState": "Indy hangs."}},
        {"sequence": 2, "source": "Doriane throws a brass torch toward Indy.", "adaptation": {
            "initialState": "Indy hangs.", "event": "Doriane throws a brass torch toward Indy.",
            "endingState": "The torch is in flight."}},
        {"sequence": 3, "source": "Indy catches the torch.", "adaptation": {
            "initialState": "The torch is in flight.", "event": "Indy catches the torch.",
            "endingState": "Indy holds the torch."}},
    ]}


def _reference_scene(sequence, opening, event):
    return {"sequence": sequence, "source": "Excluded original.",
            "adaptation": {"initialState": opening, "event": event, "endingState": "Excluded ending."}}


def _text(payload):
    return " ".join(p["text"] for p in payload["source_passages"])


@pytest.fixture
def execution(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock"))
    calls = []

    def extract(client, model, system, user, schema, temperature, max_tokens):
        data = json.loads(user)
        assert schema == lmstudio_pipeline.load("extract").CHUNK_SCHEMA
        calls.append(data)
        text = _text(data)
        known = {row["name"]: row["reference_id"] for row in data["known_references"]}
        result = {"chunk_summary": "", "characters": [], "locations": [], "objects": []}
        for name, kind in (("Indy", "characters"), ("Doriane", "characters"), ("torch", "objects")):
            if name in text:
                result[kind].append({"canonical_name": name, "reference_id": known.get(name, "new")})
        result = neutral_candidates(result, text)
        for prop in result["objects"]:
            prop["visual_traits"] = [{"attribute": "material", "description": "brass", "scope": "intrinsic",
                                      "evidence": "brass torch"}] if "brass torch" in text else []
        return result
    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", with_passage_ids(extract))
    params = {k: spec[1]["default"] for k, spec in ExtractChapterReferencesNode.INPUT_TYPES()["required"].items()
              if len(spec) > 1 and "default" in spec[1]}
    return calls, params, {"api_url": "http://localhost:1234/v1", "run_folder": "20261003120000"}


def test_multiple_chapters_ordered_sources_and_identity(execution):
    calls, params, config = execution
    chapters = [chapter(), chapter("second")]
    original = deepcopy(chapters)
    response = ExtractChapterReferencesNode().run(config, chapters, **params)
    result, summary = response["result"]
    assert response["ui"] == {"text": [summary]}
    assert chapters == original
    assert len(result) == 2
    assert [c["chapter_id"] for c in result] == ["chapter", "second"]
    assert [c["sequence"] for c in calls] == [1, 2, 3, 1, 2, 3]
    assert [_text(c) for c in calls] == [
        " ".join(s["adaptation"][p] for p in ("initialState", "event"))
        for c in chapters for s in c["sequences"]]
    assert all("phase" not in c and "chapter_context" not in c for c in calls)
    assert all(not c["known_references"] for c in calls if c["sequence"] == 1)
    assert all(c["known_references"][0]["name"] == "Indy" for c in calls if c["sequence"] == 2)
    for catalog in result:
        assert "chapter_summary" not in catalog
        assert catalog["sequences"] == chapters[0]["sequences"]
        torch = catalog["objects"][0]
        assert torch["first_sequence"] == 2
        assert "first_phase" not in torch
        states = torch["state_by_sequence"]
        assert "1" not in states and list(states["2"]) == ["source"]
        assert states["2"]["source"][0]["visual_traits"][0]["scope"] == "intrinsic"
        assert states["3"]["source"][0]["visual_traits"] == []
        assert "chapter_state" not in torch and "chapter_state" not in states["2"]["source"][0]
        assert "link_to_local_id" not in torch
        indy = [e for e in catalog["characters"] if e["canonical_name"] == "Indy"]
        assert len(indy) == 1
        assert list(indy[0]["state_by_sequence"]) == ["1", "2", "3"]
    count = len(calls)
    cached = ExtractChapterReferencesNode().run(config, chapters, **params)
    assert cached == response
    assert len(calls) == count
    chapters[0]["sequences"][0]["adaptation"]["event"] = "Indy shouts."
    ExtractChapterReferencesNode().run(config, chapters, **params)
    assert count < len(calls) <= count + 3
    count = len(calls)
    chapters[0]["sequences"][0]["source"] = "Indy shouts."
    ExtractChapterReferencesNode().run(config, chapters, **params)
    assert len(calls) == count


def test_resolved_adaptation_names_feed_extraction_and_identity(execution):
    calls, params, config = execution
    value = chapter()
    for seq in value["sequences"]:
        seq["source"] = "L'homme attend avec un objet absent de l'adaptation."
        seq["adaptation"]["endingState"] = "ENDING_ONLY_CHARACTER arrives with ENDING_ONLY_OBJECT."
    result, _ = ExtractChapterReferencesNode().run(config, [value], **params)["result"]
    assert all("L'homme" not in json.dumps(call) and "ENDING_ONLY" not in json.dumps(call) for call in calls)
    assert [e["canonical_name"] for e in result[0]["characters"]] == ["Indy", "Doriane"]
    assert result[0]["sequences"] == value["sequences"]


@pytest.mark.parametrize("field", ["initialState", "event"])
def test_one_nonempty_adaptation_field_is_sufficient(execution, field):
    calls, params, config = execution
    value = chapter()
    value["sequences"] = value["sequences"][:1]
    value["sequences"][0]["adaptation"].update(initialState="", event="")
    value["sequences"][0]["adaptation"][field] = "Indy waits."
    ExtractChapterReferencesNode().run(config, [value], **params)
    assert [_text(call) for call in calls] == ["Indy waits."]
    assert bool(calls[0]["opening_passage_ids"]) == (field == "initialState")


def test_ending_state_and_source_do_not_replace_empty_extraction_input(execution):
    calls, params, config = execution
    value = chapter()
    for seq in value["sequences"]:
        seq["adaptation"].update(initialState="", event="")
    with pytest.raises(ValueError, match="non-empty initialState or event"):
        ExtractChapterReferencesNode().run(config, [value], **params)
    assert calls == []


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
    assert [c["sequence"] for c in calls] == [1, 2, 3]


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


def test_old_catalog_reuses_source_chunk_cache(execution, tmp_path):
    calls, params, config = execution
    node = ExtractChapterReferencesNode()
    result, _ = node.run(config, [chapter()], **params)["result"]
    saved, = list(tmp_path.rglob("chapter_references.json"))
    diagnostic, = tmp_path.rglob("*.diagnostics.json")
    diagnostic.unlink()
    util.save_json(saved, {**result[0], "cache_key": "obsolete-phase-cache", "chapter_summary": ""})
    count = len(calls)

    regenerated, _ = node.run(config, [chapter()], **params)["result"]

    assert regenerated == result
    assert "chapter_summary" not in util.load_json(saved)
    assert "cache_key" not in regenerated[0]
    assert len(calls) == count


def test_catalog_schema_is_copied_and_validates_local_links(execution, tmp_path):
    _, params, config = execution
    results, _ = ExtractChapterReferencesNode().run(config, [chapter()], **params)["result"]
    schema_path, = list(tmp_path.rglob("chapter_references.schema.json"))
    assert schema_path.is_file()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    cr.validate_document(results[0], schema, "chapter_references.json")


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
    assert "1810" in _text(calls[0])
    saved, = tmp_path.rglob("chapter_references.json")
    assert util.load_json(saved) == results[0]
    count = len(calls)
    node.run(config, [source], default_temporal_context=period, **params)
    assert len(calls) == count
    changed, _ = node.run(config, [source], default_temporal_context="2040", **params)["result"]
    assert len(calls) > count
    assert "cache_key" not in changed[0]
    assert changed[0]["source"] == results[0]["source"]


def test_invalid_temporal_context_rejected_before_model_work(execution):
    calls, params, config = execution
    with pytest.raises(ValueError, match="must be a string"):
        ExtractChapterReferencesNode().run(config, [chapter()], default_temporal_context=1925, **params)
    assert calls == []


@pytest.mark.parametrize("failure", [OSError("disk"), KeyboardInterrupt()])
def test_integrity_and_cancellation_propagate(execution, monkeypatch, failure):
    _, params, config = execution

    def chat(client, model, system, user, schema, *args):
        raise failure

    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", chat)
    with pytest.raises(type(failure)):
        ExtractChapterReferencesNode().run(config, [chapter()], **params)


def test_diagnostics_preserve_editable_choices(execution, narrative_review, tmp_path):
    _, params, config = execution
    value = chapter()
    value["narrative_review"] = {**narrative_review, "warnings": [{"reason": "incomplete"}]}
    value["sequences"][0]["review"] = {"status": "incomplete"}
    results, _ = ExtractChapterReferencesNode().run(config, [value], **params)["result"]
    assert results[0]["narrative_review"]["issues"] == narrative_review["issues"]
    assert "warnings" not in results[0]["narrative_review"]
    assert "review" not in results[0]["sequences"][0]
    diagnostic, = tmp_path.rglob("*.diagnostics.json")
    assert util.load_json(diagnostic)["sequence_reviews"][0]["review"]["status"] == "incomplete"


def test_progressive_prologue_references_and_restart(execution, monkeypatch, tmp_path):
    _, params, config = execution
    chapter_data = {"chapter_name": "Prologue", "sequences": [
        _reference_scene(1, "Indy est suspendu à une corde dans la crevasse aux parois rocheuses.",
                         'Indy crie « Doriane ! »'),
        _reference_scene(2, "Indy se balance dans la crevasse.",
                         'Doriane crie « Jones ! » depuis le haut, hors champ.'),
        _reference_scene(3, "Indy se balance.", "Un filin de soie descend à côté de la corde qui le soutient."),
        _reference_scene(4, "Indy tient le filin qui vient de descendre.",
                         "Une tablette de pierre sort de la paroi de la crevasse."),
        _reference_scene(5, "Indy regarde la même tablette.",
                         "Il observe ses inscriptions en grec ancien."),
        _reference_scene(6, "Une lumière vacille sur une autre corde à Delphes.", "Le faisceau disparaît."),
    ]}
    calls = []
    stop = True

    def chat(client, model, system, user, schema, *settings):
        assert schema == lmstudio_pipeline.load("extract").CHUNK_SCHEMA and schema["strict"]
        payload = json.loads(user)
        number = payload["sequence"]
        calls.append(number)
        if number == 4 and stop:
            raise KeyboardInterrupt
        known = {ref["name"]: ref["reference_id"] for ref in payload["known_references"]}
        assert "Doriane" not in known if number <= 2 else "Doriane" in known
        assert "corde" not in known
        first, *rest = payload["source_passages"]
        event = rest[-1] if rest else first

        def row(name, passage, *, trait=None, presence="visible", speaks=None):
            traits = [] if trait is None else [{"attribute": trait[0], "description": trait[1],
                "scope": "intrinsic", "evidence_passage_ids": [passage["id"]]}]
            entry = {"canonical_name": name, "reference_id": known.get(name, "new"), "presence": presence,
                     "importance": "minor", "reference_priority": "optional", "reference_view_hints": [],
                     "evidence_passage_ids": [passage["id"]], "visual_traits": traits, "aliases": []}
            return {**entry, "speaks": speaks, "voice_description": ""} if speaks is not None else entry

        characters = [row("Indy", first, speaks=(number == 1))] if number != 6 else []
        if number == 1:
            characters[0]["evidence_passage_ids"].append(payload["source_passages"][1]["id"])
        if number in (1, 2):
            characters.append(row("Doriane", payload["source_passages"][1],
                                  presence="mentioned" if number == 1 else "offscreen", speaks=number == 2))
        locations = []
        if number in (1, 2, 4):
            locations = [row("crevasse", event if number == 4 else first,
                             trait=("walls", "parois rocheuses") if number == 1 else None)]
        if number == 6:
            locations.append(row("Delphes", first))
        objects = [row("corde", first)] if number in (1, 6) else []
        if number in (3, 4):
            objects.append(row("filin", event if number == 3 else first,
                               trait=("material", "soie") if number == 3 else None))
        if number in (4, 5):
            tablet = row("tablette", event, trait=("material", "pierre") if number == 4
                         else ("markings", "inscriptions en grec ancien"))
            if number == 5:
                tablet["evidence_passage_ids"].insert(0, first["id"])
            objects.append(tablet)
        return {"chunk_summary": "", "characters": characters, "locations": locations, "objects": objects}

    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", chat)
    node = ExtractChapterReferencesNode()
    with pytest.raises(KeyboardInterrupt):
        node.run(config, [chapter_data], **params)
    stop = False
    results, _ = node.run(config, [chapter_data], **params)["result"]
    assert calls == [1, 2, 3, 4, 4, 5, 6]
    catalog, = results
    indy, doriane = catalog["characters"]
    assert list(indy["state_by_sequence"]) == ["1", "2", "3", "4", "5"]
    assert list(doriane["state_by_sequence"]) == ["2"]
    assert doriane["speaks"] and doriane["state_by_sequence"]["2"]["source"][0]["presence"] == "offscreen"
    crevasse, = catalog["locations"]
    assert list(crevasse["state_by_sequence"]) == ["1", "2", "4"]
    filin, tablette = catalog["objects"]
    assert filin["canonical_name"] == "filin" and list(filin["state_by_sequence"]) == ["3", "4"]
    assert tablette["canonical_name"] == "tablette"
    assert {t["attribute"] for t in tablette["visual_reference"]["common_traits"]} == {"material", "markings"}
    assert list(tablette["state_by_sequence"]) == ["4", "5"]
    assert "reference_id" not in json.dumps(catalog)
    diagnostics, = tmp_path.rglob("*references.diagnostics.json")
    assert util.load_json(diagnostics)["status"] == "passed"


@pytest.mark.parametrize("known_name, local_name, source", [
    ("torchère", "torchère", "Indy tient une torche."),
    ("corde sinueuse", "corde", "Une corde descend vers Indy."),
])
def test_reused_prop_requires_its_local_name(known_name, local_name, source):
    result = {"characters": [], "locations": [], "objects": [{
        "canonical_name": local_name, "reference_id": "REF_00001", "presence": "visible",
        "visual_traits": [], "evidence_passages": [{"text": source}],
    }]}
    rejected = []
    offered = {"REF_00001": {"kind": "objects", "name": known_name, "aliases": []}}
    selected = cr._eligible_observations(result, offered, rejected)
    assert selected["objects"] == []
    assert rejected == ["objects[0].canonical_name"]


def test_atmosphere_alone_does_not_create_a_location():
    sentence = {"text": "Il oscille dans l’obscurité totale de l’abîme."}
    result = {"characters": [], "objects": [], "locations": [{
        "canonical_name": "abîme", "reference_id": "new", "presence": "visible",
        "evidence_passages": [sentence], "visual_traits": [{
            "attribute": "location_condition", "scope": "scene",
            "description": "obscurité totale de l’abîme", "evidence_passages": [sentence],
        }],
    }]}
    assert cr._eligible_observations(result, {}, [])["locations"] == []

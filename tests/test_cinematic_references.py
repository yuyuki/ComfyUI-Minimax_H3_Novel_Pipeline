from tests.neutral_reference_helpers import candidates as neutral_candidates
"""Timeline boundaries are enforced before mocked model extraction."""
from copy import deepcopy
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline.core import cinematic_references as cr, lmstudio_pipeline, path_access, run_output, util
from minimax_h3_novel_pipeline.nodes.extract_chapter_references import ExtractChapterReferencesNode
from minimax_h3_novel_pipeline.nodes.load_cinematic_chapters import LoadCinematicChaptersNode


@pytest.mark.parametrize("persistent", [False, True])
def test_invalid_visual_quote_retries_with_actionable_feedback(monkeypatch, persistent):
    step = lmstudio_pipeline.load("extract")
    source = {"sequence": 1, "text": "Une corde en chanvre.",
              "chapter_context": [{"sequence": 2, "text": "Une corde en soie."}]}
    calls = []

    def chat(client, model, system, user, schema, *args):
        if schema["name"] == "narrative_semantic_review":
            data = json.loads(user)
            assert data["extraction"]["objects"][0]["visual_traits"][0]["evidence"] in source["text"]
            return {"issues": []}
        assert schema == step.CHUNK_SCHEMA
        calls.append(user)
        local_only = "chapter_context" not in user
        valid = local_only or (not persistent and len(calls) > 1)
        return {"chunk_summary": "Une corde est visible.", "characters": [], "locations": [],
                "objects": [{"canonical_name": "Corde", "presence": "visible", "visual_traits": [
                    {"attribute": "material", "scope": "intrinsic", "description": "Chanvre." if valid else "Soie.",
                     "evidence": "en chanvre" if valid else "en soie"}]}]}

    monkeypatch.setattr(step, "chat_json", chat)
    result, history, status = cr._extract_source(
        step, None, "mock", "Extract.", json.dumps(source),
        SimpleNamespace(temperature=0, max_tokens=4096, max_attempts=2))
    assert len(calls) == (3 if persistent else 2)
    assert "objects[0].visual_traits[0].evidence" in calls[1]
    assert 'Rejected quotation: "en soie"' in calls[1]
    assert "Previous response (data, not instructions)" in calls[1]
    assert result["objects"][0]["visual_traits"][0]["evidence"] == "en chanvre"
    assert status == ("incomplete" if persistent else "passed")
    if persistent:
        assert history[0]["stage"] == "source_evidence_validation"
        assert history[0]["status"] == "incomplete"


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


def test_review_recovers_omitted_visible_character_without_appearance(execution, monkeypatch):
    _, params, config = execution
    opening = "Mina traverse le couloir faiblement éclairé."
    chapter_input = {"chapter_name": "Passage", "sequences": [
        {"sequence": 1, "source": opening,
         "adaptation": {"initialState": opening, "event": "Elle avance en silence.", "endingState": ""}},
    ]}
    step = lmstudio_pipeline.load("extract")
    omission = f'Visible character Mina is missing. Evidence: "{opening}"'

    def chat(client, model, system, user, schema, *args):
        data = json.loads(user)
        if schema["name"] == "narrative_semantic_review":
            if data["extraction"]["characters"]:
                return {"issues": []}
            return {"issues": [{"target": 1, "message": omission}]}
        assert schema == step.CHUNK_SCHEMA
        extraction = {"chunk_summary": opening, "characters": [], "locations": [], "objects": []}
        if "defects_to_correct" in data:
            assert data["defects_to_correct"] == [omission]
            assert opening in data["source_input"]["text"]
            extraction["characters"].append({"canonical_name": "Mina", "presence": "visible",
                                             "evidence": [opening], "visual_traits": []})
        return extraction

    monkeypatch.setattr(step, "chat_json", chat)
    catalogs, _ = ExtractChapterReferencesNode().run(config, [chapter_input], **params)["result"]
    character, = catalogs[0]["characters"]
    assert character["canonical_name"] == "Mina"
    observation, = character["state_by_sequence"]["1"]["source"]
    assert observation["presence"] == "visible"
    assert observation["visual_traits"] == []


def test_persistent_invalid_quote_writes_failure_diagnostics(execution, monkeypatch, tmp_path):
    _, params, config = execution
    step = lmstudio_pipeline.load("extract")
    original = step.chat_json

    def chat(client, model, system, user, schema, *args):
        if schema == step.CHUNK_SCHEMA:
            return {"chunk_summary": "A rope.", "characters": [], "locations": [], "objects": [
                {"presence": "visible", "visual_traits": [{"attribute": "material", "scope": "intrinsic",
                 "description": "Silk.", "evidence": "Unsupported silk."}]}]}
        return original(client, model, system, user, schema, *args)

    monkeypatch.setattr(step, "chat_json", chat)
    with pytest.raises(cr.InvalidResponse, match="objects\\[0\\]"):
        ExtractChapterReferencesNode().run(config, [chapter()], **params)
    diagnostic, = tmp_path.rglob("*.diagnostics.json")
    data = util.load_json(diagnostic)
    assert data["status"] == "incomplete"
    check = data["checks"][-1]
    assert check["sequence"] == 1 and check["chunk"] == 1
    assert check["stage"] == "source_evidence_validation"
    assert "Unsupported silk." in check["reason"]
    assert not list(tmp_path.rglob("chapter_references.json"))


def chapter(name="chapter"):
    return {"chapter_name": name, "source_file": "/not/read.txt", "sequences": [
        {"sequence": 1, "source": "Indy hangs from a rope.", "adaptation": {
            "initialState": "Indy hangs from a rope.", "event": "Indy waits.", "endingState": "Indy hangs."}},
        {"sequence": 2, "source": "Doriane throws a torch toward Indy.", "adaptation": {
            "initialState": "Indy hangs.", "event": "Doriane throws a torch toward Indy.",
            "endingState": "The torch is in flight."}},
        {"sequence": 3, "source": "Indy catches the torch.", "adaptation": {
            "initialState": "The torch is in flight.", "event": "Indy catches the torch.",
            "endingState": "Indy holds the torch."}},
    ]}


def test_catalog_retains_only_category_specific_image_traits(execution, monkeypatch, tmp_path):
    _, params, config = execution
    value = chapter()
    value["sequences"] = value["sequences"][:1]
    adaptation = value["sequences"][0]["adaptation"]
    adaptation.update(
        initialState="Indy porte un pantalon bleu déchiré. Une grosse corde tressée en lin lui presse le torse. "
                     "Les parois de granit sont dans le brouillard.",
        event="Le brouillard se dissipe.",
        endingState="La lumière orangée éclaire le canyon.",
    )

    def fact(attribute, description, evidence, scope="intrinsic"):
        return {"attribute": attribute, "description": description, "evidence": evidence, "scope": scope}

    step = lmstudio_pipeline.load("extract")

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema["name"] == "chapter_reference_identity":
            return {"groups": []}
        if schema["name"] == "narrative_semantic_review":
            bad = {"wardrobe_trousers_damage", "wardrobe_torso_condition", "weather_after"}
            issues = []
            for target, address in payload["visual_trait_targets"].items():
                entity = payload["extraction"][address["kind"]][address["entity_index"]]
                if entity["visual_traits"][address["trait_index"]]["attribute"] in bad:
                    issues.append({"target": int(target), "message": "This trait belongs to animation."})
            return {"issues": issues}
        assert schema == step.CHUNK_SCHEMA
        source = payload.get("source_input", payload)
        assert "lumière orangée" not in json.dumps(source, ensure_ascii=False)
        return neutral_candidates({
            "chunk_summary": "",
            "characters": [{"canonical_name": "Indy", "visual_traits": [
                fact("wardrobe_trousers", "Un pantalon bleu", "un pantalon bleu", "scene"),
                fact("wardrobe_trousers_damage", "Pantalon déchiré", "pantalon bleu déchiré", "scene"),
                fact("wardrobe_torso_condition", "Torse comprimé", "lui presse le torse", "scene"),
            ]}],
            "objects": [{"canonical_name": "corde", "visual_traits": [
                fact("construction", "Grosse corde tressée", "grosse corde tressée"),
                fact("material", "En lin", "en lin"),
            ]}],
            "locations": [{"canonical_name": "canyon", "visual_traits": [
                fact("walls_material", "Parois de granit", "parois de granit"),
                fact("weather", "Brouillard", "dans le brouillard", "scene"),
                fact("weather_after", "Ciel dégagé", "Le brouillard se dissipe", "scene"),
            ]}],
        }, source["text"])

    monkeypatch.setattr(step, "chat_json", chat)
    catalogs, _ = ExtractChapterReferencesNode().run(config, [value], **params)["result"]
    catalog, = catalogs
    character, = catalog["characters"]
    rope, = catalog["objects"]
    canyon, = catalog["locations"]
    from minimax_h3_novel_pipeline.core.reference_visuals import appearance_at

    assert appearance_at(character, "chapter", 1) == "Un pantalon bleu"
    assert appearance_at(rope, "chapter", 1) == "Grosse corde tressée En lin"
    assert appearance_at(canyon, "chapter", 1) == "Parois de granit Brouillard"
    assert catalog["sequences"][0]["adaptation"] == adaptation
    check_path, = tmp_path.rglob("*.diagnostics.json")
    check = util.load_json(check_path)["checks"][0]
    assert check["status"] == "issues"
    assert len(check["history"][-1]["rejected_visual_traits"]) == 3


def identity_reply(data):
    grouped = {}
    for entity in data["entities"]:
        key = (entity["entity_type"], " ".join(entity["canonical_name"].split()).casefold())
        grouped.setdefault(key, []).append(entity)
    return {"groups": [{"member_ids": [e["id"] for e in entries],
                        "canonical_name": entries[0]["canonical_name"], "aliases": [],
                        "source_quotes": [data["source_passages"][0]["text"]]}
                       for entries in grouped.values() if len(entries) > 1]}


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
        if schema["name"] == "chapter_reference_identity":
            return identity_reply(data)
        assert schema == lmstudio_pipeline.load("extract").CHUNK_SCHEMA
        calls.append(data)
        result = {"chunk_summary": "", "characters": [], "locations": [], "objects": []}
        for name, kind in (("Indy", "characters"), ("Doriane", "characters"), ("torch", "objects")):
            if name in data["text"]:
                result[kind].append({"canonical_name": name, "aliases": []})
        return neutral_candidates(result, data["text"])
    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", extract)
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
    assert [c["text"] for c in calls] == [
        "\n\n".join(s["adaptation"][p] for p in ("initialState", "event"))
        for c in chapters for s in c["sequences"]]
    assert all("phase" not in c and "known_references" not in c for c in calls)
    assert all("torch" not in c["text"] for c in calls if c["sequence"] == 1)
    for catalog in result:
        assert "chapter_summary" not in catalog
        assert catalog["sequences"] == chapters[0]["sequences"]
        torch = catalog["objects"][0]
        assert torch["first_sequence"] == 2
        assert "first_phase" not in torch
        states = torch["state_by_sequence"]
        assert "1" not in states and list(states["2"]) == ["source"]
        assert states["2"]["source"][0]["visual_traits"][0]["scope"] == "intrinsic"
        assert states["3"]["source"][0]["visual_traits"][0]["scope"] == "intrinsic"
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
    assert len(calls) == count + 3
    chapters[0]["sequences"][0]["source"] = "Indy shouts."
    ExtractChapterReferencesNode().run(config, chapters, **params)
    assert len(calls) == count + 3


def test_resolved_adaptation_names_feed_extraction_and_identity(execution, monkeypatch):
    calls, params, config = execution
    value = chapter()
    for seq in value["sequences"]:
        seq["source"] = "L'homme attend avec un objet absent de l'adaptation."
        seq["adaptation"]["endingState"] = "ENDING_ONLY_CHARACTER arrives with ENDING_ONLY_OBJECT."
    identity_calls = []
    original = lmstudio_pipeline.load("extract").chat_json

    def chat(client, model, system, user, schema, *args):
        if schema["name"] == "chapter_reference_identity":
            identity_calls.append(json.loads(user))
        return original(client, model, system, user, schema, *args)

    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", chat)
    result, _ = ExtractChapterReferencesNode().run(config, [value], **params)["result"]
    passages = [p["text"] for call in identity_calls for p in call["source_passages"]]
    assert passages == [call["text"] for call in calls]
    assert all("L'homme" not in text and "ENDING_ONLY" not in text for text in passages)
    assert [e["canonical_name"] for e in result[0]["characters"]] == ["Indy", "Doriane"]
    assert result[0]["sequences"] == value["sequences"]


def test_category_context_contains_only_adaptation_and_invalidates_dependent_cache(execution):
    calls, params, config = execution
    value = chapter()
    value["sequences"][0]["source"] = "Delphes, Greece. ORIGINAL_ONLY"
    value["sequences"][2]["adaptation"]["endingState"] = "ENDING_ONLY"
    node = ExtractChapterReferencesNode()
    node.run(config, [value], **params)
    expected = [{"sequence": s["sequence"], "chunk": 1,
                 "text": "\n\n".join(s["adaptation"][p] for p in ("initialState", "event"))}
                for s in value["sequences"]]
    assert all(c["chapter_context"] == expected for c in calls)
    assert "ORIGINAL_ONLY" not in json.dumps(calls) and "ENDING_ONLY" not in json.dumps(calls)
    count = len(calls)
    value["sequences"][2]["adaptation"]["event"] = "Indy catches the burning stick."
    node.run(config, [value], **params)
    assert len(calls) == count + 3
    assert "burning stick" in calls[-3]["chapter_context"][-1]["text"]


def test_oversized_category_context_keeps_local_data_and_marks_incomplete(execution, monkeypatch, tmp_path):
    calls, params, config = execution
    monkeypatch.setattr(cr.continuity_context, "REQUEST_BYTES", 10)
    results, _ = ExtractChapterReferencesNode().run(config, [chapter()], **params)["result"]
    assert results[0]["characters"]
    assert all("chapter_context" not in c for c in calls)
    diagnostic, = tmp_path.rglob("*.diagnostics.json")
    data = util.load_json(diagnostic)
    assert data["status"] == "incomplete"
    checks = [c for c in data["checks"] if c["stage"] == "asset_category_context"]
    assert len(checks) == 3 and all(c["status"] == "incomplete" for c in checks)


def test_category_correction_keeps_context_schema_and_source_observations(execution, monkeypatch):
    _, params, config = execution
    value = chapter()
    value["sequences"] = value["sequences"][:2]
    value["sequences"][0]["adaptation"].update(
        initialState="Indy est dans la crevasse obscure.",
        event="Une lueur vacillante descend au bout d'une corde.")
    value["sequences"][1]["adaptation"].update(
        initialState="Indy est dans la crevasse.", event="Indy saisit la torche descendue.")
    calls = []
    step = lmstudio_pipeline.load("extract")

    def chat(client, model, system, user, schema, *args):
        data = json.loads(user)
        calls.append((schema, data))
        if schema["name"] == "chapter_reference_identity":
            return identity_reply(data)
        if schema["name"] == "narrative_semantic_review":
            bad = any(e["canonical_name"] == "l'obscurité" for e in data["extraction"]["locations"])
            return {"issues": [{"target": 1, "message": "Darkness is a lighting condition; retain the crevasse background."}]
                    if bad else []}
        assert schema == step.CHUNK_SCHEMA and schema["strict"]
        source = data.get("source_input", data)
        corrected = "defects_to_correct" in data or source["sequence"] == 2
        name = "la crevasse" if corrected else "l'obscurité"
        return neutral_candidates({"chunk_summary": "", "characters": [], "objects": [], "locations": [
            {"canonical_name": name, "aliases": []}]}, source["text"])

    monkeypatch.setattr(step, "chat_json", chat)
    result, _ = ExtractChapterReferencesNode().run(config, [value], **params)["result"]
    assert [e["canonical_name"] for e in result[0]["locations"]] == ["la crevasse"]
    correction, = [d for _, d in calls if "defects_to_correct" in d]
    assert correction["source_input"]["chapter_context"][-1]["sequence"] == 2
    assert "torche" in correction["source_input"]["chapter_context"][-1]["text"]
    assert "torche" not in correction["source_input"]["text"]
    assert result[0]["sequences"] == value["sequences"]


@pytest.mark.parametrize("field", ["initialState", "event"])
def test_one_nonempty_adaptation_field_is_sufficient(execution, field):
    calls, params, config = execution
    value = chapter()
    value["sequences"] = value["sequences"][:1]
    value["sequences"][0]["adaptation"].update(initialState="", event="")
    value["sequences"][0]["adaptation"][field] = "Indy waits."
    ExtractChapterReferencesNode().run(config, [value], **params)
    assert [call["text"] for call in calls] == ["Indy waits."]


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
    assert "1810" in calls[0]["text"]
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


def test_type_name_index_preserves_alias_variants(execution, monkeypatch):
    _, params, config = execution
    responses = iter([
        {"characters": [{"canonical_name": "Indy", "aliases": ["Henry"]}], "locations": [],
         "objects": [{"canonical_name": "la torche", "aliases": ["flambeau"], "evidence": ["first"]}]},
        {"characters": [{"canonical_name": "Henry", "aliases": ["Indy"]}], "locations": [],
         "objects": [{"canonical_name": " LA  TORCHE ", "aliases": [], "evidence": ["second"]}]},
        {"characters": [], "locations": [{"canonical_name": "la torche", "aliases": []}],
         "objects": [{"canonical_name": "flambeau", "aliases": ["la torche"]}]},
    ])
    def chat(client, model, system, user, schema, *args):
        if schema["name"] == "narrative_semantic_review":
            return {"issues": []}
        if schema["name"] == "chapter_reference_identity":
            return identity_reply(json.loads(user))
        assert schema == lmstudio_pipeline.load("extract").CHUNK_SCHEMA
        return neutral_candidates({"chunk_summary": "", **next(responses)}, json.loads(user)["text"])
    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", chat)
    results, _ = ExtractChapterReferencesNode().run(config, [chapter()], **params)["result"]
    result, = results
    assert len(result["characters"]) == 2 and len(result["objects"]) == 2 and len(result["locations"]) == 1
    assert result["objects"][0]["evidence"] == ["first", "second"]
    assert all("link_to_local_id" not in e for k in ("characters", "locations", "objects") for e in result[k])


@pytest.mark.parametrize("failure", [ValueError("private response"), RuntimeError("private response"), {"issues": "broken"}])
def test_advisory_failure_keeps_extraction_and_retries(execution, monkeypatch, tmp_path, failure):
    _, params, config = execution
    original = lmstudio_pipeline.load("extract").chat_json
    count = []
    def chat(client, model, system, user, schema, *args):
        if schema["name"] == "narrative_semantic_review":
            count.append(1)
            if isinstance(failure, Exception):
                raise failure
            return failure
        return original(client, model, system, user, schema, *args)
    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", chat)
    node = ExtractChapterReferencesNode()
    result, _ = node.run(config, [chapter()], **params)["result"]
    assert result[0]["characters"] and "cache_key" not in result[0]
    diagnostic, = tmp_path.rglob("*.diagnostics.json")
    data = util.load_json(diagnostic)
    assert data["status"] == "incomplete" and all(c["status"] == "incomplete" for c in data["checks"])
    assert "private response" not in diagnostic.read_text(encoding="utf-8")
    node.run(config, [chapter()], **params)
    assert len(count) == 6


@pytest.mark.parametrize("failure", [OSError("disk"), KeyboardInterrupt()])
def test_integrity_and_cancellation_propagate(execution, monkeypatch, failure):
    _, params, config = execution
    original = lmstudio_pipeline.load("extract").chat_json
    def chat(client, model, system, user, schema, *args):
        if schema["name"] == "narrative_semantic_review":
            raise failure
        return original(client, model, system, user, schema, *args)
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


def test_language_and_spatial_defects_are_corrected(execution, monkeypatch, tmp_path):
    _, params, config = execution
    value = chapter()
    value["sequences"] = value["sequences"][:1]
    value["sequences"][0]["source"] = "Indy tient une torche inclinee vers le bas."
    calls = []
    def chat(client, model, system, user, schema, *args):
        data = json.loads(user)
        if schema["name"] == "narrative_semantic_review":
            assert "source narration" in system
            bad = data["extraction"]["chunk_summary"] == "English summary."
            return {"issues": [{"target": 1, "message": "Use French prose; orientation is an object property."}] if bad else []}
        calls.append(data)
        assert "actual filming environments" in system and "source narration" in system
        corrected = "defects_to_correct" in data
        entity = {"canonical_name": "la torche", "aliases": []}
        return neutral_candidates({"chunk_summary": "La torche est inclinee." if corrected else "English summary.",
                "characters": [], "locations": [] if corrected else [entity],
                "objects": [entity] if corrected else []}, data.get("source_input", data)["text"])
    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", chat)
    results, _ = ExtractChapterReferencesNode().run(config, [value], **params)["result"]
    assert len(calls) == 2
    assert results[0]["locations"] == [] and len(results[0]["objects"]) == 1
    cache, = tmp_path.rglob(".cache/*/*.json")
    assert util.load_json(cache)["extraction"]["chunk_summary"] == "La torche est inclinee."
    diagnostic, = tmp_path.rglob("*.diagnostics.json")
    assert util.load_json(diagnostic)["status"] == "passed"

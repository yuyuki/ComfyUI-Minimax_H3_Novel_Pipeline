"""Neutral references retain identity traits without staging or future-state leakage."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from tests.passages_helpers import adapt_fixture_reply, with_passage_ids

from minimax_h3_novel_pipeline.core import reference_visuals as visual, lmstudio_pipeline, util
from minimax_h3_novel_pipeline.core import cinematic_references as extraction
from minimax_h3_novel_pipeline.core import pipeline_step3_generate as generation


def trait(attribute, description, scope="intrinsic", evidence="Source quote."):
    return {"attribute": attribute, "description": description, "scope": scope, "evidence": evidence}


def observation(*traits, presence="visible"):
    return {"presence": presence, "visual_traits": list(traits)}


def reference(rows, kind="object", gid="OBJ_001"):
    timeline = {"chapter": {str(seq): {"source": obs} for seq, obs in rows.items()}}
    checks = []
    item = {"global_id": gid, "entity_type": kind, "canonical_name": "Reference entity",
            "chapters_seen": ["chapter"], "importance": "major", "reference_priority": "required",
            "distinguishing_features": [], "timeline": timeline}
    item["visual_reference"] = visual.compile_reference(timeline, checks, gid)
    item["stable_visual_description"] = visual.descriptions(item["visual_reference"]["common_traits"])
    return item, checks


def options():
    return SimpleNamespace(max_character_base_views=2, max_location_base_views=1,
                           max_object_base_views=1, image_style="digital illustration")


def test_intrinsic_material_is_reused_in_unrepeated_and_earlier_scenes():
    item, checks = reference({1: [observation()], 2: [observation(trait("material", "Made of linen."))],
                              3: [observation()]})
    assert not checks
    assert [visual.appearance_at(item, "chapter", s) for s in (1, 2, 3)] == ["Made of linen."] * 3
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([item], options())
    assert len(specs) == 1 and specs[0]["scene_bindings"] == {"chapter": [1, 2, 3]}
    assert specs[0]["view_type"] == "front_view"


def test_backdrop_fog_is_local_and_does_not_erase_terrain_or_create_presence():
    item, _ = reference({1: [observation(trait("terrain", "Granite cliffs."))],
                         2: [observation(trait("weather", "Dense fog.", "scene"))],
                         4: [observation(presence="offscreen")]}, kind="location", gid="LOC_001")
    assert visual.appearance_at(item, "chapter", 1) == "Granite cliffs."
    assert visual.appearance_at(item, "chapter", 2) == "Granite cliffs. Dense fog."
    assert visual.appearance_at(item, "chapter", 3) is None
    assert visual.appearance_at(item, "chapter", 4) is None
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([item], options())
    assert len(specs) == 2
    assert all(4 not in s["scene_bindings"]["chapter"] for s in specs)


def test_conflicting_materials_are_not_silently_propagated():
    item, checks = reference({1: [observation(trait("material", "Linen."))], 2: [observation()],
                              3: [observation(trait("material", "Hemp."))]})
    assert checks and checks[0]["status"] == "incomplete"
    assert visual.appearance_at(item, "chapter", 1) == "Linen."
    assert visual.appearance_at(item, "chapter", 2) == ""
    assert visual.appearance_at(item, "chapter", 3) == "Hemp."


def test_distinct_entities_do_not_share_material_and_name_only_objects_have_no_images():
    first, _ = reference({1: [observation(trait("material", "Linen."))]})
    second, _ = reference({1: [observation()]}, gid="OBJ_002")
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([first, second], options())
    assert len(specs) == 1 and specs[0]["linked_global_id"] == "OBJ_001"
    assert visual.appearance_at(second, "chapter", 1) == ""


def test_unspecified_material_keeps_occurrence_without_generating_empty_image():
    unknown = trait("material", "unspecified", evidence="A bell hangs in the doorway.")
    item, checks = reference({1: [observation(unknown)]})
    assert item["timeline"]["chapter"]["1"]["source"][0]["visual_traits"] == [unknown]
    assert visual.appearance_at(item, "chapter", 1) == ""
    assert lmstudio_pipeline.load("consolidate").build_picture_specs([item], options()) == []
    assert checks == []


def test_object_view_hints_can_expose_geometry_without_narrative_staging():
    item, _ = reference({1: [observation(trait("construction", "A hollow sculpted form."))]})
    item["reference_view_hints"] = ["hero_three_quarter"]
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([item], options())
    assert specs[0]["view_type"] == "hero_three_quarter"


def test_local_override_keeps_other_intrinsic_traits_and_is_reused():
    common = [trait("color", "Blue surface."), trait("material", "Oak construction.")]
    change = trait("color", "Red surface.", "scene")
    item, _ = reference({1: [observation(*common)], 2: [observation(change)], 3: [observation(change)]},
                         kind="location", gid="LOC_001")
    assert visual.appearance_at(item, "chapter", 2) == "Red surface. Oak construction."
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([item], options())
    assert len(specs) == 2
    changed, = [s for s in specs if s["variant"] != "base"]
    assert changed["scene_bindings"] == {"chapter": [2, 3]}
    assert "Blue surface" not in changed["stable_visual_description"]


def test_repeated_local_description_of_unchanged_trait_does_not_duplicate_images():
    item, _ = reference({1: [observation(trait("light", "Dim light.", "scene"))],
                         2: [observation(trait("light", "Dim light.", "scene"))]},
                         kind="location", gid="LOC_001")
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([item], options())
    assert len(specs) == 1 and specs[0]["scene_bindings"] == {"chapter": [1, 2]}


def test_ambiguous_local_override_does_not_restore_the_common_value():
    item, checks = reference({1: [observation(trait("color", "Blue."), trait("material", "Oak."))],
                              2: [observation(trait("color", "Red.", "scene"), trait("color", "Green.", "scene"))]},
                              kind="location", gid="LOC_001")
    assert checks and visual.appearance_at(item, "chapter", 2) == "Oak."
    assert item["visual_reference"]["scenes"]["chapter"]["2"]["excluded_attributes"] == ["color"]


@pytest.mark.parametrize("verdict", ["compatible", "conflicting"])
def test_cited_local_lighting_uses_spatial_scope(verdict):
    first = trait("lighting", "The entry is dim.", "scene", "The entry is dim.")
    other = "A bright alcove lies beyond it." if verdict == "compatible" else "The same entry is brightly lit."
    second = trait("lighting", other, "scene", other)
    timeline = {"chapter": {"1": {"source": [observation(first, second)]},
                            "2": {"source": [observation()]}}}
    source = f"{first['evidence']} {second['evidence']}"

    def chat(client, model, system, user, schema, *args):
        assert schema == visual.LOCAL_COMPATIBILITY_SCHEMA and schema["strict"]
        payload = json.loads(user)
        assert payload["local_source"] == source and payload["traits"] == [first, second]
        return {"verdict": verdict}

    checks = []
    resolved = visual.build_reference(timeline, checks, "LOC_001", chat, None, "mock",
                                      SimpleNamespace(temperature=0, max_tokens=1000),
                                      source_by_scene={("chapter", 1): source})
    item = {"visual_reference": resolved}
    assert visual.appearance_at(item, "chapter", 1) == (source if verdict == "compatible" else "")
    assert visual.appearance_at(item, "chapter", 2) == ""
    assert resolved["common_traits"] == []
    assert any(check["status"] == "incomplete" for check in checks) == (verdict == "conflicting")


def test_loader_rejects_an_image_state_bound_to_an_earlier_scene():
    item, _ = reference({1: [observation(trait("material", "Stone."))],
                         2: [observation(trait("light", "Sunlight.", "scene"))]},
                         kind="location", gid="LOC_001")
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([item], options())
    changed = next(s for s in specs if s["variant"] != "base")
    changed["scene_bindings"] = {"chapter": [1]}
    with pytest.raises(ValueError, match="state does not match"):
        visual.validate_registry({"entities": [item], "picture_assets": specs})


def test_bindings_select_the_exact_scene_state_and_exclude_absent_entities():
    item, _ = reference({1: [observation(trait("material", "Stone."))],
                         2: [observation(trait("light", "Sunlight.", "scene"))]},
                         kind="location", gid="LOC_001")
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([item], options())
    refs = {"entities": [item], "picture_assets": specs, "audio_assets": []}
    args = SimpleNamespace(max_pictures=9, max_pictures_per_subject=1, max_audio=0)
    for seq in (1, 2, 3):
        scene = generation.Scene("Scene", "", "", "", [item["global_id"]], [], [], False, "", seq)
        bound = generation.build_bindings(refs, scene, "chapter", args)
        assert len(bound["picture_input_order"]) == (0 if seq == 3 else 1)
        if seq < 3:
            assert bound["picture_input_order"][0]["variant"] == ("base" if seq == 1 else specs[1]["variant"])


def resolve_candidate(kind, candidate, source):
    data = {"sequence": 1, "text": source, "known_references": []}
    data["source_passages"] = visual.evidence_index(data).passages()
    wire = adapt_fixture_reply({"chunk_summary": "", kind: [
        {**observation(candidate), "canonical_name": "Subject"}]}, data, lmstudio_pipeline.load("extract").CHUNK_SCHEMA)
    return visual.resolve_evidence(wire, data)


def test_evidence_validation_rejects_traits_quoted_from_another_scene():
    data = {"text": "An unmarked surface. It becomes cracked.", "reference_state": "An unmarked surface."}
    candidate = trait("material", "Wood.", evidence="A later scene only.")
    with pytest.raises(ValueError, match="authorized source"):
        resolve_candidate("objects", candidate, data["text"])


def test_appearance_of_an_entity_first_seen_during_an_event_is_supported():
    data = {"text": "An empty room. A visitor enters wearing a red coat.", "reference_state": "An empty room."}
    candidate = trait("wardrobe_coat", "A red coat.", "scene", "wearing a red coat")
    resolve_candidate("characters", candidate, data["text"])


@pytest.mark.parametrize(("kind", "candidate"), [
    ("characters", trait("torso_condition", "A bruised torso.", "scene")),
    ("characters", trait("wardrobe_coat", "A red coat.")),
])
def test_evidence_cannot_override_category_constraints(kind, candidate):
    with pytest.raises(ValueError, match=kind):
        resolve_candidate(kind, candidate, candidate["evidence"])


def test_established_object_condition_stays_at_its_source_scene():
    source = "The wick is burning."
    candidate = trait("condition", "Burning wick.", "scene", source)
    resolved, = resolve_candidate("objects", candidate, source)["objects"]
    item, checks = reference({1: [resolved], 2: [observation()]})
    assert checks == []
    assert item["visual_reference"]["common_traits"] == []
    assert visual.appearance_at(item, "chapter", 1) == "Burning wick."
    assert visual.appearance_at(item, "chapter", 2) == ""


def test_extraction_retries_with_structured_schema_and_full_evidence(monkeypatch):
    step = lmstudio_pipeline.load("extract")
    source = {"sequence": 1, "text": "A brass surface.", "offset": 0, "opening_end": 16,
              "default_temporal_context": "", "known_references": []}
    calls = []
    def chat(client, model, system, user, schema, *args):
        calls.append((schema, user))
        quote = "Future text." if len(calls) == 1 else source["text"]
        return {"chunk_summary": "", "characters": [], "locations": [], "objects": [
            {**observation(trait("material", "Brass.", evidence=quote)), "canonical_name": "Surface",
             "aliases": [], "importance": "minor", "reference_priority": "optional", "reference_view_hints": []}]}
    monkeypatch.setattr(step, "chat_json", with_passage_ids(chat))
    args = SimpleNamespace(temperature=0.1, max_tokens=2000, max_attempts=2)
    result, rejected = extraction._extract_source(step, None, "mock", extraction.SOURCE_SYSTEM, source, args)
    assert not rejected and len(calls) == 2
    assert all(schema == step.CHUNK_SCHEMA and schema["strict"] for schema, _ in calls)
    assert all(source["text"] in user for _, user in calls)
    assert result["objects"][0]["visual_traits"][0]["description"] == "Brass."


def test_reference_prompts_are_generic_and_have_no_story_specific_examples():
    step = lmstudio_pipeline.load("consolidate")
    from minimax_h3_novel_pipeline.core import chapter_reference_identity, cinematic_adaptation
    prompts = [extraction.SOURCE_SYSTEM, lmstudio_pipeline.load("extract").EXTRACT_SYSTEM,
               step.APPEARANCE_SYSTEM, step.PICTURE_BRIEF_SYSTEM, visual.EQUIVALENCE_SYSTEM,
               chapter_reference_identity.SYSTEM, cinematic_adaptation.REVIEW_SYSTEM]
    assert all(not any(word in p.casefold() for word in ("indy", "doriane", "torch", "linen", "tablet", "crevasse"))
               for p in prompts)


def test_inputs_unchanged_and_obsolete_outputs_rejected():
    rows = {1: [observation(trait("material", "Linen."))]}
    original = deepcopy(rows)
    reference(rows)
    assert rows == original
    with pytest.raises(ValueError, match="regenerate"):
        util.require_schema({"schema_version": "minimax-h3-novel-refs.chapter.v4"}, util.CHAPTER_SCHEMA)
    with pytest.raises(ValueError, match="re-extract"):
        visual.validate_registry({"entities": [{"global_id": "old"}]})


def test_equivalent_intrinsic_wording_is_shared_without_rewriting_observations():
    item, _ = reference({1: [observation(trait("material", "Linen fibers."))],
                         2: [observation()], 3: [observation(trait("material", "Made of linen fibers."))]})
    original = deepcopy(item["timeline"])
    calls = []
    def chat(client, model, system, user, schema, *args):
        calls.append((json.loads(user), schema))
        return {"groups": [{"attribute": "material", "descriptions": ["Linen fibers.", "Made of linen fibers."],
                            "selected_description": "Made of linen fibers."}]}
    checks = []
    item["visual_reference"] = visual.build_reference(item["timeline"], checks, "OBJ_001", chat, None, "mock",
                                                      SimpleNamespace(temperature=0.1, max_tokens=2000, max_attempts=2))
    assert not checks and len(calls) == 1
    assert calls[0][1] == visual.EQUIVALENCE_SCHEMA and calls[0][1]["strict"]
    assert len(calls[0][0]["attributes"]["material"]) == 2
    assert visual.appearance_at(item, "chapter", 2) == "Made of linen fibers."
    assert item["timeline"] == original


def test_intrinsic_review_cannot_invent_descriptions_and_retries_all_alternatives():
    item, _ = reference({1: [observation(trait("material", "Oak."))],
                         2: [observation(trait("material", "Oak construction."))]})
    calls = []
    def chat(client, model, system, user, schema, *args):
        calls.append(user)
        return {"groups": [{"attribute": "material", "descriptions": ["Oak.", "Oak construction."],
                            "selected_description": "Invented material." if len(calls) == 1 else "Oak construction."}]}
    checks = []
    result = visual.build_reference(item["timeline"], checks, "OBJ_001", chat, None, "mock",
                                    SimpleNamespace(temperature=0.1, max_tokens=2000, max_attempts=2))
    assert not checks and len(calls) == 2
    assert all("Oak." in user and "Oak construction." in user for user in calls)
    assert result["common_traits"][0]["description"] == "Oak construction."


@pytest.mark.parametrize("error", [PermissionError("storage"), KeyboardInterrupt()])
def test_intrinsic_review_preserves_filesystem_errors_and_cancellation(error):
    item, _ = reference({1: [observation(trait("material", "A."))], 2: [observation(trait("material", "B."))]})
    def chat(*args):
        raise error
    with pytest.raises(type(error)):
        visual.build_reference(item["timeline"], [], "OBJ_001", chat, None, "mock",
                               SimpleNamespace(temperature=0.1, max_tokens=2000, max_attempts=2))


def test_failed_intrinsic_review_keeps_valid_local_data_and_incomplete_diagnostics():
    item, _ = reference({1: [observation(trait("material", "A."))], 2: [observation()],
                         3: [observation(trait("material", "B."))]})
    checks = []
    def chat(*args):
        raise RuntimeError("Model unavailable")
    item["visual_reference"] = visual.build_reference(item["timeline"], checks, "OBJ_001", chat, None, "mock",
                                                      SimpleNamespace(temperature=0.1, max_tokens=2000, max_attempts=2))
    assert checks and all(c["status"] == "incomplete" for c in checks)
    assert visual.appearance_at(item, "chapter", 1) == "A."
    assert visual.appearance_at(item, "chapter", 2) == ""
    assert visual.appearance_at(item, "chapter", 3) == "B."

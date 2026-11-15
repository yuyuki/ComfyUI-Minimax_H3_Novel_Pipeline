"""Neutral references retain identity traits without staging or future-state leakage."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

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


def test_damage_is_local_and_does_not_erase_material_or_create_presence():
    item, _ = reference({1: [observation(trait("material", "Made of linen."))],
                         2: [observation(trait("condition", "Frayed fibers.", "scene"))],
                         4: [observation(presence="offscreen")]})
    assert visual.appearance_at(item, "chapter", 1) == "Made of linen."
    assert visual.appearance_at(item, "chapter", 2) == "Made of linen. Frayed fibers."
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


def test_object_view_hints_can_expose_geometry_without_narrative_staging():
    item, _ = reference({1: [observation(trait("construction", "A hollow sculpted form."))]})
    item["reference_view_hints"] = ["hero_three_quarter"]
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([item], options())
    assert specs[0]["view_type"] == "hero_three_quarter"


def test_local_override_keeps_other_intrinsic_traits_and_is_reused():
    common = [trait("color", "Blue surface."), trait("material", "Oak construction.")]
    change = trait("color", "Red surface.", "scene")
    item, _ = reference({1: [observation(*common)], 2: [observation(change)], 3: [observation(change)]})
    assert visual.appearance_at(item, "chapter", 2) == "Red surface. Oak construction."
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([item], options())
    assert len(specs) == 2
    changed, = [s for s in specs if s["variant"] != "base"]
    assert changed["scene_bindings"] == {"chapter": [2, 3]}
    assert "Blue surface" not in changed["stable_visual_description"]


def test_repeated_local_description_of_unchanged_trait_does_not_duplicate_images():
    item, _ = reference({1: [observation(trait("material", "Linen."))],
                         2: [observation(trait("material", "Linen.", "scene"))]})
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([item], options())
    assert len(specs) == 1 and specs[0]["scene_bindings"] == {"chapter": [1, 2]}


def test_ambiguous_local_override_does_not_restore_the_common_value():
    item, checks = reference({1: [observation(trait("color", "Blue."), trait("material", "Oak."))],
                              2: [observation(trait("color", "Red.", "scene"), trait("color", "Green.", "scene"))]})
    assert checks and visual.appearance_at(item, "chapter", 2) == "Oak."
    assert item["visual_reference"]["scenes"]["chapter"]["2"]["excluded_attributes"] == ["color"]


def test_loader_rejects_an_image_state_bound_to_an_earlier_scene():
    item, _ = reference({1: [observation(trait("material", "Stone."))],
                         2: [observation(trait("condition", "Broken.", "scene"))]})
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([item], options())
    changed = next(s for s in specs if s["variant"] != "base")
    changed["scene_bindings"] = {"chapter": [1]}
    with pytest.raises(ValueError, match="state does not match"):
        visual.validate_registry({"entities": [item], "picture_assets": specs})


def test_bindings_select_the_exact_scene_state_and_exclude_absent_entities():
    item, _ = reference({1: [observation(trait("material", "Stone."))],
                         2: [observation(trait("condition", "Broken surface.", "scene"))]})
    specs = lmstudio_pipeline.load("consolidate").build_picture_specs([item], options())
    refs = {"entities": [item], "picture_assets": specs, "audio_assets": []}
    args = SimpleNamespace(max_pictures=9, max_pictures_per_subject=1, max_audio=0)
    for seq in (1, 2, 3):
        scene = generation.Scene("Scene", "", "", "", [item["global_id"]], [], [], False, "", seq)
        bound = generation.build_bindings(refs, scene, "chapter", args)
        assert len(bound["picture_input_order"]) == (0 if seq == 3 else 1)
        if seq < 3:
            assert bound["picture_input_order"][0]["variant"] == ("base" if seq == 1 else specs[1]["variant"])


def test_evidence_validation_rejects_traits_quoted_from_another_scene():
    data = {"text": "An unmarked surface. It becomes cracked.", "reference_state": "An unmarked surface."}
    result = {"objects": [observation(trait("condition", "Cracked.", "scene", "A later scene only."))]}
    with pytest.raises(ValueError):
        visual.validate_evidence(result, data)


def test_appearance_of_an_entity_first_seen_during_an_event_is_supported():
    data = {"text": "An empty room. A visitor enters wearing a red coat.", "reference_state": "An empty room."}
    result = {"characters": [observation(trait("clothing", "A red coat.", "scene", "wearing a red coat"))]}
    visual.validate_evidence(result, data)


def test_extraction_retries_with_structured_schema_and_full_evidence(monkeypatch):
    step = lmstudio_pipeline.load("extract")
    source = {"sequence": 1, "text": "A brass surface.", "reference_state": "A brass surface."}
    calls = []
    def chat(client, model, system, user, schema, *args):
        if schema["name"] == "narrative_semantic_review":
            return {"issues": []}
        calls.append((schema, user))
        quote = "Future text." if len(calls) == 1 else source["text"]
        return {"characters": [], "locations": [], "objects": [
            observation(trait("material", "Brass.", evidence=quote))]}
    monkeypatch.setattr(step, "chat_json", chat)
    args = SimpleNamespace(temperature=0.1, max_tokens=2000, max_attempts=2)
    result, _, status = extraction._extract_source(step, None, "mock", extraction.SOURCE_SYSTEM,
                                                  json.dumps(source), args)
    assert status == "passed" and len(calls) == 2
    assert all(schema == step.CHUNK_SCHEMA and schema["strict"] for schema, _ in calls)
    assert all(source["text"] in user for _, user in calls)
    assert result["objects"][0]["visual_traits"][0]["description"] == "Brass."


def test_reference_prompts_are_generic_and_have_no_story_specific_examples():
    step = lmstudio_pipeline.load("consolidate")
    from minimax_h3_novel_pipeline.core import chapter_reference_identity, narrative_validation
    prompts = [extraction.SOURCE_SYSTEM, lmstudio_pipeline.load("extract").EXTRACT_SYSTEM,
               step.APPEARANCE_SYSTEM, step.PICTURE_BRIEF_SYSTEM, visual.EQUIVALENCE_SYSTEM,
               chapter_reference_identity.SYSTEM, narrative_validation.SYSTEM]
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

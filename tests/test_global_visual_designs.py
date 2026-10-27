"""Global designs are intersected with chapter identities without model matching."""
from copy import deepcopy

import pytest

from minimax_h3_novel_pipeline.core import character_visual_profiles as vp, path_access, util, visual_designs as vd
from tests.test_character_visual_profiles import consolidated, design
from tests.test_image_reference_outputs import entity, options, design_checks


@pytest.fixture
def output(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path / kind)
    return tmp_path / "output"


def global_profiles():
    data = consolidated()
    alice = {**deepcopy(vp.prepare_profiles(data)[0]), **design(data)}
    alice["aliases"] = ["Lise"]
    absent = deepcopy(alice)
    absent.update(local_id="char_absent", canonical_name="Robert", aliases=[], linked_local_ids=["char_absent"])
    absent["eyes"][0]["description"] = "Blue eyes."
    return [alice, absent]


def test_global_export_alias_filter_and_manual_edits(output):
    profiles = global_profiles()
    vp.export_visual_designs(profiles, output)
    path = output / "visual_designs.json"
    payload = util.load_json(path)
    assert len(payload["entities"]) == 2
    payload["entities"][0]["added_details"]["eyes"] = "Manually chosen emerald eyes."
    util.save_json(path, payload)
    vp.export_visual_designs(profiles, output)
    chapter = entity()
    chapter.update(canonical_name="Lise", aliases=[], chapters_seen=["chapter1"])
    original = deepcopy(chapter)
    calls = []

    def chat(client, model, system, user, schema, *args):
        calls.append(schema["name"])
        return design_checks(user)

    result = vd.prepare_designs(chat, None, "model", [chapter], options(), path)
    assert chapter == original
    assert len(result["entities"]) == 1
    assert result["entities"][0]["global_id"] == chapter["global_id"]
    assert result["entities"][0]["added_details"] == {"eyes": "Manually chosen emerald eyes."}
    assert calls == ["visual_design_conflicts"]
    assert len(util.load_json(path)["entities"]) == 2


def test_temporary_clothes_accessories_and_appearance_never_export(output):
    profiles = global_profiles()
    p = profiles[0]
    ref = p["supporting_evidence"][0]
    ref["observation"]["persistence"] = "temporary"
    for field in ("default_outfit", "accessories", "hair"):
        p[field] = [{"description": "Unrelated chapter temporary state.", "classification": "inferred",
                     "evidence": [{"evidence_id": ref["evidence_id"]}]}]
    vp.export_visual_designs(profiles, output)
    assert util.load_json(output / "visual_designs.json")["entities"][0]["added_details"] == {}


@pytest.mark.parametrize("mode", ["shared_alias", "ambiguous_identity", "multiple_entities"])
def test_ambiguous_mapping_fails_without_assigning(output, mode):
    profiles = global_profiles()
    if mode == "shared_alias":
        profiles[1]["aliases"] = ["Lise"]
    if mode == "ambiguous_identity":
        profiles[0]["identity_status"] = "ambiguous"
    vp.export_visual_designs(profiles, output)
    chapter = {**entity(), "canonical_name": "Lise", "aliases": []}
    chapters = [chapter]
    if mode == "multiple_entities":
        chapters.append({**chapter, "global_id": "CHAR_002"})
    with pytest.raises(ValueError, match="Ambiguous"):
        vd.load_designs(output / "visual_designs.json", chapters)


def test_stable_id_and_unmatched_chapter_character(output):
    profiles = global_profiles()
    vp.export_visual_designs(profiles, output)
    chapter = {**entity(), "canonical_name": "Different spelling", "aliases": [],
               "character_id": profiles[0]["local_id"]}
    path = output / "visual_designs.json"
    assert vd.load_designs(path, [chapter])[chapter["global_id"]]["global_id"] == profiles[0]["local_id"]
    chapter.pop("character_id")
    with pytest.warns(UserWarning, match="Unmatched chapter character"):
        assert vd.load_designs(path, [chapter]) == {}


def test_run_local_id_collision_cannot_assign_absent_character(output):
    chapter = {**entity(), "canonical_name": "Alice", "aliases": ["Al"]}
    payload = {"schema_version": vd.DESIGN_SCHEMA_VERSION, "image_style": "cinematic photographic",
               "entities": [{"global_id": chapter["global_id"], "entity_type": "character",
                             "canonical_name": "Robert", "source_facts": vd.source_facts(chapter),
                             "added_details": {"eyes": "Blue eyes."}}]}
    path = output / "visual_designs.json"
    util.save_json(path, payload)
    with pytest.warns(UserWarning, match="Unmatched"):
        assert vd.load_designs(path, [chapter]) == {}
    payload["entities"].append({**payload["entities"][0], "global_id": "CHAR_999", "canonical_name": "Al"})
    util.save_json(path, payload)
    assert vd.load_designs(path, [chapter])[chapter["global_id"]]["global_id"] == "CHAR_999"

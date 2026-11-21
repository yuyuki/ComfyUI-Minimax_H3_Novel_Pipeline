"""Offline regression for source aliases, reusable angles and missing coverage."""
from contextlib import nullcontext
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline.core import lmstudio_pipeline, path_access, run_output, util
from minimax_h3_novel_pipeline.nodes import consolidate_references, spatial_continuity
from minimax_h3_novel_pipeline.nodes.extract_chapter_references import ExtractChapterReferencesNode
from minimax_h3_novel_pipeline.nodes.generate_h3_prompts import GenerateH3PromptsNode
from tests.h3_review_helpers import issue
from tests.test_cinematic_downstream_contract import defaults
from tests.test_h3_prompt_validation import prompt_with_shots


def test_location_alias_views_and_coverage_across_nodes(tmp_path, monkeypatch):
    cid = "00_PROLOGUE"
    geometry = "Des parois de granit encadrent une ouverture au-dessus du puits."
    southern_geometry = "Le puits méridional a des parois de calcaire."
    opening = {
        4: f"Mina oscille dans le puits du nord. {geometry}",
        5: "Mina est suspendue dans le même puits du nord, que le guide appelle l'abîme.",
        6: "Dans le puits du nord, Mina est soutenue par la corde principale, en chanvre.",
        7: "Dans le puits du nord, Mina regarde la paroi de granit.",
        8: "Face à la paroi du puits du nord, Mina lève sa lampe.",
        9: "Mina descend dans le puits du nord, sous l'ouverture.",
        10: f"Le lendemain, Mina entre dans un autre puits. {southern_geometry}",
    }
    events = {
        4: "Mina regarde vers l'ouverture.",
        5: "Un filin de soie abaisse la lampe tandis que la corde principale soutient Mina.",
        6: "Mina saisit le filin de la lampe et se tourne vers la paroi.",
        7: "Le faisceau effleure la paroi.",
        8: "Mina tire deux fois sur la corde principale.",
        9: "Mina descend parallèlement à la paroi.",
        10: "Mina s'arrête dans le puits méridional.",
    }
    chapter = {"chapter_name": cid, "duration": 8, "sequences": [
        {"sequence": n, "source": f"{text} {events[n]}",
         "adaptation": {"initialState": text, "event": events[n], "endingState": "Mina s'immobilise."}}
        for n, text in opening.items()
    ]}
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261121093004"}
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock"))
    extract = lmstudio_pipeline.load("extract")
    consolidate = lmstudio_pipeline.load("consolidate")
    generate = lmstudio_pipeline.load("generate")

    def observation(name, evidence, attribute=None, description=None):
        return {"canonical_name": name, "presence": "visible", "evidence": [evidence],
                "visual_traits": [{"attribute": attribute, "description": description,
                                   "scope": "intrinsic", "evidence": evidence}] if attribute else []}

    def extract_reply(client, model, system, user, schema, *settings):
        assert schema["strict"]
        payload = json.loads(user)
        if schema["name"] == "narrative_semantic_review":
            return {"issues": []}
        if schema["name"] == "chapter_reference_identity":
            entities = payload["entities"]
            clusters = [
                ("Puits du nord", {"Puits du nord", "L'abîme"}, [opening[4], opening[5]]),
                ("Corde principale", {"Corde principale"}, [events[5], opening[6]]),
                ("Filin de la lampe", {"Filin de la lampe"}, [events[5], events[6]]),
            ]
            assert {entity["canonical_name"] for entity in entities if entity["entity_type"] == "locations"} == {
                "Puits du nord", "L'abîme", "Puits méridional",
            }
            return {"groups": [
                {"member_ids": [e["id"] for e in entities if e["canonical_name"] in names],
                 "canonical_name": canonical, "aliases": [], "source_quotes": quotes}
                for canonical, names, quotes in clusters
            ]}
        assert schema == extract.CHUNK_SCHEMA
        n = payload["sequence"]
        assert payload["text"] == f"{opening[n]}\n\n{events[n]}"
        assert "Mina s'immobilise." not in user
        location = {
            4: observation("Puits du nord", geometry, "layout", geometry),
            5: observation("L'abîme", opening[5]),
            10: observation("Puits méridional", southern_geometry, "walls", southern_geometry),
        }.get(n, observation("Puits du nord", opening[n]))
        props = {
            5: [observation("Corde principale", events[5]),
                observation("Filin de la lampe", events[5], "material", "Soie.")],
            6: [observation("Corde principale", opening[6], "material", "Chanvre."),
                observation("Filin de la lampe", events[6])],
        }
        return {"chunk_summary": events[n], "characters": [], "locations": [location], "objects": props.get(n, [])}

    monkeypatch.setattr(extract, "chat_json", extract_reply)
    params = defaults(ExtractChapterReferencesNode)
    catalogs = ExtractChapterReferencesNode().run(config, [chapter], **params)["result"][0]
    catalog, = catalogs
    north, south = catalog["locations"]
    assert catalog["chapter_id"] == cid
    assert north["aliases"] == ["L'abîme"]
    assert set(north["state_by_sequence"]) == {str(n) for n in range(4, 10)}
    assert north["state_by_sequence"]["5"]["source"][0]["canonical_name"] == "L'abîme"
    assert set(south["state_by_sequence"]) == {"10"}
    assert {obj["canonical_name"] for obj in catalog["objects"]} == {"Corde principale", "Filin de la lampe"}
    assert all(set(obj["state_by_sequence"]) == {"5", "6"} for obj in catalog["objects"])

    planned = []
    view_types = ["wide_establishing", "secondary_angle"]
    angles = ["Vue axiale depuis le bas, cadrant l’ouverture et les deux parois de granit.",
              "Vue oblique vers la paroi de granit, avec le bord de l’ouverture en haut du cadre."]

    def consolidate_reply(client, model, system, user, schema, *settings):
        assert schema["strict"]
        if schema == consolidate.RECONCILE_SCHEMA:
            incoming = json.loads(user.split("\n\nINCOMING:\n")[1].split("\n\nCANDIDATES:\n")[0])
            return {"resolutions": [
                {**dict.fromkeys(("stable_visual_description", "voice_description"), ""),
                 **{key: item[key] for key in ("local_id", "entity_type", "canonical_name", "aliases")},
                 "match_global_id": "NEW", "distinguishing_features": [], "speaks": False,
                 "importance": "major", "reference_priority": "required", "reference_view_hints": [],
                 "reason": "First occurrence in the registry.", "confidence": 1}
                for item in incoming
            ]}
        if schema == consolidate.AUDIT_SCHEMA:
            return {"merge_groups": []}
        payload, _ = json.JSONDecoder().raw_decode(user)
        if schema == consolidate.LOCATION_VIEW_SCHEMA:
            scenes = payload["scenes"]
            if payload["location"] == "Puits méridional":
                assert [scene["sequence"] for scene in scenes] == [10]
                return {"views": [{"view_type": "wide_establishing", "framing": "Vue des parois de calcaire.",
                                   "camera_axis": "Vers la paroi", "landmark_quote": "parois de calcaire",
                                   "scene_ids": [1]}]}
            assert payload["max_views"] == 3
            assert payload["appearance"] == geometry
            assert [scene["sequence"] for scene in scenes] == list(range(4, 10))
            planned.append(scenes)
            # The first reply omits sequence 8. The retry still receives all scenes.
            return {"views": [
                {"view_type": view_types[0], "framing": angles[0], "scene_ids": [1, 2, 3],
                 "camera_axis": "Vers le haut", "landmark_quote": "une ouverture"},
                {"view_type": view_types[1], "framing": angles[1],
                 "camera_axis": "Vers la paroi", "landmark_quote": "parois de granit",
                 "scene_ids": [3, 4, 6] if len(planned) == 1 else [3, 4, 5, 6]},
            ]}
        if schema == consolidate.APPEARANCE_SCHEMA:
            return {"appearance": payload.get("appearance") or payload["stable_visual_description"]}
        assert schema == consolidate.PICTURE_BRIEF_SCHEMA
        return {"assets": [{"asset_id": spec["asset_id"], "description": "Référence réutilisable.",
                            "generation_prompt": "Cadrage lisible."} for spec in payload]}

    monkeypatch.setattr(consolidate_references, "prepare_designs", lambda *a: {"entities": []})
    monkeypatch.setattr(consolidate, "chat_json", consolidate_reply)
    node = consolidate_references.ConsolidateReferencesNode
    registry = node().run(catalogs, config, **defaults(node))["result"][0]
    assert len(planned) == 2 and planned[0] == planned[1]
    north_id = registry["chapter_entity_map"][cid][north["local_id"]]
    south_id = registry["chapter_entity_map"][cid][south["local_id"]]
    assert north_id != south_id
    assert {e["canonical_name"] for e in registry["entities"] if e["entity_type"] == "object"} == {
        "Corde principale", "Filin de la lampe",
    }
    pictures = [p for p in registry["picture_assets"] if p["linked_global_id"] == north_id]
    assert [p["view_type"] for p in pictures] == view_types
    assert [p["scene_bindings"] for p in pictures] == [{cid: [4, 5, 6]}, {cid: [6, 7, 8, 9]}]
    assert [p["description"] for p in pictures] == angles
    assert all(geometry in p["generation_prompt"] for p in pictures)

    spatial_calls = []

    def spatial_reply(client, model, system, user, schema, *settings):
        request = json.loads(user)
        n = request["target_sequence"]
        gid = south_id if n == 10 else north_id
        assert request["chapter_id"] == cid
        assert request["approved_visible_location_ids"] == [gid]
        assert schema["schema"]["properties"]["location_id"]["enum"] == [gid]
        spatial_calls.append(n)
        fields = {key: [] if spec["type"] == "array" else "Unknown" for key, spec in spatial_continuity._FIELDS.items()}
        return {**fields, "location_id": cid if spatial_calls == [4] else gid}

    monkeypatch.setattr(spatial_continuity, "chat_json", spatial_reply)
    plan = spatial_continuity.SpatialContinuityNode().run(registry, config)["result"][0]
    assert spatial_calls == [4, 4, 5, 6, 7, 8, 9, 10]
    assert [p["location_id"] for p in plan["chapters"][cid]["sequences"]] == [north_id] * 6 + [south_id]
    invalid = deepcopy(plan)
    invalid["chapters"][cid]["sequences"][0]["location_id"] = cid
    with pytest.raises(ValueError, match="consolidated location ID"):
        spatial_continuity.validate_plan(invalid, registry)

    def h3_reply(client, model, system, user, schema, *settings):
        assert schema["strict"]
        if schema["name"] == "h3_scene_metadata_repair_v1":
            # Omitted location and view preferences cannot discard approved coverage.
            from tests.test_continuity_context import expanded
            row = expanded(json.loads(user))["upstream_sequence"]
            locations = [item for item in row["catalog"] if item["entity_type"] == "location"]
            requests = [{"global_id": loc["global_id"],
                         "view_types": [view["view_type"] for view in loc["available_picture_views"]],
                         "prominence": "primary", "reason": "Use the planned sightlines."}
                        for loc in locations if loc["available_picture_views"]]
            return {"location_global_id": "", "dialogue_present": False, "visible_entity_ids": [],
                    "speaking_entity_ids": [], "reference_view_requests": requests}
        if schema["name"] == generate.CONTINUITY_REVIEW_SCHEMA["name"]:
            payload = json.loads(user)
            if not payload["continuity_plan"]["approved_location_catalog"]:
                return {"errors": [issue("The continuing visible place has no approved occurrence.",
                                         "upstream_conflict", "/continuity_plan/approved_location_catalog")]}
            return {"errors": []}
        assert schema == generate.PROMPT_SCHEMA
        bindings, _ = json.JSONDecoder().raw_decode(user.split("EXACT PER-CLIP BINDINGS\n")[1])
        prompt = prompt_with_shots("[Shot 1] The camera holds its angle against the rock.")
        if "previous_frame_sequence" in bindings:
            prompt = prompt.replace("[reference generation]", "[keyframe completion]")
        if bindings["subjects"]:
            subject, = bindings["subjects"]
            label = subject["h3_subject_label"]
            views = ", ".join(f"{p['h3_picture_label']} ({p['view_type']})" for p in subject["pictures"])
            prompt = prompt.replace("subject_definitions:\nN/A", f"subject_definitions:\n{label} {subject['canonical_name']} (location): {views}.")
            prompt = prompt.replace("retention_analysis:\nN/A", f"retention_analysis:\n{label}: fully_preserved - Fixed geometry.")
        return {"prompt_text": prompt}

    monkeypatch.setattr(generate, "chat_json", h3_reply)
    params = defaults(GenerateH3PromptsNode)
    params.update(repair_attempts=0, refine_camera=False)
    result = GenerateH3PromptsNode().run(registry, config, [chapter], spatial_continuity=plan, **params)["result"][0]
    outputs = result["chapters"][0]["outputs"]
    expected_counts = {4: 1, 5: 1, 6: 2, 7: 1, 8: 1, 9: 1, 10: 1}
    directory = tmp_path / config["run_folder"] / params["out_dir"] / cid
    for entry in outputs:
        n = entry["sequence"]
        asset = entry["bindings"]
        assert entry["valid"]
        assert len(asset["subjects"]) == 1
        assert len(asset["picture_input_order"]) == expected_counts[n]
        assert {p["subject_label"] for p in asset["picture_input_order"]} == {"<Subject 1>"}
        assert asset.get("previous_frame_sequence") == (n - 1 if n > 4 else None)
        assert north_id not in asset["copy_paste_prompt"] and "PIC_" not in asset["copy_paste_prompt"]
        assert "continuity_review" not in asset and "continuity_review" not in entry
        assert util.load_json(directory / entry["diagnostics_file"])["continuity_review"]["status"] == "approved"

    damaged = deepcopy(registry)
    wall = next(pic for pic in damaged["picture_assets"] if pic["asset_id"] == pictures[1]["asset_id"])
    wall["scene_bindings"][cid] = [6, 9]
    entity = next(e for e in damaged["entities"] if e["global_id"] == north_id)
    del entity["visual_reference"]["scenes"][cid]["8"]
    del entity["timeline"][cid]["8"]
    params["out_dir"] = "h3_gaps"
    result = GenerateH3PromptsNode().run(damaged, config, [chapter], spatial_continuity=plan, **params)["result"][0]
    directory = tmp_path / config["run_folder"] / params["out_dir"] / cid
    entries = {entry["sequence"]: entry for entry in result["chapters"][0]["outputs"]}
    for n, reason in [(7, "missing_scene_binding"), (8, "spatial_location_not_approved")]:
        entry = entries[n]
        assert entry["valid"] and entry["bindings"]["copy_paste_prompt"]
        assert entry["bindings"]["picture_input_order"] == []
        report = util.load_json(directory / entry["diagnostics_file"])
        assert report["chapter_id"] == cid and report["sequence"] == n
        assert report["continuity_review"]["status"] == "incomplete"
        assert not report["continuity_review"]["valid"]
        assert report["checks"][0]["reason"] == reason
    assert entries[9]["bindings"]["picture_input_order"]

    params.update(out_dir="h3_without_spatial", max_pictures_per_subject=1)
    result = GenerateH3PromptsNode().run(damaged, config, [chapter], **params)["result"][0]
    directory = tmp_path / config["run_folder"] / params["out_dir"] / cid
    entries = {entry["sequence"]: entry for entry in result["chapters"][0]["outputs"]}
    coverage = util.load_json(directory / entries[6]["diagnostics_file"])
    assert coverage["checks"][0]["reason"] == "unassigned_location_image"
    assert len(entries[6]["bindings"]["picture_input_order"]) == 1
    missing = util.load_json(directory / entries[8]["diagnostics_file"])
    assert missing["continuity_review"]["status"] == "incomplete"
    assert missing["continuity_review"]["errors"][0]["target"] == "upstream_conflict"
    assert entries[8]["bindings"]["picture_input_order"] == []

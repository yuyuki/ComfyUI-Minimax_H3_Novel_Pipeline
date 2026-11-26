"""Exercise the character sockets, saved documents and reference consumer offline."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import APIConnectionError
import pytest

from minimax_h3_novel_pipeline.core import (
    character_chapter_enrichment as enrichment, character_evidence as evidence,
    character_profile_batches as batches, character_profiles as profiles,
    character_visual_integration, lmstudio_pipeline, path_access, util, visual_designs,
)
from minimax_h3_novel_pipeline.nodes import (
    CharacterEvidenceExtractorNode, CharacterVisualProfileGeneratorNode,
    ConsolidateCharacterProfilesNode, ConsolidateReferencesNode,
    LoadCharacterEvidenceNode, LoadCharacterProfilesNode, LoadCharacterVisualProfilesNode,
)
from tests.passages_helpers import adapt_fixture_reply
from tests.test_character_evidence import observation
from tests.test_character_profile_batches import fake_model, request_payload
from tests.test_character_profiles import group
from tests.test_image_reference_outputs import entity, node_defaults
from tests.test_structured_json import Stream


@pytest.mark.parametrize("audit_failure", ["unavailable", "incomplete response"])
def test_saved_character_pipeline_reopens_source_and_keeps_usable_proof(tmp_path, monkeypatch, audit_failure):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path / kind)
    path = tmp_path / "input" / "chapter2.txt"
    path.parent.mkdir()
    young = "Indy avait les yeux noisette."
    identity = "Henry Jones Jr. Mais ceux qui le connaissaient l'appelaient Indy."
    baker = "La boulangère Camille avait les yeux bleus."
    sailor = "Le marin Camille avait les yeux verts."
    text = "\n\n".join((young, identity, baker, sailor))
    path.write_text(text, encoding="utf-8")
    observations = [
        observation(young, character="Indy", value="yeux noisette"),
        observation(identity, character="Henry Jones Jr.", value="Indy", attribute="identity", persistence="unknown"),
        observation(baker, character="Camille", value="yeux bleus"),
        observation(sailor, character="Camille", value="yeux verts"),
    ]
    audit_inputs = []
    compared_aliases = []

    def create(**kwargs):
        assert kwargs["response_format"]["type"] == "json_schema"
        contract = kwargs["response_format"]["json_schema"]
        assert contract["strict"] is True
        prompt = next(row["content"] for row in kwargs["messages"] if row["role"] == "user")
        payload = json.JSONDecoder().raw_decode(prompt)[0]
        if contract == evidence.EXTRACTION_SCHEMA:
            reply = {"evidence": observations}
        elif contract == evidence.AUDIT_SCHEMA:
            audit_inputs.append(payload["observations"])
            if audit_failure == "unavailable":
                raise APIConnectionError(request=httpx.Request("POST", "http://localhost/v1"))
            reply = {"verdicts": []}
        elif contract == profiles.SCHEMA:
            reply = {"profiles": [group([row["evidence_id"]], canonical_name=row["observation"]["character"])
                                  for row in payload["observations"]]}
        elif contract == batches.IDENTITY_PAIR:
            payload = request_payload(prompt)
            decisions = []
            for job in payload["jobs"]:
                names = {job[side]["canonical_name"] for side in ("left_identity", "right_identity")}
                relation = "uncertain"
                if names == {"Indy", "Henry Jones Jr."}:
                    relation = "same"
                    compared_aliases.append(job["job_id"])
                elif names == {"Camille"}:
                    relation = "different"
                decisions.append({"job_id": job["job_id"], "relation": relation,
                                  "left_support": [0], "right_support": [0]})
            reply = {"decisions": decisions}
        elif contract == enrichment.PRESENCE:
            reply = {"characters": [
                {"local_id": row["local_id"], "presence": "visible", "label": "Indy",
                 "source_quote": young} if "Indy" in row["aliases"] else
                {"local_id": row["local_id"], "presence": "absent", "label": "", "passage_ids": []}
                for row in payload["candidates"]], "unresolved": []}
        elif contract == visual_designs.DESIGN_SCHEMA:
            reply = {"added_details": []}
        elif contract == enrichment.LOCAL_STATES:
            reply = {"observations": []}
        elif contract == enrichment.LOCAL_REVIEW:
            reply = {"issues": []}
        else:
            reply = fake_model(None, None, None, prompt, contract)
        return Stream([json.dumps(adapt_fixture_reply(reply, payload, contract), ensure_ascii=False)])

    class Client:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    client = Client()
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *args: (client, "qwen3.5"))
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261126120000"}
    selection = {"chapter_paths": [str(path)]}
    extracted, extracted_path = CharacterEvidenceExtractorNode().run(
        config, selection, correction_attempts=1, character_scope="all_characters")["result"]
    assert len(audit_inputs) == 2 and audit_inputs[0] == audit_inputs[1]
    assert len(audit_inputs[0]) == len(extracted["chapters"][0]["evidence"]) == 4
    diagnostic = util.load_json(Path(extracted_path).with_name("character_evidence_diagnostics.json"))
    assert diagnostic["status"] == "incomplete"
    assert all(row["status"] == "incomplete" for row in diagnostic["chapters"][0]["observations_for_review"])

    # Introduce an invalid position in an edited copy and exercise the loader path.
    edited = deepcopy(extracted)
    wrong = deepcopy(edited["chapters"][0]["evidence"][0])
    wrong["value"] = "invented"
    wrong["provenance"]["locations"][0]["start_char"] += 1
    edited["chapters"][0]["evidence"].append(wrong)
    edited_path = Path(extracted_path).with_name("edited.json")
    util.save_json(edited_path, edited)
    incoming, _ = LoadCharacterEvidenceNode().run(str(edited_path))
    consolidated, profile_path = ConsolidateCharacterProfilesNode().run(incoming, config)["result"]
    assert len(consolidated["profiles"]) == 3
    assert len(compared_aliases) == 1
    assert len({row["local_id"] for row in consolidated["profiles"] if row["canonical_name"] == "Camille"}) == 2
    excluded = util.load_json(Path(profile_path).with_name("character_profiles_validation.json"))["excluded_evidence"]
    assert len(excluded) == 1 and excluded[0]["observation"]["value"] == "invented"
    assert "Quotation/location mismatch" in excluded[0]["observation"]["validation_warnings"][0]
    incoming, _ = LoadCharacterProfilesNode().run(str(Path(profile_path).parent))
    with monkeypatch.context() as patch:
        def no_model(*args):
            pytest.fail("Visual projection must not create a second LM Studio client.")
        patch.setattr(lmstudio_pipeline, "make_client_and_model", no_model)
        projected, visual_path = CharacterVisualProfileGeneratorNode().run(incoming, selection, config)["result"]
    assert projected["visual_designs"] == util.load_json(Path(visual_path).with_name("visual_designs.json"))
    aliases = [row for row in projected["visual_designs"]["entities"] if "Indy" in row["aliases"]]
    assert len(aliases) == 1
    assert aliases[0]["canonical_name"] == "Henry Jones Jr."
    assert aliases[0]["source_facts"]["stable_visual_description"] == "yeux noisette"
    for document in (extracted, consolidated, projected):
        evidence.reject_embedded_source(document)
        assert '"invented"' not in json.dumps(document)

    visual, _ = LoadCharacterVisualProfilesNode().run(str(visual_path))
    catalog = {"schema_version": util.CHAPTER_SCHEMA, "timeline_version": util.REFERENCE_TIMELINE_VERSION,
               "chapter_id": "chapter", "chapter_name": path.stem, "source": {"file": str(path)},
               "characters": [], "locations": [], "objects": [], "sequences": [
                   {"sequence": 1, "source": young, "adaptation": dict.fromkeys(
                       ("initialState", "event", "endingState"), young)}]}
    step = lmstudio_pipeline.load("consolidate")
    monkeypatch.setattr(step, "reconcile_chapter", lambda *args: [
        {**entity(), "canonical_name": "Indy"}, {**entity(gid="CHAR_002"), "canonical_name": "Camille"},
    ])
    monkeypatch.setattr(step, "audit_registry", lambda client, model, registry, args: registry)
    monkeypatch.setattr(step, "generate_picture_assets", lambda *args: [])
    params = {**node_defaults(ConsolidateReferencesNode), "character_visual_profiles": visual}
    with pytest.warns(UserWarning) as mapping_warnings:
        registry, _ = ConsolidateReferencesNode().run([catalog], config, **params)["result"]
    assert any("Ambiguous character registry mapping" in str(row.message) for row in mapping_warnings)
    context = character_visual_integration.sequence_context(registry, "chapter", 1)
    assert context["sequence"]["characters"][0]["source_quote"] == young
    assert context["registry_ids"] == {aliases[0]["character_id"]: "CHAR_001"}
    assert "validation" not in registry["character_enrichment"]["chapter"]

    for loader, argument, saved, document in (
            (LoadCharacterEvidenceNode, "evidence_path", extracted_path, extracted),
            (LoadCharacterProfilesNode, "profiles_path", profile_path, consolidated),
            (LoadCharacterVisualProfilesNode, "profiles_path", visual_path, projected)):
        old_path = Path(saved).with_name("legacy.json")
        util.save_json(old_path, {**document, "format_version": document["format_version"] - 1})
        with pytest.raises(ValueError, match="Regenerate"):
            loader().run(**{argument: str(old_path)})

    duplicate = deepcopy(extracted)
    duplicate["chapters"].append({**duplicate["chapters"][0], "chapter": 3})
    with pytest.raises(ValueError, match="ambiguous character source"):
        evidence.validate_document(duplicate)
    cache_key = LoadCharacterProfilesNode.IS_CHANGED(profile_path)
    reference_key = ConsolidateReferencesNode.IS_CHANGED(character_visual_profiles=visual)
    path.write_text(text + "\nChapter edited.", encoding="utf-8")
    assert LoadCharacterProfilesNode.IS_CHANGED(profile_path) != cache_key
    assert ConsolidateReferencesNode.IS_CHANGED(character_visual_profiles=visual) != reference_key
    with pytest.raises(ValueError, match="source was modified"):
        LoadCharacterProfilesNode().run(profile_path)
    path.unlink()
    with pytest.raises(ValueError, match="source file is missing"):
        LoadCharacterVisualProfilesNode().run(visual_path)

"""Editable decisions are validated offline and cannot be undone by model matching."""
from copy import deepcopy
from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from minimax_h3_novel_pipeline import (
    configuration_snapshot, editable_schemas, lmstudio_pipeline, path_access,
    pipeline_step2_consolidate as step, pipeline_step3_generate as generate,
    reference_links as links, run_output, util, visual_designs,
)
from minimax_h3_novel_pipeline.consolidate_references import ConsolidateReferencesNode
from tests.test_reference_timeline import catalog

pytest_plugins = ["tests.test_reference_timeline"]


def chapters():
    chapter = catalog()
    indy = chapter["characters"][0]
    indy["canonical_name"] = "Indy"
    for lid, name in (("LOCAL_2", "les cris"), ("LOCAL_3", "Doriane"), ("LOCAL_4", "un garde")):
        item = deepcopy(indy)
        item.update(local_id=lid, canonical_name=name)
        chapter["characters"].append(item)
    chapter["characters"][1].update(stable_visual_description="", voice_description="son inintelligible",
                                    distinguishing_features=["son inintelligible"],
                                    state_by_sequence={"1": {"event": [{"canonical_name": "les cris",
                                        "chapter_appearance": "Des cris au-dessus de lui.",
                                        "voice_description": "crie", "evidence": ["Des cris retentissaient."]}]}})
    return [chapter]


def decision(kind="attribution", source="LOCAL_2", target="LOCAL_3", status="confirmed"):
    return {"kind": kind, "source": {"chapter_id": "chapter", "local_id": source},
            "target": {"chapter_id": "chapter", "local_id": target} if target else None,
            "sequence": 1 if kind == "attribution" else None, "phase": "event" if kind == "attribution" else None,
            "relation": {"attribution": "emitted_by", "identity": "same_as", "relation": "assistant_of"}[kind],
            "status": status, "reason": "Correction utilisateur", "evidence": ["Des cris retentissaient."]}


def manifest(source):
    result = links.document(source)
    result["entities"][1]["classification"] = "manifestation"
    return result


def reconcile(source, document):
    corrected, protected, groups, events = links.plan(source, document)
    # Exercise the real reconciler with its candidate exclusions.
    args = SimpleNamespace(candidate_count=12, include_all_below=35, temperature=0.1,
                           max_tokens=8000, protected_reference_sources=protected)
    registry = []
    for chapter in corrected:
        registry = step.reconcile_chapter(None, "mock", chapter, registry, args)
    return links.apply_decisions(registry, source, document, groups), events, protected


def test_confirmed_sound_attribution_preserves_time_without_polluting_identity(mocked_reconcile):
    source = chapters()
    before = deepcopy(source)
    document = manifest(source)
    document["links"] = [decision()]
    registry, events, _ = reconcile(source, document)
    assert source == before
    assert len(registry) == 3  # Indy, Doriane, and a real unnamed guard.
    doriane = next(e for e in registry if e["canonical_name"] == "Doriane")
    assert "son inintelligible" not in doriane["distinguishing_features"]
    assert "les cris" not in doriane["aliases"]
    assert doriane["voice_description"] == "Low voice"
    obs = doriane["timeline"]["chapter"]["1"]["event"][-1]
    assert obs["evidence"] == ["Des cris retentissaient."]
    assert obs["attributed_from"]["local_id"] == "LOCAL_2"
    assert len(events) == 1
    assert "LOCAL_2" not in step.build_chapter_map(registry)["chapter"]


@pytest.mark.parametrize("status", ["proposed", "rejected", "unresolved"])
def test_uncertain_sounds_never_create_assets_or_silent_attributions(mocked_reconcile, status):
    source = chapters()
    document = manifest(source)
    document["links"] = [decision(status=status)]
    registry, events, _ = reconcile(source, document)
    assert len(registry) == 3 and events[0]["canonical_name"] == "les cris"
    assert not any(e.get("confirmed_attributions") for e in registry)


def test_confirmed_identity_and_rejected_identity_survive_matching_and_audit(mocked_reconcile):
    source = chapters()
    document = manifest(source)
    document["links"] = [decision("identity", "LOCAL_1", "LOCAL_3"),
                         decision("identity", "LOCAL_4", "LOCAL_3", "rejected")]
    registry, _, protected = reconcile(source, document)
    assert len(registry) == 2
    retained = next(e for e in registry if e["canonical_name"] == "Doriane")
    assert "Indy" in retained["aliases"]
    assert len(retained["source_entities"]) == 2
    def audit(client, model, free, args):
        assert free == []  # Neither manual group is submitted for a model merge.
        return free
    assert len(links.audit_unprotected(SimpleNamespace(audit_registry=audit), None, "mock", registry, None, protected)) == 2


def test_relation_does_not_merge_and_is_available_to_generation(mocked_reconcile):
    source = chapters()
    document = manifest(source)
    document["links"] = [decision("relation", "LOCAL_4", "LOCAL_3")]
    registry, _, _ = reconcile(source, document)
    assert len(registry) == 3
    refs = {"entities": registry, "chapter_entity_map": step.build_chapter_map(registry)}
    guard = next(e for e in generate.chapter_catalog(refs, "chapter") if e["canonical_name"] == "un garde")
    assert guard["confirmed_narrative_relations"][0]["relation"] == "assistant_of"
    assert not any(e["confirmed_narrative_relations"] for e in generate.chapter_catalog(refs, "other"))


@pytest.mark.parametrize("change,match", [
    ("digest", "source_digest"), ("status", r"links\[0\].status"), ("address", "unknown address"),
    ("duplicate", "duplicate"), ("scope", "no observation"), ("author", "multiple confirmed"),
    ("contradiction", "contradicts"), ("traits", "identity requires"),
])
def test_invalid_decisions_fail_before_execution(change, match):
    source = chapters()
    document = manifest(source)
    document["links"] = [decision()]
    if change == "digest":
        document["source_digest"] = "stale"
    elif change == "status":
        document["links"][0]["status"] = "confirm"
    elif change == "address":
        document["links"][0]["target"]["local_id"] = "missing"
    elif change == "duplicate":
        document["links"] *= 2
    elif change == "scope":
        document["links"][0]["sequence"] = 2
    elif change == "author":
        document["links"].append(decision(target="LOCAL_1"))
    elif change == "contradiction":
        document["links"] = [decision("identity", "LOCAL_1", "LOCAL_3"),
                             decision("identity", "LOCAL_3", "LOCAL_4"),
                             decision("identity", "LOCAL_1", "LOCAL_4", "rejected")]
    else:
        document["links"] = [decision("identity")]
    with pytest.raises(ValueError, match=match):
        links.validate_links(document, source)


def test_schema_exports_and_visual_traits(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    editable_schemas.export_schemas(tmp_path)
    for filename in ("reference_links.schema.json", "visual_designs.schema.json"):
        Draft202012Validator.check_schema(util.load_json(tmp_path / filename))
    data = {"$schema": "https://invalid.example/no-network", "schema_version": visual_designs.DESIGN_SCHEMA_VERSION,
            "image_style": "realistic photographic", "entities": [{"global_id": "CHAR_001", "entity_type": "character",
            "canonical_name": "Indy", "source_facts": {"stable_visual_description": "", "distinguishing_features": []},
            "added_details": {"hair": "bruns"}}]}
    path = tmp_path / "visual_designs.json"
    util.save_json(path, data)
    assert visual_designs.resolve_designs_path(str(path)) == path
    data["entities"][0]["added_details"] = {"architecture": "pierre"}
    util.save_json(path, data)
    with pytest.raises(ValueError, match=r"entities\[0\].added_details"):
        visual_designs.resolve_designs_path(str(path))


def test_links_only_exports_and_import_errors_precede_model_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: tmp_path)
    source = chapters()
    document = manifest(source)
    path = tmp_path / "reference_links.json"
    util.save_json(path, document)
    args = {key: spec[1]["default"] for section in ConsolidateReferencesNode.INPUT_TYPES().values()
            for key, spec in section.items() if len(spec) > 1 and "default" in spec[1]}
    args.update(reference_links_path=str(path), links_only=True)
    config = {"api_url": "http://127.0.0.1:1234/v1", "thinking": False, "run_folder": "20261004120000"}
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock"))
    monkeypatch.setattr(step, "chat_json", lambda *a: pytest.fail("Imported links-only run needs no inference"))
    class Blocker:
        def __init__(self, message):
            self.message = message
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", SimpleNamespace(ExecutionBlocker=Blocker))
    response = ConsolidateReferencesNode().run(source, config, **args)
    result, summary = response["result"]
    assert response["ui"] == {"text": [summary]}
    assert isinstance(result, Blocker) and "reference_links.json" in summary
    output = tmp_path / config["run_folder"] / "references"
    assert util.load_json(output / "reference_links.json") == document
    assert (output / "visual_designs.schema.json").is_file()
    assert not (output / "consolidated_references.json").exists()
    previous = ConsolidateReferencesNode.IS_CHANGED(**args)
    document["links"] = [decision(status="unresolved", target=None)]
    util.save_json(path, document)
    assert previous != ConsolidateReferencesNode.IS_CHANGED(**args)
    document["source_digest"] = "wrong"
    util.save_json(path, document)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: pytest.fail("Preflight must fail first"))
    with pytest.raises(ValueError, match="source_digest"):
        ConsolidateReferencesNode().run(source, config, **args)


def test_model_proposals_receive_sequences_and_remain_unconfirmed():
    source = chapters()
    calls = []
    def chat(client, model, system, user, schema, *args):
        data = json.loads(user)
        calls.append(data)
        entities = data["available_entities"]
        entities[1]["classification"] = "manifestation"
        return {"entities": entities, "links": [decision(status="proposed")]}
    result = links.prepare_links(chat, None, "mock", source, SimpleNamespace(temperature=0.1, max_tokens=2000))
    assert calls[0]["chapter"]["sequences"] == source[0]["sequences"]
    assert result["links"][0]["status"] == "proposed"
    assert result["source_digest"] == configuration_snapshot.content_digest(source)


def test_full_imported_pass_saves_corrected_registry_and_asset_inputs(tmp_path, monkeypatch, mocked_reconcile):
    from minimax_h3_novel_pipeline import consolidate_references as node
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock"))
    source = chapters()
    document = manifest(source)
    document["links"] = [decision(), decision("relation", "LOCAL_4", "LOCAL_3")]
    imported = tmp_path / "edited.json"
    util.save_json(imported, document)
    monkeypatch.setattr(node, "prepare_designs", lambda *a: {"entities": []})
    pictures = []
    def capture(client, model, specs, args):
        pictures.extend(specs)
        return []
    monkeypatch.setattr(step, "generate_picture_assets", capture)
    monkeypatch.setattr(step, "generate_audio_assets", lambda *a: [])
    args = {key: spec[1]["default"] for section in ConsolidateReferencesNode.INPUT_TYPES().values()
            for key, spec in section.items() if len(spec) > 1 and "default" in spec[1]}
    args.update(reference_links_path=str(imported), no_audit=True)
    config = {"api_url": "http://127.0.0.1:1234/v1", "thinking": False, "run_folder": "20261004130000"}
    registry, _ = ConsolidateReferencesNode().run(source, config, **args)["result"]
    assert {s["canonical_name"] for s in pictures} == {"Indy", "Doriane", "un garde"}
    saved = util.load_json(tmp_path / config["run_folder"] / "references/consolidated_references.json")
    assert saved == registry
    assert saved["reference_links"] == document
    assert len(saved["manifestations"]) == 1
    doriane = next(e for e in saved["entities"] if e["canonical_name"] == "Doriane")
    assert doriane["confirmed_attributions"] == [document["links"][0]]
    assert saved["source_digest"] != configuration_snapshot.content_digest(source)


def test_cross_chapter_attribution_makes_target_available_even_with_existing_local_map(mocked_reconcile):
    source = chapters()
    later = catalog("later", name="Doriane")
    source.append(later)
    document = manifest(source)
    link = decision()
    link["target"] = {"chapter_id": "later", "local_id": "LOCAL_1"}
    document["links"] = [link]
    registry, _, _ = reconcile(source, document)
    refs = {"entities": registry, "chapter_entity_map": step.build_chapter_map(registry)}
    target = next(e for e in registry if e.get("confirmed_attributions"))
    assert target["global_id"] not in refs["chapter_entity_map"]["chapter"].values()
    assert target["global_id"] in {e["global_id"] for e in generate.chapter_catalog(refs, "chapter")}


@pytest.mark.parametrize("kind,name,fragment", [
    ("locations", "la crevasse à Delphes", "la paroi rocheuse"),
    ("objects", "la corde", "les fibres rugueuses"),
])
def test_descriptive_fragment_attaches_to_entity_without_asset_or_alias(mocked_reconcile, kind, name, fragment):
    source = [catalog(kind=kind, name=name)]
    item = deepcopy(source[0][kind][0])
    item.update(local_id="LOCAL_2", canonical_name=fragment)
    source[0][kind].append(item)
    document = links.document(source)
    document["entities"][1]["classification"] = "manifestation"
    link = decision(target="LOCAL_1")
    link["relation"] = "describes"
    document["links"] = [link]
    registry, events, _ = reconcile(source, document)
    assert len(registry) == 1 and len(events) == 1
    retained = registry[0]
    assert fragment not in retained["aliases"]
    observation = retained["timeline"]["chapter"]["1"]["event"][-1]
    assert observation["chapter_state"] == item["state_by_sequence"]["1"]["event"][0]["chapter_state"]
    assert "chapter_appearance" not in observation
    assert observation["attributed_from"]["local_id"] == "LOCAL_2"

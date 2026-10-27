"""Bounded consolidation, source coverage, resumable requests and editable imports."""
from contextlib import nullcontext
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from minimax_h3_novel_pipeline.core import character_profiles as cp, character_profile_batches as batches
from minimax_h3_novel_pipeline.core.character_profile_document import make_document, validate_document
from minimax_h3_novel_pipeline.core import path_access
from minimax_h3_novel_pipeline.nodes import LoadCharacterProfilesNode
from minimax_h3_novel_pipeline.nodes import consolidate_character_profiles as node
from tests.test_character_profiles import document, group, plan, refs, settings
from tests.test_character_evidence import observation
from tests.test_structured_json import Stream, client_for


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path / kind)
    return tmp_path / "output"


def request_payload(prompt):
    payload = json.JSONDecoder().raw_decode(prompt)[0]
    table = {row["evidence_id"]: row for row in payload.get("observations", [])}
    def hydrate(value):
        if isinstance(value, dict):
            if set(value) == {"evidence_id"} and value["evidence_id"] in table:
                return table[value["evidence_id"]]
            return {k: hydrate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [hydrate(v) for v in value]
        return value
    if "jobs" in payload:
        payload["jobs"] = hydrate(payload["jobs"])
    return payload


def fake_model(*args):
    payload = request_payload(args[3])
    contract = args[4]
    if contract == cp.SCHEMA:
        groups = {}
        for row in payload["observations"]:
            groups.setdefault(row["observation"]["character"], []).append(row["evidence_id"])
        return {"profiles": [group(ids, canonical_name=name) for name, ids in groups.items()]}
    if contract == batches.TRAITS:
        rows = []
        for row in payload["observations"]:
            item = row["observation"]
            trait = ("height" if "mètre" in item["value"] else
                     "build" if "corpulent" in item["value"] else "eye_color")
            kind = "global" if item["persistence"] == "persistent" and item["classification"] == "explicit" else "context"
            if item["attribute"] in {"clothing", "accessory"} or item["persistence"] == "temporary":
                trait, kind = item["attribute"], "temporary"
            rows.append({"evidence_id": row["evidence_id"], "traits": [{"trait": trait, "kind": kind}]})
        return {"observations": rows}
    decisions = []
    for job in payload["jobs"]:
        if contract == batches.IDENTITY_PAIR:
            decisions.append({"job_id": job["job_id"], "relation": "same",
                              "evidence": refs([job["left"][0]["evidence_id"], job["right"][0]["evidence_id"]])})
        else:
            relation = "independent" if job["left"]["observation"]["persistence"] == "temporary" else "conflict"
            decisions.append({"job_id": job["job_id"], "relation": relation, "scope": "same_story_period",
                              **dict.fromkeys(("before_id", "after_id", "support_id", "temporal_quote"), "")})
    return {"decisions": decisions}


def run(data, workspace, monkeypatch, **kwargs):
    chat = Mock(side_effect=fake_model)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    result, saved = cp.consolidate(data, cp.prepare_evidence(data), workspace, None, "qwen", settings(**kwargs))
    return result, saved, chat


def fixture_document():
    return document(observation(), observation("Élise avait les yeux bleus.", value="yeux bleus"),
                    observation("Élise mesurait un mètre soixante-dix.", value="un mètre soixante-dix"),
                    observation("Élise portait une robe rouge.", attribute="clothing", value="robe rouge", persistence="temporary"),
                    observation("Élise portait une robe bleue.", attribute="clothing", value="robe bleue", persistence="temporary"))


def test_complete_pipeline_precise_conflicts_and_schema(workspace, monkeypatch):
    data = fixture_document()
    result, saved, chat = run(data, workspace, monkeypatch)
    assert len(result["profiles"]) == 1
    profile = result["profiles"][0]
    assert len(profile["supporting_evidence"]) == 5
    assert len(profile["unresolved_conflicts"]) == 1
    conflict = profile["unresolved_conflicts"][0]
    assert conflict["trait"] == "eye_color"
    assert [c["selected"] for c in conflict["choices"]] == [True, False]
    assert [t["trait"] for t in profile["global_traits"]] == ["height"]
    assert len(profile["temporary_states"]) == 2
    assert result["source_document"] == data
    assert json.loads(Path(saved).read_text(encoding="utf-8")) == result
    assert (workspace / "character_profiles.schema.json").is_file()
    assert {c.args[4]["name"] for c in chat.call_args_list} == {
        cp.SCHEMA["name"], batches.IDENTITY_PAIR["name"], batches.TRAITS["name"],
        batches.RELATIONS["name"]}
    assert all(c.args[4]["strict"] for c in chat.call_args_list)
    assert validate_document(result) is result


def test_later_build_is_global_while_earlier_outfit_stays_temporary(workspace, monkeypatch):
    data = document(
        observation("Doumas portait un manteau.", character="Doumas", attribute="clothing",
                    value="manteau", persistence="temporary"),
        observation("Doumas était corpulent.", character="Doumas", value="corpulent"))
    result, _, _ = run(data, workspace, monkeypatch)
    profile = result["profiles"][0]
    assert [trait["trait"] for trait in profile["global_traits"]] == ["build"]
    assert len(profile["temporary_states"]) == 1
    assert not profile["unresolved_conflicts"]
    assert validate_document(result) is result


def test_batches_reconcile_distant_values_no_loss_and_resume(workspace, monkeypatch):
    data = document(*(observation("Élise avait les yeux verts. " + "x" * 2500, value="yeux verts") for _ in range(8)),
                    observation("Élise avait les yeux bleus. " + "x" * 2500, value="yeux bleus"))
    # One chapter, large enough to split the identity and trait passes.
    first = data["chapters"][0]
    text = "\n".join(c["source_text"] for c in data["chapters"])
    # Separate chapters also exercise cross-chapter reconciliation; full source
    # locations remain attached to the original chapters, rather than flattened.
    assert len(text) > len(first["source_text"])
    result, _, chat = run(data, workspace, monkeypatch, context_tokens=24000, max_tokens=1024)
    records = cp.prepare_evidence(data)
    profile = result["profiles"][0]
    assert {r["evidence_id"] for r in profile["supporting_evidence"]} == set(records)
    assert len(profile["unresolved_conflicts"]) == 1
    assert len(profile["unresolved_conflicts"][0]["choices"][0]["evidence_ids"]) == 8
    traits = [c for c in chat.call_args_list if c.args[4] == batches.TRAITS]
    assert len(traits) > 1
    for call in chat.call_args_list:
        assert len((call.args[2] + call.args[3] + batches.encoded(call.args[4])).encode()) + 1024 + batches.RESERVE <= 24000
    chat.reset_mock()
    repeated, _ = cp.consolidate(data, records, workspace, None, "qwen", settings(context_tokens=24000, max_tokens=1024))
    assert repeated == result
    chat.assert_not_called()
    cache = next((workspace / ".cache" / "character_profiles").glob("*.json"))
    cache.write_text('{}', encoding="utf-8")
    cp.consolidate(data, records, workspace, None, "qwen", settings(context_tokens=24000, max_tokens=1024))
    assert chat.call_count == 1


def test_homonyms_do_not_merge_on_shared_name(workspace, monkeypatch):
    def separate(*args):
        result = fake_model(*args)
        if args[4] == batches.IDENTITY_PAIR:
            for decision in result["decisions"]:
                decision.update(relation="uncertain", evidence=[])
        return result
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", separate)
    data = document(observation(), observation("Élise la boulangère avait les yeux bleus.", value="yeux bleus"))
    result, _ = cp.consolidate(data, cp.prepare_evidence(data), workspace, None, "qwen", settings())
    assert len(result["profiles"]) == 2
    assert all(not p["unresolved_conflicts"] for p in result["profiles"])


def test_alias_reconciliation_requires_connecting_quote(workspace, monkeypatch):
    data = document(observation("Élise, aussi appelée Lise, avait les yeux verts."),
                    observation("Lise avait les yeux verts.", character="Lise"))
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", fake_model)
    result, _ = cp.consolidate(data, cp.prepare_evidence(data), workspace, None, "qwen", settings())
    assert len(result["profiles"]) == 1
    assert result["profiles"][0]["aliases"] == ["Lise"]


@pytest.mark.parametrize("relation", ["change", "uncertain"])
def test_explicit_chronology_and_uncertainty_are_not_global(workspace, monkeypatch, relation):
    data = document(observation("Élise avait une marque rouge.", value="marque rouge"),
                    observation("Après la guérison, sa marque rouge devint une cicatrice blanche.", value="cicatrice blanche"))
    def chronology(*args):
        result = fake_model(*args)
        if args[4] == batches.RELATIONS:
            payload = request_payload(args[3])
            for decision, job in zip(result["decisions"], payload["jobs"]):
                decision["relation"] = relation
                if relation == "change":
                    decision.update(before_id=job["left"]["evidence_id"], after_id=job["right"]["evidence_id"],
                                    support_id=job["right"]["evidence_id"], temporal_quote="Après la guérison")
        return result
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chronology)
    result, _ = cp.consolidate(data, cp.prepare_evidence(data), workspace, None, "qwen", settings())
    profile = result["profiles"][0]
    assert not profile["global_traits"]
    assert not profile["unresolved_conflicts"]
    assert bool(profile["chronological_changes"]) == (relation == "change")
    assert profile["trait_relations"][0]["relation"] == relation


@pytest.mark.parametrize("contract,payload,response", [
    (cp.SCHEMA, {"observations": [{"evidence_id": "e1"}, {"evidence_id": "e2"}]}, plan(["e1", "e2"])),
    (batches.TRAITS, {"observations": [{"evidence_id": "e1"}, {"evidence_id": "e2"}]},
     {"observations": [{"evidence_id": k, "traits": [{"trait": "eye_color", "kind": "global"}]} for k in ("e1", "e2")]}),
    (batches.IDENTITY_PAIR, {"jobs": [{"job_id": "e1"}, {"job_id": "e2"}]},
     {"decisions": [{"job_id": k, "relation": "uncertain", "evidence": []} for k in ("e1", "e2")]}),
    (batches.RELATIONS, {"jobs": [{"job_id": "e1"}, {"job_id": "e2"}]},
     {"decisions": [{"job_id": k, "relation": "uncertain", "scope": "unknown",
        **dict.fromkeys(("before_id", "after_id", "support_id", "temporal_quote"), "")} for k in ("e1", "e2")]}),
])
def test_actual_transport_compact_retry_preserves_schema_and_ids(workspace, contract, payload, response):
    client, create = client_for(Stream(['{']), Stream([json.dumps(response)]))
    requests = batches.Requests(workspace, client, "qwen3.5", settings())
    assert requests.call("Keep all IDs", contract, payload, lambda result: None) == response
    assert create.call_count == 2
    for call in create.call_args_list:
        assert call.kwargs["response_format"] == {"type": "json_schema", "json_schema": contract}
        assert all(key in str(call.kwargs["messages"]) for key in ("e1", "e2"))


def test_semantic_retry_full_coverage_and_partial_resume(workspace, monkeypatch):
    calls = []
    failed = False
    def broken(*args):
        nonlocal failed
        calls.append(args)
        if args[4] == batches.TRAITS and not failed:
            failed = True
            return {"observations": []}
        return fake_model(*args)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", broken)
    data = document(observation())
    result, _ = cp.consolidate(data, cp.prepare_evidence(data), workspace, None, "qwen", settings())
    traits = [a for a in calls if a[4] == batches.TRAITS]
    assert len(traits) == 2
    assert traits[0][3] in traits[1][3]
    assert result["profiles"]


def test_empty_identity_profile_corrected_with_schema_and_full_evidence(workspace):
    data = document(observation(), observation("Élise avait les yeux bleus.", value="yeux bleus"))
    records = cp.prepare_evidence(data)
    ids = list(records)
    payload = {"observations": [batches.compact_record(key, item) for key, item in records.items()]}
    invalid = {"profiles": [group(ids), group([])]}
    corrected = plan(ids)
    client, create = client_for(Stream([json.dumps(invalid)]), Stream([json.dumps(corrected)]))
    requests = batches.Requests(workspace, client, "qwen3.5", settings())
    check = lambda result: cp.build_profiles(result, records)
    result = requests.call(cp.SYSTEM, cp.SCHEMA, payload, check)
    assert result == corrected
    assert create.call_count == 2
    for call in create.call_args_list:
        contract = call.kwargs["response_format"]
        assert contract == {"type": "json_schema", "json_schema": cp.SCHEMA}
        fields = contract["json_schema"]["schema"]["properties"]["profiles"]["items"]["properties"]
        assert fields["evidence"]["minItems"] == 1
        assert "minItems" not in fields["identity_support"]
        assert all(key in str(call.kwargs["messages"]) for key in ids)
    assert "Rejected:" in str(create.call_args_list[1].kwargs["messages"])
    cache = list((workspace / ".cache" / "character_profiles").glob("*.json"))
    assert len(cache) == 1
    assert json.loads(cache[0].read_text(encoding="utf-8")) == corrected


def test_repeated_empty_identity_profile_warns_without_cache(workspace, monkeypatch, caplog):
    records = cp.prepare_evidence(document(observation()))
    chat = Mock(return_value={"profiles": [group([])]})
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    requests = batches.Requests(workspace, None, "qwen", settings(correction_attempts=2))
    payload = {"observations": [batches.compact_record(key, item) for key, item in records.items()]}
    assert requests.call(cp.SYSTEM, cp.SCHEMA, payload, lambda result: cp.build_profiles(result, records)) is None
    assert chat.call_count == 3
    assert "ignoring invalid model decision" in caplog.text
    assert requests.stats["warnings"][0]["input"] == payload
    assert not list((workspace / ".cache" / "character_profiles").glob("*.json"))
    assert not (workspace / "character_profiles.json").exists()


def test_empty_identity_profile_direct_validation_identifies_field():
    records = cp.prepare_evidence(document(observation()))
    with pytest.raises(ValueError, match=r"profiles\[0\].evidence is empty"):
        cp.build_profiles({"profiles": [group([])]}, records)


def test_inferred_identity_support_warns_and_consolidation_continues(workspace, monkeypatch, caplog):
    data = document(observation(classification="inferred"),
                    observation("Élise mesurait un mètre soixante-dix.", value="un mètre soixante-dix"))
    records = cp.prepare_evidence(data)
    inferred = next(key for key, item in records.items() if item["classification"] == "inferred")
    def unsupported(*args):
        result = fake_model(*args)
        if args[4] == cp.SCHEMA:
            for profile in result["profiles"]:
                profile["identity_support"] = profile["evidence"]
        return result
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", unsupported)
    result, saved = cp.consolidate(data, records, workspace, None, "qwen", settings(correction_attempts=2))
    assert Path(saved).is_file()
    assert validate_document(result) is result
    reason = "Identity support must cite explicit physical evidence."
    assert reason in caplog.text
    report = json.loads((workspace / "character_profiles_requests.json").read_text())
    assert any(w["reason"] == reason and w["attempts"] == 3 for w in report["warnings"])
    assert {r["evidence_id"] for p in result["profiles"] for r in p["supporting_evidence"]} == set(records)
    assert all(inferred not in p["identity_support"] for p in result["profiles"])
    assert any(t["trait"] == "height" for p in result["profiles"] for t in p["global_traits"])


@pytest.mark.parametrize("contract", [cp.SCHEMA, batches.TRAITS, batches.IDENTITY_PAIR,
                                     batches.RELATIONS])
def test_invalid_decisions_do_not_block_export_or_lose_sources(workspace, monkeypatch, contract):
    def broken(*args):
        if args[4] == contract:
            return {}
        return fake_model(*args)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", broken)
    data = fixture_document()
    records = cp.prepare_evidence(data)
    result, saved = cp.consolidate(data, records, workspace, None, "qwen", settings(correction_attempts=0))
    assert Path(saved).is_file()
    assert validate_document(result) is result
    assert {r["evidence_id"] for p in result["profiles"] for r in p["supporting_evidence"]} == set(records)
    report = json.loads((workspace / "character_profiles_requests.json").read_text())
    assert report["warnings"]
    assert all(w["schema"] == contract["name"] for w in report["warnings"])
    if contract == batches.TRAITS:
        assert all(not p["global_traits"] and not p["temporary_states"] for p in result["profiles"])
    if contract == batches.IDENTITY_PAIR:
        assert len(result["profiles"]) == len(data["chapters"])


@pytest.mark.parametrize("corrections", [0, 1])
def test_trait_batch_failure_respects_correction_setting(workspace, monkeypatch, corrections):
    records = cp.prepare_evidence(fixture_document())
    bad = next(iter(records))
    calls = []
    def broken(*args):
        if args[4] == batches.TRAITS:
            payload = request_payload(args[3])
            ids = {row["evidence_id"] for row in payload["observations"]}
            calls.append(ids)
            if bad in ids:
                return {"observations": []}
        return fake_model(*args)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", broken)
    requests = batches.Requests(workspace, None, "qwen", settings(correction_attempts=corrections))
    profile = cp.build_profiles(plan(list(records)), records)[0]
    batches.analyse_traits(profile, records, requests)
    if corrections:
        assert {bad} in calls
    else:
        assert len(calls) == 1
    assert any(len(ids) > 1 and bad in ids for ids in calls)
    applied = {key for field in ("global_traits", "temporary_states")
               for row in profile[field] for key in row["evidence_ids"]}
    assert applied == (set(records) - {bad} if corrections else set())
    assert len(profile["supporting_evidence"]) == len(records)


def test_unexpected_runtime_error_is_not_silenced(workspace, monkeypatch):
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", Mock(side_effect=RuntimeError("technical failure")))
    with pytest.raises(RuntimeError, match="technical failure"):
        run_data = document(observation())
        cp.consolidate(run_data, cp.prepare_evidence(run_data), workspace, None, "qwen", settings())


def test_cancel_between_batches_resumes_completed_requests(workspace, monkeypatch):
    class InterruptProcessingException(RuntimeError):
        pass
    def interrupted(*args):
        if args[4] == batches.TRAITS:
            raise InterruptProcessingException()
        return fake_model(*args)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", interrupted)
    data = document(observation())
    records = cp.prepare_evidence(data)
    with pytest.raises(InterruptProcessingException):
        cp.consolidate(data, records, workspace, None, "qwen", settings())
    assert len(list((workspace / ".cache" / "character_profiles").glob("*.json"))) == 1
    assert not (workspace / "character_profiles.json").exists()
    chat = Mock(side_effect=fake_model)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    cp.consolidate(data, records, workspace, None, "qwen", settings())
    assert [call.args[4] for call in chat.call_args_list] == [batches.TRAITS]


def test_oversized_observation_fails_before_network(workspace, monkeypatch):
    data = document(observation("Élise " + "x" * 100000))
    chat = Mock()
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    with pytest.raises(ValueError, match="No evidence was truncated"):
        cp.consolidate(data, cp.prepare_evidence(data), workspace, None, "qwen", settings(context_tokens=24000, max_tokens=1024))
    chat.assert_not_called()
    assert (workspace / "character_profiles_validation.json").is_file()


def editable_fixture():
    data = fixture_document()
    records = cp.prepare_evidence(data)
    profiles = cp.build_profiles({"profiles": [group([key]) for key in records]}, records)
    return make_document(data, records, profiles)


@pytest.mark.parametrize("selected", [[False, False], [True, True], [False, True]])
def test_loader_preserves_selections_renamed_and_directory(workspace, monkeypatch, selected):
    result, saved, _ = run(fixture_document(), workspace, monkeypatch)
    for choice, flag in zip(result["profiles"][0]["unresolved_conflicts"][0]["choices"], selected):
        choice["selected"] = flag
    # An external $schema value is not fetched; runtime uses its bundled schema.
    result["$schema"] = "https://invalid.example/schema.json"
    path = Path(saved)
    path.write_text(json.dumps(result), encoding="utf-8")
    loader = LoadCharacterProfilesNode()
    assert loader.run(str(workspace))[0] == result
    renamed = workspace / "edited.json"
    renamed.write_bytes(path.read_bytes())
    original = renamed.read_bytes()
    assert loader.run(str(renamed)) == (result, str(renamed.resolve()))
    assert renamed.read_bytes() == original
    before = loader.IS_CHANGED(str(renamed))
    result["profiles"][0]["canonical_name"] = "Edited name"
    renamed.write_text(json.dumps(result), encoding="utf-8")
    assert loader.IS_CHANGED(str(renamed)) != before
    assert loader.run(str(renamed))[0] == result
    assert NODE_CLASS_MAPPINGS["LoadCharacterProfilesNode"] is LoadCharacterProfilesNode
    assert NODE_DISPLAY_NAME_MAPPINGS["LoadCharacterProfilesNode"] == "Load Character Profiles"
    assert loader.RETURN_TYPES == ("MINIMAX_CHARACTER_PROFILES", "STRING")


@pytest.mark.parametrize("bad", ["missing", "self", "chain", "cycle", "duplicate", "reference", "shape", "lost"])
def test_loader_rejects_invalid_links_structure_and_references(workspace, bad):
    result = editable_fixture()
    profiles = result["profiles"]
    if bad == "missing":
        profiles[0]["link_to_local_id"] = "absent"
    elif bad == "self":
        profiles[0]["link_to_local_id"] = profiles[0]["local_id"]
    elif bad in {"chain", "cycle"}:
        profiles[0]["link_to_local_id"] = profiles[1]["local_id"]
        profiles[1]["link_to_local_id"] = profiles[2 if bad == "chain" else 0]["local_id"]
    elif bad == "duplicate":
        profiles[1]["local_id"] = profiles[0]["local_id"]
    elif bad == "reference":
        profiles[0]["global_traits"] = [{"trait": "eye_color", "value": "green", "evidence_ids": ["absent"]}]
    elif bad == "lost":
        profiles.pop()
    else:
        profiles[0]["link_to_local_id"] = 5
    workspace.mkdir()
    path = workspace / "edited.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="profiles"):
        LoadCharacterProfilesNode().run(str(path))


def test_manual_link_and_path_errors(workspace):
    result = editable_fixture()
    result["profiles"][1]["link_to_local_id"] = result["profiles"][0]["local_id"]
    workspace.mkdir()
    path = workspace / "edited.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    assert LoadCharacterProfilesNode().run(str(path))[0] == result
    with pytest.raises(ValueError):
        LoadCharacterProfilesNode().run(str(workspace.parent / "outside.json"))
    path.write_text('{', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        LoadCharacterProfilesNode().run(str(path))


def test_node_budget_settings_and_configuration(workspace, monkeypatch):
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", fake_model)
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "qwen"))
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261025120000"}
    result, saved = node.ConsolidateCharacterProfilesNode().run(document(observation()), config,
                    context_tokens=24000, max_tokens=1024)["result"]
    assert result["profiles"]
    assert Path(saved).is_relative_to(workspace / config["run_folder"])
    report = json.loads(Path(saved).with_name("character_profiles_requests.json").read_text())
    assert report["context_tokens"] == 24000
    with pytest.raises(ValueError, match="requires Qwen"):
        node.ConsolidateCharacterProfilesNode().run(document(), {"model_family": "Mistral"})


def test_cache_invalidation_for_changed_evidence_and_force(workspace, monkeypatch):
    chat = Mock(side_effect=fake_model)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    data = document(observation())
    first, _ = cp.consolidate(data, cp.prepare_evidence(data), workspace, None, "qwen", settings())
    assert chat.call_count == 2
    cp.consolidate(data, cp.prepare_evidence(data), workspace, None, "qwen", settings(force=True))
    assert chat.call_count == 4
    edited = document(observation("Élise avait les yeux bleus.", value="yeux bleus"))
    second, _ = cp.consolidate(edited, cp.prepare_evidence(edited), workspace, None, "qwen", settings())
    assert chat.call_count == 6
    assert second != first
    assert second["profiles"][0]["global_traits"][0]["value"] == "yeux bleus"


def test_shared_quotation_table_keeps_all_comparison_references():
    rows = [batches.compact_record(key, item) for key, item in cp.prepare_evidence(fixture_document()).items()]
    jobs = [{"job_id": str(i), "left": rows[0], "right": row} for i, row in enumerate(rows[1:])]
    payload = batches.decision_payload(jobs)
    assert len(payload["observations"]) == len(rows)
    assert len(payload["jobs"]) == len(jobs)
    assert request_payload(batches.encoded(payload))["jobs"] == jobs
    assert len(batches.encoded(payload)) < len(batches.encoded({"jobs": jobs}))


def test_invented_temporal_quote_is_rejected(workspace, monkeypatch):
    data = document(observation(), observation("Élise avait les yeux bleus.", value="yeux bleus"))
    def fabricated(*args):
        result = fake_model(*args)
        if args[4] == batches.RELATIONS:
            payload = request_payload(args[3])
            for decision, job in zip(result["decisions"], payload["jobs"]):
                decision.update(relation="change", before_id=job["left"]["evidence_id"],
                                after_id=job["right"]["evidence_id"], support_id=job["right"]["evidence_id"],
                                temporal_quote="Invented healing chronology")
        return result
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", fabricated)
    result, saved = cp.consolidate(data, cp.prepare_evidence(data), workspace, None, "qwen", settings())
    assert Path(saved).is_file()
    assert validate_document(result) is result
    profile = result["profiles"][0]
    assert not profile["chronological_changes"]
    assert not profile["global_traits"]
    assert profile["trait_relations"][0]["relation"] == "uncertain"
    report = json.loads((workspace / "character_profiles_requests.json").read_text())
    assert "exact explicit" in report["warnings"][0]["reason"]

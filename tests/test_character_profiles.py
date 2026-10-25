"""Cross-chapter reconciliation, provenance, conservative identities and shared transport."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
from openai import APIConnectionError
import pytest

from minimax_h3_novel_pipeline import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from minimax_h3_novel_pipeline.core import character_profiles as profiles, path_access
from minimax_h3_novel_pipeline.nodes import consolidate_character_profiles as node
from minimax_h3_novel_pipeline.nodes import LoadCharacterEvidenceNode
from tests.test_character_evidence import observation
from tests.test_structured_json import Stream, client_for


def document(*items):
    chapters = []
    for index, item in enumerate(items, 1):
        text = item["quotation"]
        name = f"chapter{index}"
        chapters.append({"chapter": index, "chapter_name": name, "source_file": name + ".txt",
                         "source_text": text, "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "location_basis": "utf8_decoded_text", "evidence": [{**item, "provenance": {
                             "chapter": index, "chapter_name": name, "file": name + ".txt", "passage": 1,
                             "locations": [{"start_char": 0, "end_char": len(text), "start_line": 1,
                                            "end_line": text.count("\n", 0, len(text) - 1) + 1}]}}]})
    return {"format_version": 1, "location_convention": "Original offsets", "chapters": chapters}


def refs(ids):
    return [{"evidence_id": key} for key in ids]


def group(ids, **overrides):
    return {"canonical_name": "Élise", "identity_status": "confirmed", "evidence": refs(ids),
            "identity_support": [], "conflicts": [], "changes": [], **overrides}


def plan(records):
    return {"profiles": [group(records)] if records else []}


def settings(**overrides):
    return argparse.Namespace(**{**dict(temperature=0.12, max_tokens=16384, thinking=True,
                                      force=False, correction_attempts=1), **overrides})


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path / kind)
    return tmp_path / "output"


def test_aliases_all_chapters_provenance_and_stable_ids():
    data = document(observation(),
                    observation("Élise était aussi appelée Lise.", attribute="alias", value="Lise"),
                    observation("Lise avait les yeux verts.  \r\n", character="Lise"))
    original = deepcopy(data)
    records = profiles.prepare_evidence(data)
    ids = list(records)
    response = {"profiles": [group(ids, identity_support=refs([ids[1]]))]}
    result = profiles.build_profiles(response, records)[0]
    assert result["aliases"] == ["Lise"]
    assert len(result["supporting_evidence"]) == 3
    assert [s["observation"] for s in result["supporting_evidence"]] == [
        {**c["evidence"][0], "validation_confidence": "high", "validation_warnings": []}
        for c in data["chapters"]]
    response["profiles"][0]["evidence"].reverse()
    data["chapters"].reverse()
    assert profiles.build_profiles(response, profiles.prepare_evidence(data)) == [result]
    data["chapters"].reverse()
    assert data == original


def test_ambiguous_alias_merge_rejected_but_separate_profiles_allowed():
    data = document(observation(), observation("Lise avait les yeux verts.", character="Lise"))
    records = profiles.prepare_evidence(data)
    ids = list(records)
    with pytest.raises(ValueError, match="connecting quotations"):
        profiles.build_profiles(plan(records), records)
    response = {"profiles": [group([ids[0]], identity_status="ambiguous"),
                             group([ids[1]], canonical_name="Lise", identity_status="ambiguous")]}
    assert len(profiles.build_profiles(response, records)) == 2
    response = {"profiles": [group(ids, identity_status="ambiguous")]}
    with pytest.raises(ValueError, match="Do not merge ambiguous"):
        profiles.build_profiles(response, records)


def test_same_name_can_remain_distinct():
    records = profiles.prepare_evidence(document(observation(), observation("Élise la boulangère attend.",
                                                                              attribute="context", value="boulangère")))
    response = {"profiles": [group([key], identity_status="ambiguous") for key in records]}
    result = profiles.build_profiles(response, records)
    assert len({item["character_id"] for item in result}) == 2
    response["profiles"].reverse()
    assert profiles.build_profiles(response, records) == result


def test_conflicts_retained_even_when_model_omits_them():
    records = profiles.prepare_evidence(document(observation(), observation("Élise avait les yeux bleus.", value="yeux bleus")))
    result = profiles.build_profiles(plan(records), records)[0]
    assert len(result["attributes"]["stable_physical"]) == 2
    assert result["unresolved_conflicts"][0]["status"] == "unresolved"
    assert set(result["unresolved_conflicts"][0]["evidence_ids"]) == set(records)
    response = plan(records)
    response["profiles"][0]["conflicts"] = [{"evidence": refs(records)}]
    assert profiles.build_profiles(response, records)[0]["unresolved_conflicts"][0]["detected_by"] == "model"


def test_temporal_changes_require_exact_explicit_support():
    data = document(observation("Élise portait du rouge.", attribute="clothing", value="rouge", persistence="temporary"),
                    observation("Plus tard, Élise remplaça sa robe rouge par une bleue.",
                                attribute="clothing", value="bleue", persistence="temporary"))
    records = profiles.prepare_evidence(data)
    before, after = records
    response = plan(records)
    change = {"before_id": before, "after_id": after, "support_id": after,
              "temporal_quote": "Plus tard, Élise remplaça sa robe rouge par une bleue."}
    response["profiles"][0]["changes"] = [change]
    result = profiles.build_profiles(response, records)[0]
    assert result["chronological_changes"][0]["temporal_quote"] == change["temporal_quote"]
    assert result["unresolved_conflicts"]  # No prior state is overwritten by chronology.
    assert len(result["attributes"]["temporary_appearance"]) == 2
    change["temporal_quote"] = "The next day she wore blue."
    with pytest.raises(ValueError, match="Temporal change"):
        profiles.build_profiles(response, records)
    change["temporal_quote"] = "Plus tard"
    records[after]["classification"] = "inferred"
    with pytest.raises(ValueError, match="Temporal change"):
        profiles.build_profiles(response, records)


def test_inferences_unknowns_actions_and_equipment_are_separate():
    records = profiles.prepare_evidence(document(
        observation(classification="inferred"),
        observation("Élise porte une épée.", attribute="accessory", value="épée", persistence="temporary"),
        observation("Élise court.", attribute="action", value="court", persistence="temporary")))
    result = profiles.build_profiles(plan(records), records)[0]
    assert result["attributes"]["stable_physical"] == []
    assert result["attributes"]["inferred_information"][0]["classification"] == "inferred"
    assert result["attributes"]["temporary_appearance"][0]["value"] == "épée"
    assert result["attributes"]["actions"][0]["value"] == "court"
    assert "physical_appearance" in result["unknown_characteristics"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign", "fabrication", "extra_quote"])
def test_rejects_lost_or_fabricated_evidence(mutation):
    records = profiles.prepare_evidence(document(observation()))
    response = plan(records)
    if mutation == "missing":
        response["profiles"] = []
    elif mutation == "duplicate":
        response["profiles"] *= 2
    elif mutation == "foreign":
        response["profiles"][0]["evidence"] = refs(["invented"])
    elif mutation == "fabrication":
        response["profiles"][0]["canonical_name"] = "Invented name"
    else:
        response["profiles"][0]["quotation"] = "Altered quotation"
    with pytest.raises(ValueError):
        profiles.build_profiles(response, records)


@pytest.mark.parametrize("field,value", [("quotation", "translated"), ("classification", "fact"),
                                        ("location", -1), ("provenance", 99)])
def test_invalid_input_is_retained_with_low_confidence(field, value, caplog):
    data = document(observation())
    item = data["chapters"][0]["evidence"][0]
    if field == "location":
        item["provenance"]["locations"][0]["start_char"] = value
    elif field == "provenance":
        item["provenance"]["chapter"] = value
    else:
        item[field] = value
    original = deepcopy(data)
    records = profiles.prepare_evidence(data)
    item = next(iter(records.values()))
    assert item["validation_confidence"] == "low"
    assert item["validation_warnings"]
    assert data == original
    assert len(caplog.records) == 1
    assert "1/1 observations excluded" in caplog.text
    report = profiles.validation_report(records)
    assert report["excluded_observations"] == 1
    assert report["excluded_evidence"][0]["observation"] == item
    assert profiles.build_profiles({"profiles": []}, records) == []


def test_validation_report_survives_failed_consolidation(workspace, monkeypatch):
    data = document(observation(), observation("invalid"))
    data["chapters"][1]["evidence"][0]["quotation"] = "invented"
    records = profiles.prepare_evidence(data)
    monkeypatch.setattr(profiles.lmstudio_json, "chat_json", Mock(side_effect=APIConnectionError(
        request=httpx.Request("POST", "http://localhost/v1"))))
    with pytest.raises(RuntimeError, match="request failed"):
        profiles.consolidate(data, records, workspace, None, "qwen", settings())
    report = json.loads((workspace / "character_profiles_validation.json").read_text(encoding="utf-8"))
    assert report["observations"] == 2
    assert report["usable_observations"] == report["excluded_observations"] == 1
    assert report["excluded_evidence"][0]["observation"]["quotation"] == "invented"


def test_node_json_loader_edits_cache_force_and_corruption(workspace, monkeypatch):
    workspace.mkdir()
    data = document(observation())
    path = workspace / "evidence.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    def respond(*args):
        observations = json.loads(args[3])["observations"]
        return plan([item["evidence_id"] for item in observations])
    chat = Mock(side_effect=respond)
    monkeypatch.setattr(profiles.lmstudio_json, "chat_json", chat)
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "qwen3.5"))
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261024120000"}
    cls = NODE_CLASS_MAPPINGS["ConsolidateCharacterProfilesNode"]
    def run(**kwargs):
        loaded, _ = LoadCharacterEvidenceNode().run(str(path))
        return cls().run(loaded, config, **kwargs)["result"]
    result, saved = run()
    assert run() == (result, saved)
    assert chat.call_count == 1
    assert json.loads(Path(saved).read_text(encoding="utf-8")) == result
    assert Path(saved).name == "character_profiles.json"
    assert result["source_document"] == data
    item = data["chapters"][0]["evidence"][0]
    item.update(value="vert sombre", classification="inferred", notes="manually corrected", warnings=["old warning"])
    path.write_text(json.dumps(data), encoding="utf-8")
    result, _ = run()
    assert chat.call_count == 2
    assert result["profiles"][0]["attributes"]["inferred_information"][0]["value"] == "vert sombre"
    assert result["source_document"] == data
    run(force=True)
    run(temperature=0.2)
    assert chat.call_count == 4
    for cache in Path(saved).parent.joinpath(".cache").glob("*.json"):
        cache.write_text('{"profiles": []}', encoding="utf-8")
    run()
    assert chat.call_count == 5
    assert cls.RETURN_TYPES == ("MINIMAX_CHARACTER_PROFILES", "STRING")
    assert cls.INPUT_TYPES()["required"]["character_evidence"] == ("MINIMAX_CHARACTER_EVIDENCE",)
    assert NODE_DISPLAY_NAME_MAPPINGS["ConsolidateCharacterProfilesNode"] == "Consolidate Character Profiles"
    assert "api_key" not in cls.INPUT_TYPES()["required"]
    with pytest.raises(ValueError, match="requires Qwen"):
        cls().run(data, {"model_family": "Mistral"})


def test_structured_transport_compact_retry_preserves_all_evidence(workspace, monkeypatch):
    monkeypatch.setattr(profiles.lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    data = document(*(observation(value=f"observation {i}") for i in range(7)))
    records = profiles.prepare_evidence(data)
    first, second = Stream(['{"profiles":']), Stream([json.dumps(plan(records))])
    client, create = client_for(first, second)
    result, _ = profiles.consolidate(data, records, workspace, client, "qwen3.5", settings())
    assert len(result["profiles"][0]["supporting_evidence"]) == 7
    assert create.call_count == 2
    for call in create.call_args_list:
        assert call.kwargs["response_format"] == {"type": "json_schema", "json_schema": profiles.SCHEMA}
        prompt = str(call.kwargs["messages"])
        assert all(key in prompt for key in records)
    assert first.closed and second.closed


def test_semantic_retry_keeps_full_input_and_rejects_bad_cache(workspace, monkeypatch):
    data = document(observation(), observation())
    records = profiles.prepare_evidence(data)
    chat = Mock(side_effect=[{"profiles": []}, plan(records)])
    monkeypatch.setattr(profiles.lmstudio_json, "chat_json", chat)
    profiles.consolidate(data, records, workspace, None, "qwen3.5", settings())
    assert chat.call_count == 2
    for call in chat.call_args_list:
        assert all(key in call.args[3] for key in records)
        assert call.args[4] == profiles.SCHEMA


def test_empty_evidence_saves_empty_profiles_without_request(workspace, monkeypatch):
    chat = Mock(side_effect=AssertionError("No model request needed"))
    monkeypatch.setattr(profiles.lmstudio_json, "chat_json", chat)
    data = document()
    result, saved = profiles.consolidate(data, {}, workspace, None, "qwen3.5", settings())
    assert result["profiles"] == []
    assert Path(saved).is_file()


@pytest.mark.parametrize("kind", ["network", "interrupt", "invalid"])
def test_failures_do_not_save_or_cache(workspace, monkeypatch, kind):
    class InterruptProcessingException(RuntimeError):
        pass
    error = {"network": APIConnectionError(message="private server detail", request=httpx.Request("POST", "http://localhost")),
             "interrupt": InterruptProcessingException(), "invalid": ValueError("invalid JSON")}[kind]
    chat = Mock(side_effect=error)
    monkeypatch.setattr(profiles.lmstudio_json, "chat_json", chat)
    data = document(observation())
    with pytest.raises(RuntimeError) as caught:
        profiles.consolidate(data, profiles.prepare_evidence(data), workspace, None, "qwen3.5", settings())
    assert "private server detail" not in str(caught.value)
    assert chat.call_count == (2 if kind == "invalid" else 1)
    assert not workspace.joinpath(".cache").exists()
    assert not workspace.joinpath("character_profiles.json").exists()


@pytest.mark.parametrize("quotation", [
    "Élise avait les yeux verts.",
    "E\u0301lise avait les yeux verts.",
    "Élise\tavait les yeux verts.",
])
def test_normalized_quotation_retains_original_source_and_confidence(quotation):
    data = document(observation("Élise  avait\r\nles yeux verts."))
    data["chapters"][0]["evidence"][0]["quotation"] = quotation
    original = deepcopy(data)
    records = profiles.prepare_evidence(data)
    result = profiles.build_profiles(plan(records), records)[0]
    evidence = result["supporting_evidence"][0]["observation"]
    assert evidence["validation_confidence"] == "medium"
    assert evidence["quotation"] == quotation
    assert evidence["provenance"] == original["chapters"][0]["evidence"][0]["provenance"]
    assert len(result["attributes"]["stable_physical"]) == 1
    assert data == original


@pytest.mark.parametrize("bad", ["fabricated", "", None, "ÉLISE AVAIT LES YEUX BLEUS.",
                               "Élise avait les yeux bleus!"])
def test_partial_consolidation_excludes_invalid_quotes(workspace, monkeypatch, bad):
    data = document(observation(), observation("Élise avait les yeux bleus.", value="yeux bleus"))
    data["chapters"][1]["evidence"][0]["quotation"] = bad
    records = profiles.prepare_evidence(data)
    def respond(*args):
        supplied = json.loads(args[3])["observations"]
        assert len(supplied) == 1
        assert supplied[0]["observation"]["validation_confidence"] == "high"
        return plan([item["evidence_id"] for item in supplied])
    monkeypatch.setattr(profiles.lmstudio_json, "chat_json", respond)
    result, saved = profiles.consolidate(data, records, workspace, None, "qwen3.5", settings())
    assert len(result["profiles"]) == 1
    profile = result["profiles"][0]
    assert len(profile["attributes"]["stable_physical"]) == 1
    assert profile["unresolved_conflicts"] == []
    excluded = result["excluded_evidence"][0]["observation"]
    assert excluded["quotation"] == bad
    assert excluded["validation_confidence"] == "low"
    assert excluded["validation_warnings"]
    assert result["source_document"] == data
    assert json.loads(Path(saved).read_text(encoding="utf-8")) == result
    from minimax_h3_novel_pipeline.core.character_visual_profiles import prepare_profiles
    assert prepare_profiles(result) == result["profiles"]


@pytest.mark.parametrize("bad", [None, {"quotation": "x"}, {"provenance": {}}])
def test_malformed_item_does_not_block_valid_siblings(workspace, monkeypatch, bad):
    data = document(observation())
    data["chapters"][0]["evidence"].append(bad)
    records = profiles.prepare_evidence(data)
    valid = [key for key, item in records.items() if item["validation_confidence"] == "high"]
    monkeypatch.setattr(profiles.lmstudio_json, "chat_json", lambda *a: plan(valid))
    result, _ = profiles.consolidate(data, records, workspace, None, "qwen3.5", settings())
    assert len(result["profiles"][0]["supporting_evidence"]) == 1
    assert len(result["excluded_evidence"]) == 1


def test_all_unusable_evidence_skips_model(workspace, monkeypatch):
    data = document(observation())
    data["chapters"][0]["evidence"][0]["quotation"] = "unverified"
    monkeypatch.setattr(profiles.lmstudio_json, "chat_json", Mock(side_effect=AssertionError))
    result, _ = profiles.consolidate(data, profiles.prepare_evidence(data), workspace, None, "qwen3.5", settings())
    assert result["profiles"] == []
    assert result["excluded_evidence"][0]["observation"]["validation_confidence"] == "low"


def test_structural_document_failure_remains_blocking():
    data = document(observation())
    del data["chapters"][0]["source_text"]
    with pytest.raises(ValueError, match="source_text"):
        profiles.prepare_evidence(data)


def test_bad_first_item_does_not_skip_remaining_chapter_evidence():
    data = document(observation())
    valid = deepcopy(data["chapters"][0]["evidence"][0])
    data["chapters"][0]["evidence"][0]["quotation"] = "fabricated"
    data["chapters"][0]["evidence"].append(valid)
    records = profiles.prepare_evidence(data)
    usable = [key for key, item in records.items() if item["validation_confidence"] == "high"]
    result = profiles.build_profiles(plan(usable), records)
    assert len(result[0]["attributes"]["stable_physical"]) == 1
    assert len(result[0]["supporting_evidence"]) == 1

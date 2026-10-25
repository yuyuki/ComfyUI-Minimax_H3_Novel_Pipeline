"""Cross-chapter reconciliation, provenance, conservative identities and shared transport."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
from openai import APIConnectionError
import pytest

from minimax_h3_novel_pipeline.core import character_profiles as profiles, path_access
from minimax_h3_novel_pipeline.core import character_profile_batches as batches
from tests.test_character_evidence import observation


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
            "identity_support": [], **overrides}


def plan(records):
    return {"profiles": [group(records)] if records else []}


def respond(*args):
    payload = json.JSONDecoder().raw_decode(args[3])[0]
    if args[4] == profiles.SCHEMA:
        return plan(row["evidence_id"] for row in payload["observations"])
    return {"observations": [{"evidence_id": row["evidence_id"],
        "traits": [{"trait": "eye_color", "kind": "global"}]} for row in payload["observations"]]}


def settings(**overrides):
    return argparse.Namespace(**{**dict(temperature=0.12, max_tokens=16384, thinking=True,
                                      force=False, correction_attempts=1), **overrides})


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path / kind)
    return tmp_path / "output"


def test_aliases_all_chapters_provenance_and_stable_ids():
    data = document(observation(),
                    observation("Élise, aussi appelée Lise, avait les yeux verts."),
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
    records = profiles.prepare_evidence(document(observation(), observation("Élise la boulangère avait les yeux bleus.", value="yeux bleus")))
    response = {"profiles": [group([key], identity_status="ambiguous") for key in records]}
    result = profiles.build_profiles(response, records)
    assert len({item["local_id"] for item in result}) == 2
    response["profiles"].reverse()
    assert profiles.build_profiles(response, records) == result






def test_inferences_unknowns_and_equipment_are_separate():
    records = profiles.prepare_evidence(document(
        observation(classification="inferred"),
        observation("Élise porte une épée.", attribute="accessory", value="épée", persistence="temporary"),
        observation("Élise avait le visage pâle.", value="visage pâle", persistence="unknown")))
    result = profiles.build_profiles(plan(records), records)[0]
    assert result["attributes"]["stable_physical"] == []
    assert result["attributes"]["inferred_information"][0]["classification"] == "inferred"
    assert result["attributes"]["temporary_appearance"][0]["value"] == "épée"
    assert result["attributes"]["other_information"][0]["value"] == "visage pâle"
    assert "unknown_characteristics" not in result


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
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", Mock(side_effect=APIConnectionError(
        request=httpx.Request("POST", "http://localhost/v1"))))
    with pytest.raises(RuntimeError, match="request failed"):
        profiles.consolidate(data, records, workspace, None, "qwen", settings())
    report = json.loads((workspace / "character_profiles_validation.json").read_text(encoding="utf-8"))
    assert report["observations"] == 2
    assert report["usable_observations"] == report["excluded_observations"] == 1
    assert report["excluded_evidence"][0]["observation"]["quotation"] == "invented"








def test_empty_evidence_saves_empty_profiles_without_request(workspace, monkeypatch):
    chat = Mock(side_effect=AssertionError("No model request needed"))
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    data = document()
    result, saved = profiles.consolidate(data, {}, workspace, None, "qwen3.5", settings())
    assert result["profiles"] == []
    assert Path(saved).is_file()


@pytest.mark.parametrize("kind", ["network", "interrupt"])
def test_failures_do_not_save_or_cache(workspace, monkeypatch, kind):
    class InterruptProcessingException(RuntimeError):
        pass
    error = {"network": APIConnectionError(message="private server detail", request=httpx.Request("POST", "http://localhost")),
             "interrupt": InterruptProcessingException(), "invalid": ValueError("invalid JSON")}[kind]
    chat = Mock(side_effect=error)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    data = document(observation())
    with pytest.raises(RuntimeError) as caught:
        profiles.consolidate(data, profiles.prepare_evidence(data), workspace, None, "qwen3.5", settings())
    assert "private server detail" not in str(caught.value)
    assert chat.call_count == 1
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




@pytest.mark.parametrize("bad", [None, {"quotation": "x"}, {"provenance": {}}])
def test_malformed_item_does_not_block_valid_siblings(workspace, monkeypatch, bad):
    data = document(observation())
    data["chapters"][0]["evidence"].append(bad)
    records = profiles.prepare_evidence(data)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", respond)
    result, _ = profiles.consolidate(data, records, workspace, None, "qwen3.5", settings())
    assert len(result["profiles"][0]["supporting_evidence"]) == 1
    assert len(result["excluded_evidence"]) == 1


def test_all_unusable_evidence_skips_model(workspace, monkeypatch):
    data = document(observation())
    data["chapters"][0]["evidence"][0]["quotation"] = "unverified"
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", Mock(side_effect=AssertionError))
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

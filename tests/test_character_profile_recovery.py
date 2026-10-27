"""Narrative context, independent recovery and bounded physical classification."""
import hashlib
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import character_profiles as cp, character_profile_batches as batches
from minimax_h3_novel_pipeline.core.character_profile_document import validate_document
from tests.test_character_evidence import observation
from tests.test_character_profiles import document, group, settings
from tests.test_character_profile_batches import fake_model, request_payload


def narrative_document(text, *items):
    data = document(*items)
    chapter = data["chapters"][0]
    chapter["source_text"] = text
    chapter["source_text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    rows = []
    for item in items:
        start = text.index(item["quotation"])
        end = start + len(item["quotation"])
        rows.append({**item, "provenance": {
            "file": chapter["source_file"], "chapter": chapter["chapter"], "chapter_name": chapter["chapter_name"],
            "passage": 1, "locations": [{"start_char": start, "end_char": end,
                "start_line": text.count("\n", 0, start) + 1, "end_line": text.count("\n", 0, end - 1) + 1}]}})
    chapter["evidence"] = rows
    data["chapters"] = [chapter]
    return data


def test_verbatim_context_connects_short_quotes_and_aliases(tmp_path, monkeypatch):
    from minimax_h3_novel_pipeline.core import path_access
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    text = "Henry Jones Jr. était appelé Indy. Il avait les yeux noisette. Indy avait de larges épaules."
    data = narrative_document(text,
        observation("les yeux noisette", character="Henry Jones Jr.", value="hazel eyes"),
        observation("de larges épaules", character="Indy", value="broad shoulders"))
    records = cp.prepare_evidence(data)
    for item in records.values():
        for ctx in item["_source_context"]:
            assert text[ctx["start_char"]:ctx["end_char"]] == ctx["text"]
    captured = []
    def model(*args):
        captured.append(args)
        return fake_model(*args)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", model)
    result, _ = cp.consolidate(data, records, tmp_path, None, "qwen", settings(correction_attempts=0))
    assert len(result["profiles"]) == 1
    profile = result["profiles"][0]
    assert profile["canonical_name"] == "Henry Jones Jr."
    assert profile["aliases"] == ["Indy"]
    assert len(profile["supporting_evidence"]) == 2
    pairs = [request_payload(a[3]) for a in captured if a[4] == batches.IDENTITY_PAIR]
    assert pairs and all(row["observation"]["source_context"] for row in pairs[0]["observations"])
    assert validate_document(result) is result


def test_source_context_is_recomputed_not_trusted():
    text = "Élise entra. Élise avait les yeux verts. Elle sourit."
    item = observation(source_context=[{"text": "Invented alias"}])
    data = narrative_document(text, item)
    record = next(iter(cp.prepare_evidence(data).values()))
    assert "Invented alias" not in str(record["_source_context"])
    assert data["chapters"][0]["evidence"][0]["source_context"] == [{"text": "Invented alias"}]


def test_invalid_identity_does_not_fragment_valid_neighbor(tmp_path, monkeypatch):
    data = narrative_document("Élise avait les yeux verts. Élise était grande. Paul était petit.",
        observation(), observation("Élise était grande.", value="grande"),
        observation("Paul était petit.", character="Paul", value="petit"))
    records = cp.prepare_evidence(data)
    ids = list(records)
    response = {"profiles": [group(ids[:2]), group(ids[2:], canonical_name="Invented") ]}
    chat = Mock(return_value=response)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    requests = batches.Requests(tmp_path, None, "qwen", settings(correction_attempts=0))
    groups = batches.partition_identities(records, requests)
    assert sorted(len(g["evidence"]) for g in groups) == [1, 2]
    assert {ref["evidence_id"] for g in groups for ref in g["evidence"]} == set(records)
    assert chat.call_count == 1
    assert not list(tmp_path.rglob("*.json"))  # Never cache a partial response as complete.


def test_partial_identity_pairs_keep_valid_decision(tmp_path, monkeypatch):
    records = cp.prepare_evidence(document(observation(), observation("Élise avait les yeux bleus.")))
    rows = [batches.compact_record(k, v) for k, v in records.items()]
    jobs = [{"job_id": str(i), "left": [rows[0]], "right": [rows[1]]} for i in range(2)]
    good = {"job_id": "0", "relation": "same", "left_support": [0], "right_support": [0]}
    bad = {"job_id": "1", "relation": "same", "left_support": [0], "right_support": []}
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", Mock(return_value={"decisions": [good, bad]}))
    def check(row, job):
        if row["relation"] == "same" and len(row["evidence"]) != 2:
            raise ValueError("Both sides required")
    requests = batches.Requests(tmp_path, None, "qwen", settings(correction_attempts=0))
    decisions = list(requests.decisions(jobs, batches.IDENTITY_SYSTEM, batches.IDENTITY_PAIR, check))
    assert {d["job_id"]: d["relation"] for d, _ in decisions} == {"0": "same", "1": "uncertain"}


def test_invalid_trait_key_does_not_remove_valid_traits(tmp_path, monkeypatch):
    records = cp.prepare_evidence(document(observation(), observation("Élise était dégingandée.", value="dégingandée")))
    ids = list(records)
    response = {"observations": [
        {"evidence_id": ids[0], "traits": [{"trait": "eye_color", "kind": "global"}]},
        {"evidence_id": ids[1], "traits": [{"trait": "exposed_skin", "kind": "global"}]}]}
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", Mock(return_value=response))
    requests = batches.Requests(tmp_path, None, "qwen", settings(correction_attempts=0))
    assignments = batches.classify_traits(records, requests)
    assert set(assignments) == {ids[0]}
    assert requests.stats["partial_responses_salvaged"] == 1


def test_global_trait_classification_batches_across_characters(tmp_path, monkeypatch):
    from minimax_h3_novel_pipeline.core import path_access
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    data = document(*(observation(f"Person{i} avait les yeux verts.", character=f"Person{i}") for i in range(5)))
    chat = Mock(side_effect=fake_model)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    result, _ = cp.consolidate(data, cp.prepare_evidence(data), tmp_path, None, "qwen", settings())
    assert len(result["profiles"]) == 5
    calls = [c for c in chat.call_args_list if c.args[4] == batches.TRAITS]
    assert len(calls) == 1
    assert len(request_payload(calls[0].args[3])["observations"]) == 5
    assert not any(c.args[4]["name"] == "character_trait_keys" for c in chat.call_args_list)


@pytest.mark.parametrize("value,quote,trait,kind", [
    ("cheveux noirs", "cheveux noirs et yeux sombres", "eye_color", "global"),
    ("pleines", "ses lèvres pleines", "hair_color", "global"),
    ("pleines", "ses lèvres pleines", "leg_shape", "global"),
    ("pleines", "ses lèvres pleines", "cheekbones", "global"),
    ("yeux brillants", "ses yeux brillants", "hair_style", "global"),
    ("yeux brillants", "ses yeux brillants", "eye_color", "global"),
    ("petit homme", "un petit homme mal habillé", "facial_hair", "temporary"),
    ("barbe de deux jours", "barbe de deux jours", "facial_hair", "global"),
    ("éclat glacial", "Ses yeux avaient un éclat glacial", "eye_color", "global"),
    ("épaisse", "son épaisse chevelure aile-de-corbeau", "hair_color", "global"),
    ("aile-de-corbeau", "son épaisse chevelure aile-de-corbeau", "hair_thickness", "global"),
    ("pieds démesurés", "ses pieds démesurés", "build", "global"),
])
def test_physical_subject_guard_rejects_observed_pollution(value, quote, trait, kind):
    with pytest.raises(ValueError):
        batches.validate_trait_subject(observation(quote, value=value), {"trait": trait, "kind": kind})


@pytest.mark.parametrize("value,quote,trait,kind", [
    ("cheveux noirs", "cheveux noirs et yeux sombres", "hair_color", "global"),
    ("pleines", "ses lèvres pleines", "lips", "global"),
    ("yeux brillants", "ses yeux brillants", "eye_appearance", "global"),
    ("petit homme", "un petit homme mal habillé", "height", "global"),
    ("barbe de deux jours", "barbe de deux jours", "facial_hair", "temporary"),
    ("épaisse", "son épaisse chevelure aile-de-corbeau", "hair_thickness", "global"),
    ("aile-de-corbeau", "son épaisse chevelure aile-de-corbeau", "hair_color", "global"),
    ("long cou à la pomme d’Adam proéminente", "long cou à la pomme d’Adam proéminente", "adams_apple", "global"),
    ("long black hair and green eyes", "long black hair and green eyes", "eye_color", "global"),
])
def test_physical_subject_guard_preserves_supported_traits(value, quote, trait, kind):
    batches.validate_trait_subject(observation(quote, value=value), {"trait": trait, "kind": kind})


def test_polluted_classification_keeps_good_assignments_without_false_conflicts(tmp_path, monkeypatch):
    from minimax_h3_novel_pipeline.core import path_access
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    data = narrative_document("Elle avait les cheveux noirs, les lèvres pleines et les yeux brillants.",
        observation("cheveux noirs", value="cheveux noirs"),
        observation("les lèvres pleines", value="pleines"),
        observation("les yeux brillants", value="yeux brillants"))
    records = cp.prepare_evidence(data)
    choices = {"cheveux noirs": ["hair_color", "eye_color"],
               "pleines": ["lips", "hair_color", "hair_style", "leg_shape", "cheekbones"],
               "yeux brillants": ["eye_appearance", "hair_style", "eye_color"]}
    def model(*args):
        if args[4] == batches.TRAITS:
            payload = request_payload(args[3])
            assert all("source_context" not in row["observation"] for row in payload["observations"])
            return {"observations": [{"evidence_id": row["evidence_id"], "traits": [
                {"trait": key, "kind": "global"} for key in choices[row["observation"]["value"]]]}
                for row in payload["observations"]]}
        assert args[4] != batches.RELATIONS  # Distinct body properties must never be compared.
        return fake_model(*args)
    chat = Mock(side_effect=model)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    result, _ = cp.consolidate(data, records, tmp_path, None, "qwen", settings(correction_attempts=0))
    profile = result["profiles"][0]
    assert {r["trait"] for r in profile["global_traits"]} == {"hair_color", "lips", "eye_appearance"}
    assert not profile["unresolved_conflicts"]
    assert len(profile["supporting_evidence"]) == 3
    assert validate_document(result) is result


def test_value_comparison_excludes_narrative_context(tmp_path, monkeypatch):
    data = narrative_document("Élise était là. Ses cheveux noirs luisaient. Ses cheveux bruns étaient longs.",
        observation("cheveux noirs", value="cheveux noirs"), observation("cheveux bruns", value="cheveux bruns"))
    records = cp.prepare_evidence(data)
    assert all(r.get("_source_context") for r in records.values())
    def respond(*args):
        assert "source_context" not in args[3]
        return fake_model(*args)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", respond)
    requests = batches.Requests(tmp_path, None, "qwen", settings())
    rows = batches.compare_values("hair_color", dict.fromkeys(records, "global"), records, requests)
    assert rows and {r["evidence_id"] for r in rows[0]["evidence"]} == set(records)


@pytest.mark.parametrize("value,quote,good,bad", [
    ("roux", "ses cheveux roux", "hair_color", "hair_style"),
    ("red hair", "red hair", "hair_color", "hair_texture"),
    ("folle chevelure", "la folle chevelure", "hair_style", "hair_color"),
    ("cheveux ondulés", "ses cheveux ondulés", "hair_texture", "hair_style"),
    ("jambes poilues", "ses jambes poilues", "body_hair", "leg_shape"),
    ("hairy legs", "hairy legs", "body_hair", "leg_shape"),
    ("jambes brunies", "ses jambes brunies", "skin_color", "leg_shape"),
    ("grand", "grand et élancé", "height", "build"),
    ("élancé", "grand et élancé", "build", "height"),
])
def test_properties_of_same_body_part_remain_distinct(value, quote, good, bad):
    item = observation(quote, value=value)
    batches.validate_trait_subject(item, {"trait": good, "kind": "global"})
    with pytest.raises(ValueError):
        batches.validate_trait_subject(item, {"trait": bad, "kind": "global"})


def test_mixed_tan_and_clothing_state_cannot_be_global():
    item = observation("ses jambes brunies, dévoilées presque jusqu’aux genoux",
                       value="jambes brunies, dévoilées presque jusqu’aux genoux")
    with pytest.raises(ValueError):
        batches.validate_trait_subject(item, {"trait": "skin_color", "kind": "global"})
    batches.validate_trait_subject(item, {"trait": "skin_color", "kind": "temporary"})


@pytest.mark.parametrize("positions", [[], [1], [0, 0], [-1], [True]])
def test_identity_side_support_rejects_missing_foreign_duplicate_positions(tmp_path, monkeypatch, positions):
    records = cp.prepare_evidence(document(observation(), observation("Élise avait les yeux bleus.")))
    rows = [batches.compact_record(k, v) for k, v in records.items()]
    jobs = [{"job_id": "0", "left": rows[:1], "right": rows[1:]}]
    result = {"decisions": [{"job_id": "0", "relation": "same", "left_support": [0], "right_support": positions}]}
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", Mock(return_value=result))
    requests = batches.Requests(tmp_path, None, "qwen", settings(correction_attempts=0))
    decisions = list(requests.decisions(jobs, batches.IDENTITY_SYSTEM, batches.IDENTITY_PAIR, lambda d, j: None))
    assert decisions[0][0]["relation"] == "uncertain"


def test_shannon_merges_across_chapters_with_side_support_and_no_color_style_conflict(tmp_path, monkeypatch):
    from minimax_h3_novel_pipeline.core import path_access
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    data = document(observation("Shannon avait les cheveux roux.", character="Shannon", value="roux"),
                    observation("Shannon avait une folle chevelure.", character="Shannon", value="folle chevelure"))
    def model(*args):
        if args[4] == batches.TRAITS:
            payload = request_payload(args[3])
            return {"observations": [{"evidence_id": row["evidence_id"], "traits": [
                {"trait": key, "kind": "global"} for key in
                (["hair_color", "hair_style"] if row["observation"]["value"] == "roux" else ["hair_style"])]}
                for row in payload["observations"]]}
        assert args[4] != batches.RELATIONS
        return fake_model(*args)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", model)
    result, _ = cp.consolidate(data, cp.prepare_evidence(data), tmp_path, None, "qwen", settings(correction_attempts=0))
    assert len(result["profiles"]) == 1
    profile = result["profiles"][0]
    assert {t["trait"] for t in profile["global_traits"]} == {"hair_color", "hair_style"}
    assert not profile["unresolved_conflicts"]
    assert len(profile["identity_support"]) == 2
    assert validate_document(result) is result

"""Structured source selections across planning, review, extraction and reload."""
import json
from types import SimpleNamespace

from jsonschema import Draft202012Validator
import pytest

from minimax_h3_novel_pipeline.core import (
    chapter_reference_identity as identity, character_evidence,
    cinematic_references, lmstudio_json, path_access, source_text,
)
from minimax_h3_novel_pipeline.nodes.load_chapter_catalogs import LoadChapterCatalogsNode


def request_payload(prompt):
    return json.JSONDecoder().raw_decode(prompt)[0]


def validate_reply(schema, reply):
    assert schema["strict"] is True
    Draft202012Validator(schema["schema"]).validate(reply)
    return reply


def test_exact_repeated_and_disjoint_passages_are_addressed_by_occurrence():
    text = "L’homme  s’arrête. À côté, l’eau étincelle. L’homme  s’arrête."
    index = source_text.PassageIndex(text, "scene:1")
    offers = index.passages()
    assert offers[0]["text"] == offers[-1]["text"] == "L’homme  s’arrête."
    selected = index.resolve([offers[-1]["id"], offers[1]["id"]])
    assert selected[0]["start"] == text.rindex("L’homme")
    assert [text[row["start"]:row["end"]] for row in selected] == [offers[-1]["text"], "À côté,"]
    with pytest.raises(ValueError, match="authorized"):
        source_text.PassageIndex(text, "scene:2").resolve([offers[-1]["id"]])
    with pytest.raises(ValueError, match="authorized"):
        source_text.PassageIndex(text + " Suite.", "scene:1").resolve([offers[-1]["id"]])


def test_extraction_traits_and_identity_keep_local_offsets_through_consumer_and_reload(tmp_path, monkeypatch):
    from minimax_h3_novel_pipeline.core import lmstudio_pipeline
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    step = lmstudio_pipeline.load("extract")
    chapter = {"chapter_name": "Test", "source_file": str(tmp_path / "book.txt"), "sequences": [
        {"sequence": number, "source": "Original prose is out of scope.",
         "adaptation": {"initialState": initial, "event": event, "endingState": "This is out of scope."}}
        for number, initial, event in [(1, "La cloche d’Élise est en bronze.", "Élise arrive."),
                                       (2, "La même cloche reste devant Élise.", "Elle s’approche.")]
    ]}
    requests = []

    def chat(client, model, system, prompt, schema, *args):
        data = request_payload(prompt)
        requests.append(schema["name"])
        if schema == step.CHUNK_SCHEMA:
            local = data.get("source_input", data)
            assert "out of scope" not in str(local)
            identifiers = [row["id"] for row in local["source_passages"]]
            ids = ["unknown"] if requests.count(schema["name"]) == 1 else identifiers
            return validate_reply(schema, {"chunk_summary": "Une cloche.", "characters": [], "locations": [],
                "objects": [{"canonical_name": "cloche",
                    "aliases": [], "importance": "major",
                    "reference_priority": "required", "reference_view_hints": [], "evidence_passage_ids": ids,
                    "presence": "visible", "reference_id": (
                        local["known_references"][0]["reference_id"] if local["sequence"] == 2 else "new"),
                    "visual_traits": [] if local["sequence"] == 2 else [{
                        "attribute": "material", "description": "en bronze", "scope": "intrinsic",
                        "evidence_passage_ids": [identifiers[0]]}]}]})
        pytest.fail(f"Unexpected request for a known reference: {schema['name']}")

    monkeypatch.setattr(step, "chat_json", chat)
    args = SimpleNamespace(temperature=0.15, max_tokens=8192, chunk_chars=4000, force=False, max_attempts=2)
    saved = cinematic_references.process_chapter(chapter, tmp_path, None, "mock", args, "chapter")
    catalogs, _ = LoadChapterCatalogsNode().run(str(saved))
    obj, = catalogs[0]["objects"]
    assert set(obj["state_by_sequence"]) == {"1", "2"}
    trait, = obj["state_by_sequence"]["1"]["source"][0]["visual_traits"]
    assert trait["evidence"] == source_text.normalize("La cloche d’Élise est en bronze.")
    assert trait["evidence_passages"] == [{"chunk": 1, "start": 0, "end": len(trait["evidence"]), "text": trait["evidence"]}]
    report = json.loads((tmp_path / "chapter_references.diagnostics.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert "passage_ids" not in saved.read_text(encoding="utf-8")
    assert requests.count(step.CHUNK_SCHEMA["name"]) == 3


def test_character_evidence_resolves_selected_occurrence_and_invalidates_source_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    quote = "Élise avait  les yeux verts."
    path = tmp_path / "chapter.txt"
    path.write_text(f"{quote} {quote}", encoding="utf-8")
    calls = []

    def chat(client, model, system, prompt, schema, *args):
        data = request_payload(prompt)
        if schema["name"] == character_evidence.AUDIT_SCHEMA["name"]:
            return validate_reply(schema, {"verdicts": [{"observation_index": 0, "verdict": "valid", "issue": "none"}]})
        offers = data["source_passages"]
        calls.append(offers[-1]["id"])
        return validate_reply(schema, {"evidence": [{
            "character": "Élise", "attribute": "physical_appearance", "value": "yeux verts", "confidence": 1,
            "classification": "explicit", "persistence": "persistent", "rationale": "",
            "quotation_passage_ids": [offers[-1]["id"]],
        }]})

    monkeypatch.setattr(lmstudio_json, "chat_json", chat)
    args = SimpleNamespace(temperature=0.15, max_tokens=8192, chunk_chars=1000, force=False,
                           correction_attempts=1, character_scope="all_characters")
    def run():
        return character_evidence.extract_chapter(path, tmp_path, None, "mock", args, 1)
    first = run()
    assert run() == first
    record, = first["evidence"]
    assert record["quotation"] == quote
    assert record["provenance"]["locations"][0]["start_char"] == len(quote) + 1
    path.write_text(f"Lina parlait. {quote}", encoding="utf-8")
    assert run()["evidence"][0]["provenance"]["locations"][0]["start_char"] == len("Lina parlait. ")
    assert calls == ["2", "2"]  # Request-local IDs are reassigned against the changed source.



def test_identity_with_exact_but_unsupported_passages_stays_incomplete(tmp_path, monkeypatch):
    from minimax_h3_novel_pipeline.core import lmstudio_pipeline

    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    step = lmstudio_pipeline.load("extract")
    jobs = [{"sequence": seq, "chunk": 1, "text": text, "result": {
        "characters": [], "locations": [], "objects": [{
            "canonical_name": "La cloche", "aliases": [], "presence": "visible", "reference_id": "new",
            "evidence": [text], "visual_traits": [],
        }],
    }} for seq, text in enumerate(["Une cloche tinte.", "Ailleurs, une cloche tinte."], 1)]
    proposals = []

    def chat(client, model, system, prompt, schema, *args):
        data = request_payload(prompt)
        if schema == identity.IDENTITY_REVIEW_SCHEMA:
            evidence = data["decisions"][0]["source_passages"]
            assert {item["sequence"] for item in evidence} == {1, 2}
            return validate_reply(schema, {"decisions": [{"decision_number": 1, "verdict": "unsupported"}]})
        proposals.append(data)
        return validate_reply(schema, {"groups": [{
            "member_ids": [row["id"] for row in data["entities"]],
            "canonical_member_id": data["entities"][0]["id"],
            "passage_ids": [row["id"] for row in data["source_passages"]],
        }]})

    monkeypatch.setattr(step, "chat_json", chat)
    diagnostics = {}
    args = SimpleNamespace(temperature=0.15, max_tokens=8192, force=False)
    entities, order = {}, 0
    for job in jobs:
        order = identity.append_occurrences(entities, job, step, order)
    result = identity.finish_dictionary(entities, jobs, step, None, "mock", args, tmp_path, diagnostics)
    assert len(result["objects"]) == 2
    assert len(proposals) == 2
    assert proposals[-1]["correction"]["reason"] == "evidence_unsupported"
    assert diagnostics["identity_consolidation"][0]["status"] == "incomplete"
    assert not list(tmp_path.glob("identity_*.json"))



def test_view_landmark_cites_approved_appearance_and_records_exact_evidence(monkeypatch):
    from minimax_h3_novel_pipeline.core import pipeline_step2_consolidate as step

    appearance = "L’arche est en grès. Les marches sont étroites."
    args = SimpleNamespace(max_location_base_views=2, max_attempts=2, max_tokens=8192, temperature=0.1,
        temporal_context_by_chapter={"chapter": {"sequences": [{"sequence": 1,
            "adaptation": {"initialState": "La lumière change.", "event": "La caméra descend."}}]}},
        reference_diagnostics=[])
    group = {"appearance": appearance, "scene_bindings": {"chapter": [1]}}
    requests = []

    def chat(client, model, system, prompt, schema, *unused):
        data = request_payload(prompt)
        if schema == step.LOCATION_VIEW_REVIEW_SCHEMA:
            assert data["views"][0]["landmark_passage_ids"] == [data["appearance_passages"][1]["id"]]
            return validate_reply(schema, {"checks": [{"view_type": "wide_establishing", "issues": []}]})
        requests.append(data)
        offers = data["appearance_passages"]
        chosen = ["wrong-scope"] if len(requests) == 1 else [offers[1]["id"]]
        return validate_reply(schema, {"views": [{"view_type": "wide_establishing",
            "camera_axis": "Vers le bas", "framing": "Vue sur les marches.",
            "landmark_passage_ids": chosen, "scene_ids": [1]}]})

    monkeypatch.setattr(step, "chat_json", chat)
    assert step.plan_location_views(None, "mock", {"global_id": "LOC_001", "canonical_name": "L’arche"}, group, args) == [
        ("wide_establishing", "Vue sur les marches.", {"chapter": [1]})]
    assert requests[0] == requests[1]
    evidence, = args.reference_diagnostics[0]["landmark_evidence"]
    assert evidence["text"] == appearance[evidence["start"]:evidence["end"]] == "Les marches sont étroites."

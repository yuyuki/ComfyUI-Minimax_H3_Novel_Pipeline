"""Structured source selections across planning, review, extraction and reload."""
from copy import deepcopy
import json
from types import SimpleNamespace

from jsonschema import Draft202012Validator
import pytest

from minimax_h3_novel_pipeline.core import (
    chapter_reference_identity as identity, character_evidence, cinematic_adaptation as adaptation,
    cinematic_references, cinematic_scene_planning as planning, lmstudio_json, path_access, source_text,
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


def timing(references):
    return {"beats": [{"events": [ref], "seconds": 1, "reason": "One action."} for ref in references],
            "pause_seconds": 0, "estimated_seconds": len(references), "reason": "Natural pace."}


def whole(index):
    return {"event_index": index, "fragment": None}


def fragment(index, first, last):
    return {"event_index": index, "fragment": {"first_word": first, "last_word": last}}


def continuity(ids=()):
    return {"reset": False, "reset_passage_ids": [],
            "updates": [{"entity": "Élise", "aspect": "support", "fact": "Élise est assise.",
                         "passage_ids": list(ids)}] if ids else []}


def scene(events):
    return {"source": "Élise est assise. L’eau  étincelle.", "initial_state": "Élise attend.",
            "events": events, "final_state": "Élise reste assise."}


def test_prologue_1_and_24_continuity_repairs_leave_valid_timing_untouched():
    original = scene(["Elle appelle."])
    calls = []

    def chat(client, model, system, prompt, schema, *args):
        data = request_payload(prompt)
        assert data["source_evidence"] == []
        selected = [row["id"] for row in data["current_source_passages"]]
        required = schema["schema"]["required"]
        calls.append(required)
        if len(calls) == 1:
            duplicate = continuity(selected)
            duplicate["updates"].append(duplicate["updates"][0].copy())
            return validate_reply(schema, {"timing": timing([whole(1)]), "continuity": duplicate})
        if len(calls) == 2:
            assert data["correction"][0]["reason"] == "duplicate_entity_aspect"
            return validate_reply(schema, {"continuity": continuity(["unknown-passage"])})
        assert data["correction"][0]["reason"] == "passage_reference"
        return validate_reply(schema, {"continuity": continuity([selected[0]])})

    result = planning.assess(chat, None, "mock", original, [], 8, 32768, 2)
    assert calls == [["timing", "continuity"], ["continuity"], ["continuity"]]
    assert result["timing"]["estimated_seconds"] == 1
    assert result["continuity"]["updates"][0]["source_quote"] == "Élise est assise."
    assert planning.cache_key(original, [], 8) != planning.cache_key({**original, "events": ["Elle se lève."]}, [], 8)


@pytest.mark.parametrize("fragments,reason", [
    ([whole(1), fragment(2, 1, 2), fragment(2, 3, 5)], None),
    ([whole(1), fragment(2, 1, 3), fragment(2, 3, 5)], "event_fragment_range"),
    ([whole(1), fragment(2, 1, 2), fragment(2, 4, 5)], "event_fragment_range"),
    ([whole(1), fragment(2, 1, 2)], "event_coverage"),
    ([whole(1), whole(2), fragment(2, 1, 5)], "whole_event_overlap"),
    ([whole(1)], "event_coverage"),
])
def test_prologue_16_references_exact_compound_event_ranges(fragments, reason):
    event = "Élise ferme,  puis elle s’assied."
    original = scene(["Elle appelle.", event])

    def chat(client, model, system, prompt, schema, *args):
        assert request_payload(prompt)["current_scene"]["events"][1] == event
        return validate_reply(schema, {"timing": timing(fragments), "continuity": continuity()})

    if reason is not None:
        with pytest.raises(planning._PlanningValidationError, match=reason):
            planning.assess(chat, None, "mock", original, [], 8, 32768, 0)
        return
    plan = planning.assess(chat, None, "mock", original, [], 8, 32768, 0)
    beats = planning._describe_plan(plan, original["events"])["timing"]["beats"]
    assert [beat["events"][0]["text"] for beat in beats] == ["Elle appelle.", "Élise ferme,  ", "puis elle s’assied."]


def adapt(source, *, checkpoints=None, **kwargs):
    return adaptation.adapt_chapter(None, "google/gemma-4-26b-it", source, chapter_name="Test",
        chunk_chars=4000, max_tokens=16384, temperature=0.15, correction_attempts=1,
        on_checkpoint=(lambda state: checkpoints.append(deepcopy(state))) if checkpoints is not None else None,
        **kwargs)


def adaptation_reply(data, schema):
    return validate_reply(schema, {"sequences": [{
        "source_end_id": data["source_segments"][-1]["id"], "initial_state": "Élise attend.",
        "events": ["Élise ouvre la porte.", "Élise sort."], "final_state": "Élise est dehors.",
    }]})


@pytest.mark.parametrize("failure", [None, "truncated", "missing", "cancelled"])
def test_prologue_8_fidelity_groups_respect_budget_and_require_every_event(monkeypatch, failure):
    groups, saved = [], []
    source = "Élise ouvre la porte. Élise sort."

    class InterruptProcessingException(RuntimeError):
        pass

    def chat(client, model, system, prompt, schema, temperature, max_tokens):
        data = request_payload(prompt)
        assert max_tokens == 16384
        assert lmstudio_json.REQUEST_OUTPUT_LIMIT.get() == 16384
        if "source_segments" in data:
            return adaptation_reply(data, schema)
        if "requested_sections" in data:
            return validate_reply(schema, {"timing": timing([whole(1), whole(2)]), "continuity": continuity()})
        focus = data.get("review_focus")
        if focus in (None, "source_fidelity"):
            raise lmstudio_json.StructuredOutputLengthError("Requested output budget exhausted.")
        if focus != "source_fidelity_event_group":
            return validate_reply(schema, {"issues": []})
        assigned = data["assigned_events"]
        groups.extend(assigned)
        assert data["sequences"][0]["source"] == source
        assert data["review_passages"]
        if assigned == [2] and failure == "truncated":
            raise lmstudio_json.StructuredOutputLengthError("Output budget exhausted.")
        if assigned == [2] and failure == "missing":
            return {"issues": [], "covered_events": []}
        if assigned == [2] and failure == "cancelled":
            raise InterruptProcessingException("Cancelled")
        return validate_reply(schema, {"issues": [], "covered_events": assigned})

    monkeypatch.setattr(lmstudio_json, "chat_json", chat)
    if failure == "cancelled":
        with pytest.raises(InterruptProcessingException):
            adapt(source, checkpoints=saved)
        assert not any(state["stage"] == "scene_committed" for state in saved)
        return
    chapter = adapt(source, checkpoints=saved)
    assert groups == [1, 2]
    assert ("review" in chapter["sequences"][0]) == bool(failure)
    if failure:
        assert chapter["sequences"][0]["review"]["status"] == "incomplete"
        assert saved[-1]["approved_reviews"] == []
    else:
        assert saved[-1]["approved_reviews"]


def test_exact_but_unsupported_continuity_preserves_timing_and_does_not_enter_history(monkeypatch):
    source = "Élise ouvre la porte. Élise sort."
    saved, calls = [], []

    def chat(client, model, system, prompt, schema, *args):
        data = request_payload(prompt)
        if "source_segments" in data:
            return adaptation_reply(data, schema)
        if "requested_sections" in data:
            sections = data["requested_sections"]
            calls.append(sections)
            passage = data["current_source_passages"][0]
            plan = {"continuity": continuity([passage["id"]])}
            if "timing" in sections:
                plan["timing"] = timing([whole(1), whole(2)])
            return validate_reply(schema, plan)
        return validate_reply(schema, {"issues": [{
            "target": 1, "check": "planning_continuity", "message": "Opening the door does not establish sitting.",
            "passage_ids": [data["review_passages"][0]["id"]],
        }]})

    monkeypatch.setattr(lmstudio_json, "chat_json", chat)
    result = adapt(source, checkpoints=saved)
    assert calls == [["timing", "continuity"], ["continuity"]]
    assert saved[-1]["scene_plans"][0]["plan"]["timing"]["estimated_seconds"] == 2
    assert saved[-1]["source_evidence"] == []
    assert result["sequences"][0]["review"]["status"] == "incomplete"
    assert any(row["reason"] == "unsupported_continuity_evidence" for row in result["narrative_review"]["warnings"])


def test_resume_rejects_changed_source_and_obsolete_planning_policy(monkeypatch):
    source, saved = "Élise ouvre la porte. Élise sort.", []

    def chat(client, model, system, prompt, schema, *args):
        data = request_payload(prompt)
        if "source_segments" in data:
            return adaptation_reply(data, schema)
        if "requested_sections" in data:
            return validate_reply(schema, {"timing": timing([whole(1), whole(2)]), "continuity": continuity()})
        return validate_reply(schema, {"issues": []})

    monkeypatch.setattr(lmstudio_json, "chat_json", chat)
    original = adapt(source, checkpoints=saved)
    checkpoint = saved[-1]
    monkeypatch.setattr(lmstudio_json, "chat_json", lambda *args: pytest.fail("Resume made an unexpected request"))
    assert adapt(source, resume_state=checkpoint) == original
    with pytest.raises(ValueError, match="Invalid cinematic resume"):
        adapt(source.replace("porte", "malle"), resume_state=checkpoint)
    with pytest.raises(ValueError, match="Invalid cinematic resume"):
        adapt(source, resume_state={**checkpoint, "review_policy": "timed-source-context-v11"})


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
                "objects": [{"canonical_name": "La cloche", "aliases": [], "importance": "major",
                    "reference_priority": "required", "reference_view_hints": [], "evidence_passage_ids": ids,
                    "presence": "visible", "visual_traits": [] if local["sequence"] == 2 else [{
                        "attribute": "material", "description": "en bronze", "scope": "intrinsic",
                        "evidence_passage_ids": [identifiers[0]]}]}]})
        if schema == identity.IDENTITY_REVIEW_SCHEMA:
            return validate_reply(schema, {"decisions": [{"decision_number": row["decision_number"], "verdict": "supported"}
                                                         for row in data["decisions"]]})
        if schema == identity.SCHEMA:
            offered = data["source_passages"]
            assert {row["sequence"] for row in offered} == {1, 2}
            return validate_reply(schema, {"groups": [{"member_ids": [e["id"] for e in data["entities"]],
                "canonical_member_id": data["entities"][0]["id"],
                "passage_ids": [next(row["id"] for row in offered if row["sequence"] == n) for n in (1, 2)]}]})
        return validate_reply(schema, {"issues": []})

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
    group, = report["identity_consolidation"][0]["groups"]
    assert [row["sequence"] for row in group["source_passages"]] == [1, 2]
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
    assert len(calls) == 2 and calls[0] != calls[1]



def test_compact_transport_preserves_independent_passage_selection():
    from tests.test_structured_json import Stream, client_for

    original = scene(["Elle appelle.", "Elle écoute."])
    index = source_text.PassageIndex(original["source"], planning.cache_key(original, [], 8))
    selected = [row["id"] for row in index.passages()]
    wire = {"timing": timing([whole(1), whole(2)]), "continuity": continuity(selected)}
    client, create = client_for(Stream(['{"timing":']), Stream([json.dumps(wire)]))
    result = planning.assess(lmstudio_json.chat_json, client, "qwen3.5-9b", original, [], 8, 8192, 0)
    assert len(result["continuity"]["updates"][0]["source_passages"]) == 2
    assert create.call_count == 2
    for call in create.call_args_list:
        response = call.kwargs["response_format"]
        assert response["type"] == "json_schema"
        schema = response["json_schema"]
        validate_reply(schema, wire)
        properties = schema["schema"]["properties"]["continuity"]["properties"]
        assert "maxItems" not in properties["updates"]["items"]["properties"]["passage_ids"]
        request = call.kwargs["messages"][-1]["content"]
        assert all(identifier in request for identifier in selected)


def test_identity_with_exact_but_unsupported_passages_stays_incomplete(tmp_path, monkeypatch):
    from minimax_h3_novel_pipeline.core import lmstudio_pipeline

    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    step = lmstudio_pipeline.load("extract")
    jobs = [{"sequence": seq, "chunk": 1, "text": text, "result": {
        "characters": [], "locations": [], "objects": [{
            "canonical_name": "La cloche", "aliases": [], "presence": "visible",
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
    result = identity.consolidate(jobs, step, None, "mock", args, tmp_path, diagnostics, tmp_path / "diagnostics.json")
    assert len(result["objects"]) == 2
    assert len(proposals) == 2
    assert proposals[-1]["correction"]["reason"] == "evidence_unsupported"
    assert diagnostics["identity_consolidation"][0]["status"] == "incomplete"
    assert not list(tmp_path.glob("identity_*.json"))



def test_interrupted_correction_resumes_only_the_invalid_section(monkeypatch):
    source, states, requests = "Élise ouvre la porte. Élise sort.", [], []

    def chat(client, model, system, prompt, schema, *args):
        data = request_payload(prompt)
        if "source_segments" in data:
            return adaptation_reply(data, schema)
        if "requested_sections" in data:
            sections = data["requested_sections"]
            requests.append(sections)
            if "timing" in sections:
                return {"timing": timing([whole(1), whole(2)]), "continuity": continuity(["unoffered"])}
            return validate_reply(schema, {"continuity": continuity()})
        return validate_reply(schema, {"issues": []})

    def stop_on_failure(state):
        if state["stage"] == "planning_attempt":
            states.append(deepcopy(state))
            raise KeyboardInterrupt

    monkeypatch.setattr(lmstudio_json, "chat_json", chat)
    with pytest.raises(KeyboardInterrupt):
        adaptation.adapt_chapter(None, "google/gemma-4-26b-it", source, chapter_name="Test",
                                 chunk_chars=4000, max_tokens=16384, temperature=0.15,
                                 correction_attempts=1, on_checkpoint=stop_on_failure)
    assert set(states[0]["pending_sequences"][0]["planning"]) == {"timing"}
    resumed = adapt(source, resume_state=states[0])
    assert requests == [["timing", "continuity"], ["continuity"]]
    assert "review" not in resumed["sequences"][0]



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

"""Event extraction, deterministic replay, and prose-preserving node integration."""
from contextlib import nullcontext
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline import narrative_nodes as nodes, narrative_state as ns
from .test_narrative_state import confirmed_review, contract, entity, extraction, opening, prologue


def test_declarations_are_absent_until_introduction_and_carry_is_immutable():
    current = {"entities": [entity("indy", "character", visible=True)]}
    snapshot = deepcopy(current)
    response = {"opening_entities": [], "new_entities": [{"id": "torch", "kind": "object"}], "events": [
        {"id": "receive", "description": "Indy receives a torch.", "source_ids": ["source_1"],
         "kind": "introduction", "changes": [
             {"entity": "torch", "field": field, "after": value}
             for field, value in {"status": "lit", "visible": True, "owner": "indy", "location": "indy", "relationship": "held"}.items()]},
        {"id": "mouth", "description": "He puts it in his mouth.", "source_ids": ["source_2"],
         "kind": "action", "changes": [{"entity": "torch", "field": "relationship", "after": "in_mouth"}]},
    ]}
    source = "Indy receives a torch. He puts it in his mouth."
    result = ns.compile_contract(response, source, current)
    assert not ns.validate_contract(result, current, source)
    assert ns.state_map(result["state_before"])["torch"]["status"] == "not_introduced"
    assert result["initial_frame"] == current
    assert result["events"][1]["changes"][0]["before"] == "held"
    assert ns.state_map(result["state_after"])["torch"]["relationship"] == "in_mouth"
    assert current == snapshot
    response["events"][0]["kind"] = "action"
    assert any("introduction" in e["expected_state"] for e in ns.validate_contract(ns.compile_contract(response, source, current)))


def test_review_still_rejects_source_unsupported_initial_state(monkeypatch):
    source = "Indy hangs suspended on a rope."
    response = extraction(contract(opening(), source), source)
    response["opening_entities"][0]["posture"] = "standing"
    conflict = ns.issue("indy", "suspended", "standing", correction="Preserve the source opening posture.")
    def chat(*args):
        if args[4] == ns.VERIFY_REVIEW_SCHEMA:
            return confirmed_review(json.loads(args[3]))
        return response if args[4] == ns.EXTRACTION_SCHEMA else {"errors": [conflict]}
    monkeypatch.setattr(ns, "chat_json", chat)
    with pytest.raises(ValueError, match="standing"):
        ns.track_scene(None, "mock", source, source, attempts=0)


@pytest.mark.parametrize("missing_acquisition", [False, True])
def test_lost_torch_review_is_verified_without_losing_real_acquisition_errors(monkeypatch, missing_acquisition):
    source = "His first torch is lost in the abyss. Indy catches the second torch."
    current = deepcopy(prologue()[2]["state_before"])
    ns.state_map(current)["torch_1"]["location"] = "abyssal dark"
    good = contract(current, source, [("torch_2", "owner", "indy"),
                                    ("torch_2", "location", "indy"), ("torch_2", "relationship", "held")])
    bad = contract(current, source)
    complaints = [
        ns.issue("torch_1", "lost in abyss", "lost in abyss", correction="Invent a prior loss event."),
        ns.issue("torch_1", "invisible", "invisible", correction="Add to initial_frame and remove from state_after."),
    ]
    acquisition = ns.issue("torch_2", "held by indy", "unowned", correction="Record the second torch acquisition.")
    generations, verifications = [], []

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema == ns.EXTRACTION_SCHEMA:
            generations.append(payload)
            beat = bad if missing_acquisition and len(generations) == 1 else good
            return extraction(beat, source, current)
        if schema == ns.REVIEW_SCHEMA:
            return {"errors": complaints + ([acquisition] if missing_acquisition and len(generations) == 1 else [])}
        verifications.append(payload)
        assert "state_after" in payload["candidate"]
        assert "Lost/invisible entities remain tracked" in system
        result = confirmed_review(payload)
        for decision in result["decisions"][:2]:
            decision.update(supported=False, reason="Lost torch is already tracked, invisible and unowned.")
        return result

    monkeypatch.setattr(ns, "chat_json", chat)
    result, report = ns.track_scene(None, "mock", source, source, current, attempts=int(missing_acquisition))
    assert result == good and report["valid"]
    assert len(generations) == len(verifications) == 1 + int(missing_acquisition)
    assert "torch_1" not in ns.state_map(result["initial_frame"])
    assert ns.state_map(result["state_after"])["torch_1"] == ns.state_map(current)["torch_1"]
    if missing_acquisition:
        assert generations[1]["validation_errors"] == [acquisition]
        assert generations[1]["candidate"] == extraction(bad, source, current)


@pytest.mark.parametrize("quote_style", ["decoded", "serialized", "pretty_event", "compact_event"])
def test_contract_review_accepts_exact_evidence_independent_of_json_format(monkeypatch, quote_style):
    source = 'Indy calls "Doriane!"\nIl attend près du câble.'
    good = contract(opening(), source)
    conflict = ns.issue("indy", "calls Doriane", "silent")
    calls = []

    def chat(*args):
        calls.append(args[4])
        if args[4] == ns.EXTRACTION_SCHEMA:
            return extraction(good, source)
        if args[4] == ns.REVIEW_SCHEMA:
            return {"errors": [conflict]}
        payload = json.loads(args[3])
        response = confirmed_review(payload)
        event = payload["candidate"]["events"][0]
        quotes = {
            "decoded": event["description"],
            "serialized": json.dumps(event, ensure_ascii=False),
            "pretty_event": json.dumps(event, ensure_ascii=True, indent=2, sort_keys=True),
            "compact_event": json.dumps(event, separators=(",", ":")),
        }
        response["decisions"][0]["candidate_evidence"] = quotes[quote_style]
        return response

    monkeypatch.setattr(ns, "chat_json", chat)
    # Confirmed findings reach the correction budget instead of failing verification.
    with pytest.raises(ValueError, match="correction budget"):
        ns.track_scene(None, "mock", source, source, attempts=0)
    assert calls == [ns.EXTRACTION_SCHEMA, ns.REVIEW_SCHEMA, ns.VERIFY_REVIEW_SCHEMA]


@pytest.mark.parametrize("evidence", [
    "", "   ", "Indy drops the torch.", "Indy catches ... torch.",
    "Indy catches the torch. He waits.",
    '{"description": "Indy catches the torch.", "visible": 1}',
    '{"description": "Indy catches the torch."}',
])
def test_contract_evidence_rejects_invented_changed_or_joined_content(evidence):
    candidate = {"events": [{"description": "Indy catches the torch.", "visible": True}],
                 "context": "He waits."}
    assert not ns._candidate_contains_evidence(candidate, evidence)


@pytest.mark.parametrize("fault", ["decisions", "source_evidence", "candidate_evidence"])
def test_contract_verification_failure_never_spends_state_correction_budget(monkeypatch, fault):
    source = "Indy calls Doriane."
    good = contract(opening(), source)
    calls = []

    def chat(*args):
        calls.append(args[4])
        if args[4] == ns.EXTRACTION_SCHEMA:
            return extraction(good, source)
        if args[4] == ns.REVIEW_SCHEMA:
            return {"errors": [ns.issue("indy", "calls", "silent")]}
        result = confirmed_review(json.loads(args[3]))
        if fault == "decisions":
            result["decisions"] = []
        else:
            result["decisions"][0][fault] = "invented evidence"
        return result

    monkeypatch.setattr(ns, "chat_json", chat)
    with pytest.raises(RuntimeError, match="Review verification failed after 3 attempts"):
        ns.track_scene(None, "mock", source, source, attempts=2)
    assert calls == [ns.EXTRACTION_SCHEMA, ns.REVIEW_SCHEMA] + [ns.VERIFY_REVIEW_SCHEMA] * 3


@pytest.mark.parametrize("fault", ["duplicate", "undeclared", "chronology", "carry"])
def test_compiler_rejects_invalid_extractions(fault):
    source = "Indy calls Doriane. Indy waits."
    response = extraction(contract(opening()), source)
    current = None
    if fault == "duplicate":
        response["new_entities"] = [{"id": "indy", "kind": "character"}]
    elif fault == "undeclared":
        response["events"][0]["changes"] = [{"entity": "missing", "field": "visible", "after": True}]
    elif fault == "chronology":
        response["events"] += [deepcopy(response["events"][0])]
        response["events"][0]["source_ids"] = ["source_2"]
    else:
        current = opening()
    with pytest.raises(ValueError):
        ns.compile_contract(response, source, current)


def test_unintroduced_physical_state_and_ownership_cycles_are_errors():
    state = opening()
    ns.state_map(state)["torch_2"]["relationship"] = "in_mouth"
    assert ns.validate_contract(contract(state))
    cyclic = {"entities": [entity("a", owner="b", location="b"), entity("b", owner="a", location="a")]}
    assert any(e["expected_state"] == "acyclic ownership" for e in ns.validate_contract(contract(cyclic)))


@pytest.mark.parametrize("use_bundle", [False, True])
def test_continuity_node_preserves_prose_and_propagates_state(tmp_path, monkeypatch, use_bundle):
    beats = prologue()
    source = "\n\n".join(b["events"][0]["description"] for b in beats)
    chapter = tmp_path / "chapter.txt"
    monkeypatch.setattr(nodes.util, "discover_inputs", lambda *a: [chapter])
    monkeypatch.setattr(nodes.util, "read_chapter", lambda *a: source)
    monkeypatch.setattr(nodes, "stage_output", lambda *a: tmp_path / "output")
    monkeypatch.setattr(nodes.lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock"))
    texts = [b["events"][0]["description"] for b in beats]
    monkeypatch.setattr(nodes.util, "split_chunks", lambda text, size, overlap: texts if size == 6000 else [text])
    current_states = []
    def chat(*args):
        assert args[4] != ns.SIMPLIFY_SCHEMA
        payload = json.loads(args[3])
        if args[4] == ns.REVIEW_SCHEMA:
            return {"errors": []}
        current_states.append(deepcopy(payload["current_state"]))
        beat = beats[texts.index(payload["original_scene"])]
        return extraction(beat, payload["original_scene"], payload["current_state"])
    monkeypatch.setattr(ns, "chat_json", chat)
    bundle = {"schema_version": "minimax-cinematic-narrative.v1", "chapters": {str(chapter.resolve()): {
        "source_digest": ns.source_digest(source), "segments": [{"original_text": t, "cinematic_text": t} for t in texts]}}}
    result = nodes.NarrativeContinuityNode().run(str(chapter), {"api_url": "unused"}, cinematic_narrative=bundle if use_bundle else None)
    assert result[1] == source
    assert current_states == [None] + [b["state_after"] for b in beats[:-1]]
    assert json.loads((tmp_path / "output/cinematic_narrative.json").read_text(encoding="utf-8")) == result[0]
    if use_bundle:
        bundle["chapters"][str(chapter.resolve())]["segments"].pop()
        with pytest.raises(ValueError, match="cover the original chapter"):
            nodes.NarrativeContinuityNode().run(str(chapter), {"api_url": "unused"}, cinematic_narrative=bundle)


def test_evidence_preserves_french_punctuation_and_unicode():
    source = "Indy se baissa pour éviter la torche. Il s’en empara après avoir attrapé le filin."
    response = {"opening_entities": [], "new_entities": [], "events": [
        {"id": "duck", "description": "Indy ducks.", "source_ids": ["source_1"], "kind": "action", "changes": []},
        {"id": "catch", "description": "Indy catches the line and then the torch.", "source_ids": ["source_2"], "kind": "action", "changes": []}]}
    result = ns.compile_contract(response, source)
    assert result["events"][1]["source_evidence"] == "Il s’en empara après avoir attrapé le filin."
    assert not ns.validate_contract(result, source=source)

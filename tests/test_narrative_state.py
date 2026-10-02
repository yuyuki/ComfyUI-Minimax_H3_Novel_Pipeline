"""Synthetic narrative tests; test_prologue_source.py covers the actual French source."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline import narrative_state as ns
from minimax_h3_novel_pipeline import pipeline_step3_generate as generate
from minimax_h3_novel_pipeline.narrative_nodes import NovelCinematicSimplifierNode
from minimax_h3_novel_pipeline.narrative_state import source_digest


def entity(id, kind="object", **fields):
    return {"id": id, "kind": kind, **dict.fromkeys(ns.FIELDS), "visible": False, **fields}


def opening():
    return {"entities": [
        entity("indy", "character", location="crevasse", position="suspended_on_rope", posture="hanging", visible=True),
        entity("torch_1", status="lost_below"), entity("torch_2", status="not_introduced"),
        entity("main_rope", status="intact", visible=True, location="crevasse"),
        entity("holder", location="crevasse", visible=True),
        entity("crevasse", "location", environment="dark", visible=True),
    ]}


def contract(state, description="Indy calls Doriane.", changes=(), kind="action"):
    before = deepcopy(state)
    after = deepcopy(state)
    index = ns.state_map(after)
    edits = []
    for key, field, value in changes:
        edits.append({"entity": key, "field": field, "before": index[key][field], "after": value})
        index[key][field] = value
    return {"state_before": before, "initial_frame": {"entities": [deepcopy(e) for e in before["entities"] if e["visible"]]},
            "events": [{"id": "event_1", "description": description, "source_evidence": description, "kind": kind, "changes": edits}],
            "state_after": after}


def extraction(beat, source, current=None):
    """Mock the event-only LM response using independently authored contract fixtures."""
    events = []
    units = ns.source_units(source)
    for event in beat["events"]:
        start = source.index(event["source_evidence"])
        end = start + len(event["source_evidence"])
        ids = [u["id"] for u in units if u["start"] < end and u["end"] > start]
        events.append({"id": event["id"], "description": event["description"], "kind": event["kind"],
                       "source_ids": ids, "changes": [{k: v for k, v in c.items() if k != "before"}
                                                      for c in event["changes"]]})
    return {"opening_entities": deepcopy(beat["state_before"]["entities"]) if current is None else [],
            "new_entities": [], "events": events}


def prologue():
    # These beats reproduce the user's supplied sequence; they are not a quotation of the full prologue.
    specs = [
        ("Indy asks Doriane for another torch.", [], "action"),
        ("A second torch descends.", [("torch_2", "status", "lit"), ("torch_2", "visible", True), ("torch_2", "location", "crevasse")], "introduction"),
        ("Indy catches it.", [("torch_2", "owner", "indy"), ("torch_2", "location", "indy"), ("torch_2", "relationship", "held")], "action"),
        ("He places it in a holder.", [("torch_2", "owner", "holder"), ("torch_2", "location", "holder"), ("torch_2", "relationship", "stored")], "action"),
        ("He retrieves it again.", [("torch_2", "owner", "indy"), ("torch_2", "location", "indy"), ("torch_2", "relationship", "held")], "action"),
        ("The main rope begins to fail.", [("main_rope", "status", "damaged")], "action"),
        ("He puts the torch between his teeth.", [("torch_2", "relationship", "in_mouth")], "action"),
        ("The rope breaks and Indy falls.", [("main_rope", "status", "broken"), ("indy", "position", "falling"), ("indy", "posture", "falling")], "action"),
    ]
    state = opening()
    result = []
    for text, edits, kind in specs:
        beat = contract(state, text, edits, kind)
        result.append(beat)
        state = beat["state_after"]
    return result


def test_prologue_state_propagation_and_no_premature_torch_or_rope_break():
    current = None
    for i, beat in enumerate(prologue()):
        assert ns.validate_contract(beat, current, beat["events"][0]["description"]) == []
        state = ns.state_map(beat["state_before"])
        if i <= 1:
            assert state["torch_2"]["status"] == "not_introduced"
        if i <= 6:
            assert state["torch_2"]["relationship"] != "in_mouth"
        assert state["main_rope"]["status"] != "broken"
        current = beat["state_after"]
    assert ns.state_map(current)["indy"]["position"] == "falling"


@pytest.mark.parametrize("key,field,value", [
    ("indy", "location", "surface"), ("indy", "position", "standing"),
    ("indy", "posture", "standing"), ("torch_2", "owner", "indy"),
    ("main_rope", "status", "broken"),
])
def test_no_silent_state_change(key, field, value):
    beat = contract(opening())
    ns.state_map(beat["state_after"])[key][field] = value
    assert any(e["entity"] == key for e in ns.validate_contract(beat))


def test_torch_in_mouth_initial_frame_leak_reports_introducing_event():
    beat = prologue()[6]
    ns.state_map(beat["initial_frame"])["torch_2"]["relationship"] = "in_mouth"
    errors = ns.validate_contract(beat)
    assert errors[0]["entity"] == "torch_2"
    assert errors[0]["introducing_event"] == "event_1"
    assert "held" in errors[0]["expected_state"]
    assert "in_mouth" in errors[0]["conflicting_state"]


def test_unintroduced_object_cannot_be_visible_or_owned():
    beat = contract(opening(), changes=[("torch_2", "visible", True), ("torch_2", "owner", "indy")])
    assert ns.validate_contract(beat)


def test_object_introduction_requires_introduction_event():
    beat = prologue()[1]
    beat["events"][0]["kind"] = "action"
    assert any("introduction" in e["expected_state"] for e in ns.validate_contract(beat))


def test_one_location_and_existing_owner():
    beat = prologue()[2]
    ns.state_map(beat["state_after"])["torch_2"]["location"] = ["indy", "holder"]
    with pytest.raises(ValueError, match="expected"):
        ns.validate_contract(beat)
    beat = contract(opening(), changes=[("torch_1", "owner", "nobody"), ("torch_1", "location", "nobody")])
    assert any("existing owner" in e["expected_state"] for e in ns.validate_contract(beat))


def test_owned_opening_prop_location_is_compiled_without_correction(monkeypatch):
    source = "Doriane holds a torch above the crevasse opening."
    state = {"entities": [
        entity("doriane", "character", location="above crevasse opening"),
        entity("torch_2", owner="doriane", location="above crevasse opening", position="held by doriane"),
    ]}
    response = extraction(contract(state, source), source)
    original_response = deepcopy(response)
    calls = []

    def chat(*args):
        calls.append(args[4])
        return response if args[4] == ns.EXTRACTION_SCHEMA else {"errors": []}

    monkeypatch.setattr(ns, "chat_json", chat)
    result, report = ns.track_scene(None, "mock", source, source, attempts=0)
    torch = ns.state_map(result["state_before"])["torch_2"]
    assert torch["location"] == "doriane"
    assert torch["position"] == "held by doriane; above crevasse opening"
    assert torch["owner"] == "doriane" and torch["visible"] is False
    assert result["state_after"] == result["state_before"]
    assert response == original_response
    assert report["valid"] and ns.validate_contract(result, source=source) == []
    assert calls == [ns.EXTRACTION_SCHEMA, ns.REVIEW_SCHEMA]


@pytest.mark.parametrize("fault", ["missing_owner", "unintroduced_owner", "unintroduced_prop", "lost_prop", "cycle", "carried_state"])
def test_opening_location_compilation_preserves_ownership_constraints(fault):
    source = "Doriane calls out."
    owner = entity("doriane", "character")
    prop = entity("torch_2", owner="doriane", location="above crevasse opening")
    state = {"entities": [owner, prop]}
    if fault == "missing_owner":
        prop["owner"] = "missing"
    elif fault == "unintroduced_owner":
        owner["status"] = "not_introduced"
    elif fault == "unintroduced_prop":
        prop["status"] = "not_introduced"
    elif fault == "lost_prop":
        prop["status"] = "lost_below"
    elif fault == "cycle":
        owner.update(owner="torch_2", location="torch_2")
    current = state if fault == "carried_state" else None
    response = extraction(contract(state, source), source, current)
    compiled = ns.compile_contract(response, source, current)
    assert ns.validate_contract(compiled, current, source)
    if current is not None:
        assert compiled["state_before"] == current


@pytest.mark.parametrize("status", ["broken", "damaged"])
def test_restoration_needs_explicit_repair(status):
    state = opening()
    ns.state_map(state)["main_rope"]["status"] = status
    beat = contract(state, "Doriane repairs the rope.", [("main_rope", "status", "intact")])
    assert ns.validate_contract(beat)
    beat["events"][0]["kind"] = "repair"
    assert not ns.validate_contract(beat)


def test_previous_state_cannot_be_rewritten_and_new_entities_start_absent():
    previous = opening()
    state = deepcopy(previous)
    ns.state_map(state)["indy"]["location"] = "surface"
    state["entities"].append(entity("new_prop", visible=True))
    errors = ns.validate_contract(contract(state), previous)
    assert {e["entity"] for e in errors} == {"indy", "new_prop"}


def test_wrong_preconditions_duplicate_ids_and_invented_evidence():
    beat = prologue()[6]
    beat["events"][0]["changes"][0]["before"] = "in_mouth"
    assert ns.validate_contract(beat)
    assert ns.validate_contract(prologue()[6], source="No such event.")
    beat["state_before"]["entities"].append(deepcopy(beat["state_before"]["entities"][0]))
    with pytest.raises(ValueError, match="unique"):
        ns.validate_contract(beat)


def test_indexed_evidence_and_replay_avoid_quote_and_precondition_retries(monkeypatch):
    source = "Indy se redresse. Il descend ensuite."
    first = contract(opening(), "Indy se redresse.", [("indy", "position", "raised")])
    second = contract(first["state_after"], "Il descend ensuite.", [("indy", "position", "lowered")])
    second["events"][0]["id"] = "event_2"
    fixed = {**first, "events": first["events"] + second["events"], "state_after": second["state_after"]}
    response = extraction(fixed, source)
    calls = []
    def chat(*args):
        calls.append(args[4])
        return deepcopy(response) if args[4] == ns.EXTRACTION_SCHEMA else {"errors": []}
    monkeypatch.setattr(ns, "chat_json", chat)
    result, report = ns.track_scene(None, "mock", source, "Indy rises. Then descends.", attempts=0)
    assert result == fixed and report["valid"]
    assert result["events"][1]["changes"][0]["before"] == "raised"
    assert calls == [ns.EXTRACTION_SCHEMA, ns.REVIEW_SCHEMA]


@pytest.mark.parametrize("ids", [[], ["invented"], ["source_2", "source_1"]])
def test_invalid_evidence_ids_exhaust_budget(monkeypatch, ids):
    response = extraction(contract(opening()), "Indy calls Doriane.")
    response["events"][0]["source_ids"] = ids
    calls = []
    def chat(*args):
        calls.append(args[4])
        return deepcopy(response)
    monkeypatch.setattr(ns, "chat_json", chat)
    with pytest.raises(ValueError, match="source_ids"):
        ns.track_scene(None, "mock", "Indy calls Doriane.", "Indy calls Doriane.", attempts=1)
    assert calls == [ns.EXTRACTION_SCHEMA, ns.EXTRACTION_SCHEMA]


def test_metaphor_normalization_and_dialogue_prompt(monkeypatch):
    source = 'Indy se balançait, suspendu tel un croissant de lune à une corde qui lui meurtrissait le torse et les aisselles. « Doriane ! »'
    normalized = 'Indy hangs by a rope passing tightly under his arms and around his torso. He sways. « Doriane ! »'
    calls = []

    def chat(client, model, system, user, schema, *args):
        calls.append((system, json.loads(user)))
        return {"cinematic_text": normalized} if schema == ns.SIMPLIFY_SCHEMA else {"errors": []}

    monkeypatch.setattr(ns, "chat_json", chat)
    result, report = ns.simplify(None, "qwen", source)
    assert result["cinematic_text"] == normalized and report["valid"]
    assert "never summarize" in calls[0][0] and "dialogue verbatim" in calls[0][0]
    assert calls[1][1]["original_scene"] == source


@pytest.mark.parametrize("bad", [{}, {"state_before": "bad"}, None])
def test_malformed_model_json_corrected_and_revalidated(monkeypatch, bad):
    good = contract(opening())
    results = iter([bad, extraction(good, "Indy calls Doriane."), {"errors": []}])
    calls = []

    def chat(*args):
        calls.append(json.loads(args[3]))
        return next(results)

    monkeypatch.setattr(ns, "chat_json", chat)
    result, report = ns.track_scene(None, "qwen", "Indy calls Doriane.", "Indy calls Doriane.", attempts=1)
    assert result == good and len(report["attempts"]) == 2
    assert calls[1]["current_state"] is None
    assert calls[1]["validation_errors"]


def test_automatic_correction_receives_expected_state_and_is_bounded(monkeypatch):
    good = prologue()[6]
    source = good["events"][0]["description"]
    response = extraction(good, source, good["state_before"])
    bad = deepcopy(response)
    bad["events"][0]["changes"].append({"entity": "torch_2", "field": "location", "after": "surface"})
    results = iter([bad, response, {"errors": []}])
    monkeypatch.setattr(ns, "chat_json", lambda *args: next(results))
    result, report = ns.track_scene(None, "qwen", source, source, good["state_before"], 1)
    assert result == good and len(report["attempts"]) == 2
    calls = []
    monkeypatch.setattr(ns, "chat_json", lambda *args: calls.append(True) or bad)
    with pytest.raises(ValueError, match="correction budget"):
        ns.track_scene(None, "qwen", source, source, good["state_before"], attempts=1)
    assert len(calls) == 2


def test_missing_optional_state_and_null_attributes(monkeypatch):
    good = contract({"entities": [entity("indy", "character")]})
    response = extraction(good, "Indy calls Doriane.")
    monkeypatch.setattr(ns, "chat_json", lambda *a: response if a[4] == ns.EXTRACTION_SCHEMA else {"errors": []})
    assert ns.track_scene(None, "qwen", "Indy calls Doriane.", "Indy calls Doriane.")[0] == good


def test_malformed_json_runtime_error_and_interruption(monkeypatch):
    def malformed(*args):
        raise RuntimeError("Invalid structured JSON after 1 attempt(s).")
    monkeypatch.setattr(ns, "chat_json", malformed)
    with pytest.raises(ValueError, match="correction budget"):
        ns.simplify(None, "qwen", "source", 0)
    def interrupted():
        raise RuntimeError("interrupted")
    monkeypatch.setattr(ns, "comfy_interrupt_check", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        ns.simplify(None, "qwen", "source", 2)


def test_semantic_review_can_reject_well_formed_normalization(monkeypatch):
    def chat(*args):
        if args[4] == ns.VERIFY_REVIEW_SCHEMA:
            return confirmed_review(json.loads(args[3]))
        return ({"cinematic_text": "Lost all actions"} if args[4] == ns.SIMPLIFY_SCHEMA
                else {"errors": [ns.issue("indy", "all actions", "omitted actions")]})
    monkeypatch.setattr(ns, "chat_json", chat)
    with pytest.raises(ValueError, match="omitted actions"):
        ns.simplify(None, "qwen", "source", 0)


@pytest.mark.parametrize("stage", ["simplify", "track_scene"])
def test_correction_feedback_is_kept_out_of_fresh_reviews(monkeypatch, stage):
    source = "Indy calls Doriane."
    good = {"cinematic_text": source} if stage == "simplify" else contract(opening(), source)
    first_error = ns.issue("Doriane", "Doriane shouts", "wrong attribution")
    second_error = ns.issue("torch_1", "lost below", "held")
    feedback = [first_error, second_error]
    generations, reviews = [], []

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema == ns.VERIFY_REVIEW_SCHEMA:
            return confirmed_review(payload)
        if schema == ns.REVIEW_SCHEMA:
            reviews.append(payload)
            # A reviewer exposed to earlier complaints repeats them even after correction.
            if "validation_errors" in payload:
                return {"errors": payload["validation_errors"]}
            return {"errors": [feedback[len(reviews) - 1]] if len(reviews) <= 2 else []}
        generations.append((system, payload))
        return deepcopy(good) if stage == "simplify" else extraction(good, source, opening())

    monkeypatch.setattr(ns, "chat_json", chat)
    if stage == "simplify":
        result, report = ns.simplify(None, "qwen", source, attempts=2)
        expected_source = {"original_scene": source}
    else:
        state = opening()
        result, report = ns.track_scene(None, "qwen", source, source, state, attempts=2,
                                        authoritative_contract=good, expected_after=good["state_after"])
        expected_source = {"original_scene": source, "cinematic_version": source, "current_state": state,
                           "authoritative_contract": good, "expected_after": good["state_after"], "source_units": ns.source_units(source)}
    assert result == good and report["valid"]
    assert [entry["errors"] for entry in report["attempts"]] == [[first_error], [second_error], []]
    assert len(generations) == len(reviews) == 3
    for review_payload in reviews:
        assert review_payload == {**expected_source, "candidate": good}
    assert generations[0][1] == expected_source
    for (system, payload), error in zip(generations[1:], feedback):
        assert "Revise the supplied candidate" in system
        assert payload == {**expected_source, "candidate": good if stage == "simplify" else extraction(good, source, state), "validation_errors": [error]}


def test_repeated_semantic_conflicts_still_exhaust_correction_budget(monkeypatch):
    conflict = ns.issue("torch_2", "not introduced until descent", "already in mouth")
    reviews = []

    def chat(client, model, system, user, schema, *args):
        if schema == ns.VERIFY_REVIEW_SCHEMA:
            return confirmed_review(json.loads(user))
        if schema == ns.REVIEW_SCHEMA:
            reviews.append(json.loads(user))
            return {"errors": [conflict]}
        return {"cinematic_text": "Indy already holds the second torch in his mouth."}

    monkeypatch.setattr(ns, "chat_json", chat)
    with pytest.raises(ValueError, match="correction budget"):
        ns.simplify(None, "qwen", "A second torch descends. Indy catches it.", attempts=2)
    assert len(reviews) == 3
    assert all("validation_errors" not in payload for payload in reviews)


def confirmed_review(payload):
    return {"decisions": [{"supported": True, "reason": "Source action differs from candidate.",
                           "source_ids": [unit["id"] for unit in ns.source_units(payload["original_scene"])],
                           "candidate_evidence": payload["candidate"].get(
                               "cinematic_text", json.dumps(payload["candidate"], ensure_ascii=False))}
                          for _ in payload["proposed_errors"]]}


@pytest.mark.parametrize("ids", [[], ["source_999"], ["source_1", "source_1"]])
def test_review_rejects_missing_unknown_or_duplicate_source_ids(ids):
    payload = {"original_scene": "Il crie : « À l’aide ! »", "candidate": {"cinematic_text": "Il attend."},
               "proposed_errors": [ns.issue("speaker", "calls", "silent")]}
    response = confirmed_review(payload)
    response["decisions"][0]["source_ids"] = ids
    with pytest.raises(ValueError, match="source_ids"):
        ns._validate_review_decisions(response, payload, payload["proposed_errors"])


def test_review_indexes_exact_source_without_asking_model_to_copy_quotes(monkeypatch):
    source = 'Il crie : « À l’aide ! »\nPuis il saisit la corde…'
    errors = [ns.issue("speaker", "calls", "silent")]
    payload = {"original_scene": source, "candidate": {"cinematic_text": "Il attend."}}

    def chat(client, model, system, user, schema, *args):
        request = json.loads(user)
        assert "source_evidence" not in schema["schema"]["properties"]["decisions"]["items"]["properties"]
        for unit in request["source_units"]:
            assert unit["text"] == source[unit["start"]:unit["end"]]
        return confirmed_review(request)

    monkeypatch.setattr(ns, "chat_json", chat)
    assert ns.verify_cinematic_review(None, "mock", payload, errors) == errors
    assert "source_units" not in payload


def test_false_review_complaints_do_not_rewrite_candidate(monkeypatch):
    source = '"Jones!" cried Doriane. Indy catches the cord, then grabs the torch. He closes his eyes. He pulls himself up.'
    candidate = '"Jones!" Doriane cries out. Indy grabs the torch after catching the cord. He closes his eyes and pulls himself up.'
    complaints = [ns.issue("Doriane", "Doriane speaks", "speaker reversed"),
                  ns.issue("Indy", "catch before grab", "actions compressed"),
                  ns.issue("Indy", "closes eyes", "missing eye closing")]
    calls = []

    def chat(client, model, system, user, schema, *args):
        calls.append(schema)
        if schema == ns.SIMPLIFY_SCHEMA:
            return {"cinematic_text": candidate}
        if schema == ns.REVIEW_SCHEMA:
            return {"errors": complaints}
        payload = json.loads(user)
        assert payload == {"original_scene": source, "candidate": {"cinematic_text": candidate},
                           "proposed_errors": complaints, "source_units": ns.source_units(source)}
        return {"decisions": [{"supported": False, "reason": reason,
                               "source_ids": [], "candidate_evidence": candidate}
                              for reason in ["Same speaker", "Same order", "Both actions present"]]}

    monkeypatch.setattr(ns, "chat_json", chat)
    result, report = ns.simplify(None, "qwen", source, attempts=0)
    assert result["cinematic_text"] == candidate and report["valid"]
    assert calls == [ns.SIMPLIFY_SCHEMA, ns.REVIEW_SCHEMA, ns.VERIFY_REVIEW_SCHEMA]


@pytest.mark.parametrize("fault", ["missing_decision", "empty_reason", "source_ids", "candidate_evidence"])
def test_invalid_review_verification_does_not_accept_candidate(monkeypatch, fault):
    payload = {"original_scene": "Indy catches the torch.", "candidate": {"cinematic_text": "Indy drops the torch."},
               "proposed_errors": [ns.issue("Indy", "catches", "drops")]}
    response = confirmed_review(payload)
    if fault == "missing_decision":
        response["decisions"] = []
    elif fault == "empty_reason":
        response["decisions"][0]["reason"] = ""
    else:
        response["decisions"][0][fault] = "invented quote"
    monkeypatch.setattr(ns, "chat_json", lambda *args: response)
    with pytest.raises(RuntimeError, match="Review verification failed after 3 attempts"):
        ns.verify_cinematic_review(None, "qwen", payload, payload["proposed_errors"])


@pytest.mark.parametrize("fault", ["missing_decision", "source_ids", "candidate_evidence", "shape", "json"])
@pytest.mark.parametrize("supported", [False, True])
def test_verification_repair_preserves_candidate_and_correction_budget(monkeypatch, fault, supported):
    source = "Indy catches the torch."
    candidate = {"cinematic_text": "Indy drops the torch."}
    conflict = ns.issue("Indy", "catches", "drops")
    generations, verifications = [], []
    invalid = None

    def chat(client, model, system, user, schema, *args):
        nonlocal invalid
        payload = json.loads(user)
        if schema == ns.SIMPLIFY_SCHEMA:
            generations.append(payload)
            return candidate if len(generations) == 1 else {"cinematic_text": source}
        if schema == ns.REVIEW_SCHEMA:
            return {"errors": [conflict] if len(generations) == 1 else []}
        verifications.append(payload)
        response = confirmed_review(payload)
        if len(verifications) == 1:
            if fault == "missing_decision":
                response["decisions"] = []
            elif fault == "shape":
                response = {}
            elif fault == "json":
                raise RuntimeError("Invalid structured JSON after 2 attempt(s).")
            else:
                response["decisions"][0][fault] = "invented quote"
            invalid = deepcopy(response)
        else:
            response["decisions"][0]["supported"] = supported
            assert "Repair the previous_verification" in system
        return response

    monkeypatch.setattr(ns, "chat_json", chat)
    result, report = ns.simplify(None, "qwen", source, attempts=int(supported))
    assert result == ({"cinematic_text": source} if supported else candidate)
    assert report["valid"]
    assert len(generations) == 1 + int(supported)
    assert len(verifications) == 2
    assert verifications[1]["candidate"] == verifications[0]["candidate"] == candidate
    assert verifications[1]["proposed_errors"] == [conflict]
    assert verifications[1]["previous_verification"] == invalid
    assert verifications[1]["verification_error"]
    if supported:
        assert generations[1]["validation_errors"] == [conflict]


def test_verification_exhaustion_does_not_rewrite_candidate(monkeypatch):
    calls = []

    def chat(*args):
        calls.append(args[4])
        if args[4] == ns.SIMPLIFY_SCHEMA:
            return {"cinematic_text": "Indy drops the torch."}
        if args[4] == ns.REVIEW_SCHEMA:
            return {"errors": [ns.issue("Indy", "catches", "drops")]}
        return {"decisions": []}

    monkeypatch.setattr(ns, "chat_json", chat)
    with pytest.raises(RuntimeError, match="Review verification failed after 3 attempts"):
        ns.simplify(None, "qwen", "Indy catches the torch.", attempts=2)
    assert calls == [ns.SIMPLIFY_SCHEMA, ns.REVIEW_SCHEMA] + [ns.VERIFY_REVIEW_SCHEMA] * 3


def test_verification_retry_checks_cancellation(monkeypatch):
    calls = []

    def chat(*args):
        calls.append(True)
        return {"decisions": []}

    def interrupt():
        if calls:
            raise RuntimeError("interrupted")

    monkeypatch.setattr(ns, "chat_json", chat)
    monkeypatch.setattr(ns, "comfy_interrupt_check", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        ns.verify_cinematic_review(None, "qwen", {}, [ns.issue("Indy", "catches", "drops")])
    assert len(calls) == 1


def test_verification_does_not_retry_unrelated_runtime_errors(monkeypatch):
    calls = []

    def chat(*args):
        calls.append(True)
        raise RuntimeError("connection failed")

    monkeypatch.setattr(ns, "chat_json", chat)
    with pytest.raises(RuntimeError, match="connection failed"):
        ns.verify_cinematic_review(None, "qwen", {}, [])
    assert len(calls) == 1


def test_review_verification_preserves_only_confirmed_errors_and_cancellation(monkeypatch):
    payload = {"original_scene": "Indy catches the torch.", "candidate": {"cinematic_text": "Indy drops the torch."},
               "proposed_errors": [ns.issue("style", "present", "past"), ns.issue("Indy", "catches", "drops")]}
    response = confirmed_review(payload)
    response["decisions"][0]["supported"] = False
    monkeypatch.setattr(ns, "chat_json", lambda *args: response)
    assert ns.verify_cinematic_review(None, "qwen", payload, payload["proposed_errors"]) == payload["proposed_errors"][1:]
    def interrupted():
        raise RuntimeError("interrupted")
    monkeypatch.setattr(ns, "comfy_interrupt_check", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        ns.verify_cinematic_review(None, "qwen", payload, payload["proposed_errors"])


def test_generation_uses_contract_after_camera_and_on_cache_hits(tmp_path, monkeypatch):
    from minimax_h3_novel_pipeline import path_access
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    beat = prologue()[6]
    source = beat["events"][0]["description"] + " Indy is still suspended in the dark crevasse. The damaged main rope supports him."
    chapter = tmp_path / "chapter.txt"
    chapter.write_text(source, encoding="utf-8")
    bundle = {"schema_version": "minimax-cinematic-narrative.v1", "correction_attempts": 1, "chapters": {
        str(chapter.resolve()): {"source_digest": source_digest(source), "cinematic_text": source,
            "segments": [{"original_text": source, "cinematic_text": source, "contract": beat}]}}}
    args = SimpleNamespace(out_dir=tmp_path / "output", force=False, chunk_chars=3000, overlap_paragraphs=2,
                           max_scenes=0, duration=8, max_shots=1, repair_attempts=1, refine_camera=True,
                           cinematic_narrative=bundle)
    scene = generate.Scene("Torch", source, source, "", [], [], [], False, "")
    planning = []
    def plan(*a):
        planning.append(a[3])
        return [scene]
    monkeypatch.setattr(generate, "plan_scenes", plan)
    monkeypatch.setattr(ns, "review", lambda *a: [])
    monkeypatch.setattr(ns, "track_scene", lambda *a: (deepcopy(beat), {"valid": True}))
    bindings = {"subjects": [{"h3_subject_label": "<Subject 1>", "canonical_name": "Indy", "pictures": [],
                              "entity_type": "character", "global_id": "indy"}],
                "picture_input_order": [], "audio": []}
    monkeypatch.setattr(generate, "build_bindings", lambda *a: bindings)
    def prompt(*a):
        assert a[-1]["narrative_state"]["state_before"] == beat["state_before"]
        return "draft"
    monkeypatch.setattr(generate, "generate_prompt", prompt)
    monkeypatch.setattr(generate, "validate_prompt", lambda *a: generate.Validation(True, [], 100))
    monkeypatch.setattr(generate, "refine_camera_prompt", lambda *a: ("premature torch in mouth", []))
    reviews = []
    def review(*a):
        reviews.append(a[3])
        return [] if a[3] == "corrected" else ["torch_2: expected held; in_mouth starts at event_1; correct opening."]
    monkeypatch.setattr(generate, "check_prompt_continuity", review)
    monkeypatch.setattr(generate, "repair_prompt", lambda *a: "corrected")
    for _ in range(2):
        manifest = generate.process_chapter(chapter, {}, None, "qwen", args)
        entry = manifest["outputs"][0]
        assert entry["valid"] and entry["narrative_state"]["state_after"] == beat["state_after"]
    assert reviews == ["premature torch in mouth", "corrected", "corrected"]
    assert len(planning) == 1 and "AUTHORITATIVE TEMPORAL CONTRACT" in planning[0]
    monkeypatch.setattr(generate, "build_bindings", lambda *a: {"subjects": [], "audio": []})
    with pytest.raises(ValueError, match="Cannot skip ordered narrative events"):
        generate.process_chapter(chapter, {}, None, "qwen", args)
    chapter.write_text("changed " * 20, encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        generate.process_chapter(chapter, {}, None, "qwen", args)


def test_new_nodes_are_optional_and_preview_outputs_are_strings():
    from minimax_h3_novel_pipeline.generate_h3_prompts import GenerateH3PromptsNode
    assert "cinematic_narrative" in GenerateH3PromptsNode.INPUT_TYPES()["optional"]
    assert list(GenerateH3PromptsNode.INPUT_TYPES()["optional"])[:2] == ["spatial_continuity", "camera_direction"]
    assert GenerateH3PromptsNode.RETURN_TYPES == ("MINIMAX_PROMPTS", "STRING", "STRING")
    assert all(t == "STRING" for t in NovelCinematicSimplifierNode.RETURN_TYPES[1:])


def test_scene_boundary_cannot_drop_actions(monkeypatch):
    beat = prologue()[6]
    monkeypatch.setattr(ns, "chat_json", lambda *a: extraction(beat, beat["events"][0]["description"], beat["state_before"]))
    with pytest.raises(ValueError, match="scene boundary"):
        ns.track_scene(None, "qwen", beat["events"][0]["description"], "Torch moves.",
                       beat["state_before"], 0, expected_after=prologue()[-1]["state_after"])


def test_out_of_order_events_rejected():
    first, second = prologue()[:2]
    beat = deepcopy(first)
    beat["events"] = [deepcopy(second["events"][0]), deepcopy(first["events"][0])]
    beat["events"][1]["id"] = "ask"
    beat["state_after"] = second["state_after"]
    source = first["events"][0]["description"] + " " + second["events"][0]["description"]
    assert any("chronological" in e["expected_state"] for e in ns.validate_contract(beat, source=source))


def test_preprocessing_node_persists_prologue_and_passes_state_between_passages(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from minimax_h3_novel_pipeline import narrative_nodes as nodes, path_access
    beats = prologue()
    sources = [beat["events"][0]["description"] for beat in beats]
    chapter = tmp_path / "prologue_sequence.txt"
    chapter.write_text("\n\n".join(sources), encoding="utf-8")
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(nodes, "stage_output", lambda *a: tmp_path / "output")
    monkeypatch.setattr(nodes.lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock-qwen"))
    monkeypatch.setattr(nodes.util, "split_chunks", lambda *a: sources)
    tracked = []

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema == ns.REVIEW_SCHEMA:
            return {"errors": []}
        index = sources.index(payload["original_scene"])
        if schema == ns.SIMPLIFY_SCHEMA:
            return {"cinematic_text": sources[index]}
        tracked.append(payload["current_state"])
        return extraction(beats[index], payload["original_scene"], payload["current_state"])

    monkeypatch.setattr(ns, "chat_json", chat)
    outputs = NovelCinematicSimplifierNode().run(str(chapter), {"api_url": "unused"})
    assert len(outputs) == 6
    bundle = outputs[0]
    saved = json.loads((tmp_path / "output/cinematic_narrative.json").read_text(encoding="utf-8"))
    assert saved == bundle
    assert tracked == [None] + [b["state_after"] for b in beats[:-1]]
    record = bundle["chapters"][str(chapter.resolve())]
    assert len(record["segments"]) == 8
    assert ns.state_map(record["segments"][-1]["contract"]["state_after"])["main_rope"]["status"] == "broken"
    for preview in outputs[2:]:
        assert isinstance(json.loads(preview), dict)

    # Run the persisted preprocessing result through all eight generated scenes.
    def plan(client, model, chapter_id, chunk, index, *args):
        source = sources[index - 1]
        return [generate.Scene(f"Beat {index}", source, source, "", [], [], [], False, "")]
    monkeypatch.setattr(generate, "plan_scenes", plan)
    bindings = {"subjects": [{"h3_subject_label": "<Subject 1>", "canonical_name": "Indy",
                              "entity_type": "character", "global_id": "indy", "pictures": []}],
                "picture_input_order": [], "audio": []}
    monkeypatch.setattr(generate, "build_bindings", lambda *a: bindings)
    openings = []
    def prompt(*a):
        state = a[-1]["narrative_state"]["state_before"]
        openings.append(state)
        return "mock H3 prompt"
    monkeypatch.setattr(generate, "generate_prompt", prompt)
    monkeypatch.setattr(generate, "validate_prompt", lambda *a: generate.Validation(True, [], 100))
    monkeypatch.setattr(generate, "check_prompt_continuity", lambda *a: [])
    args = SimpleNamespace(out_dir=tmp_path / "generated", force=False, chunk_chars=3000, overlap_paragraphs=2,
                           max_scenes=0, duration=8, repair_attempts=1, cinematic_narrative=bundle)
    manifest = generate.process_chapter(chapter, {}, None, "mock-qwen", args)
    assert manifest["saved_prompt_count"] == 8
    assert openings == [beat["state_before"] for beat in beats]
    assert all(entry["valid"] for entry in manifest["outputs"])

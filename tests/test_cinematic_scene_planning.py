"""Real planning-stage contracts with mocked transport, splitting and resume."""
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca, cinematic_scene_planning as planning
from minimax_h3_novel_pipeline.core import narrative_validation as nv


def scene(source, events=None):
    return {"source": source, "initial_state": "Avant.", "events": events or ["Action."], "final_state": "Après."}


def plan(item, seconds=2, updates=None, reset=False, reset_evidence=""):
    count = len(item["events"])
    return {"timing": {"beats": [{"events": [i], "seconds": seconds / count, "reason": "Action naturelle."}
                                  for i in range(1, count + 1)],
                       "pause_seconds": 0, "estimated_seconds": seconds, "reason": "Actions successives."},
            "continuity": {"reset": reset, "reset_evidence": reset_evidence, "updates": updates or []}}


def fact(quote, value="La tablette reste enchâssée.", entity="tablette", aspect="support"):
    return {"entity": entity, "aspect": aspect, "fact": value, "source_quote": quote}


def run(source, **kwargs):
    return ca.adapt_chapter(None, "mock", source, chunk_chars=4000, temperature=0.15,
                            max_tokens=16384, correction_attempts=kwargs.pop("correction_attempts", 1), **kwargs)


def test_one_sentence_is_split_into_three_reviewed_scenes_and_resumes_without_replanning(monkeypatch):
    spans = ["Il range son carnet et sort un filet", " dont il recouvre la tablette", ", avant de nouer les lacets."]
    source = "".join(spans)
    calls, snapshots, reviews = [], [], []

    def chat(client, model, system, user, schema, temperature, tokens):
        payload = json.loads(user)
        calls.append(schema["name"])
        assert schema["strict"] and tokens == 16384
        assert ca.lmstudio_json.REQUEST_OUTPUT_LIMIT.get() == 16384
        if schema == planning.SCHEMA:
            item = payload["current_scene"]
            assert payload["maximum_seconds"] == 8
            return plan(item, 14 if item["source"] == source else 4)
        if schema == nv.SCHEMA:
            reviews.append(payload)
            # Even an accepting semantic review cannot override an oversized estimate.
            return {"issues": []}
        if "rejected_adaptation" in payload:
            assert "14s exceeds" in payload["defects_to_correct"][0]
            assert "inside a single sentence" in system
            return {"sequences": [scene(span) for span in spans]}
        return {"sequences": [scene(source, ["Ranger et sortir.", "Recouvrir.", "Nouer."])]}

    def checkpoint(state):
        snapshots.append(deepcopy(state))
        if state["stage"] == "scene_planned" and len(state["sequences"]) == 1:
            raise KeyboardInterrupt()

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(KeyboardInterrupt):
        run(source, on_checkpoint=checkpoint)
    saved = snapshots[-1]
    assert saved["sequences"][0]["source"] == spans[0]
    assert saved["pending_sequences"][0]["source"] == spans[1]
    assert len(saved["scene_plans"]) == 1
    before = calls.count(planning.SCHEMA["name"])
    result = run(source, resume_state=saved)
    assert calls.count(planning.SCHEMA["name"]) == before + 1  # only the third scene needs planning
    assert [s["source"] for s in result["sequences"]] == spans
    assert "".join(s["source"] for s in result["sequences"]) == source
    assert len(reviews) == 4
    assert all(r["scene_plan"]["timing"]["estimated_seconds"] <= 8 for r in reviews[1:])
    assert result["narrative_review"] == {"issues": []}


def test_context_survives_omitted_generated_states_and_is_rebuilt_on_resume(monkeypatch):
    spans = ["La tablette est enchâssée.", " Il observe.", " Il attache le filet."]
    snapshots, evidence_seen = [], []

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema == planning.SCHEMA:
            item = payload["current_scene"]
            evidence_seen.append(deepcopy(payload["source_evidence"]))
            return plan(item, updates=[fact(spans[0])] if item["source"] == spans[0] else [])
        if schema == nv.SCHEMA:
            if payload["sequences"][0]["source"] == spans[2]:
                assert payload["source_evidence"] == [{**fact(spans[0]), "sequence": 1}]
                assert payload["previous_sequence"]["final_state"] == "Après."
            return {"issues": []}
        return {"sequences": [scene(span) for span in spans]}

    def checkpoint(state):
        snapshots.append(deepcopy(state))
        if state["stage"] == "scene_committed" and len(state["sequences"]) == 2:
            raise KeyboardInterrupt()

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(KeyboardInterrupt):
        run("".join(spans), on_checkpoint=checkpoint)
    run("".join(spans), resume_state=snapshots[-1])
    assert evidence_seen[0] == []
    assert evidence_seen[-1] == [{**fact(spans[0]), "sequence": 1}]
    corrupt = deepcopy(snapshots[-1])
    corrupt["source_evidence"][0]["fact"] = "Invented."
    with pytest.raises(ValueError, match="Saved physical context"):
        run("".join(spans), resume_state=corrupt)


@pytest.mark.parametrize("failure", ["length", "schema", "citation", "network"])
def test_planning_failure_retains_valid_scene_without_approval_and_continues(monkeypatch, failure):
    spans = ["Il attend.", " Il part."]
    snapshots = []

    def chat(client, model, system, user, schema, *args):
        payload, _ = json.JSONDecoder().raw_decode(user)
        if schema == planning.SCHEMA:
            item = payload["current_scene"]
            if item["source"] == spans[0]:
                if failure == "length":
                    raise ca.lmstudio_json.StructuredOutputLengthError("length")
                if failure == "network":
                    raise RuntimeError("unavailable")
                if failure == "schema":
                    return {"timing": {}}
                return plan(item, updates=[fact("Invented quotation.")])
            return plan(item)
        if schema == nv.SCHEMA:
            return {"issues": []}
        return {"sequences": [scene(span) for span in spans]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run("".join(spans), on_checkpoint=lambda s: snapshots.append(deepcopy(s)))
    assert result["sequences"][0]["review"]["status"] == "incomplete"
    assert "review" not in result["sequences"][1]
    assert result["narrative_review"]["warnings"][0]["check"] == "scene_planning"
    assert len(snapshots[-1]["approved_reviews"]) == 1
    assert snapshots[-1]["scene_plans"][0] is None


def test_persistent_oversize_is_never_approved_or_hidden_by_single_scene_proposals(monkeypatch):
    def chat(client, model, system, user, schema, *args):
        payload, _ = json.JSONDecoder().raw_decode(user)
        if schema == planning.SCHEMA:
            return plan(payload["current_scene"], 12)
        if schema == nv.SCHEMA:
            return {"issues": []}
        assert schema == ca.SCHEMA
        return {"sequences": [scene("Il noue les lacets.")]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run("Il noue les lacets.")
    assert result["sequences"][0]["review"]["status"] == "incomplete"
    assert "12s exceeds" in result["sequences"][0]["review"]["issues"][0]
    assert result["narrative_review"]["warnings"][-1]["reason"] == "duration_exceeded"


@pytest.mark.parametrize("defect", ["sum", "duplicate", "missing", "nan", "bool", "reset", "quote"])
def test_estimates_event_coverage_and_source_citations_are_validated(defect):
    item = scene("Il monte puis descend.", ["Il monte.", "Il descend."])
    value = plan(item, 4)
    if defect == "sum":
        value["timing"]["estimated_seconds"] = 1
    elif defect == "duplicate":
        value["timing"]["beats"][1]["events"] = [1]
    elif defect == "missing":
        value["timing"]["beats"].pop()
    elif defect in ("nan", "bool"):
        value["timing"]["estimated_seconds"] = float("nan") if defect == "nan" else True
    elif defect == "reset":
        value["continuity"]["reset"] = True
    else:
        value["continuity"]["updates"] = [fact("Unknown.")]
    with pytest.raises(ValueError):
        planning.validate(value, item)


def test_ledger_is_bounded_replaces_matching_aspects_and_resets_at_explicit_breaks():
    evidence = []
    for index in range(30):
        item = scene(f"Objet {index} fixé.")
        evidence = planning.advance(evidence, planning.validate(
            plan(item, updates=[fact(item["source"], entity=f"objet {index}")]), item), index + 1)
    assert len(evidence) == planning.MAX_FACTS
    item = scene("La tablette est dégagée.")
    evidence = planning.advance(evidence, planning.validate(plan(item, updates=[fact(item["source"])]), item), 31)
    update = plan(item, updates=[fact(item["source"], value="La tablette est libre.")])
    evidence = planning.advance(evidence, planning.validate(update, item), 32)
    assert sum(f["entity"] == "tablette" for f in evidence) == 1
    item = scene("Dix ans plus tard, à Paris.")
    reset = planning.validate(plan(item, reset=True, reset_evidence="Dix ans plus tard"), item)
    assert planning.advance(evidence, reset, 33) == []


@pytest.mark.parametrize("error", [KeyboardInterrupt(), PermissionError("storage"), OSError("filesystem"),
                                 type("InterruptProcessingException", (RuntimeError,), {})("Stop")])
def test_planning_cancellation_is_not_converted_to_a_warning(monkeypatch, error):
    def chat(client, model, system, user, schema, *args):
        if schema == planning.SCHEMA:
            raise error
        return {"sequences": [scene("Il attend.")]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(type(error)):
        run("Il attend.")


def test_retry_preserves_all_events_and_strict_schema(monkeypatch):
    item = scene("Il prend, déplie et attache le filet.", ["Prendre.", "Déplier.", "Attacher."])
    calls = []

    def chat(client, model, system, user, schema, temperature, tokens):
        payload, _ = json.JSONDecoder().raw_decode(user)
        calls.append(deepcopy(payload))
        assert schema == planning.SCHEMA and schema["strict"]
        assert tokens == 16384 and payload["current_scene"]["events"] == item["events"]
        value = plan(item, 9)
        if len(calls) == 1:
            value["timing"]["beats"].pop()
        return value

    result = planning.assess(chat, None, "mock", item, [], 8, 16384, 1)
    assert calls[0] == calls[1]
    assert result["timing"]["estimated_seconds"] == 9


def test_review_can_reject_a_low_estimate_and_require_a_split(monkeypatch):
    source = "Il déplie le filet et noue les lacets."
    spans = ["Il déplie le filet", " et noue les lacets."]
    reviewed = []

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema == planning.SCHEMA:
            return plan(payload["current_scene"], 2)
        if schema == nv.SCHEMA:
            reviewed.append(payload)
            assert "Independently assess" in system or "Check scene_plan timing independently" in system
            return {"issues": [{"target": 1, "message": "Unrealistically short; split the manipulations."}]
                    if payload["sequences"][0]["source"] == source else []}
        if "rejected_adaptation" in payload:
            return {"sequences": [scene(span) for span in spans]}
        return {"sequences": [scene(source)]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run(source)
    assert [s["source"] for s in result["sequences"]] == spans
    assert len(reviewed) == 3


def test_simultaneous_events_share_a_beat_and_pauses_are_counted():
    item = scene("Il marche et parle.", ["Il marche.", "Il parle."])
    value = plan(item)
    value["timing"].update(beats=[{"events": [1, 2], "seconds": 5, "reason": "Parler en marchant."}],
                           pause_seconds=1, estimated_seconds=6)
    assert planning.validate(value, item)["timing"]["estimated_seconds"] == 6


@pytest.mark.parametrize("duration", [4, 8, 15])
def test_sequential_allowance_respects_configured_duration_without_penalizing_single_beat(duration):
    item = scene("Deux actions.", ["Première.", "Seconde."])
    assert planning.duration_issues(plan(item, duration), duration)
    assert not planning.duration_issues(plan(item, duration / 1.25), duration)
    simultaneous = plan(item, duration)
    simultaneous["timing"]["beats"] = [{"events": [1, 2], "seconds": duration, "reason": "Simultanées."}]
    assert not planning.duration_issues(planning.validate(simultaneous, item), duration)
    assert not planning.duration_issues(plan(scene("Il attend."), duration), duration)


def test_prologue_scene_14_exact_eight_seconds_requires_real_split(monkeypatch):
    spans = [
        "Il brûlait d'envie d'avertir ses compagnons mais décida de conserver ses forces. "
        "Il fourra de nouveau le carnet dans son sac et en tira un filet",
        " dont il recouvrit soigneusement la tablette,",
        " avant d'en nouer les lacets à un crochet au bout de la corde.",
    ]
    original = scene("".join(spans), ["Il range le carnet.", "Il sort le filet.",
                                      "Il recouvre la tablette.", "Il noue les lacets."])
    replacements = [scene(spans[0], original["events"][:2]),
                    scene(spans[1], original["events"][2:3]), scene(spans[2], original["events"][3:])]
    repairs, planned, snapshots = [], [], []

    def chat(client, model, system, user, schema, *args):
        payload, _ = json.JSONDecoder().raw_decode(user)
        if schema == planning.SCHEMA:
            item = payload["current_scene"]
            planned.append(item["source"])
            value = plan(item, 8 if item["source"] == original["source"] else 4)
            if item["source"] == original["source"]:
                for beat, seconds in zip(value["timing"]["beats"], [2, 1, 2, 3]):
                    beat["seconds"] = seconds
            return value
        if schema == nv.SCHEMA:
            return {"issues": []}  # Accepting review cannot waive the timing allowance.
        if "rejected_adaptation" in payload:
            repairs.append(payload)
            assert payload["minimum_replacement_scenes"] == 2
            assert "25%" in payload["defects_to_correct"][0]
            if len(repairs) == 1:
                # Trying to hide the same actions in one shorter description must fail.
                return {"sequences": [scene(original["source"], ["Il prépare la tablette."])]}
            assert "at least 2 replacement scenes" in user
            return {"sequences": deepcopy(replacements)}
        return {"sequences": [deepcopy(original)]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run(original["source"], on_checkpoint=lambda s: snapshots.append(deepcopy(s)))
    assert len(repairs) == 2
    assert planned == [original["source"], *spans]
    assert [s["source"] for s in result["sequences"]] == spans
    assert "".join(s["source"] for s in result["sequences"]) == original["source"]
    assert all("review" not in s for s in result["sequences"])
    assert all(not planning.duration_issues(record["plan"], 8) for record in snapshots[-1]["scene_plans"])


def test_evidence_is_used_across_reading_windows_and_not_copied_into_source(monkeypatch):
    spans = ["La torche est fixée.", " Il attend."]
    generation_payloads = []

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema == planning.SCHEMA:
            item = payload["current_scene"]
            return plan(item, updates=[fact(spans[0], entity="torche", aspect="attachment")]
                        if item["source"] == spans[0] else [])
        if schema == nv.SCHEMA:
            return {"issues": []}
        generation_payloads.append(payload)
        result = {"sequences": [scene(payload["current_passage"])]}
        if schema == ca.WINDOW_SCHEMA:
            result["unprocessed_source"] = ""
        return result

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = ca.adapt_chapter(None, "mock", "".join(spans), chunk_chars=len(spans[0]), temperature=0.15,
                             max_tokens=16384, correction_attempts=1)
    assert len(generation_payloads) == 2
    assert generation_payloads[0]["source_evidence"] == []
    assert generation_payloads[1]["source_evidence"][0]["source_quote"] == spans[0]
    assert "".join(s["source"] for s in result["sequences"]) == "".join(spans)


def test_corrupt_pending_estimate_is_discarded_and_never_committed_to_context(monkeypatch):
    snapshots = []
    item = scene("Il attend.")

    def chat(client, model, system, user, schema, *args):
        if schema == planning.SCHEMA:
            return plan(item)
        if schema == nv.SCHEMA:
            return {"issues": []}
        return {"sequences": [deepcopy(item)]}

    def checkpoint(state):
        snapshots.append(deepcopy(state))
        if state["stage"] == "scene_planned":
            raise KeyboardInterrupt()

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(KeyboardInterrupt):
        run(item["source"], on_checkpoint=checkpoint)
    saved = snapshots[-1]
    saved["pending_sequences"][0]["planning"]["timing"]["estimated_seconds"] = -1
    result = run(item["source"], resume_state=saved,
                 on_checkpoint=lambda state: snapshots.append(deepcopy(state)))
    assert result["sequences"][0]["review"]["status"] == "incomplete"
    assert snapshots[-1]["scene_plans"] == [None]
    assert snapshots[-1]["source_evidence"] == []


def test_incomplete_held_scene_remains_resumable_when_planning_is_unavailable(monkeypatch):
    snapshots = []
    item = {**scene("Il se balance."), "events": []}

    def chat(client, model, system, user, schema, *args):
        if schema == planning.SCHEMA:
            raise RuntimeError("planning unavailable")
        return {"sequences": [deepcopy(item)]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run(item["source"], on_checkpoint=lambda state: snapshots.append(deepcopy(state)))
    assert result["sequences"][0]["adaptation"]["event"] == ""
    assert result["sequences"][0]["review"]["status"] == "incomplete"
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: pytest.fail("No completed scene regeneration"))
    assert run(item["source"], resume_state=snapshots[-1]) == result


@pytest.mark.parametrize("defect,reason,path", [
    ("schema", "schema_required", "/timing"),
    ("extra", "schema_additionalProperties", "/"),
    ("type", "schema_type", "/timing/beats/0/seconds"),
    ("coverage", "event_coverage", "/timing/beats"),
    ("sum", "timing_total_mismatch", "/timing/estimated_seconds"),
    ("finite", "timing_nonfinite", "/timing"),
    ("reset", "reset_quote_mismatch", "/continuity/reset_evidence"),
    ("quote", "source_quote_mismatch", "/continuity/updates/0/source_quote"),
    ("blank", "blank_fact_field", "/continuity/updates/0/entity"),
    ("duplicate", "duplicate_entity_aspect", "/continuity/updates/1"),
])
def test_planning_validation_exposes_only_safe_codes_and_paths(defect, reason, path):
    item = scene("Il attend.")
    value = plan(item)
    secret = "sensitive-backend-body"
    if defect == "schema":
        value["timing"] = {}
    elif defect == "extra":
        value[secret] = secret
    elif defect == "type":
        value["timing"]["beats"][0]["seconds"] = secret
    elif defect == "coverage":
        value["timing"]["beats"][0]["events"] = [2]
    elif defect == "sum":
        value["timing"]["estimated_seconds"] = 5
    elif defect == "finite":
        value["timing"]["estimated_seconds"] = float("nan")
    elif defect == "reset":
        value["continuity"].update(reset=True, reset_evidence=secret)
    elif defect == "quote":
        value["continuity"]["updates"] = [fact(secret)]
    elif defect == "blank":
        value["continuity"]["updates"] = [fact(item["source"], entity=" ")]
    else:
        value["continuity"]["updates"] = [fact(item["source"], entity=name) for name in ("torche", "La torche")]
    with pytest.raises(planning._PlanningValidationError) as caught:
        planning.validate(value, item)
    assert (caught.value.reason, caught.value.field_path) == (reason, path)
    assert secret not in str(caught.value)


def test_missing_planning_still_reviews_source_and_retains_semantic_findings(monkeypatch, caplog):
    spans = ["Il tend la main au-dessus de la corde effilochée.", " Il attend."]
    calls, snapshots = [], []

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema == planning.SCHEMA:
            item = payload["current_scene"]
            if item["source"] == spans[0]:
                return plan(item, updates=[fact("secret invalid quotation")])
            return plan(item)
        if schema == nv.SCHEMA:
            calls.append(payload)
            if payload["sequences"][0]["source"] == spans[0]:
                assert payload["scene_plan"] is None and payload["planning_status"] == "incomplete"
                return {"issues": [{"target": 1, "message": "endingState loses ABOVE the frayed section."}]}
            return {"issues": []}
        return {"sequences": [scene(span) for span in spans]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run("".join(spans), correction_attempts=0, on_checkpoint=lambda s: snapshots.append(deepcopy(s)))
    assert [call["sequences"][0]["source"] for call in calls] == spans
    review = result["sequences"][0]["review"]
    assert review["status"] == "incomplete" and "ABOVE" in review["issues"][0]
    assert review["failed_checks"] == [{"stage": "adaptation", "check": "scene_planning",
                                        "reason": "source_quote_mismatch",
                                        "field_path": "/continuity/updates/0/source_quote"}]
    assert len(snapshots[-1]["approved_reviews"]) == 1
    assert "secret invalid quotation" not in json.dumps(result) + caplog.text
    assert snapshots[-1]["scene_plans"][0] is None


def test_latest_entity_snapshot_retires_cross_aspect_conflicts_and_keeps_distinct_props():
    old = scene("La torche est fixée dans son support. La première torche est tombée.")
    evidence = planning.advance([], planning.validate(plan(old, updates=[
        fact("La torche est fixée dans son support.", entity="torche", aspect="location"),
        fact("La première torche est tombée.", entity="première torche", aspect="location"),
        fact("La torche est fixée dans son support.", entity="torchère", aspect="attachment"),
    ]), old), 13)
    before = deepcopy(evidence)
    current = scene("Il récupéra la torche et la brandit.")
    updates = [fact("Il récupéra la torche", entity="la torche", aspect="possession", value="Il la tient."),
               fact("et la brandit.", entity="La torche", aspect="motion", value="Il la lève.")]
    value = planning.validate(plan(current, updates=updates), current)
    latest = planning.advance(evidence, value, 18)
    assert evidence == before and value["continuity"]["updates"] == updates
    assert latest == [*before[1:], *[{**item, "sequence": 18} for item in updates]]
    assert not any(item["aspect"] == "location" and planning._entity_key(item["entity"]) == "torche" for item in latest)
    assert planning.advance(latest, value, 18) == latest


@pytest.mark.parametrize("left,right", [("première torche", "seconde torche"),
                                        ("corde d'Indy", "corde de la torche"),
                                        ("torche", "torchère"), ("torche 1", "torche 2")])
def test_identity_normalization_preserves_distinguishing_qualifiers(left, right):
    assert planning._entity_key(left) != planning._entity_key(right)
    assert planning._entity_key("La " + left) == planning._entity_key(left)


def test_timing_split_survives_partial_review_interruption_and_failed_replacement_planning(monkeypatch):
    spans = ["Il range le carnet et sort un filet", " puis le noue au crochet."]
    original = scene("".join(spans), ["Ranger.", "Sortir.", "Nouer."])
    snapshots, reviewed, planned = [], [], []

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema == planning.SCHEMA:
            item = payload["current_scene"]
            planned.append(item["source"])
            if item["source"] == spans[0]:
                raise RuntimeError("private backend body")
            return plan(item, 8 if item["source"] == original["source"] else 4)
        if schema == nv.SCHEMA:
            source = payload["sequences"][0]["source"]
            reviewed.append(source)
            if source == original["source"]:
                raise RuntimeError("review unavailable")
            return {"issues": []}
        if "rejected_adaptation" in payload:
            assert payload["minimum_replacement_scenes"] == 2
            assert "25%" in payload["defects_to_correct"][0]
            return {"sequences": [scene(span) for span in spans]}
        pytest.fail("No generation after pending split")

    def checkpoint(state):
        snapshots.append(deepcopy(state))
        if state["stage"] == "corrected_pending_review":
            raise KeyboardInterrupt()

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    # Start from a real source-validated checkpoint, before any planning.
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: {"sequences": [deepcopy(original)]})

    def initial(state):
        if state["stage"] == "source_validated_pending_review":
            snapshots.append(deepcopy(state))
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run(original["source"], on_checkpoint=initial)
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(KeyboardInterrupt):
        run(original["source"], resume_state=snapshots[-1], on_checkpoint=checkpoint)
    saved = snapshots[-1]
    assert saved["approved_reviews"] == []
    assert saved["narrative_review"]["warnings"][0]["status"] == "superseded_by_replacement"
    result = run(original["source"], resume_state=saved, on_checkpoint=lambda s: snapshots.append(deepcopy(s)))
    assert planned == [original["source"], *spans]
    assert reviewed == [original["source"], *spans]
    assert [item["source"] for item in result["sequences"]] == spans
    assert "".join(item["source"] for item in result["sequences"]) == original["source"]
    assert result["sequences"][0]["review"]["status"] == "incomplete"
    assert "review" not in result["sequences"][1]
    assert len(snapshots[-1]["approved_reviews"]) == 1
    assert snapshots[-1]["scene_plans"][0] is None


@pytest.mark.parametrize("old_policy", ["timed-source-context-v6", "timed-source-context-v7"])
def test_resume_rejects_old_review_policies_before_network_calls(monkeypatch, old_policy):
    snapshots = []

    def checkpoint(state):
        snapshots.append(deepcopy(state))
        raise KeyboardInterrupt()

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: {"sequences": [scene("Il attend.")]})
    with pytest.raises(KeyboardInterrupt):
        run("Il attend.", on_checkpoint=checkpoint)
    saved = snapshots[-1]
    saved["review_policy"] = old_policy
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: pytest.fail("Reject before requesting"))
    with pytest.raises(ValueError, match="Invalid cinematic resume checkpoint"):
        run("Il attend.", resume_state=saved)


def test_unplanned_pending_scene_is_reviewed_again_after_resume(monkeypatch):
    snapshots, reviews = [], []
    source = "Il attend."

    def chat(client, model, system, user, schema, *args):
        if schema == planning.SCHEMA:
            raise RuntimeError("private backend text")
        if schema == nv.SCHEMA:
            reviews.append(json.loads(user))
            return {"issues": []}
        return {"sequences": [scene(source)]}

    def checkpoint(state):
        if state["stage"] == "reviewed":
            snapshots.append(deepcopy(state))
            raise KeyboardInterrupt()

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(KeyboardInterrupt):
        run(source, on_checkpoint=checkpoint)
    assert snapshots[-1]["approved_reviews"] == []
    result = run(source, resume_state=snapshots[-1])
    assert len(reviews) == 2
    assert all(payload["planning_status"] == "incomplete" for payload in reviews)
    assert result["sequences"][0]["review"]["status"] == "incomplete"


def test_replaced_physical_snapshot_is_reconstructed_identically_on_resume(monkeypatch):
    spans = ["La torche est dans son support.", " Il récupéra la torche.", " Il attend."]
    snapshots, contexts = [], []

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema == planning.SCHEMA:
            item = payload["current_scene"]
            contexts.append(deepcopy(payload["source_evidence"]))
            updates = []
            if item["source"] == spans[0]:
                updates = [fact(spans[0], entity="torche", aspect="location", value="Dans le support.")]
            elif item["source"] == spans[1]:
                updates = [fact("Il récupéra la torche.", entity="La torche", aspect="possession", value="Indy la tient.")]
            return plan(item, updates=updates)
        if schema == nv.SCHEMA:
            return {"issues": []}
        return {"sequences": [scene(span) for span in spans]}

    def checkpoint(state):
        snapshots.append(deepcopy(state))
        if state["stage"] == "scene_committed" and len(state["sequences"]) == 2:
            raise KeyboardInterrupt()

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(KeyboardInterrupt):
        run("".join(spans), on_checkpoint=checkpoint)
    saved = snapshots[-1]
    assert len(saved["source_evidence"]) == 1
    assert saved["source_evidence"][0]["source_quote"] == "Il récupéra la torche."
    result = run("".join(spans), resume_state=saved)
    assert contexts[-1] == saved["source_evidence"]
    assert "".join(item["source"] for item in result["sequences"]) == "".join(spans)

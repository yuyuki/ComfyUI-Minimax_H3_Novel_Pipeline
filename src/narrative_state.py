"""Internal narrative contracts; these are not MiniMax H3 schema fields."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

from .lmstudio_json import chat_json
from .lmstudio_pipeline import comfy_interrupt_check


def source_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array(items):
    return {"type": "array", "items": items}


STRING = {"type": "string"}
NULLABLE = {"type": ["string", "null"]}
FIELDS = ("location", "position", "posture", "owner", "relationship", "status", "environment")
ENTITY = obj({"id": STRING, "kind": {"type": "string", "enum": ["character", "object", "location"]},
              **{key: NULLABLE for key in FIELDS}, "visible": {"type": "boolean"}})
STATE = obj({"entities": array(ENTITY)})
VALUE = {"type": ["string", "boolean", "null"]}
CHANGE = obj({"entity": STRING, "field": {"type": "string", "enum": [*FIELDS, "visible"]},
              "before": VALUE, "after": VALUE})
EVENT = obj({"id": STRING, "description": STRING, "source_evidence": STRING,
             "kind": {"type": "string", "enum": ["action", "introduction", "repair"]}, "changes": array(CHANGE)})
CONTRACT_SCHEMA = {"name": "narrative_contract_v1", "strict": True, "schema": obj({
    "state_before": STATE, "initial_frame": STATE, "events": array(EVENT), "state_after": STATE,
})}
EXTRACTION_SCHEMA = {"name": "narrative_events_v2", "strict": True, "schema": obj({
    "opening_entities": array(ENTITY),
    "new_entities": array(obj({"id": STRING, "kind": ENTITY["properties"]["kind"]})),
    "events": array(obj({"id": STRING, "description": STRING,
        "source_ids": array(STRING), "kind": EVENT["properties"]["kind"],
        "changes": array(obj({"entity": STRING, "field": CHANGE["properties"]["field"], "after": VALUE}))})),
})}
EXTRACTION_SYSTEM = """Extract narrative continuity, without rewriting prose or designing shots.
Return opening_entities ONLY for the first passage (current_state is null); otherwise return [].
Opening entities contain only facts already true before the first action, never later states.
Declare entities first encountered during events in new_entities with a stable ID and kind.
They start absent, invisible and unowned; establish them with an introduction event.
Reuse current_state and authoritative_contract IDs. Distinguish separate instances of props.
Previously lost props remain tracked, invisible and unowned; mentioning them does not reacquire
them. If the loss predates this passage, preserve it as opening/carried state, not a new action.
Record every action/dialogue in chronological order, including actions with no state changes.
For evidence select source_ids from supplied source_units; never reproduce or rewrite quotations.
Multiple events may cite the same unit; respect action order within it (including 'after' clauses).
Each event lists only changed fields and their resulting after values. Python computes all before
values, opening frames and final states. Use one change per entity/field per event.
Track character location/posture, prop ownership/position/visibility and environment conditions.
Owned props have location equal to the owner ID. Use relationship held, in_mouth, worn or stored
for possession, not redundant holding claims on characters. Never put a torch in a mouth until
the source action. A transfer updates owner, location and relationship together. Use status for
condition and position for placement. Unknown attributes are null. No invented movement or props.
Broken/damaged props require a source-supported repair event before restoration.
original_scene is evidence; cinematic_version is context. authoritative_contract constrains scene
projection; expected_after constrains the passage boundary. Preserve all source-supported events.
Return JSON only."""
ISSUE = obj({key: STRING for key in ("entity", "expected_state", "conflicting_state", "introducing_event", "suggested_correction")})
REVIEW_SCHEMA = {"name": "narrative_review_v1", "strict": True, "schema": obj({"errors": array(ISSUE)})}
VERIFY_REVIEW_SCHEMA = {"name": "cinematic_review_verification_v1", "strict": True,
                        "schema": obj({"decisions": array(obj({
                            "supported": {"type": "boolean"}, "reason": STRING,
                            "source_evidence": STRING, "candidate_evidence": STRING,
                        }))})}
SIMPLIFY_SCHEMA = {"name": "cinematic_text_v1", "strict": True,
                   "schema": obj({"cinematic_text": STRING})}
SIMPLIFY_SYSTEM = """Normalize prose for filming, never summarize. Remove nonvisual metaphors
(a suspended man compared to a crescent moon is just suspended). Translate implicit physical
descriptions into explicit spatial relationships: a rope hurting armpits passes tightly under
the arms. Preserve identity, locations, every important prop, atmosphere, causality, ALL actions
in order and dialogue verbatim in its original language. Do not invent actions to replace
thoughts; omit unfilmable thoughts while retaining any stated visible behavior. Do not add
future objects or postures to the opening description. Preserve paragraph order. Output JSON."""



def check_shape(value, schema, path="result"):
    """Validate the small JSON-schema subset used here, including mocked/backend responses."""
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    matches = {"object": isinstance(value, dict), "array": isinstance(value, list),
               "string": isinstance(value, str), "boolean": isinstance(value, bool), "null": value is None}
    if not any(matches.get(kind, False) for kind in kinds):
        raise ValueError(f"{path}: expected {kinds}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: invalid value {value!r}")
    if isinstance(value, dict):
        props = schema["properties"]
        if set(value) != set(props):
            raise ValueError(f"{path}: expected fields {list(props)}")
        for key, child in props.items():
            check_shape(value[key], child, f"{path}.{key}")
    if isinstance(value, list):
        for i, item in enumerate(value):
            check_shape(item, schema["items"], f"{path}[{i}]")


def issue(entity, expected, conflicting, event="", correction="Restore the expected state until the introducing event."):
    return dict(entity=str(entity), expected_state=str(expected), conflicting_state=str(conflicting),
                introducing_event=str(event), suggested_correction=correction)


def state_map(state):
    check_shape(state, STATE, "state")
    result = {e["id"]: e for e in state["entities"]}
    if len(result) != len(state["entities"]) or any(not key.strip() for key in result):
        raise ValueError("State entity IDs must be unique and nonempty.")
    return result


def validate_contract(contract, current_state=None, source=""):
    """Replay transitions without trusting the model's claimed end or opening frame."""
    check_shape(contract, CONTRACT_SCHEMA["schema"])
    before = state_map(contract["state_before"])
    after = state_map(contract["state_after"])
    initial = state_map(contract["initial_frame"])
    errors = []
    previous = state_map(current_state) if current_state is not None else {}
    for key, entity in previous.items():
        if before.get(key) != entity:
            errors.append(issue(key, entity, before.get(key), correction="Carry the previous state_after unchanged into state_before."))
    for key, entity in before.items():
        if current_state is not None and key not in previous and (
                entity["status"] != "not_introduced" or entity["visible"] or entity["owner"] is not None or entity["location"] is not None):
            errors.append(issue(key, "not_introduced, invisible, no owner/location", entity))
    for key, entity in initial.items():
        if entity != before.get(key) or not entity["visible"] or entity["status"] == "not_introduced":
            introducing = next((e["id"] for e in contract["events"] if any(c["entity"] == key for c in e["changes"])), "")
            errors.append(issue(key, before.get(key), entity, introducing))

    def invariants(state, event):
        for key, entity in state.items():
            if entity["status"] in {"not_introduced", "lost_below"} and (entity["visible"] or entity["owner"] is not None):
                errors.append(issue(key, "invisible and unowned", entity, event))
            if entity["status"] == "not_introduced" and entity["location"] is not None:
                errors.append(issue(key, "no location before introduction", entity["location"], event))
            if entity["status"] == "not_introduced" and any(entity[f] is not None for f in FIELDS if f != "status"):
                errors.append(issue(key, "no physical state before introduction", entity, event))
            owner = entity["owner"]
            if owner is not None and (owner not in state or entity["location"] != owner or state[owner]["status"] == "not_introduced"):
                errors.append(issue(key, "one existing owner; location equals owner ID", entity, event,
                                    correction="Use a declared, already introduced owner ID. For an owned prop set location "
                                    "to that exact owner ID; keep spatial descriptions in position. Correct opening_entities "
                                    "for an initial-state error, or update owner and location together in the introducing "
                                    "event. Do not invent an owner or change carried current_state."))
            if entity["relationship"] in {"held", "in_mouth", "worn"} and owner is None:
                errors.append(issue(key, "owner for held/in_mouth/worn object", entity, event))
            visited = {key}
            while owner in state:
                if owner in visited:
                    errors.append(issue(key, "acyclic ownership", owner, event))
                    break
                visited.add(owner)
                owner = state[owner]["owner"]

    replay = deepcopy(before)
    invariants(replay, "initial state")
    seen = set()
    evidence_position = 0
    for event in contract["events"]:
        eid = event["id"]
        if not eid.strip() or eid in seen:
            errors.append(issue("events", "unique nonempty IDs", eid))
        seen.add(eid)
        if not event["source_evidence"].strip() or (source and event["source_evidence"] not in source):
            errors.append(issue("events", "verbatim evidence from source", event["source_evidence"], eid,
                                correction="Replace this event's source_evidence with an exact contiguous excerpt from original_scene, "
                                "in the original language and punctuation. Do not quote cinematic_version, translate, "
                                "paraphrase or insert ellipses. Preserve the source-supported event."))
        elif source:
            position = source.find(event["source_evidence"], evidence_position)
            if position < 0:
                errors.append(issue("events", "source evidence in chronological order", event["source_evidence"], eid,
                                    correction="Order events by their excerpts in original_scene; replay all changes in that order."))
            else:
                evidence_position = position
        changed = set()
        for change in event["changes"]:
            key, field = change["entity"], change["field"]
            if key not in replay:
                errors.append(issue(key, "entity declared in state_before", "missing", eid))
                continue
            if (key, field) in changed:
                errors.append(issue(key, "one atomic change per field per event", field, eid))
            changed.add((key, field))
            old, new = change["before"], change["after"]
            if (field == "visible" and not isinstance(new, bool)) or (field != "visible" and new is not None and not isinstance(new, str)):
                errors.append(issue(key, f"valid {field} type", new, eid))
                continue
            if replay[key][field] != old:
                errors.append(issue(key, f"{field}={replay[key][field]}", f"{field}={old}", eid,
                                    correction=f"Set this change.before to the running {field} value "
                                    f"{json.dumps(replay[key][field], ensure_ascii=False)} after preceding events. "
                                    "If a preceding change is unsupported, correct it against original_scene instead. "
                                    "Replay all events and recompute state_after; do not reset to the opening state."))
            if field == "status":
                if old == "not_introduced" and new != old and event["kind"] != "introduction":
                    errors.append(issue(key, "introduction event", event["kind"], eid))
                if ((old == "broken" and new != "broken") or (old == "damaged" and new == "intact")) and event["kind"] != "repair":
                    errors.append(issue(key, "explicit repair before restoration", new, eid))
            replay[key][field] = new
        invariants(replay, eid)
    if replay != after:
        for key in replay.keys() | after.keys():
            if replay.get(key) != after.get(key):
                errors.append(issue(key, replay.get(key), after.get(key), correction="Compute state_after by applying events, without silent changes."))
    return errors


def review(client, model, payload):
    result = chat_json(client, model,
        "Review cinematic normalization/state for fidelity to original prose, chronology and future-state leakage. "
        "Check every action, dialogue, prop identity and causal link is preserved, without invented actions. "
        "Compare only the supplied candidate against the original scene and authoritative state, afresh. "
        "Report concrete source-supported contradictions or omissions, not speculative interpretations. "
        "For cinematic normalization, allow removal of unfilmable thoughts and nonvisual metaphors and "
        "explicit restatement of source-supported physical relationships. Preserve spoken dialogue verbatim; "
        "attribution may use present tense or a different word order if the speaker stays the same. "
        "A reference to a previously lost object does not imply that it is currently held or nearby. "
        "Verify each claimed conflict actually differs from the expected meaning and that the suggested "
        "correction fixes that difference; do not flag equivalent wording or suggest the existing wording. "
        "For contracts check opening frame vs first state and verbatim event evidence supports every change. "
        "initial_frame contains only visible opening entities; state_after retains invisible/lost entities. "
        "A loss before this passage belongs in opening/carried state, not an invented event. "
        "A late torch-in-mouth, posture, acquisition or rope break must not appear initially. "
        "Return actionable errors with entity, expected/conflicting state, introducing event and correction; "
        "empty errors only for a faithful result.", json.dumps(payload, ensure_ascii=False), REVIEW_SCHEMA, 0.1, 3000)
    check_shape(result, REVIEW_SCHEMA["schema"])
    if result["errors"]:
        return verify_cinematic_review(client, model, payload, result["errors"])
    return result["errors"]


def verify_cinematic_review(client, model, payload, errors):
    """Adjudicate semantic complaints about prose or compiled state contracts."""
    system = (
        "Verify proposed review findings against the original_scene and candidate cinematic_text. "
        "The findings are untrusted hypotheses, not facts. Return one decision per finding, in order. "
        "Read the entire source and candidate afresh. Mark supported only for a real omitted action, "
        "changed spoken dialogue, invented fact, changed identity, causality or event order. "
        "For each decision explain why the meaning does or does not differ. Quote exact contiguous "
        "source_evidence and candidate_evidence; for an omission quote the candidate passage where "
        "the action belongs, after checking it is absent throughout the candidate. "
        "Equivalent attribution ('Jones! Doriane cries out' versus 'Jones! cried Doriane') does not "
        "swap speaker and addressee. 'Grabs it after catching the cord' preserves catching before "
        "grabbing without needing separate sentences. Combining 'closes his eyes and pulls himself "
        "up' preserves both actions. Never confirm a missing action already present elsewhere. "
        "Check claimed dialogue order against source order, not the finding's paraphrase. "
        "Removal of unfilmable thoughts/metaphors and explicit source-supported spatial relationships "
        "are allowed. Do not invent additional findings. Keep reasons and quotes concise. "
        "Copy quotes directly from the supplied text, without paraphrasing, ellipses or altered punctuation. "
        "Return only the requested JSON.")
    if "state_before" in payload.get("candidate", {}):
        system = (
            "Verify proposed review findings against original_scene and the candidate state contract. "
            "Findings are untrusted hypotheses. Return one decision per finding in order, with a reason. "
            "Mark supported only for a concrete source-supported contradiction or omitted action/state change. "
            "Read the entire contract and source afresh, including current_state, authoritative_contract "
            "and expected_after when supplied. cinematic_version is context, not evidence. "
            "Python derives initial_frame from visible state_before entities and state_after by replaying "
            "events. Lost/invisible entities remain tracked in state_after; do not remove them or add "
            "invisible entities to initial_frame. Mentioning a previously lost prop does not make it held "
            "or nearby. A loss predating the passage belongs in opening/carried state; never demand an "
            "invented loss event. Different props keep separate IDs. Read owner, location, relationship, "
            "status and visible together. Equivalent wording or identical expected/conflicting meanings "
            "alone are not contradictions. Check the actual candidate even if the finding quotes it wrongly. "
            "Confirm real missing acquisitions, changed identity, unsupported opening facts or wrong action "
            "order. A suggested correction must address the conflict without violating these contract rules. "
            "For supported findings quote exact contiguous source_evidence from original_scene and "
            "candidate_evidence from the candidate's JSON as supplied (including JSON escaping). For an "
            "omission quote the relevant existing event/state after checking the entire contract. "
            "Do not invent additional findings. Keep reasons and quotes concise. Return only requested JSON.")
    request = {**payload, "proposed_errors": errors}
    # Verification failures describe the reviewer, not the candidate.
    # Repair that response locally rather than spending the correction budget.
    for attempt in range(3):
        comfy_interrupt_check()
        result = None
        try:
            result = chat_json(client, model, system, json.dumps(request, ensure_ascii=False),
                               VERIFY_REVIEW_SCHEMA, 0.0, 4000)
            return _validate_review_decisions(result, payload, errors)
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and not str(exc).startswith("Invalid structured JSON after"):
                raise
            if attempt == 2:
                raise RuntimeError(f"Review verification failed after 3 attempts: {exc}") from exc
            request = {**payload, "proposed_errors": errors, "previous_verification": result,
                       "verification_error": str(exc)}
            system += (" Repair the previous_verification using verification_error. Return the complete "
                        f"decisions array with exactly {len(errors)} decisions in finding order, including "
                        "rejected findings. Do not rewrite the candidate or change the findings.")


def _validate_review_decisions(result, payload, errors):
    check_shape(result, VERIFY_REVIEW_SCHEMA["schema"])
    decisions = result["decisions"]
    if len(decisions) != len(errors):
        raise ValueError(f"Review verification must decide every proposed error: expected {len(errors)} decisions, got {len(decisions)}.")
    confirmed = []
    for index, (error, decision) in enumerate(zip(errors, decisions)):
        if not decision["reason"].strip():
            raise ValueError(f"Review verification requires a reason for every decision: decisions[{index}].reason is blank.")
        if decision["supported"]:
            candidate = payload["candidate"]
            candidate_text = (candidate["cinematic_text"] if "cinematic_text" in candidate
                              else json.dumps(candidate, ensure_ascii=False))
            for field, text in (("source_evidence", payload["original_scene"]),
                                ("candidate_evidence", candidate_text)):
                evidence = decision[field]
                if not evidence.strip() or evidence not in text:
                    raise ValueError(f"Review verification requires verbatim {field} for a confirmed error: decisions[{index}].{field}.")
            confirmed.append(error)
    return confirmed


def checked_pass(client, model, system, schema, payload, attempts, validator, compile_result=None):
    if not isinstance(attempts, int) or not 0 <= attempts <= 10:
        raise ValueError("Correction attempts must be between 0 and 10.")
    source_payload = deepcopy(payload)
    history = []
    for attempt in range(attempts + 1):
        comfy_interrupt_check()
        result = None
        candidate = None
        try:
            correction = (
                " Revise the supplied candidate using validation_errors as feedback, checking each claim "
                "against the original scene and authoritative state. Fix supported errors without deleting "
                "source facts or changing dialogue. Return the complete corrected JSON in the requested "
                "schema, not a patch or the review."
            ) if attempt else ""
            result = chat_json(client, model, system + correction, json.dumps(payload, ensure_ascii=False), schema, 0.15, 8000)
            check_shape(result, schema["schema"])
            candidate = deepcopy(result)
            if compile_result is not None:
                result = compile_result(result)
            errors = validator(result)
            if not errors:
                # Previous review feedback belongs only in the correction request. Passing it
                # to the reviewer anchors fresh assessments to stale or mistaken complaints.
                errors = review(client, model, {**source_payload, "candidate": result})
        except ValueError as exc:
            errors = [issue("schema", "valid structured JSON", str(exc))]
        except RuntimeError as exc:
            if not str(exc).startswith("Invalid structured JSON after"):
                raise
            errors = [issue("schema", "valid structured JSON", str(exc))]
        history.append({"attempt": attempt, "errors": errors})
        if not errors:
            return result, {"valid": True, "attempts": history}
        payload = {**source_payload, "candidate": candidate if candidate is not None else result, "validation_errors": errors}
    raise ValueError("Narrative validation failed after correction budget: " + json.dumps(history, ensure_ascii=False))


def simplify(client, model, original, attempts=2):
    return checked_pass(client, model, SIMPLIFY_SYSTEM, SIMPLIFY_SCHEMA, {"original_scene": original}, attempts,
                        lambda result: [] if result["cinematic_text"].strip() else [issue("text", "nonempty cinematic text", "empty")])


def source_units(source):
    """Index exact source slices; IDs remove quotation reproduction from model work."""
    units = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+|\n+", source):
        end = match.start()
        if source[start:end].strip():
            units.append({"id": f"source_{len(units) + 1}", "start": start, "end": end,
                          "text": source[start:end]})
        start = match.end()
    if source[start:].strip():
        units.append({"id": f"source_{len(units) + 1}", "start": start, "end": len(source), "text": source[start:]})
    return units


def compile_contract(extraction, source, current_state=None):
    """Build the public v1 contract from model facts plus deterministic event replay."""
    check_shape(extraction, EXTRACTION_SCHEMA["schema"])
    if current_state is not None and extraction["opening_entities"]:
        raise ValueError("opening_entities must be empty when carrying current_state; use events for changes.")
    before = deepcopy(current_state) if current_state is not None else {"entities": deepcopy(extraction["opening_entities"])}
    if current_state is None:
        opening = state_map(before)
        for entity in opening.values():
            owner = opening.get(entity["owner"])
            if (entity["kind"] == "object" and owner is not None
                    and owner["status"] != "not_introduced"
                    and entity["status"] not in {"not_introduced", "lost_below"}
                    and entity["location"] != entity["owner"]):
                # Ownership already establishes the canonical location. Preserve the
                # model's physical placement without inventing a narrative transition.
                location = entity["location"]
                if location:
                    entity["position"] = "; ".join(filter(None, (entity["position"], location)))
                entity["location"] = entity["owner"]
    for entity in extraction["new_entities"]:
        before["entities"].append({**entity, **dict.fromkeys(FIELDS), "status": "not_introduced", "visible": False})
    replay = deepcopy(state_map(before))
    units = {unit["id"]: unit for unit in source_units(source)}
    events = []
    last_start = -1
    for event in extraction["events"]:
        ids = event["source_ids"]
        if not ids or any(key not in units for key in ids):
            raise ValueError(f"Event {event['id']}: source_ids must select existing source units.")
        selected = [units[key] for key in ids]
        starts = [unit["start"] for unit in selected]
        if starts != sorted(set(starts)) or starts[0] < last_start:
            raise ValueError(f"Event {event['id']}: source_ids must follow chronological source order.")
        last_start = starts[0]
        changes = []
        for change in event["changes"]:
            key, field = change["entity"], change["field"]
            if key not in replay:
                raise ValueError(f"Event {event['id']}: undeclared entity {key}.")
            changes.append({**change, "before": deepcopy(replay[key][field])})
            replay[key][field] = change["after"]
        events.append({"id": event["id"], "description": event["description"], "kind": event["kind"],
                       "source_evidence": source[selected[0]["start"]:selected[-1]["end"]], "changes": changes})
    return {"state_before": before,
            "initial_frame": {"entities": [deepcopy(e) for e in before["entities"] if e["visible"]]},
            "events": events, "state_after": {"entities": list(replay.values())}}


def track_scene(client, model, original, cinematic, current_state=None, attempts=2, authoritative_contract=None, expected_after=None):
    if current_state is not None:
        state_map(current_state)
    def validate(result):
        errors = validate_contract(result, current_state, original)
        if expected_after is not None and state_map(result["state_after"]) != state_map(expected_after):
            errors.append(issue("scene boundary", expected_after, result["state_after"],
                                correction="Preserve all planned events and stable IDs to reach the authoritative ending state."))
        return errors
    return checked_pass(client, model, EXTRACTION_SYSTEM, EXTRACTION_SCHEMA,
                        {"original_scene": original, "cinematic_version": cinematic, "current_state": current_state,
                         "source_units": source_units(original),
                         "authoritative_contract": authoritative_contract, "expected_after": expected_after}, attempts, validate,
                        lambda result: compile_contract(result, original, current_state))

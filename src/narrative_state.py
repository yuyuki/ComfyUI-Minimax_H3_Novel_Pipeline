"""Internal narrative contracts; these are not MiniMax H3 schema fields."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

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
ISSUE = obj({key: STRING for key in ("entity", "expected_state", "conflicting_state", "introducing_event", "suggested_correction")})
REVIEW_SCHEMA = {"name": "narrative_review_v1", "strict": True, "schema": obj({"errors": array(ISSUE)})}
SIMPLIFY_SCHEMA = {"name": "cinematic_text_v1", "strict": True,
                   "schema": obj({"cinematic_text": STRING})}
SIMPLIFY_SYSTEM = """Normalize prose for filming, never summarize. Remove nonvisual metaphors
(a suspended man compared to a crescent moon is just suspended). Translate implicit physical
descriptions into explicit spatial relationships: a rope hurting armpits passes tightly under
the arms. Preserve identity, locations, every important prop, atmosphere, causality, ALL actions
in order and dialogue verbatim in its original language. Do not invent actions to replace
thoughts; omit unfilmable thoughts while retaining any stated visible behavior. Do not add
future objects or postures to the opening description. Preserve paragraph order. Output JSON."""
STATE_SYSTEM = """Extract a chronological film state contract from the supplied scene.
Use stable entity IDs, distinguish multiple instances (torch_1 and torch_2). Track characters,
locations and objects, their location, position, posture, owner, relationship, status, visibility
and environment. Null means unknown, not permission to invent. An owned object's location is
its owner's ID; relationship describes held, in_mouth, worn or stored. No duplicate entities.
Copy current_state exactly into state_before. You may add newly encountered entities, but when
current_state exists start new entities not_introduced, invisible, with null owner and location;
an introduction event establishes them. For the first scene infer ONLY its opening state.
initial_frame is a subset of state_before containing only visible entities with identical fields.
Every event has an ID, a verbatim source_evidence excerpt, and atomic changes with exact before
and after values. Apply changes in strict story order to produce state_after. Even dialogue or
looks with no state change are events. Movement, acquisition, posture changes and rope damage
must be events. Never place a late torch-in-mouth action in state_before. Lost and unintroduced
objects are invisible. A broken object needs an explicit repair event before restoration.
Do not skip or invent actions; preserve dialogue. Return only the requested JSON contract."""


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
            owner = entity["owner"]
            if owner is not None and (owner not in state or entity["location"] != owner or state[owner]["status"] == "not_introduced"):
                errors.append(issue(key, "one existing owner; location equals owner ID", entity, event))
            if entity["relationship"] in {"held", "in_mouth", "worn"} and owner is None:
                errors.append(issue(key, "owner for held/in_mouth/worn object", entity, event))

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
            errors.append(issue("events", "verbatim evidence from source", event["source_evidence"], eid))
        elif source:
            position = source.find(event["source_evidence"], evidence_position)
            if position < 0:
                errors.append(issue("events", "source evidence in chronological order", event["source_evidence"], eid))
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
                errors.append(issue(key, f"{field}={replay[key][field]}", f"{field}={old}", eid))
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
        "For contracts check opening frame vs first state and verbatim event evidence supports every change. "
        "A late torch-in-mouth, posture, acquisition or rope break must not appear initially. "
        "Return actionable errors with entity, expected/conflicting state, introducing event and correction; "
        "empty errors only for a faithful result.", json.dumps(payload, ensure_ascii=False), REVIEW_SCHEMA, 0.1, 3000)
    check_shape(result, REVIEW_SCHEMA["schema"])
    return result["errors"]


def checked_pass(client, model, system, schema, payload, attempts, validator):
    if not isinstance(attempts, int) or not 0 <= attempts <= 10:
        raise ValueError("Correction attempts must be between 0 and 10.")
    history = []
    for attempt in range(attempts + 1):
        comfy_interrupt_check()
        result = None
        try:
            result = chat_json(client, model, system, json.dumps(payload, ensure_ascii=False), schema, 0.15, 8000)
            check_shape(result, schema["schema"])
            errors = validator(result)
            if not errors:
                errors = review(client, model, {**payload, "candidate": result})
        except ValueError as exc:
            errors = [issue("schema", "valid structured JSON", str(exc))]
        except RuntimeError as exc:
            if not str(exc).startswith("Invalid structured JSON after"):
                raise
            errors = [issue("schema", "valid structured JSON", str(exc))]
        history.append({"attempt": attempt, "errors": errors})
        if not errors:
            return result, {"valid": True, "attempts": history}
        payload = {**payload, "candidate": result, "validation_errors": errors}
    raise ValueError("Narrative validation failed after correction budget: " + json.dumps(history, ensure_ascii=False))


def simplify(client, model, original, attempts=2):
    return checked_pass(client, model, SIMPLIFY_SYSTEM, SIMPLIFY_SCHEMA, {"original_scene": original}, attempts,
                        lambda result: [] if result["cinematic_text"].strip() else [issue("text", "nonempty cinematic text", "empty")])


def track_scene(client, model, original, cinematic, current_state=None, attempts=2, authoritative_contract=None, expected_after=None):
    if current_state is not None:
        state_map(current_state)
    def validate(result):
        errors = validate_contract(result, current_state, original)
        if expected_after is not None and state_map(result["state_after"]) != state_map(expected_after):
            errors.append(issue("scene boundary", expected_after, result["state_after"],
                                correction="Preserve all planned events and stable IDs to reach the authoritative ending state."))
        return errors
    return checked_pass(client, model, STATE_SYSTEM +
                        " Reuse IDs and facts from authoritative_contract when supplied, but apply only the "
                        "current scene's events. Reach expected_after at this segment boundary if supplied.", CONTRACT_SCHEMA,
                        {"original_scene": original, "cinematic_version": cinematic, "current_state": current_state,
                         "authoritative_contract": authoritative_contract, "expected_after": expected_after}, attempts, validate)

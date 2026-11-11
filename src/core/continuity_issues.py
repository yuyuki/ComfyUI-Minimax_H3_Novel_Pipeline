"""Strict, evidence-linked continuity issues and bounded metadata corrections."""
from copy import deepcopy

from jsonschema import Draft202012Validator

from . import continuity_context
from .generation_requests import digest


ISSUE = {"type": "object", "properties": {
    "target": {"type": "string", "enum": ["prompt", "scene_plan", "binding", "upstream_conflict"]},
    "message": {"type": "string", "minLength": 1},
    "evidence_paths": {"type": "array", "minItems": 1, "items": {"type": "string"}},
}, "required": ["target", "message", "evidence_paths"], "additionalProperties": False}
SCHEMA = {"name": "scene_continuity_review_v2", "strict": True, "schema": {
    "type": "object", "properties": {"errors": {"type": "array", "items": ISSUE}},
    "required": ["errors"], "additionalProperties": False}}


class UnresolvedEvidenceError(ValueError):
    def __init__(self, paths):
        self.paths = tuple(paths)
        super().__init__(
            "LM Studio returned an invalid scene continuity review: unresolved evidence "
            f"path(s) {', '.join(repr(path) for path in self.paths)}."
        )


INSTRUCTIONS = """
Classify each issue's target: prompt for incorrect generated wording; scene_plan
for incorrect current-beat dialogue/visibility/reference metadata; binding for
an incorrect assigned reference table; upstream_conflict for conflicting approved
source/choices/staging or a problem requiring changed story beats. Metadata repair
cannot rewrite source, visual_event, adaptation_notes, sequence order or operator
choices. Never ask a prompt-only correction to change those inputs.
Each issue needs an actionable message and evidence_paths: absolute JSON pointers
to the actual supplied request, e.g. /current_scene/visual_event or
/continuity_plan/sequence_plan. Cite the supplied evidence supporting the issue,
not an imagined record or a record from a different batch. Return errors: [] only
when there are no issues. The current_prompt field is one complete string; cite
/current_prompt for evidence from the whole prompt. For section-specific prompt
evidence, cite /current_prompt_sections/<section_name>, such as
/current_prompt_sections/detailed_description. These values contain the exact
section text from current_prompt. Do not append section names to /current_prompt.
Use the supplied structured schema exactly.
"""


def resolve(payload, path):
    if not path.startswith("/"):
        raise ValueError("Evidence path must be an absolute JSON pointer.")
    value = payload
    for part in path[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not part.isdecimal():
                raise ValueError("Invalid evidence array index.")
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise ValueError("Evidence path continues past a scalar value.")
    return value


def validate(result, payload):
    if not Draft202012Validator(SCHEMA["schema"]).is_valid(result):
        raise ValueError("LM Studio returned an invalid scene continuity review.")
    issues = []
    unresolved_paths = []
    for issue in result["errors"]:
        if not issue["message"].strip():
            raise ValueError("LM Studio returned an invalid scene continuity review: empty correction.")
        evidence = []
        for path in issue["evidence_paths"]:
            try:
                value = resolve(payload, path)
            except (KeyError, IndexError, TypeError, ValueError):
                unresolved_paths.append(path)
                continue
            # Keep exact cited values with the issue: later repairs may run
            # after this batch is gone. No second interpretation or truncation.
            evidence.append({"path": path, "value": deepcopy(value)})
        issues.append({**issue, "evidence": evidence})
    if unresolved_paths:
        raise UnresolvedEvidenceError(unresolved_paths)
    return issues


def unique(issues):
    return list({digest(issue): issue for issue in issues}.values())


def metadata_schema(catalog):
    ids = sorted({item["global_id"] for item in catalog})
    locations = sorted(item["global_id"] for item in catalog if item.get("entity_type") == "location")
    speakers = sorted(item["global_id"] for item in catalog if item.get("entity_type") == "character")
    array = {"type": "array", "items": {"type": "string", **({"enum": ids} if ids else {})}}
    if not ids:
        array["maxItems"] = 0
    fields = {"dialogue_present": {"type": "boolean"},
              "visible_entity_ids": deepcopy(array), "speaking_entity_ids": deepcopy(array),
              "location_global_id": {"type": "string", "enum": ["", *locations]}}
    fields["speaking_entity_ids"]["items"] = {"type": "string", **({"enum": speakers} if speakers else {})}
    if not speakers:
        fields["speaking_entity_ids"]["maxItems"] = 0
    request_fields = {
        "global_id": {"type": "string", **({"enum": ids} if ids else {})},
        "view_types": {"type": "array", "items": {"type": "string"}},
        "prominence": {"type": "string", "enum": ["primary", "secondary", "background"]},
        "reason": {"type": "string"},
    }
    fields["reference_view_requests"] = {"type": "array", "items": {
        "type": "object", "properties": request_fields, "required": list(request_fields), "additionalProperties": False}}
    if not ids:
        fields["reference_view_requests"]["maxItems"] = 0
    return {"name": "h3_scene_metadata_repair_v1", "strict": True, "schema": {
        "type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}}


def repair_metadata(client, model, scene, row, issues, args, request):
    """Correct only current-beat metadata; immutable story decisions stay intact."""
    schema = metadata_schema(row["catalog"])
    system = (
        "Correct only the current beat's dialogue, visible/speaking entity IDs, location and reference-view metadata. "
        "Use the exact supplied beat and approved upstream evidence. Never advance neighbouring events, "
        "add dialogue or turn offscreen/mentioned characters into visible characters. "
        "Return only the schema fields. Preserve existing correct values. Empty means unknown, not invented. "
        "Use only catalog global IDs and available_picture_views. Never supply picture asset IDs as entity IDs. "
        "Source, selected choices, beat, dialogue text and phases are immutable."
    ) + continuity_context.INSTRUCTIONS
    payload = {"current_scene": scene, "upstream_sequence": row, "issues": issues}
    if not continuity_context.fits(system, payload, schema):
        raise ValueError("H3 metadata repair exceeds the safe request budget; no evidence was discarded.")
    result = request(client, model, system, continuity_context.pack(payload), schema, .1, args.max_tokens)
    if not Draft202012Validator(schema["schema"]).is_valid(result):
        raise ValueError("Invalid H3 scene metadata repair.")
    if not result["dialogue_present"] and result["speaking_entity_ids"]:
        raise ValueError("Scene metadata repair assigns speakers to a beat without dialogue.")
    views = {item["global_id"]: {view["view_type"] for view in item.get("available_picture_views", [])}
             for item in row["catalog"]}
    seen = set()
    for requested in result["reference_view_requests"]:
        gid = requested["global_id"]
        if (gid in seen or gid not in result["visible_entity_ids"]
                or not set(requested["view_types"]) <= views[gid]):
            raise ValueError("Scene metadata repair requests an unavailable or invisible reference.")
        seen.add(gid)
    context = row.get("character_context")
    if context and context.get("sequence"):
        mapped = context["registry_ids"]
        presence = dict.fromkeys(mapped.values(), "absent")
        presence.update({mapped[r["local_id"]]: r["presence"] for r in context["sequence"]["characters"]})
        if any(presence.get(gid, "visible") != "visible" for gid in result["visible_entity_ids"]):
            raise ValueError("Scene metadata repair contradicts approved character visibility.")
        if any(presence.get(gid, "offscreen") not in ("visible", "offscreen") for gid in result["speaking_entity_ids"]):
            raise ValueError("Scene metadata repair contradicts approved speech presence.")
    # Picture selection is rebuilt from approved assets, never model asset IDs.
    candidate = {**deepcopy(scene), **result}
    if candidate["location_global_id"] and candidate["location_global_id"] not in candidate["visible_entity_ids"]:
        candidate["visible_entity_ids"].insert(0, candidate["location_global_id"])
    return candidate

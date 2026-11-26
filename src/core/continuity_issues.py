"""Strict, evidence-linked continuity issues and bounded metadata corrections."""
from copy import deepcopy

from jsonschema import Draft202012Validator

from . import continuity_context
from .generation_requests import digest, response_validation




ISSUE = {"type": "object", "properties": {
    "target": {"type": "string", "enum": ["prompt", "scene_plan", "binding", "upstream_conflict"]},
    "message": {"type": "string", "minLength": 1},
    "evidence_paths": {"type": "array", "minItems": 1, "items": {"type": "string"}},
}, "required": ["target", "message", "evidence_paths"], "additionalProperties": False}
SCHEMA = {"name": "scene_continuity_review_v3", "strict": True, "schema": {
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
when there are no issues. current_prompt_sections contains the complete current
prompt wording, organized by section. Cite /current_prompt_sections for the whole
prompt or /current_prompt_sections/<section_name> for a specific section, such as
/current_prompt_sections/detailed_description. A current_prompt string is supplied
only when raw text is needed to preserve malformed or unparsed content; cite
/current_prompt only if that field is actually supplied. Do not append section
names to /current_prompt.
Select evidence_paths only from the enum supplied in the schema. Cite a parent container when a nested field is not offered. adaptation_notes belongs to /current_scene, never /current_prompt_sections. Use the supplied structured schema exactly.
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


def review_schema(payload):
    """Offer only existing evidence addresses, with bounded schema overhead.

    Container citations retain every nested value. Lists expose each record,
    and scene/section objects expose their immediate fields.
    """
    paths = []
    def escaped(key):
        return str(key).replace("~", "~0").replace("/", "~1")
    for key, value in payload.items():
        if key == "review_validation_feedback":
            continue
        root = "/" + escaped(key)
        paths.append(root)
        if isinstance(value, dict):
            paths.extend(root + "/" + escaped(field) for field in value)
        elif isinstance(value, list):
            paths.extend(root + "/" + str(index) for index in range(len(value)))
    schema = deepcopy(SCHEMA)
    schema["schema"]["properties"]["errors"]["items"]["properties"]["evidence_paths"]["items"]["enum"] = paths
    return schema


def validate(result, payload):
    if not Draft202012Validator(SCHEMA["schema"]).is_valid(result):
        raise ValueError("LM Studio returned an invalid scene continuity review.")
    issues = []
    unresolved_paths = []
    for issue in result["errors"]:
        if not issue["message"].strip():
            raise ValueError("LM Studio returned an invalid scene continuity review: empty correction.")
        evidence = []
        evidence_paths = []
        for path in issue["evidence_paths"]:
            try:
                value = resolve(payload, path)
            except (KeyError, IndexError, TypeError, ValueError):
                unresolved_paths.append(path)
                continue
            # Keep exact cited values with the issue: later repairs may run
            # after this batch is gone. No second interpretation or truncation.
            evidence.append({"path": path, "value": deepcopy(value)})
            evidence_paths.append(path)
        issues.append({**issue, "evidence_paths": evidence_paths, "evidence": evidence})
    if unresolved_paths:
        raise UnresolvedEvidenceError(unresolved_paths)
    return issues


def unique(issues):
    return list({digest(issue): issue for issue in issues}.values())


def visible_locations(row):
    return [item for item in row["catalog"]
            if item["entity_type"] == "location" and item.get("presence") == "visible"]


def location_coverage(row, bindings, excluded_images=()):
    """Diagnose approved visibility and image coverage without estimating presence."""
    locations = visible_locations(row)
    spatial = (row.get("spatial_plan") or {}).get("location_id")
    bound = {s["global_id"]: {p["asset_id"] for p in s.get("pictures", [])} for s in bindings["subjects"]}
    checks = []
    if spatial and spatial not in {item["global_id"] for item in locations}:
        checks.append({"reason": "spatial_location_not_approved", "entity_id": spatial,
                       "message": "The spatial location has no approved visible occurrence in this sequence. "
                                  "Check the resolved adaptation, extraction and consolidated identity."})
    for item in locations:
        gid = item["global_id"]
        if gid in excluded_images:
            continue
        expected = {view["asset_id"] for view in item["available_picture_views"]}
        if not expected:
            checks.append({"reason": "missing_scene_binding", "entity_id": gid,
                           "message": "The approved visible location has no explicitly scene-bound image. "
                                      "Check its appearance evidence and view plan in consolidation."})
        elif missing := (expected - bound.get(gid, set())):
            checks.append({"reason": "unassigned_location_image", "entity_id": gid,
                           "asset_ids": sorted(missing),
                           "message": "An approved location view was available but was not assigned. "
                                      "Check this scene's image budget and view selection."})
    return [{"stage": "location_reference_coverage", "status": "incomplete", **check} for check in checks]


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
        "Only request a view for a visible entity or the visible location selected for this beat. "
        "Use the canonical catalog location when it is visibly established. Choose scene-bound "
        "location views by their framing and landmarks. Request both only if the shot needs both. "
        "Submit at most one view request per entity. Omit preferences when no catalog view fits. "
        "Source, selected choices, beat, dialogue text and phases are immutable."
    ) + continuity_context.INSTRUCTIONS
    payload = {"current_scene": scene, "upstream_sequence": row, "issues": issues}
    if not continuity_context.fits(system, payload, schema):
        raise ValueError("H3 metadata repair exceeds the safe request budget; no evidence was discarded.")
    validator = Draft202012Validator(schema["schema"])

    def validate(result):
        return validate_metadata(result, row, validator)

    with response_validation(validate):
        result = request(client, model, system, continuity_context.pack(payload), schema, .1, args.max_tokens)
    visible = validate(result)
    approved = visible_locations(row)
    if len(approved) == 1:
        result = {**result, "location_global_id": approved[0]["global_id"]}
    # Picture selection is rebuilt from approved assets, never model asset IDs.
    return {**deepcopy(scene), **deepcopy(result), "visible_entity_ids": visible}


def validate_metadata(result, row, validator):
    """Validate current-scene visibility before checking image preferences."""
    if not validator.is_valid(result):
        raise ValueError("Invalid H3 scene metadata repair.")
    if not result["dialogue_present"] and result["speaking_entity_ids"]:
        raise ValueError("Scene metadata repair assigns speakers to a beat without dialogue.")
    catalog = {item["global_id"]: item for item in row["catalog"]}
    visible = list(result["visible_entity_ids"])
    location = result["location_global_id"]
    if location and location not in visible:
        visible.insert(0, location)
    for gid in visible:
        if catalog[gid].get("presence", "visible") != "visible":
            raise ValueError(f"Scene metadata selects entity {gid!r} without approved visibility in this sequence.")
    visible = list(dict.fromkeys([*visible, *(item["global_id"] for item in visible_locations(row))]))
    visible_ids = set(visible)
    seen = set()
    for requested in result["reference_view_requests"]:
        gid = requested["global_id"]
        if gid in seen:
            raise ValueError(f"Scene metadata contains duplicate view requests for entity {gid!r}.")
        if gid not in visible_ids:
            raise ValueError(f"Scene metadata requests a view for entity {gid!r} outside the visible selection.")
        views = {view["view_type"] for view in catalog[gid].get("available_picture_views", [])}
        unavailable = set(requested["view_types"]) - views
        if unavailable:
            raise ValueError(
                f"Scene metadata requests unavailable views for entity {gid!r}: "
                f"{sorted(unavailable)!r}. Allowed views: {sorted(views)!r}."
            )
        seen.add(gid)
    context = row.get("character_context")
    if context and context.get("sequence"):
        mapped = context["registry_ids"]
        presence = dict.fromkeys(mapped.values(), "absent")
        presence.update({mapped[r["local_id"]]: r["presence"] for r in context["sequence"]["characters"]})
        if any(presence.get(gid, "visible") != "visible" for gid in visible):
            raise ValueError("Scene metadata repair contradicts approved character visibility.")
        if any(presence.get(gid, "offscreen") not in ("visible", "offscreen") for gid in result["speaking_entity_ids"]):
            raise ValueError("Scene metadata repair contradicts approved speech presence.")
    return visible

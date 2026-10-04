"""Offline JSON schemas shared by editor exports and runtime import validation."""
from jsonschema import Draft202012Validator

from . import util

TEXT = {"type": "string", "minLength": 1, "pattern": r"\S"}
TRAITS = {
    "character": ["age_appearance", "skin", "facial_features", "hair", "eyes", "build", "default_outfit", "footwear", "accessories"],
    "location": ["materials", "colors", "architecture", "layout", "fixtures", "vegetation"],
    "object": ["shape", "materials", "colors", "dimensions", "construction", "markings"],
}


def obj(properties, required=None):
    return {"type": "object", "properties": properties, "required": list(properties) if required is None else required,
            "additionalProperties": False}


ADDRESS = obj({"chapter_id": TEXT, "local_id": TEXT})
LINK = obj({
    "kind": {"enum": ["identity", "attribution", "relation"]},
    "source": ADDRESS, "target": {"anyOf": [ADDRESS, {"type": "null"}]},
    "sequence": {"type": ["integer", "null"], "minimum": 1},
    "phase": {"enum": [None, "initialState", "event", "endingState"]},
    "relation": TEXT, "status": {"enum": ["proposed", "confirmed", "rejected", "unresolved"]},
    "reason": {"type": "string"}, "evidence": {"type": "array", "items": TEXT},
})
MENTION = obj({"chapter_id": TEXT, "local_id": TEXT, "canonical_name": TEXT,
               "entity_type": {"enum": list(TRAITS)}, "classification": {"enum": ["entity", "manifestation"]}})
LINK_VERSION = "minimax-h3-reference-links.v1"
LINK_SCHEMA = obj({"$schema": {"type": "string"}, "schema_version": {"const": LINK_VERSION},
                   "source_digest": TEXT, "entities": {"type": "array", "items": MENTION},
                   "links": {"type": "array", "items": LINK}},
                  ["schema_version", "source_digest", "entities", "links"])
LINK_SCHEMA["properties"]["links"]["items"] = {**LINK, "allOf": [
    {"if": {"properties": {"kind": {"const": "identity"}}}, "then": {"properties": {
        "target": ADDRESS, "relation": {"const": "same_as"}, "sequence": {"type": "null"}, "phase": {"type": "null"}}}},
    {"if": {"properties": {"kind": {"const": "attribution"}}}, "then": {"properties": {
        "sequence": {"type": "integer", "minimum": 1},
        "phase": {"enum": ["initialState", "event", "endingState"]}}}},
    {"if": {"properties": {"kind": {"const": "relation"}}}, "then": {"properties": {"target": ADDRESS}}},
    {"if": {"properties": {"status": {"const": "confirmed"}}}, "then": {"properties": {"target": ADDRESS}}},
]}
DESIGN_SCHEMA = obj({
    "$schema": {"type": "string"}, "schema_version": {"const": "minimax-h3-visual-designs.v1"},
    "image_style": TEXT,
    "entities": {"type": "array", "items": {
        **obj({"global_id": TEXT, "entity_type": {"enum": list(TRAITS)}, "canonical_name": TEXT,
               "source_facts": obj({"stable_visual_description": {"type": "string"},
                                    "distinguishing_features": {"type": "array", "items": {"type": "string"}}}),
               "added_details": {"type": "object"}}),
        "allOf": [{"if": {"properties": {"entity_type": {"const": kind}}},
                   "then": {"properties": {"added_details": obj({trait: TEXT for trait in traits}, [])}}}
                  for kind, traits in TRAITS.items()],
    }},
}, ["schema_version", "image_style", "entities"])


def validate_document(payload, schema, filename):
    # Local schema objects only: never retrieve an editable document's $schema URL.
    error = next(Draft202012Validator(schema).iter_errors(payload), None)
    if error:
        path = "".join(f"[{x}]" if isinstance(x, int) else f".{x}" for x in error.absolute_path).lstrip(".")
        raise ValueError(f"{filename}: {path or '$'}: {error.message}")


def export_schemas(output):
    for name, schema in (("reference_links", LINK_SCHEMA), ("visual_designs", DESIGN_SCHEMA)):
        util.save_json(output / f"{name}.schema.json", {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema})

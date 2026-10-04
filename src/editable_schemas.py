"""Bundled JSON schemas shared by editor exports and runtime import validation."""
import json
from pathlib import Path
from shutil import copyfile

from jsonschema import Draft202012Validator

from .path_access import output_path

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
LINK_SCHEMA = json.loads((SCHEMA_DIR / "reference_links.schema.json").read_text(encoding="utf-8"))
DESIGN_SCHEMA = json.loads((SCHEMA_DIR / "visual_designs.schema.json").read_text(encoding="utf-8"))
LINK_VERSION = LINK_SCHEMA["properties"]["schema_version"]["const"]
TRAITS = {
    rule["if"]["properties"]["entity_type"]["const"]:
        list(rule["then"]["properties"]["added_details"]["properties"])
    for rule in DESIGN_SCHEMA["properties"]["entities"]["items"]["allOf"]
}


def obj(properties, required=None):
    return {"type": "object", "properties": properties, "required": list(properties) if required is None else required,
            "additionalProperties": False}


def validate_document(payload, schema, filename):
    # Local schema objects only: never retrieve an editable document's $schema URL.
    error = next(Draft202012Validator(schema).iter_errors(payload), None)
    if error:
        path = "".join(f"[{x}]" if isinstance(x, int) else f".{x}" for x in error.absolute_path).lstrip(".")
        raise ValueError(f"{filename}: {path or '$'}: {error.message}")


def export_schemas(output):
    output = output_path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("reference_links", "visual_designs"):
        filename = f"{name}.schema.json"
        copyfile(SCHEMA_DIR / filename, output_path(output / filename))

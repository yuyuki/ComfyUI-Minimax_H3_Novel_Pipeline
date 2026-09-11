"""Bounded semantic validation on top of the existing JSON request retries."""
import json

from . import lmstudio_json
from .lmstudio_pipeline import comfy_interrupt_check


def validated_request(chat, client, model, system, user, schema, args, validate):
    original = user if isinstance(user, str) else json.dumps(user, ensure_ascii=False)
    correction = ""
    for attempt in range(lmstudio_json.QWEN35_LENGTH_RETRIES + 1):
        comfy_interrupt_check()
        result = chat(client, model, system, original + correction, schema, args.temperature, args.max_tokens)
        try:
            validate(result)
            return result
        except ValueError as exc:
            if attempt == lmstudio_json.QWEN35_LENGTH_RETRIES:
                raise ValueError(f"Invalid {schema['name']} response after bounded retries: {exc}") from exc
            correction = f"\nPrevious response was invalid: {exc}. Return the complete corrected JSON, including every requested ID."


def validate_assets(result, specs):
    expected = {spec["asset_id"] for spec in specs}
    items = result.get("assets")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("assets must be a list of objects")
    ids = [item.get("asset_id") for item in items]
    if not all(isinstance(value, str) for value in ids):
        raise ValueError("Every asset requires a string asset_id")
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Return exactly once each asset_id: " + ", ".join(sorted(expected)))
    for item in items:
        for field in ("description", "generation_prompt"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"{item['asset_id']} needs non-empty {field}")

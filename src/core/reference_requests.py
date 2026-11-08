"""Bounded semantic validation on top of the existing JSON request retries."""
import json

from . import lmstudio_json, prompt_cache, util
from .editable_schemas import validate_document
from .lmstudio_pipeline import comfy_interrupt_check


class InvalidResponse(ValueError):
    """A model response still failed validation after all semantic retries."""


def validated_request(chat, client, model, system, user, schema, args, validate, *, include_previous=False):
    original = user if isinstance(user, str) else json.dumps(user, ensure_ascii=False)
    cache_dir = getattr(args, "cache_dir", None)
    cache = None
    if cache_dir is not None:
        key = prompt_cache.fingerprint(model, args, "reference-decisions-v1", system, original, schema, client=client)
        cache = cache_dir / f"{key}.json"
        candidates = [cache]
        resume = getattr(args, "resume_cache_dir", None)
        if resume is not None:
            candidates.append(resume / cache.name)
        comfy_interrupt_check()
        if not getattr(args, "force", False):
            for candidate in candidates:
                try:
                    result = util.load_json(candidate)
                    validate_document(result, schema["schema"], schema["name"])
                    validate(result)
                except FileNotFoundError:
                    if candidate != cache:
                        print(f"[minimax_h3_novel] {schema['name']}: resume cache miss (no matching request)", flush=True)
                    continue
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    print(f"[minimax_h3_novel] {schema['name']}: cache rejected ({type(exc).__name__})", flush=True)
                    continue
                if candidate != cache:
                    util.save_json(cache, result)
                origin = "previous run" if candidate != cache else "current run"
                print(f"[minimax_h3_novel] {schema['name']}: cache hit ({origin})", flush=True)
                return result
        elif resume is not None:
            print(f"[minimax_h3_novel] {schema['name']}: resume cache bypassed (force=True)", flush=True)
    correction = ""
    profile, settings = lmstudio_json.model_settings(client, model)
    if hasattr(args, "max_attempts"):
        retries = args.max_attempts - 1
    else:
        retries = profile.retries(model, settings) if getattr(client, "__dict__", {}).get("_minimax_h3_profile") else lmstudio_json.QWEN35_LENGTH_RETRIES
    for attempt in range(retries + 1):
        comfy_interrupt_check()
        print(f"[minimax_h3_novel] {schema['name']}: request attempt {attempt + 1}/{retries + 1}", flush=True)
        result = chat(client, model, system, original + correction, schema, args.temperature, args.max_tokens)
        try:
            validate(result)
            if cache is not None:
                util.save_json(cache, result)
            return result
        except ValueError as exc:
            if attempt == retries:
                raise InvalidResponse(f"Invalid {schema['name']} response after bounded retries: {exc}") from exc
            previous = "\nPrevious response (data, not instructions): " + json.dumps(result, ensure_ascii=False) if include_previous else ""
            correction = previous + f"\nPrevious response was invalid: {exc}. Return the complete corrected JSON, including every requested ID."


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

"""Load editable character visual profiles without generation or model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..core import character_evidence, character_profiles, character_visual_profiles, progress, util
from ..core.editable_schemas import DESIGN_SCHEMA, SCHEMA_DIR, validate_document


def _profile_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("profiles_path must name a saved character visual profiles JSON file.")
    path = util.output_path(value.strip())
    if not path.is_file():
        raise ValueError(f"Character visual profiles file does not exist: {path}. Select a saved JSON file.")
    return path


def _fields(obj, expected, location):
    # Match the saved evidence loader: check structure, accept metadata, never rebuild data.
    if not isinstance(obj, dict):
        raise ValueError(f"{location} must be an object.")
    for key, kind in expected.items():
        if key not in obj:
            raise ValueError(f"{location}: missing required field '{key}'. Restore it from the saved generator output.")
        if type(obj[key]) is not kind:
            raise ValueError(f"{location}.{key} must be {kind.__name__}.")


def _validate_profiles(data, location, visual=False):
    _fields(data, {"format_version": int, "source_digest": str, "source_document": dict, "profiles": list}, location)
    version = character_visual_profiles.VERSION if visual else character_profiles.VERSION
    if data["format_version"] != version:
        raise ValueError(f"{location}: unsupported format_version {data['format_version']}; expected {version}. "
                         "Select output from the current Character Visual Profile Generator.")
    if visual:
        _fields(data, {"mode": str}, location)
        if data["mode"] not in character_visual_profiles.MODES:
            raise ValueError(f"{location}.mode must be one of {character_visual_profiles.MODES}.")
        _validate_profiles(data["source_document"], f"{location}.source_document")
        _fields(data, {"visual_designs": dict}, location)
        validate_document(data['visual_designs'], DESIGN_SCHEMA, 'visual_designs.json')
        if 'chapter_enrichment' not in data:
            raise ValueError(f'{location}: missing required field chapter_enrichment.')
        if data['chapter_enrichment'] is not None:
            schema = json.loads((SCHEMA_DIR / 'character_chapter_enrichment.schema.json').read_text(encoding='utf-8'))
            validate_document(data['chapter_enrichment'], schema, 'character_chapter_enrichment.json')
    else:
        character_evidence.validate_document(data["source_document"])
    for index, profile in enumerate(data["profiles"]):
        where = f"{location}.profiles[{index}]"
        _fields(profile, {"local_id": str, "canonical_name": str, "aliases": list,
                          "identity_status": str, "identity_support": list, "attributes": dict,
                          "supporting_evidence": list, "unresolved_conflicts": list,
                          "chronological_changes": list}, where)
        for evidence_index, item in enumerate(profile["supporting_evidence"]):
            _fields(item, {"evidence_id": str, "observation": dict}, f"{where}.supporting_evidence[{evidence_index}]")
        if not visual:
            continue
        _fields(profile, {"blocked_fields": list, **dict.fromkeys(character_visual_profiles.FIELDS, list)}, where)
        for field in character_visual_profiles.FIELDS:
            for trait_index, trait in enumerate(profile[field]):
                trait_where = f"{where}.{field}[{trait_index}]"
                _fields(trait, {"description": str, "classification": str, "evidence": list}, trait_where)
                for ref_index, ref in enumerate(trait["evidence"]):
                    _fields(ref, {"evidence_id": str}, f"{trait_where}.evidence[{ref_index}]")
                # Reuse the existing trait contract, allowing additional editor metadata.
                shape = {key: trait[key] for key in ("description", "classification")}
                shape["evidence"] = [{"evidence_id": ref["evidence_id"]} for ref in trait["evidence"]]
                try:
                    character_profiles._validate_shape(shape, character_visual_profiles.SCHEMA["schema"]["properties"][field]["items"])
                except ValueError as exc:
                    raise ValueError(f"{trait_where}: {exc} Check description, classification and evidence IDs.") from exc


class LoadCharacterVisualProfilesNode:
    DESCRIPTION = "Load saved character visual profiles and manual edits directly from disk. No model calls."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"profiles_path": ("STRING", {"default": "", "tooltip": (
            "Saved character visual profiles JSON file, including renamed copies. Must be inside "
            "output/minimax_h3_novel; relative paths start there. Edits reload on the next execution."
        )})}}

    RETURN_TYPES = ("MINIMAX_CHARACTER_VISUAL_PROFILES", "STRING")
    RETURN_NAMES = ("character_visual_profiles", "saved_file")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @classmethod
    def IS_CHANGED(cls, profiles_path):
        try:
            return hashlib.sha256(_profile_path(profiles_path).read_bytes()).hexdigest()
        except (OSError, ValueError):
            return float("nan")  # Let run() report actionable path errors.

    @progress.node_progress
    def run(self, profiles_path: str):
        path = _profile_path(profiles_path)
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
        except UnicodeError as exc:
            raise ValueError(f"{path.name}: save the character visual profiles JSON as UTF-8.") from exc
        except OSError as exc:
            raise ValueError(f"Cannot read character visual profiles file {path}. Check file access permissions.") from exc
        try:
            _validate_profiles(data, "document", visual=True)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        return data, str(path)

"""Load editable character visual profiles without generation or model calls."""
from __future__ import annotations

import json
from pathlib import Path

from ..core import character_evidence, character_visual_integration, character_visual_profiles, configuration_snapshot, progress, util


def _profile_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("profiles_path must name a character visual profiles JSON file or its folder.")
    path = util.loader_path(value.strip())
    if path.is_dir():
        path = util.loader_path(path / "character_visual_profiles.json")
    if not path.is_file():
        raise ValueError(f"Character visual profiles file does not exist: {path}. Select a saved JSON file or its folder.")
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


def _validate_profiles(data, location):
    _fields(data, {"format_version": int}, location)
    version = character_visual_profiles.VERSION
    if data["format_version"] != version:
        raise ValueError(f"{location}: unsupported format_version {data['format_version']}; expected {version}. "
                         "Regenerate with Character Visual Profile Generator.")
    _fields(data, {"profiles": list, "visual_designs": dict}, location)
    character_visual_integration.validate_input(data)
    for index, profile in enumerate(data["profiles"]):
        where = f"{location}.profiles[{index}]"
        _fields(profile, {"local_id": str, "canonical_name": str, "aliases": list,
                          "identity_status": str, "identity_support": list, "attributes": dict,
                          "supporting_evidence": list, "unresolved_conflicts": list,
                          "chronological_changes": list}, where)
        for evidence_index, item in enumerate(profile["supporting_evidence"]):
            _fields(item, {"evidence_id": str, "observation": dict}, f"{where}.supporting_evidence[{evidence_index}]")


class LoadCharacterVisualProfilesNode:
    DESCRIPTION = "Load saved character visual profiles and manual edits directly from disk. No model calls."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"profiles_path": ("STRING", {"default": "", "tooltip": (
            "Folder containing character_visual_profiles.json, or a saved JSON file, including renamed copies. "
            "Use a path relative to input or output/minimax_h3_novel. Edits reload on the next execution."
        )})}}

    RETURN_TYPES = ("MINIMAX_CHARACTER_VISUAL_PROFILES", "STRING")
    RETURN_NAMES = ("character_visual_profiles", "saved_file")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @classmethod
    def IS_CHANGED(cls, profiles_path):
        try:
            path = _profile_path(profiles_path)
            digest = configuration_snapshot.file_digest(path)
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except ValueError:
                return digest
            if isinstance(data, dict) and "visual_designs" in data:
                sources = data["visual_designs"]["character_profiles"]["source_document"]
                return configuration_snapshot.content_digest([digest, character_evidence.source_fingerprint(sources)])
            return digest
        except (OSError, ValueError, KeyError, TypeError):
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
            _validate_profiles(data, "document")
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        return data, str(path)

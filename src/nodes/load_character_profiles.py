"""Load editable character profiles without extraction or model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..core import progress, util
from ..core.character_profile_document import validate_document


def _profiles_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("profiles_path must name a saved character profiles JSON file or its directory.")
    path = util.output_path(value.strip())
    if path.is_dir():
        path = util.output_path(path / "character_profiles.json")
    if not path.is_file():
        raise ValueError(f"Character profiles file does not exist: {path}. Select a saved JSON file or its directory.")
    return path




class LoadCharacterProfilesNode:
    DESCRIPTION = "Load saved character profiles and manual edits directly from disk. No model calls."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"profiles_path": ("STRING", {"default": "", "tooltip": (
            "Saved character profiles JSON file (including renamed copies), or a directory containing "
            "character_profiles.json. Must be inside "
            "output/minimax_h3_novel; relative paths start there. Edits reload on the next execution."
        )})}}

    RETURN_TYPES = ("MINIMAX_CHARACTER_PROFILES", "STRING")
    RETURN_NAMES = ("character_profiles", "saved_file")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @classmethod
    def IS_CHANGED(cls, profiles_path):
        try:
            return hashlib.sha256(_profiles_path(profiles_path).read_bytes()).hexdigest()
        except (OSError, ValueError):
            return float("nan")  # Let run() report actionable path errors.

    @progress.node_progress
    def run(self, profiles_path: str):
        path = _profiles_path(profiles_path)
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
        except UnicodeError as exc:
            raise ValueError(f"{path.name}: save the character profiles JSON as UTF-8.") from exc
        except OSError as exc:
            raise ValueError(f"Cannot read character profiles file {path}. Check file access permissions.") from exc
        try:
            validate_document(data)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        return data, str(path)

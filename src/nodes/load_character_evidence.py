"""Load editable character evidence without extraction or model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..core import progress, util
from ..core.character_evidence import validate_document


def _evidence_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evidence_path must name a saved character evidence JSON file.")
    path = util.output_path(value.strip())
    if not path.is_file():
        raise ValueError(f"Character evidence file does not exist: {path}. Select a saved JSON file.")
    return path




class LoadCharacterEvidenceNode:
    DESCRIPTION = "Load saved character evidence and manual edits directly from disk. No model calls."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"evidence_path": ("STRING", {"default": "", "tooltip": (
            "Saved character evidence JSON file, including renamed copies. Must be inside "
            "output/minimax_h3_novel; relative paths start there. Edits reload on the next execution."
        )})}}

    RETURN_TYPES = ("MINIMAX_CHARACTER_EVIDENCE", "STRING")
    RETURN_NAMES = ("character_evidence", "saved_file")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @classmethod
    def IS_CHANGED(cls, evidence_path):
        try:
            return hashlib.sha256(_evidence_path(evidence_path).read_bytes()).hexdigest()
        except (OSError, ValueError):
            return float("nan")  # Let run() report actionable path errors.

    @progress.node_progress
    def run(self, evidence_path: str):
        path = _evidence_path(evidence_path)
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
        except UnicodeError as exc:
            raise ValueError(f"{path.name}: save the character evidence JSON as UTF-8.") from exc
        except OSError as exc:
            raise ValueError(f"Cannot read character evidence file {path}. Check file access permissions.") from exc
        try:
            validate_document(data)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        return data, str(path)

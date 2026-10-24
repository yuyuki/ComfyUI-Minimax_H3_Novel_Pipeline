"""Load editable character evidence without extraction or model calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..core import progress, util
from ..core.character_evidence import VERSION


def _evidence_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("evidence_path must name a saved character evidence JSON file.")
    path = util.output_path(value.strip())
    if not path.is_file():
        raise ValueError(f"Character evidence file does not exist: {path}. Select a saved JSON file.")
    return path


def _validate_document(data) -> None:
    """Check the saved envelope, retaining even extractor observations flagged for review.

    The existing response validator annotates LLM observations and is not a saved
    document validator. Do not re-run it or rebuild provenance on manually edited data.
    """
    def fields(obj, expected, location):
        if not isinstance(obj, dict):
            raise ValueError(f"{location} must be an object.")
        for key, kind in expected.items():
            if key not in obj:
                raise ValueError(f"{location}: missing required field '{key}'. Restore it from the saved extractor output.")
            if type(obj[key]) is not kind:
                raise ValueError(f"{location}.{key} must be {kind.__name__}.")

    fields(data, {"format_version": int, "location_convention": str, "chapters": list}, "document")
    if data["format_version"] != VERSION:
        raise ValueError(f"Unsupported character evidence format_version {data['format_version']}; "
                         f"expected {VERSION}. Select output from the current Character Evidence Extractor.")
    for index, chapter in enumerate(data["chapters"]):
        where = f"chapters[{index}]"
        fields(chapter, {"chapter": int, "chapter_name": str, "source_file": str,
                         "source_text": str, "source_text_sha256": str, "location_basis": str,
                         "evidence": list}, where)
        for item_index, item in enumerate(chapter["evidence"]):
            item_where = f"{where}.evidence[{item_index}]"
            fields(item, {"provenance": dict}, item_where)
            fields(item["provenance"], {"chapter": int, "chapter_name": str, "file": str,
                                        "passage": int, "locations": list}, f"{item_where}.provenance")
            for loc_index, location in enumerate(item["provenance"]["locations"]):
                fields(location, dict.fromkeys(("start_char", "end_char", "start_line", "end_line"), int),
                       f"{item_where}.provenance.locations[{loc_index}]")


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
            _validate_document(data)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        return data, str(path)

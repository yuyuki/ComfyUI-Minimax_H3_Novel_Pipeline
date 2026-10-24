"""Load a manually corrected spatial plan without calling LM Studio."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..core import progress, util
from .spatial_continuity import validate_plan


def _plan_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("spatial_path must name a spatial JSON file or its folder.")
    path = util.output_path(value.strip())
    if path.is_dir():
        path = util.output_path(path / "spatial_continuity.json")
    if not path.is_file():
        raise ValueError(f"Spatial continuity file does not exist: {path}")
    return path


class LoadSpatialContinuityNode:
    DESCRIPTION = "Load and validate a manually corrected spatial JSON against its consolidation. No model calls."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "spatial_path": ("STRING", {"default": "", "tooltip": (
                "JSON file (including renamed copies), or folder containing spatial_continuity.json. "
                "Must be inside output/minimax_h3_novel; relative paths start there. "
                "Saved edits are detected on the next workflow execution."
            )}),
            "consolidated_references": ("MINIMAX_REGISTRY",),
        }}

    RETURN_TYPES = ("MINIMAX_SPATIAL_CONTINUITY", "STRING", "STRING")
    RETURN_NAMES = ("spatial_continuity", "spatial_json", "summary")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @classmethod
    def IS_CHANGED(cls, spatial_path: str, consolidated_references=None):
        # Hash content rather than timestamps: editors can preserve size and mtime.
        # Upstream socket changes are tracked separately by ComfyUI.
        try:
            return hashlib.sha256(_plan_path(spatial_path).read_bytes()).hexdigest()
        except (OSError, ValueError):
            return float("nan")  # Force run() to report missing/invalid paths.

    @progress.node_progress
    def run(self, spatial_path: str, consolidated_references: dict):
        util.require_schema(consolidated_references, util.REGISTRY_SCHEMA)
        path = _plan_path(spatial_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path.name}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc
        try:
            validate_plan(payload, consolidated_references)
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        count = sum(len(chapter["sequences"]) for chapter in payload["chapters"].values())
        summary = f"Loaded spatial continuity: {count} sequences from {path}"
        return payload, json.dumps(payload, ensure_ascii=False, indent=2), summary

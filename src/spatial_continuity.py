"""Operator-approved spatial anchors for the scene continuity pass."""
from __future__ import annotations

import json
from typing import Any

from . import progress


class SpatialContinuityNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"anchors_json": ("STRING", {"multiline": True, "default": "{\n  \"tablet.wall\": \"right wall\",\n  \"main_rope.side\": \"right side of crevasse\"\n}"})}}

    RETURN_TYPES = ("MINIMAX_SPATIAL_CONTINUITY", "STRING")
    RETURN_NAMES = ("spatial_continuity", "anchor_summary")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @progress.node_progress
    def run(self, anchors_json: str) -> tuple[dict[str, Any], str]:
        try:
            anchors = json.loads(anchors_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid spatial anchors JSON: {exc}. Enter a JSON object of stable relations, e.g. {{\"tablet.wall\": \"right wall\"}}.") from exc
        if not isinstance(anchors, dict) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
            for key, value in anchors.items()
        ):
            raise ValueError("Spatial anchors must be a JSON object with non-empty string keys and values.")
        return {"schema_version": "minimax-spatial-continuity.v1", "anchors": anchors}, "\n".join(f"{k}: {v}" for k, v in anchors.items())

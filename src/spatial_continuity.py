"""Enable model-driven continuity using the consolidated visual designs."""
from __future__ import annotations

import json
from typing import Any

from . import progress


class SpatialContinuityNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("MINIMAX_SPATIAL_CONTINUITY", "STRING")
    RETURN_NAMES = ("spatial_continuity", "anchor_summary")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @progress.node_progress
    def run(self, anchors_json: str = "{}") -> tuple[dict[str, Any], str]:
        # Retain direct-call compatibility; new workflows need no JSON settings.
        try:
            anchors = json.loads(anchors_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid spatial anchors JSON: {exc}. Enter a JSON object of stable relations, e.g. {{\"tablet.wall\": \"right wall\"}}.") from exc
        if not isinstance(anchors, dict) or any(
            not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value.strip()
            for key, value in anchors.items()
        ):
            raise ValueError("Spatial anchors must be a JSON object with non-empty string keys and values.")
        summary = "\n".join(f"{k}: {v}" for k, v in anchors.items())
        return {"schema_version": "minimax-spatial-continuity.v1", "anchors": anchors}, summary or "Automatic scene continuity enabled using consolidated visual designs."

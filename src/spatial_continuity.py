"""Enable model-driven continuity using the consolidated visual designs."""
from __future__ import annotations

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
    def run(self) -> tuple[dict[str, Any], str]:
        return {"schema_version": "minimax-spatial-continuity.v1", "anchors": {}}, "Automatic scene continuity enabled using consolidated visual designs."

"""ComfyUI entrypoint for the MiniMax H3 Novel Pipeline custom node.

ComfyUI-Manager clones this repository directly into ``custom_nodes`` and
therefore requires this file at the repository root. The implementation stays
in ``src/minimax_h3_novel_pipeline``.
"""
from __future__ import annotations

from .src.minimax_h3_novel_pipeline import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    comfy_entrypoint,
)

# This path is relative to this root custom-node package.
WEB_DIRECTORY = "./web/js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "comfy_entrypoint",
]

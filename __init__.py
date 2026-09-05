"""ComfyUI entrypoint for the MiniMax H3 Novel Pipeline custom node.

ComfyUI-Manager clones this repository directly into ``custom_nodes`` and
therefore requires this file at the repository root. The implementation stays
in ``src``.
"""
from __future__ import annotations

if __package__:
    from .src import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
else:
    # Pytest imports this entrypoint as a bare module in hyphenated checkouts.
    from src import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )

# This path is relative to this root custom-node package.
WEB_DIRECTORY = "./web/js"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

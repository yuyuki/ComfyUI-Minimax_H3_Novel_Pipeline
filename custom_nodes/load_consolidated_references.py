"""ComfyUI node for reusing a saved Step 2 consolidation output."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import util


class LoadConsolidatedReferencesNode:
    """Load a saved consolidated registry for prompt generation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "consolidated_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Folder containing consolidated_references.json, or "
                            "the consolidated_references.json file itself."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MINIMAX_REGISTRY",)
    RETURN_NAMES = ("consolidated_references",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(
        self, consolidated_path: str
    ) -> tuple[dict[str, Any]]:
        if not isinstance(consolidated_path, str) or not consolidated_path.strip():
            raise ValueError(
                "consolidated_path must name a consolidation output folder or "
                "consolidated_references.json file."
            )

        supplied_path = Path(consolidated_path.strip()).expanduser()
        json_path = (
            supplied_path / "consolidated_references.json"
            if supplied_path.is_dir()
            else supplied_path
        )
        if not json_path.is_file():
            raise ValueError(f"Consolidated references file does not exist: {json_path}")
        if json_path.name != "consolidated_references.json":
            raise ValueError(
                "Expected a file named consolidated_references.json, or a folder containing it: "
                f"{json_path}"
            )

        registry = util.load_json(json_path)
        if not isinstance(registry, dict):
            raise ValueError(f"{json_path.name}: expected a JSON object.")

        required = {"entities", "chapter_entity_map", "entity_asset_index", "picture_assets", "audio_assets"}
        missing = sorted(required - registry.keys())
        if missing:
            raise ValueError(
                f"{json_path.name}: this is not a complete consolidation output; "
                "missing: " + ", ".join(missing)
            )
        pictures = registry["picture_assets"]
        audio = registry["audio_assets"]
        if not isinstance(pictures, list) or not all(isinstance(item, dict) for item in pictures):
            raise ValueError(f"{json_path.name}: picture_assets must be a list of objects.")
        if not isinstance(audio, list) or not all(isinstance(item, dict) for item in audio):
            raise ValueError(f"{json_path.name}: audio_assets must be a list of objects.")

        print(
            "[minimax_h3_novel] Loaded consolidated references from "
            f"{json_path} ({len(pictures)} picture and {len(audio)} audio brief(s))",
            flush=True,
        )
        return (registry,)

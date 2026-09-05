"""ComfyUI node for reusing chapter catalogs saved by Step 1."""
from __future__ import annotations

from typing import Any

from . import util


class LoadChapterCatalogsNode:
    """Load saved ``*_references.json`` files for the consolidation node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "catalog_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Folder containing *_references.json files, or one "
                            "such JSON file saved by Extract Chapter References. "
                            "Must be inside output/minimax_h3_novel; relative paths start there."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MINIMAX_CHAPTERS", "STRING")
    RETURN_NAMES = ("chapter_catalogs", "catalog_summary")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(self, catalog_path: str) -> tuple[list[dict[str, Any]], str]:
        if not isinstance(catalog_path, str) or not catalog_path.strip():
            raise ValueError("catalog_path must name a saved chapter catalog JSON file or folder.")

        path = util.output_path(catalog_path.strip())
        if path.is_file():
            paths = [path]
        elif path.is_dir():
            paths = sorted(
                (
                    candidate
                    for candidate in path.glob("*_references.json")
                    if candidate.is_file() and candidate.name != "consolidated_references.json"
                ),
                key=lambda candidate: util.natural_key(candidate.name),
            )
        else:
            raise ValueError(f"Catalog path does not exist: {path}")

        if not paths:
            raise ValueError(f"No *_references.json files found in: {path}")

        catalogs: list[dict[str, Any]] = []
        for json_path in paths:
            data = util.load_json(json_path)
            util.require_schema(data, util.CHAPTER_SCHEMA)
            if not isinstance(data, dict):
                raise ValueError(f"{json_path.name}: expected a chapter catalog JSON object.")
            if not data.get("chapter_id") or not data.get("schema_version"):
                raise ValueError(
                    f"{json_path.name}: this is not a valid saved chapter catalog "
                    "(chapter_id and schema_version are required)."
                )
            catalogs.append(data)

        print(f"[minimax_h3_novel] Loaded {len(catalogs)} chapter catalog(s) from {path}", flush=True)
        return catalogs, util.catalog_summary(catalogs)

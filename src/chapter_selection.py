"""Reusable chapter picker for MiniMax H3 pipeline nodes."""
from __future__ import annotations

from . import progress

from pathlib import Path
from typing import Any

from . import util


def saved_chapter_choices() -> list[str]:
    try:
        import folder_paths

        root = Path(folder_paths.get_input_directory()) / "minimax_h3_novel"
        files = sorted(
            (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in util.SUPPORTED_EXTENSIONS),
            key=lambda path: path.name.lower(),
        ) if root.is_dir() else []
        return [""] + [f"minimax_h3_novel/{path.name}" for path in files]
    except Exception:
        return [""]


def chapter_path_list(selection: Any) -> list[str]:
    """Return paths from the structured Select Chapters payload."""
    if not isinstance(selection, dict):
        raise TypeError("chapter_selection must come from Select Chapters.")
    paths = selection.get("chapter_paths")
    if not isinstance(paths, list) or any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError("chapter_selection requires a chapter_paths list of non-empty strings.")
    return paths


class SelectChaptersNode:
    """Choose uploaded chapters once and share that selection downstream."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "chapter_paths": ("STRING", {"multiline": True, "default": "", "tooltip": "One chapter file or folder per line, inside ComfyUI's input directory. Relative paths start there."}),
            "saved_chapter": (saved_chapter_choices(), {"tooltip": "Previously uploaded chapter."}),
        }}

    RETURN_TYPES = ("MINIMAX_CHAPTER_SELECTION",)
    RETURN_NAMES = ("chapter_selection",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @progress.node_progress
    def run(self, chapter_paths: str, saved_chapter: str) -> tuple[dict[str, list[str]]]:
        """Return a structured chapter-path list for downstream nodes."""
        paths = [path.strip() for path in chapter_paths.splitlines() if path.strip()]
        if not paths and saved_chapter.strip():
            paths = [saved_chapter.strip()]
        return ({"chapter_paths": paths},)

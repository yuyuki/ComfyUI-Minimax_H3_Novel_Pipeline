"""Reusable chapter picker for MiniMax H3 pipeline nodes."""
from __future__ import annotations

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


def chapter_paths(selection: Any, saved_chapter: str = "") -> str:
    """Return paths from the shared selection or a single saved-chapter fallback."""
    if isinstance(selection, dict):
        selection = selection.get("chapter_paths", "")
    if isinstance(selection, (list, tuple)):
        selection = "\n".join(str(path) for path in selection)
    if isinstance(selection, str) and selection.strip():
        return selection
    return str(saved_chapter or "")


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

    def run(self, chapter_paths: str, saved_chapter: str) -> tuple[dict[str, str]]:
        return ({"chapter_paths": chapter_paths or saved_chapter or ""},)

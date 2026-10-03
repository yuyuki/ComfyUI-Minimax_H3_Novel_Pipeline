"""ComfyUI node for reusing saved cinematic chapter adaptations."""
from __future__ import annotations

from . import cinematic_references, progress, util


class LoadCinematicChaptersNode:
    """Load adapter JSON files for Extract Chapter References."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "cinematic_path": ("STRING", {
                "default": "",
                "tooltip": (
                    "Folder containing *.cinematic.json files, or one saved cinematic chapter JSON file. "
                    "Must be inside output/minimax_h3_novel; relative paths start there."
                ),
            }),
        }}

    RETURN_TYPES = ("MINIMAX_CINEMATIC_CHAPTERS", "STRING")
    RETURN_NAMES = ("cinematic_chapters", "saved_files")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @progress.node_progress
    def run(self, cinematic_path: str) -> tuple[list[dict], str]:
        if not isinstance(cinematic_path, str) or not cinematic_path.strip():
            raise ValueError("cinematic_path must name a saved cinematic chapter JSON file or folder.")
        path = util.output_path(cinematic_path.strip())
        if path.is_file():
            paths = [path]
        elif path.is_dir():
            paths = sorted(
                (candidate for candidate in path.glob("*.cinematic.json") if candidate.is_file()),
                key=lambda candidate: util.natural_key(candidate.name),
            )
        else:
            raise ValueError(f"Cinematic chapter path does not exist: {path}")
        if not paths:
            raise ValueError(f"No *.cinematic.json files found in: {path}")

        chapters, saved_files = [], []
        for json_path in progress.steps(paths):
            json_path = util.output_path(json_path)
            try:
                chapter = cinematic_references.parse_chapters([util.load_json(json_path)])[0]
            except ValueError as exc:
                raise ValueError(f"{json_path.name}: {exc}") from exc
            chapter["saved_file"] = str(json_path)
            chapters.append(chapter)
            saved_files.append(str(json_path))
        return chapters, "\n".join(saved_files)

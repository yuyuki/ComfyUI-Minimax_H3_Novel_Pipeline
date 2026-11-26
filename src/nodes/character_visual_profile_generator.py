"""ComfyUI entry point for stable character visual profiles."""
from __future__ import annotations

from pathlib import Path
from ..core import character_visual_profiles, progress, util, visual_designs
from ..core.path_access import input_path
from ..core.run_output import stage_output
from .chapter_selection import chapter_path_list


class CharacterVisualProfileGeneratorNode:
    DESCRIPTION = "Project consolidated character identities and permanent physical traits into visual designs."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "character_profiles": ("MINIMAX_CHARACTER_PROFILES",),
            "chapter_selection": ("MINIMAX_CHAPTER_SELECTION",),
        }, "optional": {
            "visual_designs_path": ("STRING", {"default": "", "tooltip": "Reuse edited physical defaults."}),
            "out_dir": ("STRING", {"default": "character_visual_profiles"}),
        }}

    RETURN_TYPES = ("MINIMAX_CHARACTER_VISUAL_PROFILES", "STRING")
    RETURN_NAMES = ("character_visual_profiles", "saved_file")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @progress.node_progress
    def run(self, character_profiles, chapter_selection, lmstudio_config,
            out_dir="character_visual_profiles", visual_designs_path=""):
        if not isinstance(out_dir, str) or not out_dir.strip():
            raise ValueError("out_dir must be a non-empty relative subfolder.")
        selected = [input_path(Path(p)) for p in chapter_path_list(chapter_selection)]
        for path in selected:
            if not path.exists():
                raise ValueError(f"Selected character source is missing: {path}. Restore it and retry.")
        paths = util.discover_inputs(selected)
        designs_path = visual_designs.resolve_designs_path(visual_designs_path)
        previous = util.load_json(designs_path) if designs_path else None
        output = stage_output(lmstudio_config, out_dir.strip())
        data, saved = character_visual_profiles.generate(character_profiles, paths, output, previous)
        return {"ui": {"text": [saved]}, "result": (data, saved)}

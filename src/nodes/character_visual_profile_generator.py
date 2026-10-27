"""ComfyUI entry point for stable character visual profiles."""
from __future__ import annotations

import argparse

from ..core import character_visual_profiles, character_chapter_enrichment, lmstudio_pipeline, progress, util, visual_designs
from ..core.run_output import stage_output


class CharacterVisualProfileGeneratorNode:
    DESCRIPTION = "Generate evidence-linked English character designs with Qwen, preserving canon and unresolved conflicts."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "character_profiles": ("MINIMAX_CHARACTER_PROFILES",),
            "cinematic_chapters": ("MINIMAX_CINEMATIC_CHAPTERS",),
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "out_dir": ("STRING", {"default": "character_visual_profiles"}),
            "mode": (list(character_visual_profiles.MODES), {"default": "balanced"}),
            "thinking": ("BOOLEAN", {"default": True}),
            "temperature": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0}),
            "max_tokens": ("INT", {"default": 16384, "min": 256, "max": 100000}),
            "correction_attempts": ("INT", {"default": 2, "min": 0, "max": 10}),
            "force": ("BOOLEAN", {"default": False}),
            "context_tokens": ("INT", {"default": 65536, "min": 4096, "max": 1000000}),
        }, "optional": {"visual_designs_path": ("STRING", {"default": "", "tooltip": "Reuse edited visual_designs.json from a previous chapter."})}}

    RETURN_TYPES = ("MINIMAX_CHARACTER_VISUAL_PROFILES", "STRING")
    RETURN_NAMES = ("character_visual_profiles", "saved_file")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @progress.node_progress
    def run(self, character_profiles, lmstudio_config, out_dir="character_visual_profiles", mode="balanced",
            thinking=True, temperature=0.12, max_tokens=16384, correction_attempts=2, force=False,
            cinematic_chapters=None, context_tokens=65536, visual_designs_path=""):
        if not isinstance(lmstudio_config, dict):
            raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        if lmstudio_config.get("model_family", "Qwen") != "Qwen":
            raise ValueError("Character Visual Profile Generator requires Qwen in LM Studio Configuration.")
        if not isinstance(out_dir, str) or not out_dir.strip():
            raise ValueError("out_dir must be a non-empty relative subfolder.")
        if max_tokens < 1 or correction_attempts < 0 or not 0 <= temperature <= 2:
            raise ValueError("Invalid token budget, correction attempts or temperature.")
        if mode not in character_visual_profiles.MODES:
            raise ValueError("Unknown generation mode.")
        profiles = character_visual_profiles.prepare_profiles(character_profiles)
        chapter = character_chapter_enrichment.selected_chapter(character_profiles, cinematic_chapters)
        designs_path = visual_designs.resolve_designs_path(visual_designs_path)
        previous = util.load_json(designs_path) if designs_path else None
        config = {**lmstudio_config, "thinking": bool(thinking)}
        output = stage_output(config, out_dir.strip())
        args = argparse.Namespace(max_tokens=int(max_tokens), temperature=float(temperature),
                                  correction_attempts=int(correction_attempts), thinking=bool(thinking),
                                  force=bool(force), mode=mode, base_url=config["api_url"], context_tokens=int(context_tokens))
        client, model = lmstudio_pipeline.make_client_and_model(lmstudio_pipeline.load("consolidate"),
                                                               config["api_url"], config)
        with client:
            data, saved = character_visual_profiles.generate(character_profiles, profiles, output, client, model, args,
                                                             chapter, previous)
        return {"ui": {"text": [saved]}, "result": (data, saved)}

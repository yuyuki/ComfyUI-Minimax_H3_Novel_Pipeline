"""ComfyUI entry point for stable character visual profiles."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ..core import character_visual_profiles, character_visual_review, lmstudio_pipeline, progress, util, visual_designs
from ..core.run_output import stage_output
from .chapter_selection import chapter_path_list


class CharacterVisualProfileGeneratorNode:
    DESCRIPTION = "Generate permanent physical designs, then review candidate duplicate identities with LM Studio and update the same files."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "character_profiles": ("MINIMAX_CHARACTER_PROFILES",),
            "chapter_selection": ("MINIMAX_CHAPTER_SELECTION",),
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "out_dir": ("STRING", {"default": "character_visual_profiles"}),
        }, "optional": {
            "visual_designs_path": ("STRING", {"default": "", "tooltip": "Reuse edited physical defaults."}),
            "thinking": ("BOOLEAN", {"default": False}),
            "temperature": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0}),
            "max_tokens": ("INT", {"default": 8192, "min": 256, "max": 100000}),
            "context_tokens": ("INT", {"default": 65536, "min": 8192, "max": 2000000}),
            "correction_attempts": ("INT", {"default": 2, "min": 0, "max": 10}),
            "force": ("BOOLEAN", {"default": False}),
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
            out_dir="character_visual_profiles", visual_designs_path="", thinking=False,
            temperature=0.12, max_tokens=8192, context_tokens=65536, correction_attempts=2, force=False):
        if not isinstance(out_dir, str) or not out_dir.strip():
            raise ValueError("out_dir must be a non-empty relative subfolder.")
        if max_tokens < 1 or context_tokens < 8192 or correction_attempts < 0 or not 0 <= temperature <= 2:
            raise ValueError("Invalid identity review token budget, correction attempts or temperature.")
        paths = util.discover_inputs([Path(p) for p in chapter_path_list(chapter_selection)])
        designs_path = visual_designs.resolve_designs_path(visual_designs_path)
        previous = util.load_json(designs_path) if designs_path else None
        output = stage_output(lmstudio_config, out_dir.strip())
        data, saved = character_visual_profiles.generate(character_profiles, paths, output, previous)
        candidates = character_visual_review.candidates(data['visual_designs'])
        if candidates:
            config = {**lmstudio_config, 'thinking': bool(thinking)}
            args = SimpleNamespace(temperature=float(temperature), max_tokens=int(max_tokens),
                                   context_tokens=int(context_tokens), correction_attempts=int(correction_attempts),
                                   force=bool(force), thinking=bool(thinking), base_url=config['api_url'])
            client, model = lmstudio_pipeline.make_client_and_model(lmstudio_pipeline.load('consolidate'),
                                                                   config['api_url'], config)
            with client:
                corrected = character_visual_review.repair(data['visual_designs'], candidates, output, client, model, args)
            lmstudio_pipeline.comfy_interrupt_check()
            data, saved = character_visual_profiles.generate(corrected['character_profiles'], paths, output, corrected)
        return {"ui": {"text": [saved]}, "result": (data, saved)}

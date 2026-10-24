"""ComfyUI entry point for source-grounded character profiles."""
from __future__ import annotations

import argparse

from ..core import character_profiles, lmstudio_pipeline, progress
from ..core.run_output import stage_output


class ConsolidateCharacterProfilesNode:
    DESCRIPTION = "Reconcile all character evidence with Qwen, retaining exact sources and conflicting observations."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "character_evidence": ("MINIMAX_CHARACTER_EVIDENCE",),
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "out_dir": ("STRING", {"default": "character_profiles"}),
            "thinking": ("BOOLEAN", {"default": True}),
            "temperature": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0}),
            "max_tokens": ("INT", {"default": 16384, "min": 256, "max": 100000}),
            "correction_attempts": ("INT", {"default": 2, "min": 0, "max": 10}),
            "force": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("MINIMAX_CHARACTER_PROFILES", "STRING")
    RETURN_NAMES = ("character_profiles", "saved_file")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @progress.node_progress
    def run(self, character_evidence, lmstudio_config, out_dir="character_profiles",
            thinking=True, temperature=0.12, max_tokens=16384, correction_attempts=2, force=False):
        if not isinstance(lmstudio_config, dict):
            raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        if lmstudio_config.get("model_family", "Qwen") != "Qwen":
            raise ValueError("Consolidate Character Profiles requires Qwen in LM Studio Configuration.")
        if not isinstance(out_dir, str) or not out_dir.strip():
            raise ValueError("out_dir must be a non-empty relative subfolder.")
        if max_tokens < 1 or correction_attempts < 0 or not 0 <= temperature <= 2:
            raise ValueError("Invalid token budget, correction attempts or temperature.")
        records = character_profiles.prepare_evidence(character_evidence)
        config = {**lmstudio_config, "thinking": bool(thinking)}
        output = stage_output(config, out_dir.strip())
        args = argparse.Namespace(max_tokens=int(max_tokens), temperature=float(temperature),
                                  correction_attempts=int(correction_attempts), thinking=bool(thinking),
                                  force=bool(force), base_url=config["api_url"])
        client, model = lmstudio_pipeline.make_client_and_model(lmstudio_pipeline.load("consolidate"),
                                                               config["api_url"], config)
        with client:
            data, saved = character_profiles.consolidate(character_evidence, records, output, client, model, args)
        return {"ui": {"text": [saved]}, "result": (data, saved)}

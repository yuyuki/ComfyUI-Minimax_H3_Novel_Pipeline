"""LM Studio-backed MiniMax H3 prompt-generation ComfyUI node."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import lmstudio_pipeline, util


def _saved_chapter_choices() -> list[str]:
    try:
        import folder_paths
        root = Path(folder_paths.get_input_directory()) / "minimax_h3_novel"
        return [f"minimax_h3_novel/{p.name}" for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in util.SUPPORTED_EXTENSIONS] if root.is_dir() else [""]
    except Exception:
        return [""]


def _default_output_dir() -> str:
    try:
        import folder_paths
        return str(Path(folder_paths.get_output_directory()) / "minimax_h3_novel" / "h3_prompts")
    except Exception:
        return "output/minimax_h3_novel/h3_prompts"


class GenerateH3PromptsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "consolidated_references": ("MINIMAX_REGISTRY",), "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "chapter_paths": ("STRING", {"multiline": True, "default": ""}), "saved_chapter": (_saved_chapter_choices(),), "duration": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 3600.0}),
            "chunk_chars": ("INT", {"default": 14000, "min": 3000, "max": 1000000}), "overlap_paragraphs": ("INT", {"default": 2, "min": 0, "max": 100}), "scenes_per_chunk": ("INT", {"default": 4, "min": 1, "max": 100}), "max_scenes": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "max_pictures": ("INT", {"default": 8, "min": 1, "max": 100}), "max_pictures_per_subject": ("INT", {"default": 4, "min": 1, "max": 10}), "max_audio": ("INT", {"default": 4, "min": 0, "max": 100}), "temperature": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 2.0, "step": 0.05}), "max_tokens": ("INT", {"default": 8000, "min": 256, "max": 100000}),
            "repair_attempts": ("INT", {"default": 2, "min": 0, "max": 10}), "force": ("BOOLEAN", {"default": False}), "out_dir": ("STRING", {"default": _default_output_dir()}),
        }}

    RETURN_TYPES = ("MINIMAX_PROMPTS",); RETURN_NAMES = ("prompts",); FUNCTION = "run"; CATEGORY = "MiniMax H3 Novel"

    def run(self, consolidated_references: dict[str, Any], lmstudio_config: dict[str, Any], chapter_paths: str, saved_chapter: str, out_dir: str, **params: Any) -> tuple[dict[str, Any]]:
        if not isinstance(consolidated_references, dict): raise TypeError("consolidated_references must be a registry object.")
        if not isinstance(out_dir, str) or not out_dir.strip(): raise ValueError("out_dir must be a non-empty string.")
        if not isinstance(lmstudio_config, dict): raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        paths = util.discover_inputs([Path(p.strip()) for p in (chapter_paths or saved_chapter).splitlines() if p.strip()])
        if not paths: raise ValueError("No supported chapter files found.")
        pipeline = lmstudio_pipeline.load("generate")
        lmstudio_pipeline.configure_qwen(pipeline, thinking=bool(lmstudio_config["thinking"]), chat_backend=str(lmstudio_config["chat_backend"]), max_output_tokens=int(lmstudio_config["qwen35_max_output_tokens"]), length_retries=int(lmstudio_config["qwen35_length_retries"]))
        client, resolved_model = lmstudio_pipeline.make_client_and_model(pipeline, str(lmstudio_config["api_url"]), str(lmstudio_config["api_key"]), str(lmstudio_config.get("model", "")))
        client = lmstudio_pipeline.make_interruptible_client(client)
        keys = ("duration", "chunk_chars", "overlap_paragraphs", "scenes_per_chunk", "max_scenes", "max_pictures", "max_pictures_per_subject", "max_audio", "temperature", "max_tokens", "repair_attempts", "force")
        args = argparse.Namespace(**{key: params[key] for key in keys}, delay=0.0, out_dir=Path(out_dir.strip()))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        manifests = []
        for path in paths:
            lmstudio_pipeline.comfy_interrupt_check()
            manifests.append(pipeline.process_chapter(path, consolidated_references, client, resolved_model, args))
        for manifest in manifests:
            target = args.out_dir / manifest["chapter_id"]
            for scene in manifest["outputs"]:
                if not scene.get("prompt_file"):
                    continue
                scene["prompt_text"] = (target / scene["prompt_file"]).read_text(encoding="utf-8").rstrip()
                bindings = util.load_json(target / scene["assets_file"]); scene["bindings"] = bindings
                scene["picture_asset_ids"] = [x["asset_id"] for x in bindings["picture_input_order"]]
                scene["audio_asset_ids"] = [x["asset_id"] for x in bindings["audio_input_order"]]
        return ({"schema_version": "minimax-h3-novel-prompts.v3", "model": resolved_model, "chapters": manifests},)

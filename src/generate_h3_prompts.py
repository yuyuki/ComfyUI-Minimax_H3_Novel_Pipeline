"""LM Studio-backed MiniMax H3 prompt-generation ComfyUI node."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from . import lmstudio_pipeline, util
from .path_access import confined_path


def _saved_chapter_choices() -> list[str]:
    try:
        import folder_paths
        root = Path(folder_paths.get_input_directory()) / "minimax_h3_novel"
        # ``saved_chapter`` is only a single-file fallback.  Keep an explicit
        # empty enum value so workflows using the multi-file ``chapter_paths``
        # field pass ComfyUI validation.
        files = sorted(root.iterdir()) if root.is_dir() else []
        return [""] + [f"minimax_h3_novel/{p.name}" for p in files if p.is_file() and p.suffix.lower() in util.SUPPORTED_EXTENSIONS]
    except Exception:
        return [""]


def _default_output_dir() -> str:
    return "h3_prompts"


class GenerateH3PromptsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "consolidated_references": ("MINIMAX_REGISTRY",), "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "chapter_paths": ("STRING", {"multiline": True, "default": "", "tooltip": "One chapter file or folder per line, inside ComfyUI's input directory. Relative paths start there."}), "saved_chapter": (_saved_chapter_choices(),), "duration": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 3600.0}),
            "chunk_chars": ("INT", {"default": 14000, "min": 3000, "max": 1000000}), "overlap_paragraphs": ("INT", {"default": 2, "min": 0, "max": 100}), "scenes_per_chunk": ("INT", {"default": 4, "min": 1, "max": 100}), "max_scenes": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "max_pictures": ("INT", {"default": 8, "min": 1, "max": 100}), "max_pictures_per_subject": ("INT", {"default": 4, "min": 1, "max": 10}), "max_audio": ("INT", {"default": 4, "min": 0, "max": 100}), "temperature": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 2.0, "step": 0.05}), "max_tokens": ("INT", {"default": 8000, "min": 256, "max": 100000}),
            "repair_attempts": ("INT", {"default": 2, "min": 0, "max": 10}), "force": ("BOOLEAN", {"default": False}), "out_dir": ("STRING", {"default": _default_output_dir(), "tooltip": "Folder inside ComfyUI's output/minimax_h3_novel directory. Relative paths start there."}),
        }}

    RETURN_TYPES = ("MINIMAX_PROMPTS", "STRING")
    RETURN_NAMES = ("prompts", "prompt_text")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(self, consolidated_references: dict[str, Any], lmstudio_config: dict[str, Any], chapter_paths: str, saved_chapter: str, out_dir: str, **params: Any) -> tuple[dict[str, Any], str]:
        if not isinstance(consolidated_references, dict): raise TypeError("consolidated_references must be a registry object.")
        if not isinstance(out_dir, str) or not out_dir.strip(): raise ValueError("out_dir must be a non-empty string.")
        output = util.output_path(out_dir.strip())
        util.require_schema(consolidated_references, util.REGISTRY_SCHEMA)
        if not isinstance(lmstudio_config, dict): raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        paths = util.discover_inputs([Path(p.strip()) for p in (chapter_paths or saved_chapter).splitlines() if p.strip()])
        if not paths: raise ValueError("No supported chapter files found.")
        pipeline = lmstudio_pipeline.load("generate")
        lmstudio_pipeline.configure_qwen(thinking=bool(lmstudio_config["thinking"]), max_output_tokens=int(lmstudio_config["qwen35_max_output_tokens"]), length_retries=int(lmstudio_config["qwen35_length_retries"]), safe_chunk_chars=int(lmstudio_config["qwen35_safe_chunk_chars"]), top_k=int(lmstudio_config["qwen35_top_k"]), min_p=float(lmstudio_config["qwen35_min_p"]), repeat_penalty=float(lmstudio_config["qwen35_repeat_penalty"]))
        client, resolved_model = lmstudio_pipeline.make_client_and_model(pipeline, str(lmstudio_config["api_url"]))
        with client:
            keys = ("duration", "chunk_chars", "overlap_paragraphs", "scenes_per_chunk", "max_scenes", "max_pictures", "max_pictures_per_subject", "max_audio", "temperature", "max_tokens", "repair_attempts", "force")
            args = argparse.Namespace(**{key: params[key] for key in keys}, out_dir=output)
            args.out_dir.mkdir(parents=True, exist_ok=True)
            manifests = []
            for path in paths:
                lmstudio_pipeline.comfy_interrupt_check()
                manifests.append(pipeline.process_chapter(path, consolidated_references, client, resolved_model, args))
            for manifest in manifests:
                target = confined_path(manifest["chapter_id"], args.out_dir)
                for scene in manifest["outputs"]:
                    if not scene.get("prompt_file"):
                        continue
                    scene["prompt_text"] = confined_path(scene["prompt_file"], target).read_text(encoding="utf-8").rstrip()
                    bindings = util.load_json(confined_path(scene["assets_file"], target))
                    scene["bindings"] = bindings
                    scene["picture_asset_ids"] = [x["asset_id"] for x in bindings["picture_input_order"]]
                    scene["audio_asset_ids"] = [x["asset_id"] for x in bindings["audio_input_order"]]
            prompt_text = "\n\n".join(
                f"# {manifest.get('chapter_id', 'Chapter')} — Scene {scene.get('index', '?')}\n\n{scene['prompt_text']}"
                for manifest in manifests
                for scene in manifest["outputs"]
                if scene.get("prompt_text")
            )
            return ({"schema_version": "minimax-h3-novel-prompts.v3", "model": resolved_model, "chapters": manifests}, prompt_text)

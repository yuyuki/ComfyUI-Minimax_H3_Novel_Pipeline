"""Standalone MiniMax H3 prompt-generation ComfyUI node."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
from . import standalone_pipeline, util

def _saved_chapter_choices() -> list[str]:
    try:
        import folder_paths
        root = Path(folder_paths.get_input_directory()) / "minimax_h3_novel"
        return [f"minimax_h3_novel/{p.name}" for p in sorted(root.iterdir()) if p.suffix.lower() in util.SUPPORTED_EXTENSIONS] if root.is_dir() else [""]
    except Exception: return [""]

def _default_output_dir() -> str:
    try:
        import folder_paths
        return str(Path(folder_paths.get_output_directory()) / "minimax_h3_novel" / "h3_prompts")
    except Exception: return "output/minimax_h3_novel/h3_prompts"

class GenerateH3PromptsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"consolidated_references": ("MINIMAX_REGISTRY",), "clip": ("CLIP",), "chapter_paths": ("STRING", {"multiline": True, "default": ""}), "saved_chapter": (_saved_chapter_choices(),), "out_dir": ("STRING", {"default": _default_output_dir()}), "duration": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 3600.0}), "max_pictures": ("INT", {"default": 8, "min": 1, "max": 100}), "max_audio": ("INT", {"default": 4, "min": 0, "max": 100})}}
    RETURN_TYPES = ("MINIMAX_PROMPTS",); RETURN_NAMES = ("prompts",); FUNCTION = "run"; CATEGORY = "MiniMax H3 Novel"
    def run(self, consolidated_references: dict[str, Any], clip: Any, chapter_paths: str, saved_chapter: str, out_dir: str, duration: float, max_pictures: int, max_audio: int) -> tuple[dict[str, Any]]:
        if not isinstance(consolidated_references, dict): raise TypeError("consolidated_references must be a registry object.")
        if not out_dir.strip(): raise ValueError("out_dir must not be empty.")
        paths = util.discover_inputs([Path(p.strip()) for p in (chapter_paths or saved_chapter).splitlines() if p.strip()])
        if not paths: raise ValueError("No supported chapter files found.")
        args = argparse.Namespace(out_dir=Path(out_dir), duration=float(duration), max_pictures=int(max_pictures), max_audio=int(max_audio)); args.out_dir.mkdir(parents=True, exist_ok=True)
        manifests = [standalone_pipeline.process_chapter(path, consolidated_references, None, type(clip).__name__, args) for path in paths]
        for manifest in manifests:
            target = args.out_dir / manifest["chapter_id"]
            for scene in manifest["outputs"]:
                scene["prompt_text"] = (target / scene["prompt_file"]).read_text(encoding="utf-8").rstrip(); bindings = util.load_json(target / scene["assets_file"]); scene["bindings"] = bindings
                scene["picture_asset_ids"] = [x["asset_id"] for x in bindings["picture_input_order"]]; scene["audio_asset_ids"] = [x["asset_id"] for x in bindings["audio_input_order"]]
        return ({"schema_version": "minimax-h3-novel-prompts.v3", "model": type(clip).__name__, "chapters": manifests},)

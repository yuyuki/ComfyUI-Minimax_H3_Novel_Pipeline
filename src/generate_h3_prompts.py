"""LM Studio-backed MiniMax H3 prompt-generation ComfyUI node."""
from __future__ import annotations

from . import progress

import argparse
from pathlib import Path
from typing import Any

from . import configuration_snapshot, lmstudio_pipeline, util
from .chapter_selection import chapter_paths as selected_chapter_paths
from .path_access import confined_path
from .run_output import stage_output
from .image_prompt_export import export_image_prompts


def _default_output_dir() -> str:
    return "h3_prompts"


class GenerateH3PromptsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "consolidated_references": ("MINIMAX_REGISTRY",), "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "chapter_selection": ("MINIMAX_CHAPTER_SELECTION", {"tooltip": "Output of Select Chapters."}), "duration": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 3600.0}),
            "chunk_chars": ("INT", {"default": 14000, "min": 3000, "max": 1000000}), "overlap_paragraphs": ("INT", {"default": 2, "min": 0, "max": 100}), "scenes_per_chunk": ("INT", {"default": 4, "min": 1, "max": 100}), "max_scenes": ("INT", {"default": 0, "min": 0, "max": 10000}),
            "max_pictures": ("INT", {"default": 8, "min": 1, "max": 100}), "max_pictures_per_subject": ("INT", {"default": 4, "min": 1, "max": 10}), "max_audio": ("INT", {"default": 4, "min": 0, "max": 100}), "temperature": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 2.0, "step": 0.05}), "max_tokens": ("INT", {"default": 8000, "min": 256, "max": 100000}),
            "repair_attempts": ("INT", {"default": 2, "min": 0, "max": 10}), "force": ("BOOLEAN", {"default": False}), "out_dir": ("STRING", {"default": _default_output_dir(), "tooltip": "Subfolder of the current timestamped run inside output/minimax_h3_novel."}),
        }, "optional": {"spatial_continuity": ("MINIMAX_SPATIAL_CONTINUITY",)}}

    RETURN_TYPES = ("MINIMAX_PROMPTS", "STRING", "STRING")
    RETURN_NAMES = ("prompts", "prompt_text", "image_prompt_text")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @progress.node_progress
    def run(self, consolidated_references: dict[str, Any], lmstudio_config: dict[str, Any], chapter_selection: Any, out_dir: str, **params: Any) -> tuple[dict[str, Any], str, str]:
        if not isinstance(consolidated_references, dict): raise TypeError("consolidated_references must be a registry object.")
        if not isinstance(out_dir, str) or not out_dir.strip(): raise ValueError("out_dir must be a non-empty string.")
        output = stage_output(lmstudio_config, out_dir.strip())
        util.require_schema(consolidated_references, util.REGISTRY_SCHEMA)
        if not isinstance(lmstudio_config, dict): raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        selected_paths = selected_chapter_paths(chapter_selection)
        paths = util.discover_inputs([Path(p.strip()) for p in selected_paths.splitlines() if p.strip()])
        if not paths: raise ValueError("No supported chapter files found.")
        pipeline = lmstudio_pipeline.load("generate")
        client, resolved_model = lmstudio_pipeline.make_client_and_model(pipeline, str(lmstudio_config["api_url"]), lmstudio_config)
        with client:
            keys = ("duration", "chunk_chars", "overlap_paragraphs", "scenes_per_chunk", "max_scenes", "max_pictures", "max_pictures_per_subject", "max_audio", "temperature", "max_tokens", "repair_attempts", "force")
            args = argparse.Namespace(**{key: params[key] for key in keys}, out_dir=output)
            continuity = params.get("spatial_continuity")
            if continuity is not None:
                if (not isinstance(continuity, dict) or continuity.get("schema_version") != "minimax-spatial-continuity.v1"
                        or not isinstance(continuity.get("anchors"), dict)
                        or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
                               for k, v in continuity["anchors"].items())):
                    raise ValueError("Invalid spatial_continuity. Connect Spatial Continuity and enter non-empty JSON string anchors.")
                args.spatial_anchors = continuity["anchors"]
            args.out_dir.mkdir(parents=True, exist_ok=True)
            snapshot = configuration_snapshot.start(
                output, "generate", lmstudio_config, resolved_model, args, out_dir=out_dir,
                inputs={"chapters": [str(path) for path in paths],
                        "consolidated_references_sha256": configuration_snapshot.content_digest(consolidated_references)},
            )
            image_records, image_text = export_image_prompts(consolidated_references, output / "image_prompts")
            manifests = []
            for index, path in enumerate(paths):
                lmstudio_pipeline.comfy_interrupt_check()
                with progress.scope(0.98 * index / len(paths), 0.98 * (index + 1) / len(paths)):
                    manifests.append(pipeline.process_chapter(path, consolidated_references, client, resolved_model, args))
            for manifest in manifests:
                target = confined_path(manifest["chapter_id"], args.out_dir)
                for scene in manifest["outputs"]:
                    if not scene.get("prompt_file"):
                        continue
                    scene["asset_sheet_text"] = confined_path(scene["prompt_file"], target).read_text(encoding="utf-8").rstrip()
                    bindings = util.load_json(confined_path(scene["assets_file"], target))
                    scene["prompt_text"] = bindings["copy_paste_prompt"]
                    scene["bindings"] = bindings
                    scene["picture_asset_ids"] = [x["asset_id"] for x in bindings["picture_input_order"]]
                    scene["audio_asset_ids"] = [x["asset_id"] for x in bindings["audio_input_order"]]
            prompt_text = "\n\n".join(
                f"# {manifest.get('chapter_id', 'Chapter')} — Scene {scene.get('index', '?')}\n\n{scene['asset_sheet_text']}"
                for manifest in manifests
                for scene in manifest["outputs"]
                if scene.get("prompt_text")
            )
            configuration_snapshot.complete(snapshot, [output / "image_prompts"] + [
                confined_path(manifest["chapter_id"], output) for manifest in manifests
            ])
            return ({"schema_version": "minimax-h3-novel-prompts.v3", "model": resolved_model,
                     "chapters": manifests, "image_prompts": image_records}, prompt_text, image_text)

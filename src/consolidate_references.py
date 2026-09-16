"""LM Studio-backed ComfyUI node for cross-chapter consolidation."""
from __future__ import annotations

from . import progress

import argparse
import hashlib
from typing import Any, Iterable

from . import configuration_snapshot, lmstudio_pipeline, util
from .run_output import stage_output
from .visual_designs import IMAGE_STYLES, prepare_designs, resolve_designs_path
from .image_prompt_export import export_image_prompts


def _default_output_dir() -> str:
    return "references"


def _log(message: str) -> None:
    print(f"[minimax_h3_novel] {message}", flush=True)


class ConsolidateReferencesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "chapter_catalogs": ("MINIMAX_CHAPTERS",), "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "candidate_count": ("INT", {"default": 12, "min": 1, "max": 1000}), "include_all_below": ("INT", {"default": 35, "min": 0, "max": 100000}),
            "picture_threshold": (["optional", "recommended", "required"], {"default": "recommended"}), "audio_threshold": (["optional", "recommended", "required"], {"default": "recommended"}),
            "max_character_base_views": ("INT", {"default": 4, "min": 1, "max": 7}), "max_location_base_views": ("INT", {"default": 3, "min": 1, "max": 6}), "max_object_base_views": ("INT", {"default": 2, "min": 1, "max": 4}), "asset_batch_size": ("INT", {"default": 4, "min": 1, "max": 1000}),
            "no_variants": ("BOOLEAN", {"default": False}), "no_audit": ("BOOLEAN", {"default": False}), "audit_max_entities": ("INT", {"default": 120, "min": 0, "max": 100000}),
            "temperature": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0, "step": 0.05}), "max_tokens": ("INT", {"default": 8500, "min": 256, "max": 100000}),
            "out_dir": ("STRING", {"default": _default_output_dir(), "tooltip": "Subfolder of the current timestamped run inside output/minimax_h3_novel."}),
            "audit_similarity": ("FLOAT", {"default": 0.68, "min": 0.0, "max": 1.0, "step": 0.01}),
            "audit_cluster_size": ("INT", {"default": 24, "min": 2, "max": 120}),
        }, "optional": {
            "image_style": (list(IMAGE_STYLES), {"default": "realistic photographic"}),
            "visual_designs_path": ("STRING", {"default": "", "tooltip": "Optional existing visual_designs.json to import inside output/minimax_h3_novel. Leave empty on the first run; consolidation saves this file automatically."}),
            "image_asset_scope": (["all entities", "existing priority threshold"], {"default": "all entities"}),
        }}

    RETURN_TYPES = ("MINIMAX_REGISTRY", "STRING")
    RETURN_NAMES = ("consolidated_references", "registry_summary")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @progress.node_progress
    def run(self, chapter_catalogs: Iterable[dict[str, Any]], lmstudio_config: dict[str, Any], out_dir: str, **params: Any) -> tuple[dict[str, Any], str]:
        chapters = list(chapter_catalogs or [])
        if not chapters: raise ValueError("No chapter catalogs were supplied.")
        if not isinstance(out_dir, str) or not out_dir.strip(): raise ValueError("out_dir must be a non-empty string.")
        designs_path = resolve_designs_path(params.get("visual_designs_path", ""))
        output = stage_output(lmstudio_config, out_dir.strip())
        for chapter in chapters:
            util.require_schema(chapter, util.CHAPTER_SCHEMA)
        if not isinstance(lmstudio_config, dict): raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        pipeline = lmstudio_pipeline.load("consolidate")
        client, resolved_model = lmstudio_pipeline.make_client_and_model(pipeline, str(lmstudio_config["api_url"]), lmstudio_config)
        with client:
            keys = ("candidate_count", "include_all_below", "picture_threshold", "audio_threshold", "max_character_base_views", "max_location_base_views", "max_object_base_views", "asset_batch_size", "no_variants", "no_audit", "audit_max_entities", "temperature", "max_tokens")
            args = argparse.Namespace(**{key: params[key] for key in keys}, audit_similarity=float(params["audit_similarity"]), audit_cluster_size=max(2, int(params["audit_cluster_size"])))
            args.image_style = params.get("image_style", "realistic photographic")
            args.image_asset_scope = params.get("image_asset_scope", "all entities")
            if args.image_style not in IMAGE_STYLES:
                raise ValueError("Unknown image style.")
            if args.image_asset_scope not in {"all entities", "existing priority threshold"}:
                raise ValueError("Unknown image asset scope.")
            _log(f"LM Studio consolidation: model={resolved_model}, chapters={len(chapters)}")
            snapshot = configuration_snapshot.start(
                output, "consolidate", lmstudio_config, resolved_model, args, out_dir=out_dir,
                inputs={"chapter_catalogs_sha256": configuration_snapshot.content_digest(chapters),
                        "chapter_ids": [chapter["chapter_id"] for chapter in chapters],
                        "visual_designs_sha256": configuration_snapshot.file_digest(designs_path) if designs_path else None},
                extra={"visual_designs_path": str(designs_path) if designs_path else ""},
            )
            registry: list[dict[str, Any]] = []
            for chapter in progress.steps(chapters, 0, 0.3):
                lmstudio_pipeline.comfy_interrupt_check()
                registry = pipeline.reconcile_chapter(client, resolved_model, chapter, registry, args)
            lmstudio_pipeline.comfy_interrupt_check()
            with progress.scope(0.3, 0.4):
                registry = pipeline.audit_registry(client, resolved_model, registry, args)
            registry.sort(key=lambda item: ({"character": 0, "location": 1, "object": 2}[item["entity_type"]], pipeline.natural_key(item["global_id"])))
            with progress.scope(0.4, 0.55):
                designs = prepare_designs(pipeline.chat_json, client, resolved_model, registry, args,
                                          designs_path)
            util.save_json(output / "visual_designs.json", designs)
            args.visual_designs = {item["global_id"]: item for item in designs["entities"]}
            lmstudio_pipeline.comfy_interrupt_check()
            with progress.scope(0.55, 0.85):
                pictures = pipeline.generate_picture_assets(client, resolved_model, pipeline.build_picture_specs(registry, args), args)
            lmstudio_pipeline.comfy_interrupt_check()
            with progress.scope(0.85, 0.98):
                audio = pipeline.generate_audio_assets(client, resolved_model, pipeline.build_audio_specs(registry, args), args)
            digest = hashlib.sha256("\n".join(f"{c['chapter_id']}:{c.get('source', {}).get('sha256', '')}" for c in chapters).encode()).hexdigest()
            payload = {"schema_version": util.REGISTRY_SCHEMA, "source_digest": digest, "llm": {"base_url": lmstudio_config["api_url"], "model": resolved_model, "thinking": bool(lmstudio_config["thinking"]), "chat_backend": "structured-json"}, "chapters": [{"chapter_id": c["chapter_id"], "source_file": c.get("source", {}).get("file", ""), "source_sha256": c.get("source", {}).get("sha256", "")} for c in chapters], "entities": registry, "picture_assets": pictures, "audio_assets": audio, "video_assets": [], "chapter_entity_map": pipeline.build_chapter_map(registry), "entity_asset_index": pipeline.build_entity_asset_index(registry, pictures, audio), "label_note": "canonical_label is only a convenient full-registry ordering. MiniMax H3 labels are request-local."}
            payload["visual_designs"] = designs
            payload["image_style"] = args.image_style
            util.save_json(output / "consolidated_references.json", payload)
            export_image_prompts(payload, output / "image_prompts")
            pipeline.write_asset_prompts(util.output_path(output / "reference_asset_prompts.txt"), pictures, audio)
            configuration_snapshot.complete(snapshot, [output / name for name in (
                "consolidated_references.json", "visual_designs.json", "reference_asset_prompts.txt", "image_prompts",
            )])
            return payload, util.registry_summary(payload)

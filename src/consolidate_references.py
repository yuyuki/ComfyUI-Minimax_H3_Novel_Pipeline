"""LM Studio-backed ComfyUI node for cross-chapter consolidation."""
from __future__ import annotations

import argparse
import hashlib
from typing import Any, Iterable

from . import lmstudio_pipeline, util


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
            "max_character_base_views": ("INT", {"default": 4, "min": 1, "max": 7}), "max_location_base_views": ("INT", {"default": 3, "min": 1, "max": 6}), "max_object_base_views": ("INT", {"default": 2, "min": 1, "max": 4}), "asset_batch_size": ("INT", {"default": 16, "min": 1, "max": 1000}),
            "no_variants": ("BOOLEAN", {"default": False}), "no_audit": ("BOOLEAN", {"default": False}), "audit_max_entities": ("INT", {"default": 120, "min": 0, "max": 100000}),
            "temperature": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0, "step": 0.05}), "max_tokens": ("INT", {"default": 8500, "min": 256, "max": 100000}),
            "out_dir": ("STRING", {"default": _default_output_dir(), "tooltip": "Folder inside ComfyUI's output/minimax_h3_novel directory. Relative paths start there."}),
            "audit_similarity": ("FLOAT", {"default": 0.68, "min": 0.0, "max": 1.0, "step": 0.01}),
            "audit_cluster_size": ("INT", {"default": 24, "min": 2, "max": 120}),
        }}

    RETURN_TYPES = ("MINIMAX_REGISTRY",)
    RETURN_NAMES = ("consolidated_references",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(self, chapter_catalogs: Iterable[dict[str, Any]], lmstudio_config: dict[str, Any], out_dir: str, **params: Any) -> tuple[dict[str, Any]]:
        chapters = list(chapter_catalogs or [])
        if not chapters: raise ValueError("No chapter catalogs were supplied.")
        if not isinstance(out_dir, str) or not out_dir.strip(): raise ValueError("out_dir must be a non-empty string.")
        output = util.output_path(out_dir.strip())
        for chapter in chapters:
            util.require_schema(chapter, util.CHAPTER_SCHEMA)
        if not isinstance(lmstudio_config, dict): raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        pipeline = lmstudio_pipeline.load("consolidate")
        lmstudio_pipeline.configure_qwen(thinking=bool(lmstudio_config["thinking"]), max_output_tokens=int(lmstudio_config["qwen35_max_output_tokens"]), length_retries=int(lmstudio_config["qwen35_length_retries"]), safe_chunk_chars=int(lmstudio_config["qwen35_safe_chunk_chars"]), top_k=int(lmstudio_config["qwen35_top_k"]), min_p=float(lmstudio_config["qwen35_min_p"]), repeat_penalty=float(lmstudio_config["qwen35_repeat_penalty"]))
        client, resolved_model = lmstudio_pipeline.make_client_and_model(pipeline, str(lmstudio_config["api_url"]))
        with client:
            keys = ("candidate_count", "include_all_below", "picture_threshold", "audio_threshold", "max_character_base_views", "max_location_base_views", "max_object_base_views", "asset_batch_size", "no_variants", "no_audit", "audit_max_entities", "temperature", "max_tokens")
            args = argparse.Namespace(**{key: params[key] for key in keys}, audit_similarity=float(params["audit_similarity"]), audit_cluster_size=max(2, int(params["audit_cluster_size"])))
            _log(f"LM Studio consolidation: model={resolved_model}, chapters={len(chapters)}")
            registry: list[dict[str, Any]] = []
            for chapter in chapters:
                lmstudio_pipeline.comfy_interrupt_check()
                registry = pipeline.reconcile_chapter(client, resolved_model, chapter, registry, args)
            lmstudio_pipeline.comfy_interrupt_check()
            registry = pipeline.audit_registry(client, resolved_model, registry, args)
            registry.sort(key=lambda item: ({"character": 0, "location": 1, "object": 2}[item["entity_type"]], pipeline.natural_key(item["global_id"])))
            lmstudio_pipeline.comfy_interrupt_check()
            pictures = pipeline.generate_picture_assets(client, resolved_model, pipeline.build_picture_specs(registry, args), args)
            lmstudio_pipeline.comfy_interrupt_check()
            audio = pipeline.generate_audio_assets(client, resolved_model, pipeline.build_audio_specs(registry, args), args)
            digest = hashlib.sha256("\n".join(f"{c['chapter_id']}:{c.get('source', {}).get('sha256', '')}" for c in chapters).encode()).hexdigest()
            payload = {"schema_version": util.REGISTRY_SCHEMA, "source_digest": digest, "llm": {"base_url": lmstudio_config["api_url"], "model": resolved_model, "thinking": bool(lmstudio_config["thinking"]), "chat_backend": "structured-json"}, "chapters": [{"chapter_id": c["chapter_id"], "source_file": c.get("source", {}).get("file", ""), "source_sha256": c.get("source", {}).get("sha256", "")} for c in chapters], "entities": registry, "picture_assets": pictures, "audio_assets": audio, "video_assets": [], "chapter_entity_map": pipeline.build_chapter_map(registry), "entity_asset_index": pipeline.build_entity_asset_index(registry, pictures, audio), "label_note": "canonical_label is only a convenient full-registry ordering. MiniMax H3 labels are request-local."}
            util.save_json(output / "consolidated_references.json", payload)
            pipeline.write_asset_prompts(util.output_path(output / "reference_asset_prompts.txt"), pictures, audio)
            return (payload,)

"""LM Studio-backed ComfyUI node for cross-chapter consolidation."""
from __future__ import annotations

from ..core import progress
from ..core.reference_timeline import refresh_first_occurrence, validate_catalogs
from ..core.cinematic_references import parse_chapters

import argparse
from typing import Any, Iterable

from ..core import configuration_snapshot, lmstudio_pipeline, util
from ..core import character_visual_integration
from ..core.run_output import stage_output
from ..core.visual_designs import IMAGE_STYLES, prepare_designs, resolve_designs_path
from ..core.image_prompt_export import export_image_prompts
from ..core.editable_schemas import export_schemas


def _default_output_dir() -> str:
    return "references"


def _log(message: str) -> None:
    print(f"[minimax_h3_novel] {message}", flush=True)


class ConsolidateReferencesNode:
    DESCRIPTION = (
        "Consolidate references while asking the selected LM Studio model to preserve "
        "the source language of supplied descriptions and generated briefs. Language "
        "support depends on the model. JSON keys and fixed image framing remain unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "chapter_catalogs": ("MINIMAX_CHAPTERS",), "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "candidate_count": ("INT", {"default": 12, "min": 1, "max": 1000}), "include_all_below": ("INT", {"default": 35, "min": 0, "max": 100000}),
            "audio_threshold": (["optional", "recommended", "required"], {"default": "recommended"}),
            "max_character_base_views": ("INT", {"default": 4, "min": 1, "max": 7}), "max_location_base_views": ("INT", {"default": 3, "min": 1, "max": 6}), "max_object_base_views": ("INT", {"default": 2, "min": 1, "max": 4}), "asset_batch_size": ("INT", {"default": 4, "min": 1, "max": 1000}),
            "no_audit": ("BOOLEAN", {"default": False}), "audit_max_entities": ("INT", {"default": 120, "min": 0, "max": 100000}),
            "thinking": ("BOOLEAN", {"default": True, "tooltip": "Enable Qwen reasoning for this node. Ignored by Mistral."}),
            "temperature": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0, "step": 0.05}), "max_tokens": ("INT", {"default": 8500, "min": 256, "max": 100000}),
            "out_dir": ("STRING", {"default": _default_output_dir(), "tooltip": "Subfolder of the current timestamped run inside output/minimax_h3_novel."}),
            "audit_similarity": ("FLOAT", {"default": 0.68, "min": 0.0, "max": 1.0, "step": 0.01}),
            "audit_cluster_size": ("INT", {"default": 24, "min": 2, "max": 120}),
        }, "optional": {
            "character_visual_profiles": ("MINIMAX_CHARACTER_VISUAL_PROFILES",),
            "image_style": (list(IMAGE_STYLES), {"default": "cinematic photographic"}),
            "visual_designs_path": ("STRING", {"default": "", "tooltip": "Optional chapter or global visual_designs.json inside output/minimax_h3_novel; only matching chapter entities are imported. Leave empty on the first run; consolidation saves this file automatically."}),
        }}

    RETURN_TYPES = ("MINIMAX_REGISTRY", "STRING")
    RETURN_NAMES = ("consolidated_references", "registry_summary")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **params):
        # ComfyUI must rerun when the contents of an imported file change.
        path = params.get("visual_designs_path", "").strip()
        if not path:
            return ""
        resolved = util.output_path(path)
        try:
            return configuration_snapshot.file_digest(resolved)
        except OSError:
            # Cache inspection may include nodes outside the requested branch.
            # Invalidate unavailable imports; run() validates them if executed.
            return float("nan")

    @progress.node_progress
    def run(self, chapter_catalogs: Iterable[dict[str, Any]], lmstudio_config: dict[str, Any], out_dir: str, thinking: bool = True, **params: Any) -> dict[str, Any]:
        chapters = list(chapter_catalogs or [])
        if not chapters: raise ValueError("No chapter catalogs were supplied.")
        if not isinstance(out_dir, str) or not out_dir.strip(): raise ValueError("out_dir must be a non-empty string.")
        validate_catalogs(chapters)
        chapters = parse_chapters(chapters)
        character_visual_profiles = params.get("character_visual_profiles")
        character_visual_integration.validate_input(character_visual_profiles, chapters)
        designs_path = resolve_designs_path(params.get("visual_designs_path", ""))
        output = stage_output(lmstudio_config, out_dir.strip())
        if not isinstance(lmstudio_config, dict): raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        lmstudio_config = {**lmstudio_config, "thinking": bool(thinking)}
        pipeline = lmstudio_pipeline.load("consolidate")
        client, resolved_model = lmstudio_pipeline.make_client_and_model(pipeline, str(lmstudio_config["api_url"]), lmstudio_config)
        with client:
            keys = ("candidate_count", "include_all_below", "audio_threshold", "max_character_base_views", "max_location_base_views", "max_object_base_views", "asset_batch_size", "no_audit", "audit_max_entities", "temperature", "max_tokens")
            args = argparse.Namespace(**{key: params[key] for key in keys}, audit_similarity=float(params["audit_similarity"]), audit_cluster_size=max(2, int(params["audit_cluster_size"])))
            args.image_style = params.get("image_style", "cinematic photographic")
            if args.image_style not in IMAGE_STYLES:
                raise ValueError("Unknown image style.")
            _log(f"LM Studio consolidation: model={resolved_model}, chapters={len(chapters)}")
            args.thinking = bool(thinking)
            snapshot = configuration_snapshot.start(
                output, "consolidate", lmstudio_config, resolved_model, args, out_dir=out_dir,
                inputs={"chapter_catalogs_sha256": configuration_snapshot.content_digest(chapters),
                        "character_visual_profiles_sha256": configuration_snapshot.content_digest(character_visual_profiles)
                        if character_visual_profiles is not None else None,
                        "chapter_ids": [chapter["chapter_id"] for chapter in chapters],
                        "visual_designs_sha256": configuration_snapshot.file_digest(designs_path) if designs_path else None},
                extra={"visual_designs_path": str(designs_path) if designs_path else ""},
            )
            export_schemas(output)
            registry: list[dict[str, Any]] = []
            for chapter in progress.steps(chapters, 0, 0.3):
                lmstudio_pipeline.comfy_interrupt_check()
                registry = pipeline.reconcile_chapter(client, resolved_model, chapter, registry, args)
            lmstudio_pipeline.comfy_interrupt_check()
            with progress.scope(0.3, 0.4):
                registry = pipeline.audit_registry(client, resolved_model, registry, args)
            for entity in registry:
                refresh_first_occurrence(entity, [c["chapter_id"] for c in chapters])
            registry.sort(key=lambda item: ({"character": 0, "location": 1, "object": 2}[item["entity_type"]], pipeline.natural_key(item["global_id"])))
            args.temporal_context_by_chapter = {
                c["chapter_id"]: {"default_temporal_context": c.get("default_temporal_context", ""),
                                  "sequences": c["sequences"]} for c in chapters
            }
            character_enrichment = character_visual_integration.apply(character_visual_profiles, chapters, registry)
            args.character_designs = {}
            if character_visual_profiles is not None:
                global_designs = {d['global_id']: d for d in character_visual_profiles['visual_designs']['entities']}
                args.character_designs = {e['global_id']: global_designs[e['character_id']]
                                          for e in registry if e.get('character_id') in global_designs}
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
            digest = configuration_snapshot.content_digest(chapters)
            payload = {"schema_version": util.REGISTRY_SCHEMA, "source_digest": digest, "llm": {"base_url": lmstudio_config["api_url"], "model": resolved_model, "thinking": bool(lmstudio_config["thinking"]), "chat_backend": "structured-json"}, "chapters": [{"chapter_id": c["chapter_id"], "source_file": c.get("source", {}).get("file", ""), "source_sha256": c.get("source", {}).get("sha256", "")} for c in chapters], "entities": registry, "picture_assets": pictures, "audio_assets": audio, "video_assets": [], "chapter_entity_map": pipeline.build_chapter_map(registry), "entity_asset_index": pipeline.build_entity_asset_index(registry, pictures, audio), "label_note": "canonical_label is only a convenient full-registry ordering. MiniMax H3 labels are request-local."}
            payload["chapter_timelines"] = {c["chapter_id"]: {
                "default_temporal_context": c.get("default_temporal_context", ""),
                "sequences": c["sequences"],
                "narrative_review": c.get("narrative_review", {"issues": []}),
                "references": {kind: c.get(kind, []) for kind in ("characters", "locations", "objects")},
            } for c in chapters}
            payload["visual_designs"] = designs
            if character_enrichment:
                payload["character_enrichment"] = character_enrichment
            payload["image_style"] = args.image_style
            util.save_json(output / "consolidated_references.json", payload)
            export_image_prompts(payload, output / "image_prompts")
            pipeline.write_asset_prompts(util.output_path(output / "reference_asset_prompts.txt"), pictures, audio)
            configuration_snapshot.complete(snapshot, [output / name for name in (
                "consolidated_references.json", "visual_designs.json", "reference_asset_prompts.txt", "image_prompts",
                "visual_designs.schema.json",
            )])
            summary = util.registry_summary(payload)
            return {"ui": {"text": [summary]}, "result": (payload, summary)}

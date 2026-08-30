"""Standalone ComfyUI node for book-level reference consolidation."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Iterable

from . import util
from .extract_chapter_references import _decode_json_with_clip
from . import standalone_pipeline as pipeline


def _load_pipeline_script(path_hint: str = "", pipeline_dir: str = "") -> Any:
    """Locate Step 2 in either a source checkout or a ComfyUI install.

    The bundled module is deliberately used even when CLI scripts happen to
    exist beside the custom node.
    """
    """Compatibility shim retained for workflows created with older nodes."""
    return pipeline


def _default_output_dir() -> str:
    """Use ComfyUI's managed output directory by default."""
    try:
        import folder_paths
        return str(Path(folder_paths.get_output_directory()) / "minimax_h3_novel" / "references")
    except Exception:
        return "output/minimax_h3_novel/references"


def _log(message: str) -> None:
    print(f"[minimax_h3_novel] {message}", flush=True)


class ConsolidateReferencesNode:
    """Reconcile chapter catalogs and generate reusable picture/voice briefs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chapter_catalogs": ("MINIMAX_CHAPTERS",),
                "clip": ("CLIP",),
                "candidate_count": ("INT", {"default": 12, "min": 1, "max": 1000}),
                "include_all_below": ("INT", {"default": 35, "min": 0, "max": 100000}),
                "picture_threshold": (["optional", "recommended", "required"], {"default": "recommended"}),
                "audio_threshold": (["optional", "recommended", "required"], {"default": "recommended"}),
                "max_character_base_views": ("INT", {"default": 4, "min": 1, "max": 7}),
                "max_location_base_views": ("INT", {"default": 3, "min": 1, "max": 6}),
                "max_object_base_views": ("INT", {"default": 2, "min": 1, "max": 4}),
                "asset_batch_size": ("INT", {"default": 16, "min": 1, "max": 1000}),
                "no_variants": ("BOOLEAN", {"default": False}),
                "no_audit": ("BOOLEAN", {"default": False}),
                "audit_max_entities": ("INT", {"default": 120, "min": 0, "max": 100000}),
                "temperature": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 8500, "min": 256, "max": 100000}),
                "out_dir": ("STRING", {"default": _default_output_dir(), "tooltip": "Folder for consolidated_references.json and reference_asset_prompts.txt."}),
            },
        }

    RETURN_TYPES = ("MINIMAX_REGISTRY",)
    RETURN_NAMES = ("consolidated_references",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(self, chapter_catalogs: Iterable[dict[str, Any]], clip: Any, out_dir: str = "", **params: Any) -> tuple[dict[str, Any]]:
        if not isinstance(out_dir, str):
            raise TypeError("out_dir must be a string")
        if not out_dir.strip():
            raise ValueError("out_dir must not be empty")
        chapters = list(chapter_catalogs or [])
        if not chapters:
            raise ValueError("No chapter catalogs were supplied by ExtractChapterReferencesNode.")
        if not all(callable(getattr(clip, name, None)) for name in ("tokenize", "generate", "decode")):
            raise TypeError(
                "ConsolidateReferencesNode requires a generative CLIP. Load a generative "
                "text model through Load CLIP; standard SD CLIP encoders cannot reconcile catalogs."
            )

        # Retain the source script's schema checks and legacy upgrade behaviour.
        for chapter in chapters:
            schema = chapter.get("schema_version")
            if schema not in {pipeline.INPUT_SCHEMA, pipeline.LEGACY_INPUT_SCHEMA}:
                raise ValueError(
                    f"{chapter.get('chapter_id', '<unknown>')}: unsupported schema {schema!r}; "
                    f"expected {pipeline.INPUT_SCHEMA!r} or {pipeline.LEGACY_INPUT_SCHEMA!r}."
                )
            if schema == pipeline.LEGACY_INPUT_SCHEMA:
                for key in ("characters", "locations", "objects"):
                    for entity in chapter.get(key, []):
                        entity.setdefault("reference_view_hints", [])

        args = argparse.Namespace(
            candidate_count=int(params["candidate_count"]),
            include_all_below=int(params["include_all_below"]),
            picture_threshold=params["picture_threshold"],
            audio_threshold=params["audio_threshold"],
            max_character_base_views=int(params["max_character_base_views"]),
            max_location_base_views=int(params["max_location_base_views"]),
            max_object_base_views=int(params["max_object_base_views"]),
            asset_batch_size=int(params["asset_batch_size"]),
            no_variants=bool(params["no_variants"]),
            no_audit=bool(params["no_audit"]),
            audit_max_entities=int(params["audit_max_entities"]),
            temperature=float(params["temperature"]),
            max_tokens=int(params["max_tokens"]),
            delay=0.0,
        )
        # Reuse the canonical reconciliation and asset-planning code with the
        # ComfyUI CLIP JSON transport.
        # All of Step 2's LLM calls flow through ``chat_json``.
        model = type(clip).__name__
        _log(f"Consolidating {len(chapters)} chapter catalog(s) with {model}")
        registry, pictures, audio = pipeline.consolidate(chapters, args)
        _log(f"Generated {len(pictures)} picture and {len(audio)} audio asset brief(s)")

        digest_source = "\n".join(
            f"{chapter['chapter_id']}:{chapter.get('source', {}).get('sha256', '')}"
            for chapter in chapters
        )
        payload = {
            "schema_version": pipeline.OUTPUT_SCHEMA,
            "source_digest": hashlib.sha256(digest_source.encode()).hexdigest(),
            "llm": {"backend": "comfyui-clip", "model": model, "thinking": False},
            "chapters": [{"chapter_id": chapter["chapter_id"], "source_file": chapter.get("source", {}).get("file", ""), "source_sha256": chapter.get("source", {}).get("sha256", "")} for chapter in chapters],
            "entities": registry,
            "picture_assets": pictures,
            "audio_assets": audio,
            "chapter_entity_map": pipeline.chapter_map(registry),
            "entity_asset_index": pipeline.asset_index(registry, pictures, audio),
            "label_note": (
                "canonical_label is only a convenient full-registry ordering. MiniMax H3 labels are request-local. "
                "Step 3 maps the exact subset used by each clip to <Picture 1>..., <Audio 1>..., while multiple pictures may define the same <Subject N>."
            ),
        }
        output = Path(out_dir.strip())
        util.save_json(output / "consolidated_references.json", payload)
        pipeline.write_asset_prompts(output / "reference_asset_prompts.txt", pictures, audio)
        _log(f"Saved consolidation outputs to {output}")
        return (payload,)

"""ComfyUI node equivalent of :mod:`02_consolidate_references`.

The canonical script remains the single source of truth for reconciliation,
asset planning, and structured LLM calls. This node adapts its in-memory
chapter catalogs to ComfyUI rather than maintaining a divergent reimplementation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from . import util
from .extract_chapter_references import _decode_json_with_clip


def _load_pipeline_script(path_hint: str = "") -> Any:
    """Locate Step 2 in either a source checkout or a ComfyUI install.

    A custom node is commonly copied on its own to ``custom_nodes`` while the
    pipeline checkout remains in a sibling directory.  The original resolver
    only supported the source-tree layout, which made that installation fail.
    ``MINIMAX_H3_PIPELINE_DIR`` is an escape hatch for non-standard layouts.
    When the node's output directory is inside the source checkout, that
    directory also provides a reliable, zero-configuration location hint.
    """
    package_dir = Path(__file__).resolve().parent
    script_name = "02_consolidate_references.py"
    configured_dir = os.environ.get("MINIMAX_H3_PIPELINE_DIR", "").strip()
    candidates = [
        Path(configured_dir).expanduser() / script_name if configured_dir else None,
        package_dir.parents[1] / "02_consolidate_references.py",
        package_dir.parent / "02_consolidate_references.py",
        package_dir / script_name,
    ]
    if path_hint.strip():
        # ``out_dir`` is commonly ``<pipeline checkout>/comfyui_plugins/out``.
        # Check it and its parents so a custom node copied into a separate
        # ComfyUI install can still reuse its original checkout automatically.
        hint = Path(path_hint).expanduser()
        try:
            hint = hint.resolve()
        except OSError:
            hint = hint.absolute()
        candidates.extend(parent / script_name for parent in (hint, *hint.parents))
    # Standard ComfyUI layout: custom_nodes/<this plugin> next to a checkout
    # such as custom_nodes/minimax_h3_novel_pipeline_v2_4_1_FLAT/.
    custom_nodes_dir = package_dir.parent
    if custom_nodes_dir.is_dir():
        candidates.extend(sorted(custom_nodes_dir.glob(f"*/{script_name}")))
    candidates = [path for path in candidates if path is not None]
    script_path = next((path for path in candidates if path.is_file()), None)
    if script_path is None:
        searched = "\n  - ".join(str(path) for path in candidates)
        raise RuntimeError(
            "The canonical 02_consolidate_references.py was not found. "
            "Copy the three numbered pipeline scripts into the custom-node "
            "folder, keep the pipeline checkout in a sibling custom_nodes "
            "folder, or set MINIMAX_H3_PIPELINE_DIR to that checkout. "
            f"Locations checked:\n  - {searched}"
        )
    spec = importlib.util.spec_from_file_location("_minimax_h3_consolidation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load consolidation implementation: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


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
                "delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1}),
            },
            "optional": {
                "out_dir": ("STRING", {"default": "", "tooltip": "Optional folder for consolidated_references.json and reference_asset_prompts.txt."}),
            },
        }

    RETURN_TYPES = ("MINIMAX_REGISTRY", "MINIMAX_PICTURE_BRIEFS", "MINIMAX_AUDIO_BRIEFS")
    RETURN_NAMES = ("consolidated_references", "picture_asset_briefs", "audio_asset_briefs")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(self, chapter_catalogs: Iterable[dict[str, Any]], clip: Any, out_dir: str = "", **params: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        if not isinstance(out_dir, str):
            raise TypeError("out_dir must be a string")
        if not out_dir.strip():
            raise ValueError("out_dir must not be empty")
        pipeline = _load_pipeline_script(out_dir)
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
            delay=float(params["delay"]),
        )
        # Reuse the canonical reconciliation and asset-planning code with the
        # ComfyUI CLIP JSON transport.
        # All of Step 2's LLM calls flow through ``chat_json``.
        model = type(clip).__name__
        pipeline.THINKING_ENABLED = False
        pipeline.CHAT_BACKEND = "comfyui-clip"
        pipeline.chat_json = lambda _client, _model, system, user, schema, temperature, max_tokens: _decode_json_with_clip(
            clip, system, user, schema, temperature, max_tokens
        )
        client = None
        _log(f"Consolidating {len(chapters)} chapter catalog(s) with {model}")

        registry: list[dict[str, Any]] = []
        for index, chapter in enumerate(chapters, start=1):
            _log(f"Reconciling {index}/{len(chapters)}: {chapter['chapter_id']}")
            registry = pipeline.reconcile_chapter(client, model, chapter, registry, args)
            if args.delay:
                pipeline.time.sleep(args.delay)
        if not args.no_audit:
            try:
                registry = pipeline.audit_registry(client, model, registry, args)
            except Exception as exc:
                # Match the CLI: an audit failure must not discard an otherwise
                # valid reconciled registry.
                _log(f"WARNING: duplicate audit failed; continuing: {exc}")
        registry.sort(key=lambda entity: ({"character": 0, "location": 1, "object": 2}[entity["entity_type"]], pipeline.natural_key(entity["global_id"])))

        picture_specs = pipeline.build_picture_specs(registry, args)
        audio_specs = pipeline.build_audio_specs(registry, args)
        _log(f"Generating {len(picture_specs)} picture and {len(audio_specs)} audio asset brief(s)")
        pictures = pipeline.generate_picture_assets(client, model, picture_specs, args)
        audio = pipeline.generate_audio_assets(client, model, audio_specs, args)

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
            "video_assets": [],
            "chapter_entity_map": pipeline.build_chapter_map(registry),
            "entity_asset_index": pipeline.build_entity_asset_index(registry, pictures, audio),
            "label_note": (
                "canonical_label is only a convenient full-registry ordering. MiniMax H3 labels are request-local. "
                "Step 3 maps the exact subset used by each clip to <Picture 1>..., <Audio 1>..., while multiple pictures may define the same <Subject N>."
            ),
        }
        output = Path(out_dir.strip())
        util.save_json(output / "consolidated_references.json", payload)
        pipeline.write_asset_prompts(output / "reference_asset_prompts.txt", pictures, audio)
        _log(f"Saved consolidation outputs to {output}")
        return payload, pictures, audio

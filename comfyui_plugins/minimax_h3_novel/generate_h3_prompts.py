"""ComfyUI adapter for :mod:`03_generate_h3_prompts`."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from . import util
from .extract_chapter_references import _decode_json_with_clip


_CHAPTER_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def _log(message: str) -> None:
    print(f"[minimax_h3_novel] {message}", flush=True)


def _saved_chapter_choices() -> list[str]:
    try:
        import folder_paths
        root = Path(folder_paths.get_input_directory()) / "minimax_h3_novel"
        if not root.is_dir():
            return [""]
        files = sorted((p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _CHAPTER_EXTENSIONS), key=lambda p: p.name.lower())
        return [f"minimax_h3_novel/{p.name}" for p in files] or [""]
    except Exception:
        return [""]


def _default_output_dir() -> str:
    """Return a ComfyUI-managed destination without relying on its CWD."""
    try:
        import folder_paths
        return str(Path(folder_paths.get_output_directory()) / "minimax_h3_novel" / "h3_prompts")
    except Exception:
        # Keep a useful fallback when the module is imported outside ComfyUI.
        return "output/minimax_h3_novel/h3_prompts"


def _load_pipeline_script(path_hint: str = "") -> Any:
    """Load Step 3 from a source checkout or a sibling ComfyUI checkout.

    ``out_dir`` is often configured inside the original pipeline checkout,
    even when this custom node itself has been copied into a separate ComfyUI
    installation.  Searching that path and its parents makes this common
    layout work without requiring an environment variable.
    """
    package_dir = Path(__file__).resolve().parent
    script_name = "03_generate_h3_prompts.py"
    configured_dir = os.environ.get("MINIMAX_H3_PIPELINE_DIR", "").strip()
    candidates = [
        Path(configured_dir).expanduser() / script_name if configured_dir else None,
        package_dir.parents[1] / script_name,
        package_dir.parent / script_name,
        package_dir / script_name,
    ]
    if path_hint.strip():
        hint = Path(path_hint).expanduser()
        try:
            hint = hint.resolve()
        except OSError:
            hint = hint.absolute()
        candidates.extend(parent / script_name for parent in (hint, *hint.parents))
    custom_nodes_dir = package_dir.parent
    if custom_nodes_dir.is_dir():
        candidates.extend(sorted(custom_nodes_dir.glob(f"*/{script_name}")))
    candidates = [path for path in candidates if path is not None]
    script_path = next((path for path in candidates if path.is_file()), None)
    if script_path is None:
        searched = "\n  - ".join(str(path) for path in candidates)
        raise RuntimeError(
            "The canonical 03_generate_h3_prompts.py was not found. "
            "Copy the numbered pipeline scripts into the custom-node folder, "
            "keep the pipeline checkout in a sibling custom_nodes folder, or "
            "set MINIMAX_H3_PIPELINE_DIR to that checkout. "
            f"Locations checked:\n  - {searched}"
        )
    spec = importlib.util.spec_from_file_location("_minimax_h3_prompt_generation", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load prompt-generation implementation: {script_path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses in the canonical script resolve annotations through
    # sys.modules while the module body executes.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


class GenerateH3PromptsNode:
    """Generate validated H3 prompts using the canonical Step 3 workflow.

    ``consolidated_references`` is the authoritative hand-off: it already
    contains entities, picture/audio assets, indexes, and chapter mappings.
    Picture and audio briefs are optional ports so all three Consolidate node
    outputs can be wired; they only fill absent embedded asset lists.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "consolidated_references": ("MINIMAX_REGISTRY",), "clip": ("CLIP",),
                "chapter_paths": ("STRING", {"multiline": True, "default": "", "tooltip": "Chapter files/folders, one per line."}),
                "saved_chapter": (_saved_chapter_choices(), {"tooltip": "Previously uploaded chapter."}),
                "out_dir": ("STRING", {"default": _default_output_dir(), "tooltip": "Folder for prompt files, bindings, manifests, and cache. Defaults to ComfyUI's output folder."}),
                "duration": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 3600.0, "step": 0.1}),
                "chunk_chars": ("INT", {"default": 14000, "min": 3000, "max": 1000000}),
                "overlap_paragraphs": ("INT", {"default": 2, "min": 0, "max": 100}),
                "scenes_per_chunk": ("INT", {"default": 4, "min": 1, "max": 100}),
                "scenes_per_chapter": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "0 keeps every planned scene."}),
                "max_pictures": ("INT", {"default": 8, "min": 1, "max": 100}),
                "max_pictures_per_subject": ("INT", {"default": 4, "min": 1, "max": 20}),
                "max_audio": ("INT", {"default": 4, "min": 0, "max": 100}),
                "temperature": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 2048, "min": 512, "max": 100000, "tooltip": "2048 is sufficient for one H3 prompt and prevents very long malformed-JSON runs."}),
                "repair_attempts": ("INT", {"default": 2, "min": 0, "max": 10}), "force": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "picture_asset_briefs": ("MINIMAX_PICTURE_BRIEFS",), "audio_asset_briefs": ("MINIMAX_AUDIO_BRIEFS",),
                "delay": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("MINIMAX_PROMPTS",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(self, consolidated_references: dict[str, Any], clip: Any, chapter_paths: str, saved_chapter: str, out_dir: str, picture_asset_briefs: Iterable[dict[str, Any]] | None = None, audio_asset_briefs: Iterable[dict[str, Any]] | None = None, **params: Any) -> tuple[dict[str, Any]]:
        if not isinstance(consolidated_references, dict):
            raise TypeError(
                "consolidated_references must be the first output from "
                "ConsolidateReferencesNode or LoadConsolidatedReferencesNode."
            )
        if not isinstance(out_dir, str) or not out_dir.strip():
            raise ValueError("out_dir must be a non-empty folder path.")
        if not all(callable(getattr(clip, name, None)) for name in ("tokenize", "generate", "decode")):
            raise TypeError("GenerateH3PromptsNode requires a generative CLIP, not a standard text encoder.")

        refs = dict(consolidated_references)
        if not refs.get("picture_assets") and picture_asset_briefs is not None:
            refs["picture_assets"] = list(picture_asset_briefs)
        if not refs.get("audio_assets") and audio_asset_briefs is not None:
            refs["audio_assets"] = list(audio_asset_briefs)
        missing = sorted({"entities", "chapter_entity_map", "entity_asset_index", "picture_assets", "audio_assets"} - refs.keys())
        if missing:
            raise ValueError("Registry is incomplete; missing: " + ", ".join(missing))

        raw_paths = chapter_paths or saved_chapter
        paths = util.discover_inputs([Path(line.strip()) for line in raw_paths.splitlines() if line.strip()])
        if not paths:
            raise ValueError("No supported chapter files found. Provide chapter_paths or select saved_chapter.")

        pipeline = _load_pipeline_script(out_dir)
        model = type(clip).__name__
        pipeline.THINKING_ENABLED = False
        pipeline.CHAT_BACKEND = "comfyui-clip"
        pipeline.chat_json = lambda _client, _model, system, user, schema, temperature, max_tokens: _decode_json_with_clip(clip, system, user, schema, temperature, max_tokens)
        args = argparse.Namespace(out_dir=Path(out_dir.strip()), duration=float(params["duration"]), chunk_chars=int(params["chunk_chars"]), overlap_paragraphs=int(params["overlap_paragraphs"]), scenes_per_chunk=int(params["scenes_per_chunk"]), max_scenes=int(params["scenes_per_chapter"]), max_pictures=int(params["max_pictures"]), max_pictures_per_subject=int(params["max_pictures_per_subject"]), max_audio=int(params["max_audio"]), temperature=float(params["temperature"]), max_tokens=int(params["max_tokens"]), repair_attempts=int(params["repair_attempts"]), delay=float(params.get("delay", 0.0)), force=bool(params["force"]))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Generating H3 prompts for {len(paths)} chapter(s) with {model}")
        manifests = [pipeline.process_chapter(path, refs, None, model, args) for path in paths]
        for manifest in manifests:
            chapter_dir = args.out_dir / manifest["chapter_id"]
            for entry in manifest.get("outputs", []):
                if entry.get("prompt_file"):
                    entry["prompt_text"] = (chapter_dir / entry["prompt_file"]).read_text(encoding="utf-8").rstrip()
        result = {"schema_version": "minimax-h3-novel-prompts.v2", "model": model, "chapters": manifests}
        _log(f"H3 prompt generation complete: {sum(x['saved_prompt_count'] for x in manifests)} prompt(s) saved")
        return (result,)

"""LM Studio-backed ComfyUI node for chapter-reference extraction."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Iterable

from . import lmstudio_pipeline, util
from .chapter_selection import chapter_paths as selected_chapter_paths, saved_chapter_choices

def _default_output_dir() -> str:
    return "chapter_catalogs"


def _log(message: str) -> None:
    print(f"[minimax_h3_novel] {message}", flush=True)


class ExtractChapterReferencesNode:
    """Extract reference catalogs through LM Studio; no ComfyUI CLIP is loaded."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "chapter_paths": ("STRING", {"multiline": True, "default": "", "tooltip": "One chapter file or folder per line, inside ComfyUI's input directory. Relative paths start there."}),
            "saved_chapter": (saved_chapter_choices(), {"tooltip": "Previously uploaded chapter."}),
            "chunk_chars": ("INT", {"default": 5500, "min": 1000, "max": 1000000}),
            "overlap_paragraphs": ("INT", {"default": 2, "min": 0, "max": 100}),
            "temperature": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 2.0, "step": 0.05}),
            "max_tokens": ("INT", {"default": 2200, "min": 256, "max": 32768, "tooltip": "Normal JSON output budget per extraction/merge call."}),
            "force": ("BOOLEAN", {"default": False, "tooltip": "Ignore compatible cached chapter results."}),
            "out_dir": ("STRING", {"default": _default_output_dir(), "tooltip": "Folder inside ComfyUI's output/minimax_h3_novel directory. Relative paths start there."}),
            "merge_batch_size": ("INT", {"default": 6, "min": 2, "max": 32, "tooltip": "Partial catalogs per merge call."}),
        }, "optional": {
            "chapter_selection": ("MINIMAX_CHAPTER_SELECTION", {"tooltip": "Output of Select Chapters. Takes precedence over the legacy chapter fields."}),
        }}

    RETURN_TYPES = ("MINIMAX_CHAPTERS", "STRING")
    RETURN_NAMES = ("chapter_catalogs", "catalog_summary")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(self, lmstudio_config: dict[str, Any], chapter_paths: Iterable[Path], saved_chapter: str, out_dir: str, **params: Any) -> tuple[list[dict[str, Any]], str]:
        if not isinstance(out_dir, str) or not out_dir.strip():
            raise ValueError("out_dir must be a non-empty string.")
        output = util.output_path(out_dir.strip())
        raw_paths = selected_chapter_paths(params.get("chapter_selection"), chapter_paths, saved_chapter)
        items = [Path(x.strip()) for x in raw_paths.splitlines() if x.strip()] if isinstance(raw_paths, str) else [Path(x) for x in raw_paths]
        paths = util.discover_inputs(items)
        if not paths:
            raise ValueError("No supported chapter files found.")
        if not isinstance(lmstudio_config, dict):
            raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        pipeline = lmstudio_pipeline.load("extract")
        lmstudio_pipeline.configure_qwen(
            thinking=bool(lmstudio_config["thinking"]),
            length_retries=int(lmstudio_config["qwen35_length_retries"]),
            top_k=int(lmstudio_config["qwen35_top_k"]),
            min_p=float(lmstudio_config["qwen35_min_p"]),
            repeat_penalty=float(lmstudio_config["qwen35_repeat_penalty"]),
        )
        client, resolved_model = lmstudio_pipeline.make_client_and_model(pipeline, str(lmstudio_config["api_url"]))
        with client:
            args = argparse.Namespace(merge_batch_size=max(2, int(params["merge_batch_size"])), chunk_chars=int(params["chunk_chars"]), overlap_paragraphs=int(params["overlap_paragraphs"]), temperature=float(params["temperature"]), max_tokens=int(params["max_tokens"]), force=bool(params["force"]), base_url=lmstudio_config["api_url"])
            output.mkdir(parents=True, exist_ok=True)
            _log(f"LM Studio extraction: model={resolved_model}, chapters={len(paths)}")
            started = time.perf_counter()
            results = []
            for path in paths:
                lmstudio_pipeline.comfy_interrupt_check()
                saved = pipeline.process_chapter(path, output, client, resolved_model, args)
                results.append(util.load_json(saved))
            _log(f"Extraction complete in {time.perf_counter() - started:.1f}s")
            return results, util.catalog_summary(results)

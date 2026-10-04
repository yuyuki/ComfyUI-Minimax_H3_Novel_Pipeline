"""Standalone cinematic chapter adaptation node using the shared LM Studio client."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import cinematic_adaptation, configuration_snapshot, lmstudio_pipeline, progress, util
from .chapter_selection import chapter_path_list
from .run_output import stage_output


class CinematicChapterAdapterNode:
    """Save one chapter object, independently of reference extraction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "chapter_selection": ("MINIMAX_CHAPTER_SELECTION",),
            "out_dir": ("STRING", {"default": "cinematic_chapters"}),
            "chunk_chars": ("INT", {"default": 4000, "min": 1000, "max": 30000}),
            "temperature": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0, "step": 0.05}),
            "max_tokens": ("INT", {"default": 8192, "min": 256, "max": 32768}),
            "correction_attempts": ("INT", {"default": 2, "min": 0, "max": 10,
                                          "tooltip": "Retries for invalid fields or incomplete source coverage."}),
        }}

    RETURN_TYPES = ("MINIMAX_CINEMATIC_CHAPTERS", "STRING")
    RETURN_NAMES = ("cinematic_chapters", "saved_files")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"
    OUTPUT_NODE = True

    @progress.node_progress
    def run(self, lmstudio_config, chapter_selection, out_dir="cinematic_chapters",
            chunk_chars=4000, temperature=0.15, max_tokens=8192, correction_attempts=2):
        if not isinstance(out_dir, str) or not out_dir.strip():
            raise ValueError("out_dir must be a non-empty string.")
        if not isinstance(lmstudio_config, dict):
            raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        if chunk_chars < 1 or correction_attempts < 0 or max_tokens < 1:
            raise ValueError("Invalid adaptation size, token budget or correction attempts.")
        paths = util.discover_inputs([Path(path) for path in chapter_path_list(chapter_selection)])
        if not paths:
            raise ValueError("No supported chapter files found.")
        output = stage_output(lmstudio_config, out_dir.strip())
        pipeline = lmstudio_pipeline.load("extract")
        client, model = lmstudio_pipeline.make_client_and_model(pipeline, lmstudio_config["api_url"], lmstudio_config)
        settings = dict(chunk_chars=int(chunk_chars), temperature=float(temperature),
                        max_tokens=int(max_tokens), correction_attempts=int(correction_attempts))
        chapters, artifacts = [], []
        with client:
            snapshot = configuration_snapshot.start(
                output, "cinematic_adapter", lmstudio_config, model, argparse.Namespace(**settings), out_dir=out_dir,
                inputs={"chapters": [{"file": str(p), "sha256": configuration_snapshot.file_digest(p)} for p in paths]},
                extra={"schema_version": "minimax-cinematic-chapters.v2"},
            )
            for index, path in enumerate(paths):
                lmstudio_pipeline.comfy_interrupt_check()
                with progress.scope(index / len(paths), (index + 1) / len(paths)):
                    chapter = cinematic_adaptation.adapt_chapter(
                        client, model, util.read_chapter(path), chapter_name=path.stem, **settings,
                    )
                    # Persist completed work before progress or cancellation can interrupt it.
                    # The index distinguishes identically named chapters from separate folders.
                    saved = output / f"{index + 1:03d}_{path.stem}.cinematic.json"
                    util.save_json(saved, chapter)
                artifacts.append(saved)
                chapters.append({"source_file": str(path), "saved_file": str(saved), **chapter})
                lmstudio_pipeline.comfy_interrupt_check()
            configuration_snapshot.complete(snapshot, artifacts)
        return {"ui": {"text": [json.dumps(chapters, ensure_ascii=False, indent=2)]},
                "result": (chapters, "\n".join(str(path) for path in artifacts))}

"""ComfyUI entry point for independent, editable character evidence."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..core import character_evidence, lmstudio_json, lmstudio_pipeline, progress, util
from ..core.path_access import output_path
from ..core.run_output import stage_output
from .chapter_selection import chapter_path_list


class CharacterEvidenceExtractorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "chapter_selection": ("MINIMAX_CHAPTER_SELECTION",),
            "out_dir": ("STRING", {"default": "character_evidence"}),
            "chunk_chars": ("INT", {"default": 5500, "min": 1000, "max": 30000}),
            "thinking": ("BOOLEAN", {"default": True}),
            "temperature": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 2.0}),
            "max_tokens": ("INT", {"default": 8192, "min": 256, "max": 32768}),
            "correction_attempts": ("INT", {"default": 2, "min": 0, "max": 10}),
            "force": ("BOOLEAN", {"default": False}),
            "character_scope": (list(character_evidence.SCOPES), {"default": "proper_names"}),
        }}

    RETURN_TYPES = ("MINIMAX_CHARACTER_EVIDENCE", "STRING")
    RETURN_NAMES = ("character_evidence", "saved_file")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Recheck source files and disk caches even when widget values are unchanged.
        return float("nan")

    @progress.node_progress
    def run(self, lmstudio_config, chapter_selection, out_dir="character_evidence",
            chunk_chars=5500, thinking=True, temperature=0.15, max_tokens=8192,
            correction_attempts=2, force=False, character_scope="proper_names"):
        started = time.perf_counter()
        if character_scope not in character_evidence.SCOPES:
            raise ValueError("Invalid character_scope.")
        if not isinstance(lmstudio_config, dict):
            raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        if lmstudio_config.get("model_family", "Qwen") != "Qwen":
            raise ValueError("Character Evidence Extractor requires Qwen in LM Studio Configuration.")
        if not isinstance(out_dir, str) or not out_dir.strip():
            raise ValueError("out_dir must be a non-empty relative subfolder.")
        if chunk_chars < 1 or max_tokens < 1 or correction_attempts < 0 or not 0 <= temperature <= 2:
            raise ValueError("Invalid chunk size, token budget, correction attempts or temperature.")
        paths = util.discover_inputs([Path(p) for p in chapter_path_list(chapter_selection)])
        if not paths:
            raise ValueError("No supported chapters found. Select all novel chapter files or their folder.")
        config = {**lmstudio_config, "thinking": bool(thinking)}
        output = stage_output(config, out_dir.strip())
        args = argparse.Namespace(chunk_chars=int(chunk_chars), max_tokens=int(max_tokens),
                                  temperature=float(temperature), correction_attempts=int(correction_attempts),
                                  thinking=bool(thinking), force=bool(force), base_url=config["api_url"],
                                  character_scope=character_scope)
        sources = []
        for path in paths:
            lmstudio_pipeline.comfy_interrupt_check()
            sources.append(character_evidence.read_source(path))
        occurrences = character_evidence.index_names(sources) if character_scope == "proper_names" else [[] for _ in paths]
        diagnostics = {"character_scope": character_scope, "chapters": [], "llm_requests": 0,
                       "estimated_input_tokens": 0, "estimated_output_tokens": 0,
                       "token_estimation": "ceil(characters / 4); includes transport retries; not tokenizer usage",
                       "preprocessing_seconds": round(time.perf_counter() - started, 3)}
        client, model = lmstudio_pipeline.make_client_and_model(lmstudio_pipeline.load("extract"),
                                                               config["api_url"], config)
        chapters = []
        token = lmstudio_json.REQUEST_METRICS.set(diagnostics)
        try:
            with client:
                for index, path in enumerate(paths):
                    lmstudio_pipeline.comfy_interrupt_check()
                    with progress.scope(index / len(paths), (index + 1) / len(paths)):
                        chapters.append(character_evidence.extract_chapter(
                            path, output, client, model, args, index + 1,
                            source=sources[index], occurrences=occurrences[index],
                            cache_dir=output_path(".cache/character_evidence"), diagnostics=diagnostics["chapters"]))
        finally:
            lmstudio_json.REQUEST_METRICS.reset(token)
        data = {"format_version": character_evidence.VERSION,
                "location_convention": "Zero-based Unicode character offsets, exclusive end; one-based lines in source_text.",
                "chapters": chapters}
        saved = output / "character_evidence.json"
        util.save_json(saved, data)
        diagnostics["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        util.save_json(output / "character_evidence_diagnostics.json", diagnostics)
        return {"ui": {"text": [str(saved)]}, "result": (data, str(saved))}

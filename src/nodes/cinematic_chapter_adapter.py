"""Standalone cinematic chapter adaptation node using the shared LM Studio client."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core import cinematic_adaptation, cinematic_diagnostics, configuration_snapshot, lmstudio_pipeline, progress, util
from .chapter_selection import chapter_path_list
from ..core.run_output import stage_output


class CinematicChapterAdapterNode:
    """Save one chapter object, independently of reference extraction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "chapter_selection": ("MINIMAX_CHAPTER_SELECTION",),
            "thinking": ("BOOLEAN", {"default": True, "tooltip": "Request reasoning for Qwen, Gemma and Mistral through LM Studio. GPT-OSS always reasons; OpenAI has no thinking control."}),
            "max_tokens": ("INT", {"default": 16384, "min": 256, "max": 32768,
                                   "tooltip": "Output budget per request for generation, timing/context planning, scene validation and focused checks. "
                                   "Reasoning and JSON share this budget; retries cannot exceed it."}),
            "chunk_chars": ("INT", {"default": 4000, "min": 1000, "max": 30000,
                                    "tooltip": "Maximum source characters per reading window. Incomplete trailing scenes "
                                    "are read again with new source text. Instructions and review text add to request size."}),
            "temperature": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "duration": ("FLOAT", {"default": 8.0, "min": 4.0, "max": 15.0, "tooltip": "Maximum duration per cinematic scene, not a target to fill. Use the same duration in Generate H3 Prompts."}),
            "correction_attempts": ("INT", {"default": 2, "min": 0, "max": 10,
                                          "tooltip": "Retries for invalid fields, source coverage or semantic defects. "
                                          "When nonzero, persistent semantic defects trigger one final targeted repair round. "
                                          "Remaining narrative doubts become editable proposals; the first is selected automatically. "
                                          "Persistent timing overruns retain the scene with incomplete-review diagnostics."}),
            "out_dir": ("STRING", {"default": "cinematic_chapters"}),
        }, "optional": {
            "resume_run": ("STRING", {"default": "", "tooltip":
                "Previous timestamped run folder, e.g. 20261112124902. Resume matching scenes and drafts "
                "using the same out_dir, chapters, model and settings. Leave empty for a fresh run."}),
        }}

    RETURN_TYPES = ("MINIMAX_CINEMATIC_CHAPTERS", "STRING")
    RETURN_NAMES = ("cinematic_chapters", "saved_files")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @progress.node_progress
    def run(self, lmstudio_config, chapter_selection, out_dir="cinematic_chapters",
            chunk_chars=4000, temperature=0.15, max_tokens=16384, correction_attempts=2, thinking: bool = True, duration=8.0,
            resume_run=""):
        if not isinstance(out_dir, str) or not out_dir.strip():
            raise ValueError("out_dir must be a non-empty string.")
        if not isinstance(lmstudio_config, dict):
            raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        if chunk_chars < 1 or correction_attempts < 0 or max_tokens < 1:
            raise ValueError("Invalid adaptation size, token budget or correction attempts.")
        if type(duration) not in (int, float) or not 4 <= duration <= 15:
            raise ValueError("duration must be between 4 and 15 seconds.")
        paths = util.discover_inputs([Path(path) for path in chapter_path_list(chapter_selection)])
        if not paths:
            raise ValueError("No supported chapter files found.")
        lmstudio_config = {**lmstudio_config, "thinking": bool(thinking)}
        output = stage_output(lmstudio_config, out_dir.strip())
        if not isinstance(resume_run, str):
            raise ValueError("resume_run must be a previous timestamped run folder name.")
        resume_run = resume_run.strip()
        resume_cache = None
        if resume_run:
            resume_cache = stage_output({"run_folder": resume_run}, out_dir.strip()) / ".cache"
            if resume_cache == output / ".cache":
                raise ValueError("resume_run must refer to a previous run, not the current run.")
            if not resume_cache.is_dir():
                raise ValueError("Previous run cache does not exist; use the same out_dir as that run.")
        pipeline = lmstudio_pipeline.load("extract")
        client, model = lmstudio_pipeline.make_client_and_model(pipeline, lmstudio_config["api_url"], lmstudio_config)
        settings = dict(duration=float(duration), chunk_chars=int(chunk_chars), temperature=float(temperature),
                        max_tokens=int(max_tokens), correction_attempts=int(correction_attempts))
        chapters, artifacts, diagnostic_artifacts = [], [], []
        with client:
            snapshot = configuration_snapshot.start(
                output, "cinematic_adapter", lmstudio_config, model, argparse.Namespace(**settings, thinking=bool(thinking)), out_dir=out_dir,
                inputs={"chapters": [{"file": str(p), "sha256": configuration_snapshot.file_digest(p)} for p in paths]},
                extra={"schema_version": "minimax-cinematic-chapters.v2", "resume_run": resume_run},
            )
            if resume_cache is not None:
                previous = util.load_json(resume_cache.parent / "cinematic_adapter_configuration.json")
                def comparable(record):
                    return {**{k: record[k] for k in ("lmstudio_config", "resolved_model", "model_controls", "inputs")},
                            "node_settings": {k: v for k, v in record["node_settings"].items() if k != "resume_run"}}
                if comparable(previous) != comparable(snapshot[1]):
                    raise ValueError("resume_run does not match the chapters, model or settings; restore them or clear resume_run.")
            for index, path in enumerate(paths):
                lmstudio_pipeline.comfy_interrupt_check()
                cache_dir = output / ".cache" / f"{index + 1:03d}_{path.stem}"
                persisted_scenes = 0
                latest_state = None
                diagnostics_path = output / f"{index + 1:03d}_{path.stem}.cinematic.diagnostics.json"
                resume_state = None
                if resume_cache is not None:
                    previous_checkpoint = resume_cache / cache_dir.name / "progress.json"
                    if previous_checkpoint.is_file():
                        resume_state = util.load_json(previous_checkpoint)
                        if (resume_state.get("schema_version") != "minimax-cinematic-checkpoint.v2"
                                or resume_state.get("source_file") != str(path)
                                or resume_state.get("model") != model
                                or resume_state.get("settings") != {**settings, "thinking": bool(thinking)}):
                            raise ValueError(f"Invalid or mismatched cinematic checkpoint for {path.name}.")

                def checkpoint(state):
                    nonlocal persisted_scenes, latest_state
                    # Replace atomically so interruption leaves the last complete
                    # checkpoint readable. Drafts never masquerade as final output.
                    data = {"schema_version": "minimax-cinematic-checkpoint.v2",
                            "source_file": str(path), "model": model,
                            "settings": {**settings, "thinking": bool(thinking)}, **state}
                    pending = cache_dir / "progress.json.tmp"
                    util.save_json(pending, data)
                    pending.replace(cache_dir / "progress.json")
                    for scene in state["sequences"][persisted_scenes:]:
                        target = cache_dir / f"scene_{scene['sequence']:03d}.json"
                        pending = target.with_suffix(".json.tmp")
                        util.save_json(pending, scene)
                        pending.replace(target)
                        persisted_scenes += 1
                    latest_state = state
                    diagnostics = cinematic_diagnostics.report(
                        state, chapter_name=path.stem, source_file=str(path), maximum_seconds=float(duration))
                    pending = diagnostics_path.with_suffix(".json.tmp")
                    util.save_json(pending, diagnostics)
                    pending.replace(diagnostics_path)
                    print(f"    Cinematic checkpoint: {state['stage']}; "
                          f"saved_scenes={persisted_scenes}, pending_scenes={len(state['pending_sequences'])}.", flush=True)

                with progress.scope(index / len(paths), (index + 1) / len(paths)):
                    chapter = cinematic_adaptation.adapt_chapter(
                        client, model, util.read_chapter(path), chapter_name=path.stem, **settings,
                        on_checkpoint=checkpoint,
                        resume_state=resume_state,
                    )
                    # Persist completed work before progress or cancellation can interrupt it.
                    # The index distinguishes identically named chapters from separate folders.
                    saved = output / f"{index + 1:03d}_{path.stem}.cinematic.json"
                    chapter = {**chapter, "source_file": str(path)}
                    diagnostics = cinematic_diagnostics.report(
                        latest_state or chapter, chapter_name=path.stem, source_file=str(path),
                        maximum_seconds=float(duration), completed=True)
                    chapter = cinematic_diagnostics.downstream_chapter(chapter)
                    util.save_json(saved, chapter)
                    pending = diagnostics_path.with_suffix(".json.tmp")
                    util.save_json(pending, diagnostics)
                    pending.replace(diagnostics_path)
                artifacts.append(saved)
                diagnostic_artifacts.append(diagnostics_path)
                chapters.append({"source_file": str(path), "saved_file": str(saved), **chapter})
                lmstudio_pipeline.comfy_interrupt_check()
            configuration_snapshot.complete(snapshot, artifacts + diagnostic_artifacts)
        return {"ui": {"text": [json.dumps(chapters, ensure_ascii=False, indent=2)]},
                "result": (chapters, "\n".join(str(path) for path in artifacts))}

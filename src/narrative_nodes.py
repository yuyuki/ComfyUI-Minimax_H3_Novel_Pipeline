"""Optional cinematic preprocessing with inspectable narrative state outputs."""
from __future__ import annotations

import json
from pathlib import Path

from . import lmstudio_pipeline, narrative_state, progress, util
from .chapter_selection import chapter_paths
from .run_output import stage_output


class NovelCinematicSimplifierNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "chapter_selection": ("MINIMAX_CHAPTER_SELECTION",),
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "out_dir": ("STRING", {"default": "cinematic_narrative"}),
            "chunk_chars": ("INT", {"default": 6000, "min": 1000, "max": 30000}),
            "correction_attempts": ("INT", {"default": 2, "min": 0, "max": 10}),
        }}

    RETURN_TYPES = ("MINIMAX_CINEMATIC_NARRATIVE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cinematic_narrative", "cinematic_text", "validation_report", "state_before_json", "events_json", "state_after_json")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @progress.node_progress
    def run(self, chapter_selection, lmstudio_config, out_dir="cinematic_narrative", chunk_chars=6000, correction_attempts=2):
        paths = util.discover_inputs([Path(p.strip()) for p in chapter_paths(chapter_selection).splitlines() if p.strip()])
        if not paths:
            raise ValueError("No supported chapter files found.")
        output = stage_output(lmstudio_config, out_dir)
        pipeline = lmstudio_pipeline.load("generate")
        client, model = lmstudio_pipeline.make_client_and_model(pipeline, lmstudio_config["api_url"], lmstudio_config)
        records = {}
        with client:
            for path in progress.steps(paths):
                original = util.read_chapter(path)
                texts, reports, segments = [], [], []
                current = None
                for chunk in util.split_chunks(original, chunk_chars, 0):
                    result, report = narrative_state.simplify(client, model, chunk, correction_attempts)
                    contract, state_report = narrative_state.track_scene(
                        client, model, chunk, result["cinematic_text"], current, correction_attempts)
                    current = contract["state_after"]
                    texts.append(result["cinematic_text"])
                    reports.append({"simplification": report, "state_tracking": state_report})
                    segments.append({"original_text": chunk, "cinematic_text": result["cinematic_text"], "contract": contract})
                records[str(path.resolve())] = {"source_digest": narrative_state.source_digest(original),
                    "cinematic_text": "\n\n".join(texts), "validation_report": reports, "segments": segments}
        bundle = {"schema_version": "minimax-cinematic-narrative.v1", "chapters": records,
                  "correction_attempts": correction_attempts, "resolved_model": model}
        output.mkdir(parents=True, exist_ok=True)
        (output / "cinematic_narrative.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        return (bundle, "\n\n".join(r["cinematic_text"] for r in records.values()), json.dumps(
            {key: r["validation_report"] for key, r in records.items()}, ensure_ascii=False, indent=2), *[
                json.dumps({key: [s["contract"][field] for s in r["segments"]] for key, r in records.items()}, ensure_ascii=False, indent=2)
                for field in ("state_before", "events", "state_after")])

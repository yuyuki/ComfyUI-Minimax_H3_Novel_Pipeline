"""Narrative continuity orchestration and the compatible cinematic simplifier node."""
from __future__ import annotations

import json
from pathlib import Path

from . import lmstudio_pipeline, narrative_state, progress, util
from .chapter_selection import chapter_paths
from .run_output import stage_output


class NarrativeContinuityNode:
    """Track selected prose without stylistic changes or reference catalog generation."""

    simplify_prose = False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "chapter_selection": ("MINIMAX_CHAPTER_SELECTION",),
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "out_dir": ("STRING", {"default": "cinematic_narrative"}),
            "chunk_chars": ("INT", {"default": 6000, "min": 1000, "max": 30000}),
            "correction_attempts": ("INT", {"default": 2, "min": 0, "max": 10}),
        }, "optional": {"cinematic_narrative": ("MINIMAX_CINEMATIC_NARRATIVE",)}}

    RETURN_TYPES = ("MINIMAX_CINEMATIC_NARRATIVE", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("cinematic_narrative", "cinematic_text", "validation_report", "state_before_json", "events_json", "state_after_json")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @progress.node_progress
    def run(self, chapter_selection, lmstudio_config, out_dir="cinematic_narrative", chunk_chars=6000, correction_attempts=2,
            cinematic_narrative=None):
        paths = util.discover_inputs([Path(p.strip()) for p in chapter_paths(chapter_selection).splitlines() if p.strip()])
        if not paths:
            raise ValueError("No supported chapter files found.")
        # Validate supplied preprocessing before opening the client or making requests.
        prepared = []
        for path in paths:
            original = util.read_chapter(path)
            if cinematic_narrative is None:
                passages = [{"original_text": text, "cinematic_text": text}
                            for text in util.split_chunks(original, chunk_chars, 0)]
            else:
                if (not isinstance(cinematic_narrative, dict)
                        or cinematic_narrative.get("schema_version") != "minimax-cinematic-narrative.v1"):
                    raise ValueError("Invalid cinematic_narrative bundle.")
                chapters = cinematic_narrative.get("chapters")
                if not isinstance(chapters, dict):
                    raise ValueError("Invalid cinematic narrative chapters.")
                record = chapters.get(str(path.resolve()))
                if not isinstance(record, dict) or record.get("source_digest") != narrative_state.source_digest(original):
                    raise ValueError(f"Missing or stale cinematic narrative for {path.name}; rerun the simplifier.")
                passages = record.get("segments")
                if not isinstance(passages, list) or not passages or any(
                    not isinstance(s, dict) or any(not isinstance(s.get(k), str) or not s[k].strip()
                        for k in ("original_text", "cinematic_text")) for s in passages
                ):
                    raise ValueError("Cinematic narrative requires nonempty original/cinematic passages.")
                # split_chunks normalizes paragraph separators; compare using that same rule.
                reconstructed = "\n\n".join(s["original_text"] for s in passages)
                if (util.split_chunks(reconstructed, len(reconstructed) + 1, 0)
                        != util.split_chunks(original, len(original) + 1, 0)):
                    raise ValueError("Cinematic passages must cover the original chapter in order.")
            if not passages:
                raise ValueError(f"No narrative text in {path.name}.")
            prepared.append((path, original, passages))
        output = stage_output(lmstudio_config, out_dir)
        pipeline = lmstudio_pipeline.load("generate")
        client, model = lmstudio_pipeline.make_client_and_model(pipeline, lmstudio_config["api_url"], lmstudio_config)
        records = {}
        with client:
            for path, original, passages in progress.steps(prepared):
                texts, reports, segments = [], [], []
                current = None
                for passage in passages:
                    chunk = passage["original_text"]
                    if self.simplify_prose:
                        result, report = narrative_state.simplify(client, model, chunk, correction_attempts)
                    else:
                        result = {"cinematic_text": passage["cinematic_text"]}
                        report = {"valid": True, "skipped": True}
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


class NovelCinematicSimplifierNode(NarrativeContinuityNode):
    """Preserve existing workflows: simplify, then invoke the shared continuity layer."""

    simplify_prose = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": super().INPUT_TYPES()["required"]}

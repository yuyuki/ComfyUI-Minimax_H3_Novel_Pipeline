"""LM Studio-backed ComfyUI node for chapter-reference extraction."""
from __future__ import annotations

from . import progress

import argparse
from typing import Any

from . import cinematic_references, configuration_snapshot, lmstudio_pipeline, util
from .run_output import stage_output

def _default_output_dir() -> str:
    return "chapter_catalogs"


def _log(message: str) -> None:
    print(f"[minimax_h3_novel] {message}", flush=True)


class ExtractChapterReferencesNode:
    """Extract independent cinematic timelines with sequence/phase provenance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "cinematic_chapters": ("MINIMAX_CINEMATIC_CHAPTERS", {"tooltip": "Ordered chapter timelines from Cinematic Chapter Adapter."}),
            "chunk_chars": ("INT", {"default": 5500, "min": 1000, "max": 1000000}),
            "temperature": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 2.0, "step": 0.05}),
            "max_tokens": ("INT", {"default": 8192, "min": 256, "max": 32768, "tooltip": "JSON output budget per sequence-phase extraction call. Dense catalogs may need more tokens."}),
            "force": ("BOOLEAN", {"default": False, "tooltip": "Ignore compatible cached chapter results."}),
            "out_dir": ("STRING", {"default": _default_output_dir(), "tooltip": "Subfolder of the current timestamped run inside output/minimax_h3_novel."}),
        }}

    RETURN_TYPES = ("MINIMAX_CHAPTERS", "STRING")
    RETURN_NAMES = ("chapter_catalogs", "catalog_summary")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @progress.node_progress
    def run(self, lmstudio_config: dict[str, Any], cinematic_chapters: Any, out_dir: str, **params: Any) -> tuple[list[dict[str, Any]], str]:
        if not isinstance(out_dir, str) or not out_dir.strip():
            raise ValueError("out_dir must be a non-empty string.")
        output = stage_output(lmstudio_config, out_dir.strip())
        chapters = cinematic_references.parse_chapters(cinematic_chapters)
        if not isinstance(lmstudio_config, dict):
            raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        pipeline = lmstudio_pipeline.load("extract")
        client, resolved_model = lmstudio_pipeline.make_client_and_model(pipeline, str(lmstudio_config["api_url"]), lmstudio_config)
        with client:
            args = argparse.Namespace(chunk_chars=int(params["chunk_chars"]), temperature=float(params["temperature"]), max_tokens=int(params["max_tokens"]), force=bool(params["force"]), base_url=lmstudio_config["api_url"])
            output.mkdir(parents=True, exist_ok=True)
            snapshot = configuration_snapshot.start(
                output, "extract", lmstudio_config, resolved_model, args, out_dir=out_dir,
                inputs={"chapters": [{"chapter_name": c["chapter_name"], "sha256": cinematic_references.chapter_digest(c)} for c in chapters]},
            )
            _log(f"LM Studio extraction: model={resolved_model}, chapters={len(chapters)}")
            results = []
            artifacts = []
            reserved_ids = {pipeline.slug(c["chapter_name"]) for c in chapters}
            used_ids = set()
            for index, chapter in enumerate(chapters):
                lmstudio_pipeline.comfy_interrupt_check()
                with progress.scope(index / len(chapters), (index + 1) / len(chapters)):
                    chapter_id = pipeline.slug(chapter["chapter_name"])
                    if sum(pipeline.slug(c["chapter_name"]) == chapter_id for c in chapters) > 1:
                        base_id = f"{chapter_id}_{index + 1:03d}"
                        chapter_id = base_id
                        suffix = 1
                        while chapter_id in reserved_ids or chapter_id in used_ids:
                            chapter_id = f"{base_id}_{suffix}"
                            suffix += 1
                    used_ids.add(chapter_id)
                    saved = cinematic_references.process_chapter(chapter, output, client, resolved_model, args, chapter_id)
                    results.append(util.load_json(saved))
                    artifacts.append(saved)
            configuration_snapshot.complete(snapshot, artifacts)
            return results, util.catalog_summary(results)

"""LM Studio-backed MiniMax H3 prompt-generation ComfyUI node."""
from __future__ import annotations

from ..core import progress

import argparse
from typing import Any

from ..core import cinematic_references, configuration_snapshot, lmstudio_pipeline, util
from ..core.path_access import confined_path
from ..core.run_output import stage_output
from ..core.image_prompt_export import export_image_prompts
from .spatial_continuity import validate_plan


def _default_output_dir() -> str:
    return "h3_prompts"


class GenerateH3PromptsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "thinking": ("BOOLEAN", {"default": True, "tooltip": "Reasoning for reference metadata, prompt generation and format repairs. Camera and continuity review have separate controls. GPT-OSS always reasons; OpenAI has no thinking control."}),
            "max_tokens": ("INT", {"default": 16384, "min": 256, "max": 100000, "tooltip": "Output token budget for prompt generation, repairs and continuity planning/review, including reasoning. Reasoning-limit retries use at least 16384 tokens."}),
            "temperature": ("FLOAT", {"default": 0.38, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "consolidated_references": ("MINIMAX_REGISTRY",),
            "cinematic_chapter": ("MINIMAX_CINEMATIC_CHAPTERS", {"tooltip": "Cinematic Chapter Adapter or Load Cinematic Chapters; one H3 scene per supplied sequence."}),
            "max_shots": ("INT", {"default": 1, "min": 1, "max": 6, "tooltip": "Maximum camera shots per scene. Qwen may use fewer; a shot needs at least 2.5 seconds. Default 1 preserves existing workflows."}),
            "max_scenes": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Maximum scenes per chapter; 0 generates all scenes."}),
            "max_pictures": ("INT", {"default": 8, "min": 1, "max": 9}), "max_pictures_per_subject": ("INT", {"default": 4, "min": 1, "max": 9}), "max_audio": ("INT", {"default": 3, "min": 0, "max": 3}),
            "repair_attempts": ("INT", {"default": 2, "min": 0, "max": 10}), "force": ("BOOLEAN", {"default": False}), "out_dir": ("STRING", {"default": _default_output_dir(), "tooltip": "Subfolder of the current timestamped run inside output/minimax_h3_novel."}),
            "refine_camera": ("BOOLEAN", {"default": True, "tooltip": "After prompt validation, ask LM Studio for camera-only descriptions for each shot. Adds one request per scene."}),
        }, "optional": {
            "spatial_continuity": ("MINIMAX_SPATIAL_CONTINUITY", {"tooltip": "Connect Spatial Continuity or Load Spatial Continuity to use its staging plan and review final prompts for consistency."}),
            "camera_direction": ("STRING", {"multiline": True, "default": "", "tooltip": "Optional camera axis, screen placement, starting view, movement and ending view. Applies to each generated scene."}),
            "camera_thinking": ("BOOLEAN", {"default": True, "tooltip": "Reasoning for the optional camera-only pass. Independent of generation thinking. GPT-OSS always reasons; OpenAI has no thinking control."}),
            "review_thinking": ("BOOLEAN", {"default": True, "tooltip": "Reasoning for continuity reviews and their repairs. Off avoids spending the review budget on reasoning. GPT-OSS always reasons; OpenAI has no thinking control."}),
            "resume_run": ("STRING", {"default": "", "tooltip":
                "Reuse matching prompts and completed request checkpoints from a previous timestamped run with the same out_dir. "
                "Example: 20261031202452. Changed requests are recomputed; force ignores caches."}),
        }}

    RETURN_TYPES = ("MINIMAX_PROMPTS", "STRING", "STRING")
    RETURN_NAMES = ("prompts", "prompt_text", "image_prompt_text")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"
    OUTPUT_NODE = True

    @classmethod
    def validate_parameters(cls, params: dict[str, Any]) -> None:
        """Reject invalid saved/API values before filesystem or network work."""
        inputs = cls.INPUT_TYPES()
        unknown = params.keys() - (inputs["required"].keys() | inputs["optional"].keys())
        if unknown:
            raise ValueError(f"Unknown Generate H3 parameters: {', '.join(sorted(unknown))}.")
        for name, (kind, *options) in cls.INPUT_TYPES()["required"].items():
            if kind not in ("INT", "FLOAT"):
                continue
            limits = options[0]
            value = params.get(name, limits["default"])
            valid_type = type(value) is int if kind == "INT" else type(value) in (int, float)
            if not valid_type or not limits["min"] <= value <= limits["max"]:
                raise ValueError(
                    f"{name} must be {kind.lower()} between {limits['min']} and {limits['max']}. "
                    "Set it on Generate H3 Prompts."
                )

    @progress.node_progress
    def run(self, consolidated_references: dict[str, Any], lmstudio_config: dict[str, Any], cinematic_chapter: Any, out_dir: str, thinking: bool = True, **params: Any) -> dict[str, Any]:
        self.validate_parameters(params)
        if not isinstance(consolidated_references, dict): raise TypeError("consolidated_references must be a registry object.")
        if not isinstance(out_dir, str) or not out_dir.strip(): raise ValueError("out_dir must be a non-empty string.")
        output = stage_output(lmstudio_config, out_dir.strip())
        resume_run = params.get("resume_run", "")
        if not isinstance(resume_run, str):
            raise ValueError("resume_run must be a previous timestamped run folder name.")
        resume_run = resume_run.strip()
        resume_cache_dir = None
        if resume_run:
            resume_cache_dir = stage_output({"run_folder": resume_run}, out_dir.strip()) / ".cache"
            if resume_cache_dir == output / ".cache":
                raise ValueError("resume_run must refer to a previous run, not the current run.")
            if not resume_cache_dir.is_dir():
                raise ValueError("Previous run cache does not exist; use the same out_dir as that run.")
        util.require_schema(consolidated_references, util.REGISTRY_SCHEMA)
        if not isinstance(lmstudio_config, dict): raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        chapters = cinematic_references.parse_chapters(cinematic_chapter)
        for chapter in chapters:
            duration = chapter.get("duration")
            if type(duration) not in (int, float) or not 4 <= duration <= 15:
                raise ValueError("Cinematic input requires duration (4-15 seconds); rerun Cinematic Chapter Adapter.")
        lmstudio_config = {**lmstudio_config, "thinking": bool(thinking)}
        pipeline = lmstudio_pipeline.load("generate")
        selected_ids = set()
        for chapter in chapters:
            chapter_id, _ = pipeline.selected_timeline(consolidated_references, chapter)
            if chapter_id in selected_ids:
                raise ValueError("Multiple cinematic chapters resolve to the same consolidated chapter; "
                                 "re-extract and reconsolidate with source provenance.")
            selected_ids.add(chapter_id)
        client, resolved_model = lmstudio_pipeline.make_client_and_model(pipeline, str(lmstudio_config["api_url"]), lmstudio_config)
        with client:
            keys = ("max_scenes", "max_pictures", "max_pictures_per_subject", "max_audio", "temperature", "max_tokens", "repair_attempts", "force")
            args = argparse.Namespace(**{key: params[key] for key in keys}, out_dir=output,
                                      max_shots=int(params.get("max_shots", 1)))
            if not 1 <= args.max_shots <= 6:
                raise ValueError("max_shots must be between 1 and 6. Set it on Generate H3 Prompts.")
            args.camera_direction = str(params.get("camera_direction", "")).strip()
            args.refine_camera = bool(params.get("refine_camera", False))
            args.camera_thinking = bool(params.get("camera_thinking", True))
            args.review_thinking = bool(params.get("review_thinking", True))
            continuity = params.get("spatial_continuity")
            if continuity is not None:
                validate_plan(continuity, consolidated_references)
                args.spatial_plan = continuity
            args.out_dir.mkdir(parents=True, exist_ok=True)
            args.thinking = bool(thinking)
            snapshot = configuration_snapshot.start(
                output, "generate", lmstudio_config, resolved_model, args, out_dir=out_dir,
                inputs={"chapters": [{"chapter_name": c["chapter_name"], "duration": c["duration"], "sha256": cinematic_references.chapter_digest(c)} for c in chapters],
                        "consolidated_references_sha256": configuration_snapshot.content_digest(consolidated_references)},
                extra={"resume_run": resume_run},
            )
            if resume_cache_dir is not None:
                args.resume_cache_dir = resume_cache_dir
                try:
                    previous = util.load_json(resume_cache_dir.parent / "generate_configuration.json")
                    current = snapshot[1]
                    def settings(record):
                        return {k: v for k, v in record["node_settings"].items()
                                if k not in {"resume_run", "force", "resume_cache_dir"}}
                    matching = (
                        not args.force and settings(previous) == settings(current)
                        and all(previous.get(k) == current.get(k) for k in
                                ("lmstudio_config", "resolved_model", "model_controls", "inputs"))
                    )
                    if matching:
                        args.resume_scene_files = resume_cache_dir.parent
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            image_records, image_text = export_image_prompts(consolidated_references, output / "image_prompts")
            manifests = []
            for index, chapter in enumerate(chapters):
                lmstudio_pipeline.comfy_interrupt_check()
                chapter_args = argparse.Namespace(**vars(args), duration=float(chapter["duration"]))
                with progress.scope(0.98 * index / len(chapters), 0.98 * (index + 1) / len(chapters)):
                    manifests.append(pipeline.process_chapter(chapter, consolidated_references, client, resolved_model, chapter_args))
            for manifest in manifests:
                target = confined_path(manifest["chapter_id"], args.out_dir)
                for scene in manifest["outputs"]:
                    if not scene.get("prompt_file"):
                        continue
                    scene["asset_sheet_text"] = confined_path(scene["prompt_file"], target).read_text(encoding="utf-8").rstrip()
                    bindings = util.load_json(confined_path(scene["assets_file"], target))
                    scene["prompt_text"] = bindings["copy_paste_prompt"]
                    scene["bindings"] = bindings
                    scene["picture_asset_ids"] = [x["asset_id"] for x in bindings["picture_input_order"]]
                    scene["audio_asset_ids"] = [x["asset_id"] for x in bindings["audio_input_order"]]
            prompt_text = "\n\n".join(
                f"# {manifest.get('chapter_id', 'Chapter')} — Scene {scene.get('index', '?')}\n\n{scene['asset_sheet_text']}"
                for manifest in manifests
                for scene in manifest["outputs"]
                if scene.get("prompt_text")
            )
            configuration_snapshot.complete(snapshot, [output / "image_prompts"] + [
                confined_path(manifest["chapter_id"], output) for manifest in manifests
            ])
            return {"ui": {"text": [prompt_text, image_text]}, "result": (
                {"schema_version": "minimax-h3-novel-prompts.v3", "model": resolved_model,
                 "chapters": manifests, "image_prompts": image_records}, prompt_text, image_text)}

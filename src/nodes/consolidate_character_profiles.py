"""ComfyUI entry point for source-grounded character profiles."""
from __future__ import annotations

import argparse
from copy import deepcopy

from ..core import character_profiles, configuration_snapshot, lmstudio_pipeline, progress
from ..core.run_output import stage_output


class ConsolidateCharacterProfilesNode:
    DESCRIPTION = "Reconcile selected character evidence with Qwen or GPT-OSS, retaining exact sources and conflicting observations."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "character_evidence": ("MINIMAX_CHARACTER_EVIDENCE",),
            "lmstudio_config": ("MINIMAX_LMSTUDIO_CONFIG",),
            "out_dir": ("STRING", {"default": "character_profiles"}),
            "thinking": ("BOOLEAN", {"default": False}),
            "temperature": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 2.0, "step": 0.01, "round": 0.01}),
            "max_tokens": ("INT", {"default": 16384, "min": 256, "max": 100000}),
            "context_tokens": ("INT", {"default": 65536, "min": 8192, "max": 2000000,
                "tooltip": "Must not exceed the loaded LM Studio context. Includes output and retry reserves."}),
            "correction_attempts": ("INT", {"default": 0, "min": 0, "max": 10,
                "tooltip": "0 skips invalid decisions immediately, without correction or subdivision requests."}),
            "force": ("BOOLEAN", {"default": False}),
            "persistence_filter": (["persistent", "everything"], {"default": "persistent",
                "tooltip": "Consolidate only persistent observations, or include temporary and unknown observations too."}),
        }, "optional": {
            "resume_run": ("STRING", {"default": "", "tooltip":
                "Reuse validated decisions from a previous timestamped run with the same out_dir. "
                "Example: 20261031202452. Changed requests are recomputed; force ignores caches."}),
        }}

    RETURN_TYPES = ("MINIMAX_CHARACTER_PROFILES", "STRING")
    RETURN_NAMES = ("character_profiles", "saved_file")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @progress.node_progress
    def run(self, character_evidence, lmstudio_config, out_dir="character_profiles",
            thinking=False, temperature=0.12, max_tokens=16384, correction_attempts=0, force=False, context_tokens=65536,
            persistence_filter="persistent", resume_run=""):
        if not isinstance(lmstudio_config, dict):
            raise TypeError("lmstudio_config must come from LM Studio Configuration.")
        if lmstudio_config.get("model_family", "Qwen") not in {"Qwen", "GPT-OSS"}:
            raise ValueError("Consolidate Character Profiles requires Qwen or GPT-OSS in LM Studio Configuration.")
        if not isinstance(out_dir, str) or not out_dir.strip():
            raise ValueError("out_dir must be a non-empty relative subfolder.")
        if max_tokens < 1 or correction_attempts < 0 or not 0 <= temperature <= 2:
            raise ValueError("Invalid token budget, correction attempts or temperature.")
        if persistence_filter not in {"persistent", "everything"}:
            raise ValueError("persistence_filter must be persistent or everything.")
        records = character_profiles.prepare_evidence(character_evidence)
        if persistence_filter == "persistent":
            # Keep the saved source document and its coverage contract aligned with
            # the selected observations, without mutating the upstream evidence.
            character_evidence = deepcopy(character_evidence)
            for chapter in character_evidence["chapters"]:
                chapter["evidence"] = [item for item in chapter["evidence"]
                                       if isinstance(item, dict) and item.get("persistence") == "persistent"]
            records = {key: item for key, item in records.items() if item.get("persistence") == "persistent"}
        config = {**lmstudio_config, "thinking": bool(thinking)}
        output = stage_output(config, out_dir.strip())
        if not isinstance(resume_run, str):
            raise ValueError("resume_run must be a previous timestamped run folder name.")
        resume_cache_dir = None
        if resume_run.strip():
            resume_cache_dir = stage_output({'run_folder': resume_run.strip()}, out_dir.strip()) / '.cache'
            if resume_cache_dir == output / '.cache':
                raise ValueError("resume_run must refer to a previous run, not the current run.")
            if not resume_cache_dir.is_dir():
                raise ValueError("Previous run cache does not exist; use the same out_dir as that run.")
        args = argparse.Namespace(max_tokens=int(max_tokens), temperature=float(temperature),
                                  correction_attempts=int(correction_attempts), thinking=bool(thinking),
                                  force=bool(force), context_tokens=int(context_tokens), base_url=config["api_url"])
        # Path is operational state, excluded from the scalar request fingerprint.
        if resume_cache_dir is not None:
            args.resume_cache_dir = resume_cache_dir
        client, model = lmstudio_pipeline.make_client_and_model(lmstudio_pipeline.load("consolidate"),
                                                               config["api_url"], config)
        with client:
            # Record UI controls, not operational Path objects used by cache lookup.
            snapshot_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k != 'resume_cache_dir'})
            snapshot = configuration_snapshot.start(
                output, 'character_profiles', config, model, snapshot_args, out_dir=out_dir.strip(),
                inputs={'character_evidence_sha256': configuration_snapshot.content_digest(character_evidence)},
                extra={'persistence_filter': persistence_filter, 'resume_run': resume_run.strip()},
            )
            data, saved = character_profiles.consolidate(character_evidence, records, output, client, model, args)
            configuration_snapshot.complete(snapshot, [output / name for name in (
                'character_profiles.json', 'character_profiles.schema.json',
                'character_profiles_validation.json', 'character_profiles_requests.json',
            )])
        return {"ui": {"text": [saved]}, "result": (data, saved)}

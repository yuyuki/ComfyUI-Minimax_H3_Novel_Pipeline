"""Secure provider configuration node for the MiniMax H3 workflow."""
from __future__ import annotations

from ..core import progress

from typing import Any

from ..core import lmstudio_settings
from ..core.run_output import reserve_run


class LMStudioConfigurationNode:
    """Share the selected provider's saved, non-secret settings."""

    DESCRIPTION = """\
Share LM Studio or OpenAI settings across the workflow. Connect `lmstudio_config`
to Extract, Consolidate and Generate H3 Prompts.

Enable and save a provider, API key and model in ComfyUI Settings: `MiniMax H3 Novel → Providers`.
Choose an available provider here. Its saved model is displayed automatically; the URL and key
come from settings. The key is not saved in the workflow. Qwen controls appear only for Qwen
models in LM Studio. GPT-OSS reasoning is configured in LM Studio. Set max_tokens on each processing node.
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (["LM Studio", "OpenAI"], {
                    "default": "LM Studio",
                    "tooltip": "Enabled providers with a saved API key and model. Configure them in Settings → MiniMax H3 Novel → Providers.",
                }),
                "qwen35_length_retries": ("INT", {
                    "default": 2, "min": 0, "max": 10,
                    "tooltip": "Number of compact retries after incomplete Qwen3.5 JSON output.",
                }),
                "qwen35_top_k": ("INT", {
                    "default": 20, "min": 1, "max": 200,
                    "tooltip": "Qwen3.5/LM Studio sampling: limits token choices for more stable JSON output.",
                }),
                "qwen35_min_p": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Qwen3.5/LM Studio sampling. 0 disables this filter; keep 0 for structured extraction.",
                }),
                "qwen35_repeat_penalty": ("FLOAT", {
                    "default": 1.05, "min": 0.8, "max": 2.0, "step": 0.01,
                    "tooltip": "Penalizes Qwen3.5 repetition; 1.05 reduces loops while preserving JSON list detail.",
                }),
            },
        }

    RETURN_TYPES = ("MINIMAX_LMSTUDIO_CONFIG", "STRING")
    RETURN_NAMES = ("lmstudio_config", "configuration_status")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Saving a new run is intentional even when the queued graph is unchanged.
        return float("nan")

    @progress.node_progress
    def run(self, backend: str = "LM Studio",
            qwen35_length_retries: int = 2,
            qwen35_top_k: int = 20,
            qwen35_min_p: float = 0.0, qwen35_repeat_penalty: float = 1.05) -> tuple[dict[str, Any], str]:
        connection = lmstudio_settings.get_provider_configuration(backend)
        config = {
            **connection,
            "qwen35_length_retries": max(0, int(qwen35_length_retries)),
            "qwen35_top_k": max(1, int(qwen35_top_k)),
            "qwen35_min_p": min(1.0, max(0.0, float(qwen35_min_p))),
            "qwen35_repeat_penalty": min(2.0, max(0.8, float(qwen35_repeat_penalty))),
            "api_key_source": "ComfyUI Settings",
            "run_folder": reserve_run(),
        }
        status = (
            f"{backend}: {config['api_url']} | model: {config['openai_model'] if backend == 'OpenAI' else config['lmstudio_model']} | "
            f"API key: ComfyUI Settings (hidden) | run: {config['run_folder']}"
        )
        return config, status

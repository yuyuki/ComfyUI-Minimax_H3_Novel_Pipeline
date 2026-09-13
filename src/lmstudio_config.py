"""Secure LM Studio configuration node for the MiniMax H3 workflow."""
from __future__ import annotations

from . import progress

from typing import Any

from . import lmstudio_settings, lmstudio_models
from .run_output import reserve_run


class LMStudioConfigurationNode:
    """Share non-secret LM Studio settings and validate the operator's endpoint."""

    DESCRIPTION = """\
Share LM Studio settings across the workflow. Connect `lmstudio_config`
to Extract, Consolidate and Generate H3 Prompts.

Set the API key in ComfyUI Settings: `MiniMax H3 Novel → LM Studio → API Key`.
The key is not saved in the workflow. Choose Qwen or Mistral and configure the URL here. Thinking and qwen35 controls apply only to Qwen. Set max_tokens on each processing node.
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "http://127.0.0.1:1234/v1",
                    "tooltip": "Must match the server's MINIMAX_H3_LMSTUDIO_BASE_URL (default: http://127.0.0.1:1234/v1).",
                }),
                "thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Disable thinking for faster structured JSON output. Qwen3.5/3.8 requests "
                        "also prefill a closed thinking block so LM Studio continues directly "
                        "with JSON. Check thinking and reasoning_chars in the console to verify "
                        "the setting with your model and runtime."
                    ),
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
            "optional": {
                "model_family": (list(lmstudio_models.PROFILES), {
                    "default": "Qwen",
                    "tooltip": "Select a matching model exposed by LM Studio. Load the model there first. Mistral ignores thinking and qwen35 controls.",
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
    def run(self, api_url: str, thinking: bool = False,
            qwen35_length_retries: int = 2,
            qwen35_top_k: int = 20,
            qwen35_min_p: float = 0.0, qwen35_repeat_penalty: float = 1.05,
            model_family: str = "Qwen") -> tuple[dict[str, Any], str]:
        api_url = lmstudio_settings.validate_api_url(api_url)

        lmstudio_models.get_profile(model_family)

        api_key = lmstudio_settings.get_api_key()
        if not api_key:
            raise RuntimeError(
                "No API key in ComfyUI settings. Open Settings → "
                "MiniMax H3 Novel → LM Studio → API Key, save the key, then retry."
            )

        config = {
            "api_url": api_url.strip(),
            "model_family": model_family,
            "thinking": bool(thinking),
            "qwen35_length_retries": max(0, int(qwen35_length_retries)),
            "qwen35_top_k": max(1, int(qwen35_top_k)),
            "qwen35_min_p": min(1.0, max(0.0, float(qwen35_min_p))),
            "qwen35_repeat_penalty": min(2.0, max(0.8, float(qwen35_repeat_penalty))),
            "api_key_source": "ComfyUI Settings",
            "run_folder": reserve_run(),
        }
        status = (
            f"LM Studio: {config['api_url']} | model family: {model_family} (select matching model) | "
            f"thinking: {config['thinking'] if model_family == 'Qwen' else 'not used'} | "
            f"API key: ComfyUI Settings (hidden) | run: {config['run_folder']}"
        )
        return config, status

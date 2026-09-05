"""Secure LM Studio configuration node for the MiniMax H3 workflow."""
from __future__ import annotations

from typing import Any

from . import lmstudio_settings


class LMStudioConfigurationNode:
    """Share non-secret LM Studio settings and validate the operator's endpoint."""

    DESCRIPTION = """\
Share LM Studio settings across the workflow. Connect `lmstudio_config`
to Extract, Consolidate and Generate H3 Prompts.

Set the API key in ComfyUI Settings: `MiniMax H3 Novel → LM Studio → API Key`.
The key is not saved in the workflow. Configure the URL, model, thinking, and Qwen3.5 output and retry limits here.
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
                    "tooltip": "Disable thinking for faster structured JSON output.",
                }),
                "qwen35_max_output_tokens": ("INT", {
                    "default": 3500, "min": 256, "max": 32768,
                    "tooltip": "Qwen3.5 output limit; streaming stops once the JSON is complete.",
                }),
                "qwen35_length_retries": ("INT", {
                    "default": 2, "min": 0, "max": 10,
                    "tooltip": "Number of compact retries after incomplete Qwen3.5 JSON output.",
                }),
                "qwen35_safe_chunk_chars": ("INT", {
                    "default": 3600, "min": 3000, "max": 20000,
                    "tooltip": "Maximum passage size during Qwen3.5 extraction. Shorter passages help prevent truncated JSON.",
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
                "model": ("STRING", {
                    "default": "",
                    "tooltip": "Exact ID of the model loaded in LM Studio. Leave empty for automatic selection.",
                }),
            },
        }

    RETURN_TYPES = ("MINIMAX_LMSTUDIO_CONFIG", "STRING")
    RETURN_NAMES = ("lmstudio_config", "configuration_status")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(self, api_url: str, model: str = "", thinking: bool = False,
            qwen35_max_output_tokens: int = 3500, qwen35_length_retries: int = 2,
            qwen35_safe_chunk_chars: int = 3600, qwen35_top_k: int = 20,
            qwen35_min_p: float = 0.0, qwen35_repeat_penalty: float = 1.05) -> tuple[dict[str, Any], str]:
        api_url = lmstudio_settings.validate_api_url(api_url)

        api_key = lmstudio_settings.get_api_key()
        if not api_key:
            raise RuntimeError(
                "No API key in ComfyUI settings. Open Settings → "
                "MiniMax H3 Novel → LM Studio → API Key, save the key, then retry."
            )

        config = {
            "api_url": api_url.strip(),
            "model": (model or "").strip(),
            "thinking": bool(thinking),
            "qwen35_max_output_tokens": max(256, int(qwen35_max_output_tokens)),
            "qwen35_length_retries": max(0, int(qwen35_length_retries)),
            "qwen35_safe_chunk_chars": max(3000, int(qwen35_safe_chunk_chars)),
            "qwen35_top_k": max(1, int(qwen35_top_k)),
            "qwen35_min_p": min(1.0, max(0.0, float(qwen35_min_p))),
            "qwen35_repeat_penalty": min(2.0, max(0.8, float(qwen35_repeat_penalty))),
            "api_key_source": "ComfyUI Settings",
        }
        selected_model = config["model"] or "auto-select loaded Qwen/first model"
        status = (
            f"LM Studio: {config['api_url']} | model: {selected_model} | "
            f"thinking: {config['thinking']} | "
            "API key: ComfyUI Settings (hidden)"
        )
        return config, status

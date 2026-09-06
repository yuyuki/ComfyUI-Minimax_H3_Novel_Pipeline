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
The key is not saved in the workflow. Configure the URL, thinking, and Qwen3.5 retry and sampling controls here. Set max_tokens on each processing node.
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
                        "Disable thinking for faster structured JSON output. If reasoning_chars "
                        "is not 0 while this is disabled, the model's thinking flag probably "
                        "is not working: add `{%- set enable_thinking = false %}` at the "
                        "beginning of its Chat Template Jinja content."
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
        }

    RETURN_TYPES = ("MINIMAX_LMSTUDIO_CONFIG", "STRING")
    RETURN_NAMES = ("lmstudio_config", "configuration_status")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(self, api_url: str, thinking: bool = False,
            qwen35_length_retries: int = 2,
            qwen35_top_k: int = 20,
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
            "thinking": bool(thinking),
            "qwen35_length_retries": max(0, int(qwen35_length_retries)),
            "qwen35_top_k": max(1, int(qwen35_top_k)),
            "qwen35_min_p": min(1.0, max(0.0, float(qwen35_min_p))),
            "qwen35_repeat_penalty": min(2.0, max(0.8, float(qwen35_repeat_penalty))),
            "api_key_source": "ComfyUI Settings",
        }
        status = (
            f"LM Studio: {config['api_url']} | model: auto-select loaded Qwen/first model | "
            f"thinking: {config['thinking']} | "
            "API key: ComfyUI Settings (hidden)"
        )
        return config, status

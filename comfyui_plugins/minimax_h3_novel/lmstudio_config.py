"""Secure LM Studio configuration node for the MiniMax H3 workflow."""
from __future__ import annotations

from typing import Any

from . import lmstudio_pipeline, lmstudio_settings


class LMStudioConfigurationNode:
    """Read LM Studio settings from environment variables once per workflow."""

    DESCRIPTION = """\
Centralise les réglages LM Studio pour tout le workflow. Connectez la sortie
`lmstudio_config` aux nodes Extract, Consolidate et Generate H3 Prompts.

La clé API est définie dans les réglages ComfyUI : `MiniMax H3 Novel → LM
Studio → API Key`. Elle n'est pas enregistrée dans le workflow. Les autres
paramètres sont configurables ici : URL, modèle, backend Qwen, thinking et
limites de sortie/réessais Qwen3.5.
"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "http://127.0.0.1:1234/v1",
                    "tooltip": "URL du serveur OpenAI-compatible de LM Studio.",
                }),
                "chat_backend": (["auto", "openai-chat", "qwen35-chatml"], {
                    "default": "auto",
                    "tooltip": "auto utilise le ChatML streamé pour les modèles Qwen3.5.",
                }),
                "thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Désactivez-le pour accélérer les sorties JSON structurées.",
                }),
                "qwen35_max_output_tokens": ("INT", {
                    "default": 3500, "min": 256, "max": 32768,
                    "tooltip": "Plafond de sécurité Qwen3.5 ; le stream s'arrête dès que le JSON est complet.",
                }),
                "qwen35_length_retries": ("INT", {
                    "default": 2, "min": 0, "max": 10,
                    "tooltip": "Nombre de réessais compacts après un JSON Qwen3.5 incomplet.",
                }),
            },
            "optional": {
                "model": ("STRING", {
                    "default": "",
                    "tooltip": "ID exact du modèle chargé dans LM Studio. Vide = sélection automatique.",
                }),
            },
        }

    RETURN_TYPES = ("MINIMAX_LMSTUDIO_CONFIG", "STRING")
    RETURN_NAMES = ("lmstudio_config", "configuration_status")
    FUNCTION = "run"
    CATEGORY = "MiniMax H3 Novel"

    def run(self, api_url: str, model: str = "", chat_backend: str = "auto", thinking: bool = False,
            qwen35_max_output_tokens: int = 3500, qwen35_length_retries: int = 2) -> tuple[dict[str, Any], str]:
        if not api_url.strip():
            raise ValueError("api_url ne peut pas être vide.")

        api_key = lmstudio_settings.get_api_key()
        if not api_key:
            raise RuntimeError(
                "Aucune clé API dans les réglages ComfyUI. Ouvrez Settings → "
                "MiniMax H3 Novel → LM Studio → API Key, enregistrez la clé, puis réessayez."
            )

        config = {
            "api_url": api_url.strip(),
            "api_key": api_key,
            "model": (model or "").strip(),
            "chat_backend": chat_backend,
            "thinking": bool(thinking),
            "qwen35_max_output_tokens": max(256, int(qwen35_max_output_tokens)),
            "qwen35_length_retries": max(0, int(qwen35_length_retries)),
            "api_key_source": "ComfyUI Settings",
        }
        selected_model = config["model"] or "auto-select loaded Qwen/first model"
        status = (
            f"LM Studio: {config['api_url']} | model: {selected_model} | "
            f"backend: {config['chat_backend']} | thinking: {config['thinking']} | "
            "API key: ComfyUI Settings (hidden)"
        )
        return config, status

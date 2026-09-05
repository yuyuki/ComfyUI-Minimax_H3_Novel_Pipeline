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
                    "tooltip": "auto utilise d'abord le JSON contraint de LM Studio pour Qwen3.5; qwen35-chatml conserve le mode compatible ancien.",
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
                "qwen35_safe_chunk_chars": ("INT", {
                    "default": 3600, "min": 3000, "max": 20000,
                    "tooltip": "Taille maximale d'un passage pendant l'extraction Qwen3.5. Des passages plus courts évitent les JSON tronqués.",
                }),
                "qwen35_top_k": ("INT", {
                    "default": 20, "min": 1, "max": 200,
                    "tooltip": "Échantillonnage Qwen3.5/LM Studio: limite les choix de tokens pour des JSON plus stables.",
                }),
                "qwen35_min_p": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Échantillonnage Qwen3.5/LM Studio. 0 désactive ce filtre; conserver 0 pour l'extraction structurée.",
                }),
                "qwen35_repeat_penalty": ("FLOAT", {
                    "default": 1.05, "min": 0.8, "max": 2.0, "step": 0.01,
                    "tooltip": "Pénalise les répétitions Qwen3.5; 1.05 réduit les boucles sans appauvrir les listes JSON.",
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
            qwen35_max_output_tokens: int = 3500, qwen35_length_retries: int = 2,
            qwen35_safe_chunk_chars: int = 3600, qwen35_top_k: int = 20,
            qwen35_min_p: float = 0.0, qwen35_repeat_penalty: float = 1.05) -> tuple[dict[str, Any], str]:
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
            "qwen35_safe_chunk_chars": max(3000, int(qwen35_safe_chunk_chars)),
            "qwen35_top_k": max(1, int(qwen35_top_k)),
            "qwen35_min_p": min(1.0, max(0.0, float(qwen35_min_p))),
            "qwen35_repeat_penalty": min(2.0, max(0.8, float(qwen35_repeat_penalty))),
            "api_key_source": "ComfyUI Settings",
        }
        selected_model = config["model"] or "auto-select loaded Qwen/first model"
        status = (
            f"LM Studio: {config['api_url']} | model: {selected_model} | "
            f"backend: {config['chat_backend']} | thinking: {config['thinking']} | "
            "API key: ComfyUI Settings (hidden)"
        )
        return config, status

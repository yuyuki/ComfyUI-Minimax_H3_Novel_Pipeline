"""Load the LM Studio pipeline implementation bundled with this node package."""
from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from . import lmstudio_settings, lmstudio_json

_SCRIPT_FILES = {
    # These deliberately do not share names with the ComfyUI node modules.
    # Loading ``extract_chapter_references.py`` here would reload the node as a
    # top-level module, where its ``from . import ...`` imports cannot work.
    "extract": "pipeline_step1_extract.py",
    "consolidate": "pipeline_step2_consolidate.py",
    "generate": "pipeline_step3_generate.py",
}


@lru_cache(maxsize=None)
def load(step: str) -> ModuleType:
    """Import the packaged stage; historical scripts are never searched."""
    return importlib.import_module(f".{Path(_SCRIPT_FILES[step]).stem}", __package__)


def configure_qwen(*, thinking: bool,
                   length_retries: int,
                   safe_chunk_chars: int = 3600, top_k: int = 20,
                   min_p: float = 0.0, repeat_penalty: float = 1.05) -> None:
    lmstudio_json.THINKING_ENABLED = bool(thinking)
    lmstudio_json.QWEN35_LENGTH_RETRIES = max(0, int(length_retries))
    lmstudio_json.QWEN35_SAFE_CHUNK_CHARS = max(3000, int(safe_chunk_chars))
    lmstudio_json.QWEN35_TOP_K = max(1, int(top_k))
    lmstudio_json.QWEN35_MIN_P = min(1.0, max(0.0, float(min_p)))
    lmstudio_json.QWEN35_REPEAT_PENALTY = min(2.0, max(0.8, float(repeat_penalty)))


def make_client_and_model(module: ModuleType, api_url: str) -> tuple[object, str]:
    # Recheck here: downstream nodes can receive forged or cached configuration.
    api_url = lmstudio_settings.validate_api_url(api_url)
    api_key = lmstudio_settings.get_api_key()
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must not be empty (LM Studio accepts 'lm-studio' by default).")
    import httpx

    # Redirects and ambient proxies must not reroute the operator's credential.
    transport = httpx.Client(follow_redirects=False, trust_env=False, timeout=300.0)
    try:
        client = module.make_client(api_url, api_key.strip(), http_client=transport)
        return client, module.select_model(client, None)
    except BaseException:
        transport.close()
        raise


def comfy_interrupt_check() -> None:
    """Raise ComfyUI's normal interruption exception when Stop was pressed."""
    try:
        import comfy.model_management as model_management
    except ImportError:
        return
    model_management.throw_exception_if_processing_interrupted()



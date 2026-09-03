"""Load the LM Studio pipeline implementation bundled with this node package."""
from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from typing import Any, Callable, Iterator

_SCRIPT_FILES = {
    # These deliberately do not share names with the ComfyUI node modules.
    # Loading ``extract_chapter_references.py`` here would reload the node as a
    # top-level module, where its ``from . import ...`` imports cannot work.
    "extract": "pipeline_step1_extract.py",
    "consolidate": "pipeline_step2_consolidate.py",
    "generate": "pipeline_step3_generate.py",
}


def api_key_from_environment(variable_name: str) -> str:
    """Read the only secret in the configuration from the server environment."""
    normalized = str(variable_name or "").strip().upper()
    if not normalized:
        raise ValueError("api_key_environment_variable must not be empty.")
    api_key = os.environ.get(normalized, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing {normalized}. Set it in the environment running ComfyUI, then restart ComfyUI."
        )
    return api_key


def _bundled_pipeline_dir() -> Path:
    """Return the package directory containing the self-contained scripts."""
    path = Path(__file__).resolve().parent
    if path.is_dir():
        return path
    raise RuntimeError(f"Bundled pipeline directory not found: {path}.")


@lru_cache(maxsize=None)
def load(step: str) -> ModuleType:
    """Load a bundled pipeline step without invoking its CLI entrypoint."""
    try:
        path = _bundled_pipeline_dir() / _SCRIPT_FILES[step]
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline step: {step}") from exc
    module_name = f"minimax_h3_novel_canonical_{step}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load bundled pipeline script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        sys.modules.pop(module_name, None)
        if exc.name == "openai":
            raise RuntimeError(
                "The LM Studio nodes require the OpenAI Python client. "
                "Install comfyui_plugins/minimax_h3_novel/requirements.txt in ComfyUI's Python environment."
            ) from exc
        raise
    return module


def configure_qwen(module: Any, *, thinking: bool, chat_backend: str,
                   max_output_tokens: int, length_retries: int) -> None:
    module.THINKING_ENABLED = bool(thinking)
    module.CHAT_BACKEND = chat_backend
    module.QWEN35_MAX_OUTPUT_TOKENS = max(256, int(max_output_tokens))
    module.QWEN35_LENGTH_RETRIES = max(0, int(length_retries))


def make_client_and_model(module: ModuleType, api_url: str, api_key: str, model: str) -> tuple[object, str]:
    if not isinstance(api_url, str) or not api_url.strip():
        raise ValueError("api_url must be a non-empty LM Studio OpenAI-compatible URL.")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must not be empty (LM Studio accepts 'lm-studio' by default).")
    client = module.make_client(api_url.strip(), api_key.strip())
    return client, module.select_model(client, model.strip() or None)


def comfy_interrupt_check() -> None:
    """Raise ComfyUI's normal interruption exception when Stop was pressed."""
    try:
        import comfy.model_management as model_management
    except ImportError:
        return
    check = getattr(model_management, "throw_exception_if_processing_interrupted", None)
    if callable(check):
        check()


class _InterruptibleStream:
    """Check ComfyUI cancellation before consuming each SSE event."""

    def __init__(self, stream: Any, interrupt_check: Callable[[], None]) -> None:
        self._stream = stream
        self._interrupt_check = interrupt_check

    def __iter__(self) -> Iterator[Any]:
        try:
            for event in self._stream:
                self._interrupt_check()
                yield event
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _ChatCompletionsProxy:
    def __init__(self, completions: Any, interrupt_check: Callable[[], None]) -> None:
        self._completions = completions
        self._interrupt_check = interrupt_check

    def create(self, *args: Any, **kwargs: Any) -> Any:
        # The canonical pipeline expects a complete response for this path. Ask
        # LM Studio for SSE instead and assemble that response locally, checking
        # ComfyUI's Stop flag after each chunk.
        if kwargs.get("stream"):
            return _InterruptibleStream(self._completions.create(*args, **kwargs), self._interrupt_check)
        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        stream = _InterruptibleStream(self._completions.create(*args, **stream_kwargs), self._interrupt_check)
        content: list[str] = []
        reasoning: list[str] = []
        try:
            for event in stream:
                if not getattr(event, "choices", None):
                    continue
                delta = getattr(event.choices[0], "delta", None)
                if delta is None:
                    continue
                piece = getattr(delta, "content", None)
                if piece:
                    content.append(piece)
                thought = getattr(delta, "reasoning_content", None)
                if thought:
                    reasoning.append(thought)
        finally:
            stream.close()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="".join(content),
                reasoning_content="".join(reasoning) or None,
            ))]
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _InterruptibleClient:
    def __init__(self, client: Any, interrupt_check: Callable[[], None]) -> None:
        self._client = client
        self.chat = SimpleNamespace(completions=_ChatCompletionsProxy(client.chat.completions, interrupt_check))
        self.completions = SimpleNamespace(create=self._streaming_completion(client.completions.create, interrupt_check))

    @staticmethod
    def _streaming_completion(create: Callable[..., Any], interrupt_check: Callable[[], None]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = create(*args, **kwargs)
            return _InterruptibleStream(result, interrupt_check) if kwargs.get("stream") else result
        return wrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def make_interruptible_client(client: object) -> object:
    """Return a client that closes LM Studio's active stream when ComfyUI stops."""
    return _InterruptibleClient(client, comfy_interrupt_check)

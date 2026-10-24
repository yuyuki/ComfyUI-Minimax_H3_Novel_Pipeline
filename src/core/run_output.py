"""Reserve one confined output directory per ComfyUI execution."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re
from threading import Lock

from .path_access import confined_path, output_path, storage_root

_lock = Lock()
_runs: dict[tuple[str, str], str] = {}


def execution_id() -> str | None:
    try:
        from comfy_execution.utils import get_executing_context
    except ImportError:
        return None
    context = get_executing_context()
    return context.prompt_id if context else None


def reserve_run() -> str:
    root = storage_root("output")
    prompt_id = execution_id()
    key = (str(root), prompt_id) if prompt_id else None
    with _lock:
        if key is not None and key in _runs:
            return _runs[key]
        stamp = datetime.now().replace(microsecond=0)
        root.mkdir(parents=True, exist_ok=True)
        while True:
            name = stamp.strftime("%Y%m%d%H%M%S")
            target = confined_path(name, root)
            try:
                target.mkdir(exist_ok=False)
                break
            except FileExistsError:
                stamp += timedelta(seconds=1)
        if key is not None:
            _runs[key] = name
        return name


def stage_output(config: dict, value: str) -> Path:
    """Resolve a relative stage folder inside the configured execution run."""
    if not isinstance(config, dict):
        raise TypeError("lmstudio_config must come from LM Studio Configuration.")
    if Path(value).anchor:
        raise ValueError("out_dir must be a relative subfolder of the current run.")
    name = config.get("run_folder")
    if not isinstance(name, str) or not re.fullmatch(r"\d{14}", name):
        raise ValueError("Invalid run folder; reconnect LM Studio Configuration.")
    return confined_path(value, output_path(name))

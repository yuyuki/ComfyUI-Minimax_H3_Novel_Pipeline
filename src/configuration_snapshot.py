"""Non-secret execution settings saved alongside stage results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from . import lmstudio_json, lmstudio_models, util


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def start(output, stage, config, model, args, *, out_dir, inputs, extra=None):
    """Use an allowlist: never serialize the client or arbitrary config fields."""
    from .lmstudio_config import LMStudioConfigurationNode

    public = {}
    for fields in LMStudioConfigurationNode.INPUT_TYPES().values():
        for key, spec in fields.items():
            public[key] = config.get(key, spec[1]["default"])
    profile = lmstudio_models.get_profile(public["model_family"])
    settings = profile.settings_from_config(public)
    _, request_extra, top_p = profile.request_settings(model, "", "", settings)
    node_settings = {key: value for key, value in vars(args).items() if key not in {"base_url", "out_dir"}}
    node_settings.update(extra or {})
    node_settings["out_dir"] = out_dir
    snapshot = {
        "schema_version": "minimax-h3-configuration.v1",
        "stage": stage,
        "run_folder": config["run_folder"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "output_directory": str(output),
        "lmstudio_config": public,
        "resolved_model": model,
        "model_controls": {
            "profile": profile.NAME, "settings": settings,
            "request_extra_body": request_extra, "top_p": top_p,
            "length_retries": profile.retries(model, settings),
            "chat_backend": lmstudio_json.CHAT_BACKEND,
            "chatml_fallback_allowed": profile.allows_chatml(model, settings),
            "compact_schema_version": lmstudio_json.COMPACT_SCHEMA_VERSION,
        },
        "node_settings": node_settings,
        "inputs": inputs,
        "outputs": [],
    }
    path = output / f"{stage}_configuration.json"
    util.save_json(path, snapshot)
    return path, snapshot


def complete(handle, artifacts):
    """Hash only this invocation's artifacts; caches are not result files."""
    path, snapshot = handle
    files = set()
    for artifact in artifacts:
        artifact = util.output_path(artifact)
        candidates = artifact.rglob("*") if artifact.is_dir() else [artifact]
        for candidate in candidates:
            candidate = util.output_path(candidate)
            if candidate.is_file() and ".cache" not in candidate.relative_to(path.parent).parts:
                files.add(candidate)
    snapshot["outputs"] = [
        {"file": str(file.relative_to(path.parent)), "sha256": file_digest(file)}
        for file in sorted(files)
    ]
    snapshot["status"] = "completed"
    snapshot["completed_at"] = datetime.now(timezone.utc).isoformat()
    util.save_json(path, snapshot)

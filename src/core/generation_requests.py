"""Exact-request checkpoints and task metrics for H3 generation only."""
from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import time

from jsonschema import Draft202012Validator

from . import lmstudio_json, lmstudio_pipeline
from .path_access import confined_path


VERSION = 2
# Bump the affected contract when helpers, validation or transport behavior
# changes cache semantics without changing the serialized inputs/prompts/schema.
# These versions replace runtime inspection of the package's Python sources.
CACHE_CONTRACTS = {
    "transport": 1,  # Shared JSON transport and every model profile.
    "evidence": 1,  # Catalog, timeline and character-context projection.
    "metadata": 1,  # Cinematic scene metadata and binding construction.
    "draft": 1,
    "format": 1,
    "camera": 1,
    "continuity": 1,  # Packing, evidence validation, review and repair.
    "final_prompt": 1,
}
_TASK_CONTRACTS = {
    "scene_metadata": ("metadata",), "scene_metadata_repair": ("metadata", "continuity"),
    "generation": ("draft",), "format_repair": ("format",), "camera": ("camera", "format"),
    "continuity_review": ("continuity",), "continuity_repair": ("continuity",),
}
_SESSION = ContextVar("h3_generation_requests", default=None)
_TASK = ContextVar("h3_generation_task", default="planning")
_DETAIL = ContextVar("h3_generation_detail", default={})


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def contracts(*names):
    """Snapshot explicit cache versions; unknown names are programming errors."""
    return {name: CACHE_CONTRACTS[name] for name in names}


def atomic_json(path, value):
    """Never leave a partially written checkpoint looking complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".checkpoint-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def backend_signature(client, model):
    profile, settings = lmstudio_json.model_settings(client, model)
    return {"profile": profile.NAME, "settings": settings,
            "endpoint_digest": digest(str(getattr(client, "base_url", ""))),
            "transport": lmstudio_json.CHAT_BACKEND,
            "contracts": contracts("transport"),
            "compact_schema_version": lmstudio_json.COMPACT_SCHEMA_VERSION}


@contextmanager
def task(name, thinking=None):
    """Scope overrides to this task, including exceptions and cancellation."""
    token = _TASK.set(name)
    reasoning_token = lmstudio_json.REQUEST_THINKING.set(thinking)
    try:
        yield
    finally:
        lmstudio_json.REQUEST_THINKING.reset(reasoning_token)
        _TASK.reset(token)


def position(label):
    state = _SESSION.get()
    if state is not None:
        state["position"] = label


def cache_context():
    """Chapter-local cache scope shared by artifacts and coverage records."""
    return _SESSION.get()


def cache_event(stage, reason, *, source=None, changed=()):
    """Bounded diagnostics containing labels and hashes only, never payloads."""
    state = _SESSION.get()
    if state is None:
        return
    counts = state.setdefault("cache_reasons", {}).setdefault(stage, {})
    counts[reason] = counts.get(reason, 0) + 1
    events = state.setdefault("cache_events", [])
    events.append({"stage": stage, "reason": reason, "position": state["position"],
                   "detail": _DETAIL.get(), "source": source, "changed_components": list(changed)})
    if len(events) > 100:
        del events[:-100]
        state["cache_events_dropped"] = state.get("cache_events_dropped", 0) + 1


def cache_dependencies(stage, dependencies):
    """Compare diagnostic snapshots at the same logical slot, never grant reuse.

    Only component digests are persisted. A missing/corrupt snapshot means the
    previous dependencies are unknown; authoritative cache checks still run.
    """
    state = _SESSION.get()
    if state is None:
        return
    components = {name: digest(value) for name, value in dependencies.items()}
    slot = digest([stage, state["position"], _DETAIL.get()])
    relative = Path("diagnostics") / f"{slot}.json"
    try:
        for index, directory in enumerate(state["sources"]):
            try:
                record = json.loads(confined_path(relative, directory).read_text(encoding="utf-8"))
                previous = record["components"]
                if not isinstance(previous, dict) or record["digest"] != digest(previous):
                    raise ValueError("Invalid diagnostic snapshot")
                # Emit only current, caller-owned component names, never file contents.
                changed = sorted(name for name in components if previous.get(name) != components[name])
                if changed or previous.keys() != components.keys():
                    cache_event(stage, "dependencies_changed", source=index, changed=changed)
                break
            except FileNotFoundError:
                continue
            except (ValueError, KeyError, TypeError):
                cache_event(stage, "diagnostic_invalid", source=index)
        atomic_json(confined_path(relative, state["directory"]),
                    {"components": components, "digest": digest(components)})
    except OSError:
        cache_event(stage, "diagnostic_unavailable")


@contextmanager
def detail(**values):
    """Attach bounded progress labels without changing request identity."""
    token = _DETAIL.set({**_DETAIL.get(), **values})
    try:
        yield
    finally:
        _DETAIL.reset(token)


def write_metrics(state, stage, status, **values):
    state["last_metrics"] = (stage, status, values)
    try:
        atomic_json(confined_path("request_metrics.json", state["directory"]),
                    {"last_position": state["position"], "last_stage": stage,
                     "last_status": status, "detail": _DETAIL.get(),
                     "tasks": state["tasks"], "artifacts": state.get("artifacts", {}),
                     "cache_reasons": state.get("cache_reasons", {}),
                     "cache_events": state.get("cache_events", []),
                     "cache_events_dropped": state.get("cache_events_dropped", 0),
                     "review_coverage": state.get("review_coverage"),
                     "chapter_status": state.get("chapter_status", "running"), **values})
    except OSError:
        print("    H3: could not persist request metrics.", flush=True)


@contextmanager
def chapter(chapter_id, args):
    directory = confined_path(args.out_dir / ".cache" / chapter_id, args.out_dir)
    sources = [directory]
    previous = getattr(args, "resume_cache_dir", None)
    if previous is not None:
        sources.append(confined_path(chapter_id, previous))
    state = {"directory": directory, "sources": sources, "force": args.force,
             "position": "planning", "tasks": {}}
    token = _SESSION.set(state)
    started = time.perf_counter()
    try:
        yield
        state["chapter_status"] = "completed"
    except BaseException:
        state["chapter_status"] = "failed_or_interrupted"
        raise
    finally:
        stage, status, values = state.get("last_metrics", ("chapter", state.get("chapter_status"), {}))
        write_metrics(state, stage, status, **values)
        _SESSION.reset(token)
        print(f"  H3 requests for {chapter_id}: {json.dumps(state['tasks'], sort_keys=True)}, "
              f"elapsed={time.perf_counter() - started:.1f}s", flush=True)


def chat_json(client, model, system, user, schema, temperature, max_tokens):
    """Memoize completed structured responses, never exceptions or partial streams.

    A response is a draft, not a validation result. All caller-side format and
    semantic checks still run on cache hits, including each continuity batch.
    """
    state = _SESSION.get()
    if state is None:
        return lmstudio_json.chat_json(client, model, system, user, schema, temperature, max_tokens)
    lmstudio_pipeline.comfy_interrupt_check()
    stage = _TASK.get()
    stats = state["tasks"].setdefault(stage, {
        "calls": 0, "cache_hits": 0, "llm_requests": 0,
        "estimated_input_tokens": 0, "estimated_output_tokens": 0, "seconds": 0.0,
        "request_bytes": 0, "cache_misses": 0, "invalid_checkpoints": 0,
    })
    stats["calls"] += 1
    label = f"{state['position']} / {stage} / request {stats['calls']}"
    task_contract = contracts(*_TASK_CONTRACTS.get(stage, ()))
    key = digest([VERSION, task_contract, model, backend_signature(client, model),
                  system, user, schema, temperature, max_tokens])
    diagnostic_stage = f"request:{stage}"
    cache_dependencies(diagnostic_stage, dict(version=VERSION, task_contract=task_contract, model=model,
                       backend=backend_signature(client, model), system=system, user=user,
                       schema=schema, temperature=temperature, max_tokens=max_tokens))
    relative = Path("requests") / f"{key}.json"
    target = confined_path(relative, state["directory"])
    validator = Draft202012Validator(schema["schema"])
    started = time.perf_counter()
    metrics = {"llm_requests": 0, "estimated_input_tokens": 0, "estimated_output_tokens": 0}
    metric_token = lmstudio_json.REQUEST_METRICS.set(metrics)
    usage_token = lmstudio_json.REQUEST_USAGE.set(True)
    status = "failed"
    request_bytes = len(system.encode("utf-8")) + len(user.encode("utf-8")) + len(
        json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    started_at = time.time()
    try:
        if state["force"]:
            cache_event(diagnostic_stage, "force")
        for index, source in enumerate(state["sources"] if not state["force"] else []):
            try:
                record = json.loads(confined_path(relative, source).read_text(encoding="utf-8"))
                result = record["result"]
                reason = ("contract_mismatch" if record.get("version") != VERSION else
                          "key_mismatch" if record.get("key") != key else
                          "integrity_mismatch" if record.get("result_digest") != digest(result) else
                          "validation_failed" if not isinstance(result, dict) or not validator.is_valid(result) else None)
                if reason:
                    cache_event(diagnostic_stage, reason, source=index)
                    stats["invalid_checkpoints"] += 1
                    continue
            except FileNotFoundError:
                cache_event(diagnostic_stage, "missing", source=index)
                continue
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                cache_event(diagnostic_stage, "unreadable_record", source=index)
                stats["invalid_checkpoints"] += 1
                continue
            lmstudio_pipeline.comfy_interrupt_check()
            if source != state["directory"]:
                atomic_json(target, record)
            stats["cache_hits"] += 1
            cache_event(diagnostic_stage, "hit", source=index)
            status = "cached"
            print(f"    H3 {label}: cached", flush=True)
            return result
        print(f"    H3 {label}: requesting structured JSON", flush=True)
        stats["cache_misses"] += 1
        stats["request_bytes"] += request_bytes
        write_metrics(state, stage, "requesting", request_started_at=started_at,
                      request_bytes=request_bytes,
                      cache_reason="force" if state["force"] else "no_valid_exact_checkpoint")
        result = lmstudio_json.chat_json(client, model, system, user, schema, temperature, max_tokens)
        lmstudio_pipeline.comfy_interrupt_check()
        if isinstance(result, dict) and validator.is_valid(result):
            atomic_json(target, {"version": VERSION, "key": key, "result": result,
                                 "result_digest": digest(result)})
        else:
            cache_event(diagnostic_stage, "not_cacheable")
        status = "completed"
        return result
    except Exception:
        # Do not wrap cancellation or echo arbitrary server errors/secrets.
        print(f"    H3 {label}: interrupted or failed. Completed requests are in "
              f"{state['directory']}. Resume this run with the same out_dir.", flush=True)
        raise
    finally:
        lmstudio_json.REQUEST_METRICS.reset(metric_token)
        lmstudio_json.REQUEST_USAGE.reset(usage_token)
        for name, value in metrics.items():
            stats[name] = stats.get(name, 0) + value
        stats["seconds"] = round(stats["seconds"] + time.perf_counter() - started, 3)
        write_metrics(state, stage, status, request_started_at=started_at,
                      request_elapsed_seconds=round(time.perf_counter() - started, 3))

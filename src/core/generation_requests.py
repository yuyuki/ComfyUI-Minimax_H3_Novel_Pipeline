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


VERSION = 1
_SESSION = ContextVar("h3_generation_requests", default=None)
_TASK = ContextVar("h3_generation_task", default="planning")
_DETAIL = ContextVar("h3_generation_detail", default={})


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


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
            "transport_source": sha256(Path(lmstudio_json.__file__).read_bytes()).hexdigest(),
            "profile_source": sha256(Path(profile.__file__).read_bytes()).hexdigest(),
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
    key = digest([VERSION, model, backend_signature(client, model),
                  system, user, schema, temperature, max_tokens])
    relative = Path("requests") / f"{key}.json"
    target = confined_path(relative, state["directory"])
    validator = Draft202012Validator(schema["schema"])
    started = time.perf_counter()
    metrics = {"llm_requests": 0, "estimated_input_tokens": 0, "estimated_output_tokens": 0}
    metric_token = lmstudio_json.REQUEST_METRICS.set(metrics)
    status = "failed"
    request_bytes = len(system.encode("utf-8")) + len(user.encode("utf-8")) + len(
        json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    started_at = time.time()
    try:
        for source in state["sources"] if not state["force"] else []:
            try:
                record = json.loads(confined_path(relative, source).read_text(encoding="utf-8"))
                result = record["result"]
                if (record.get("version") != VERSION or record.get("key") != key
                        or record.get("result_digest") != digest(result)
                        or not isinstance(result, dict) or not validator.is_valid(result)):
                    stats["invalid_checkpoints"] += 1
                    continue
            except FileNotFoundError:
                continue
            except (OSError, ValueError, KeyError, TypeError):
                stats["invalid_checkpoints"] += 1
                continue
            lmstudio_pipeline.comfy_interrupt_check()
            if source != state["directory"]:
                atomic_json(target, record)
            stats["cache_hits"] += 1
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
        status = "completed"
        return result
    except Exception:
        # Do not wrap cancellation or echo arbitrary server errors/secrets.
        print(f"    H3 {label}: interrupted or failed. Completed requests are in "
              f"{state['directory']}. Resume this run with the same out_dir.", flush=True)
        raise
    finally:
        lmstudio_json.REQUEST_METRICS.reset(metric_token)
        for name, value in metrics.items():
            stats[name] += value
        stats["seconds"] = round(stats["seconds"] + time.perf_counter() - started, 3)
        write_metrics(state, stage, status, request_started_at=started_at,
                      request_elapsed_seconds=round(time.perf_counter() - started, 3))

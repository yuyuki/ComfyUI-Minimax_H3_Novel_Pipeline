"""Structured LM Studio JSON requests shared by all three ComfyUI stages."""
from __future__ import annotations

import copy
import json
import re
import time
from typing import Any

from openai import OpenAI

THINKING_ENABLED = False
CHAT_BACKEND = "structured-json"
QWEN35_LENGTH_RETRIES = 2
QWEN35_SAFE_CHUNK_CHARS = 3600
QWEN35_TOP_K = 20
QWEN35_MIN_P = 0.0
QWEN35_REPEAT_PENALTY = 1.05

def _is_comfy_interrupt(error: BaseException) -> bool:
    """Do not retry a ComfyUI Stop request as though it were an LLM error."""
    return error.__class__.__name__ == "InterruptProcessingException"

def select_model(client: OpenAI, requested: str | None) -> str:
    if requested:
        return requested
    models = list(client.models.list().data)
    if not models:
        raise RuntimeError("LM Studio exposes no models. Load one first.")
    return next((m.id for m in models if "qwen" in m.id.casefold()), models[0].id)

def parse_json(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise

def _is_qwen35_model(model: str) -> bool:
    normalized = model.casefold().replace("_", "").replace("-", "").replace(".", "")
    return "qwen35" in normalized

def _complete_json_prefix(text: str) -> str | None:
    """Return the first complete top-level JSON object, or None if incomplete.

    This is intentionally a small streaming parser. It tracks string/escape state
    so braces inside JSON strings do not affect nesting depth.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

def _qwen35_compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a retry schema small enough to finish under a short token cap."""
    compact = copy.deepcopy(schema)
    root_props = compact["schema"]["properties"]
    for name in ("characters", "locations", "objects"):
        if name in root_props:
            root_props[name]["maxItems"] = min(3, int(root_props[name].get("maxItems", 3)))

    def limit(node: dict[str, Any], field_name: str = "") -> None:
        if node.get("type") == "string" and "maxLength" in node:
            limit_by_field = {
                "chunk_summary": 240,
                "canonical_name": 80,
                "stable_visual_description": 180,
                "chapter_appearance": 160,
                "chapter_state": 160,
                "voice_description": 100,
                "evidence": 80,
            }
            node["maxLength"] = min(int(node["maxLength"]), limit_by_field.get(field_name, 80))
        if node.get("type") == "array":
            item_limit = {"aliases": 2, "distinguishing_features": 3, "reference_view_hints": 2, "evidence": 1}
            if field_name in item_limit:
                node["maxItems"] = min(int(node.get("maxItems", item_limit[field_name])), item_limit[field_name])
            items = node.get("items")
            if isinstance(items, dict):
                limit(items, field_name)
        for name, value in node.get("properties", {}).items():
            if isinstance(value, dict):
                limit(value, name)

    limit(compact["schema"])
    return compact


def chat_json(client: OpenAI, model: str, system: str, user: str,
              schema: dict[str, Any], temperature: float, max_tokens: int) -> dict[str, Any]:
    from .lmstudio_pipeline import comfy_interrupt_check

    qwen = _is_qwen35_model(model)
    # Structured-output backends can occasionally end a response mid-string
    # even for non-Qwen3.5 models. Always allow one compact retry instead of
    # failing the whole ComfyUI run on that transient malformed response.
    retries = QWEN35_LENGTH_RETRIES if qwen else 1
    for attempt in range(retries + 1):
        comfy_interrupt_check()
        request_schema = _qwen35_compact_schema(schema) if attempt else schema
        note = "\nReturn compact JSON with short descriptions and finish within the output limit." if attempt else ""
        extra = {"chat_template_kwargs": {"enable_thinking": THINKING_ENABLED}}
        if qwen:
            extra.update(top_k=QWEN35_TOP_K, min_p=QWEN35_MIN_P, repeat_penalty=QWEN35_REPEAT_PENALTY)
        started = time.perf_counter()
        # Transport/authentication failures propagate; only malformed output is retried.
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": ("/think" if THINKING_ENABLED else "/no_think") + "\n\n" + system},
                      {"role": "user", "content": user + note}],
            temperature=min(temperature, 0.12) if attempt else temperature,
            top_p=0.8 if qwen else 0.9, max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": request_schema},
            extra_body=extra, stream=True,
        )
        raw = ""
        content_chars = reasoning_chars = 0
        finish_reason = "not_received"
        local_stop = "stream_end"
        try:
            for event in stream:
                comfy_interrupt_check()
                if not event.choices:
                    continue
                choice = event.choices[0]
                reason = getattr(choice, "finish_reason", None)
                if reason is not None:
                    # Only log known metadata, never arbitrary server text.
                    finish_reason = reason if reason in {"stop", "length", "content_filter", "tool_calls", "function_call"} else "other"
                content = choice.delta.content or ""
                content_chars += len(content)
                for field in ("reasoning_content", "reasoning"):
                    reasoning = getattr(choice.delta, field, None)
                    if isinstance(reasoning, str):
                        reasoning_chars += len(reasoning)
                raw += content
                complete = _complete_json_prefix(raw)
                if complete is not None:
                    raw = complete
                    local_stop = "json_complete"
                    break
        except BaseException:
            local_stop = "interrupted_or_error"
            raise
        finally:
            stream.close()
            diagnostics = (
                f"attempt={attempt + 1}, max_tokens={max_tokens}, "
                f"content_chars={content_chars}, reasoning_chars={reasoning_chars}, "
                f"finish_reason={finish_reason}, local_stop={local_stop}"
            )
            print(f"    LLM stream: {diagnostics}", flush=True)
        try:
            result = parse_json(raw)
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object.")
            print(f"    LLM: structured JSON, {time.perf_counter() - started:.1f}s, attempt={attempt + 1}")
            return result
        except (ValueError, TypeError) as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Invalid structured JSON after {attempt + 1} attempt(s). {diagnostics}"
                ) from error
    raise AssertionError("Unreachable")

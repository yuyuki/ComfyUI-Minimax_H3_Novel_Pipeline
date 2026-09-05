#!/usr/bin/env python3
"""
STEP 2/3 — Consolidate chapter reference JSONs into a book-level registry and
build multi-view image/audio asset specifications.

Input:
    *_references.json files produced by 01_extract_chapter_references.py

Outputs:
    consolidated_references.json
    reference_asset_prompts.txt

Main v2 feature:
    One important entity may own multiple picture assets, e.g.
      PIC_CHAR_001_FACE_FRONT
      PIC_CHAR_001_FULL_BODY_FRONT
      PIC_CHAR_001_THREE_QUARTER
      PIC_CHAR_001_BACK_VIEW

MiniMax <Picture N>/<Subject N> labels are NOT permanently assigned here.
Step 3 maps the subset used by each clip to request-local H3 labels.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

# Qwen thinking control. Non-thinking is the default for this pipeline.
THINKING_ENABLED = False
CHAT_BACKEND = "auto"
QWEN35_MAX_OUTPUT_TOKENS = 3500
QWEN35_LENGTH_RETRIES = 2
SCRIPT_VERSION = "2.4.1"


def _is_comfy_interrupt(error: BaseException) -> bool:
    """Do not retry a ComfyUI Stop request as though it were an LLM error."""
    return error.__class__.__name__ == "InterruptProcessingException"


INPUT_SCHEMA = "minimax-h3-novel-refs.chapter.v2"
LEGACY_INPUT_SCHEMA = "minimax-h3-novel-refs.chapter.v1"
OUTPUT_SCHEMA = "minimax-h3-novel-refs.consolidated.v2"

IMPORTANCE_ORDER = {"background": 0, "minor": 1, "recurring": 2, "major": 3}
PRIORITY_ORDER = {"optional": 0, "recommended": 1, "required": 2}
TYPE_PREFIX = {"character": "CHAR", "location": "LOC", "object": "OBJ"}

CHARACTER_VIEWS = [
    "face_front",
    "full_body_front",
    "three_quarter",
    "back_view",
    "profile",
    "expression_closeup",
    "costume_detail",
]
LOCATION_VIEWS = [
    "wide_establishing",
    "secondary_angle",
    "reverse_angle",
    "key_detail",
    "interior_zone",
    "exterior_approach",
]
OBJECT_VIEWS = [
    "hero_three_quarter",
    "side_profile",
    "detail_closeup",
    "scale_context",
]
ALLOWED_VIEWS = {
    "character": CHARACTER_VIEWS,
    "location": LOCATION_VIEWS,
    "object": OBJECT_VIEWS,
}


def reconciliation_item_schema() -> dict[str, Any]:
    props = {
        "local_id": {"type": "string"},
        "entity_type": {"type": "string", "enum": ["character", "location", "object"]},
        "match_global_id": {"type": "string"},
        "canonical_name": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "stable_visual_description": {"type": "string"},
        "distinguishing_features": {"type": "array", "items": {"type": "string"}},
        "voice_description": {"type": "string"},
        "speaks": {"type": "boolean"},
        "importance": {"type": "string", "enum": list(IMPORTANCE_ORDER)},
        "reference_priority": {"type": "string", "enum": list(PRIORITY_ORDER)},
        "reference_view_hints": {"type": "array", "items": {"type": "string"}},
        "variant_reference_recommended": {"type": "boolean"},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


RECONCILE_SCHEMA = {
    "name": "chapter_to_global_reconciliation_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "resolutions": {"type": "array", "items": reconciliation_item_schema()}
        },
        "required": ["resolutions"],
        "additionalProperties": False,
    },
}


AUDIT_ITEM = {
    "keep_global_id": {"type": "string"},
    "merge_global_ids": {"type": "array", "items": {"type": "string"}},
    "canonical_name": {"type": "string"},
    "aliases": {"type": "array", "items": {"type": "string"}},
    "stable_visual_description": {"type": "string"},
    "distinguishing_features": {"type": "array", "items": {"type": "string"}},
    "voice_description": {"type": "string"},
    "reference_view_hints": {"type": "array", "items": {"type": "string"}},
    "reason": {"type": "string"},
}
AUDIT_SCHEMA = {
    "name": "global_duplicate_audit_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "merge_groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": AUDIT_ITEM,
                    "required": list(AUDIT_ITEM),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["merge_groups"],
        "additionalProperties": False,
    },
}

PICTURE_BRIEF_ITEM = {
    "asset_id": {"type": "string"},
    "description": {"type": "string"},
    "generation_prompt": {"type": "string"},
}
PICTURE_BRIEF_SCHEMA = {
    "name": "picture_asset_briefs_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "assets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": PICTURE_BRIEF_ITEM,
                    "required": list(PICTURE_BRIEF_ITEM),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["assets"],
        "additionalProperties": False,
    },
}

AUDIO_BRIEF_ITEM = {
    "asset_id": {"type": "string"},
    "description": {"type": "string"},
    "generation_prompt": {"type": "string"},
}
AUDIO_BRIEF_SCHEMA = {
    "name": "audio_asset_briefs_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "assets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": AUDIO_BRIEF_ITEM,
                    "required": list(AUDIO_BRIEF_ITEM),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["assets"],
        "additionalProperties": False,
    },
}


def natural_key(value: str) -> list[Any]:
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", value)]


def norm_name(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold(), flags=re.UNICODE).split())


def dedupe(values: Iterable[str], max_items: int = 1000) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        value = re.sub(r"\s+", " ", str(raw)).strip()
        key = value.casefold()
        if value and key not in seen:
            out.append(value)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


def stronger(a: str, b: str, order: dict[str, int]) -> str:
    return a if order.get(a, 0) >= order.get(b, 0) else b


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=300.0, max_retries=2)


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


def _use_qwen35_chatml(model: str) -> bool:
    if CHAT_BACKEND == "qwen35-chatml":
        return True
    if CHAT_BACKEND == "openai-chat":
        return False
    return _is_qwen35_model(model)



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


def _qwen35_stream_json_completion(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[str, float, int | None, str]:
    """Stream a manual ChatML completion and stop as soon as valid JSON closes.

    Qwen3.5 chat-tuned GGUFs used through /v1/completions do not always emit
    <|im_end|> promptly. Waiting for that token can make a compact JSON request
    run until max_tokens or the HTTP timeout. Streaming lets us terminate once
    the root JSON object is syntactically complete.
    """
    started = time.perf_counter()
    chunks: list[str] = []
    complete: str | None = None
    finish_reason = "json_complete"
    stream = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<END_JSON>"],
        stream=True,
    )
    try:
        for event in stream:
            if not event.choices:
                continue
            choice = event.choices[0]
            piece = choice.text or ""
            if piece:
                chunks.append(piece)
                current = "".join(chunks)
                complete = _complete_json_prefix(current)
                if complete is not None:
                    break
            if getattr(choice, "finish_reason", None):
                finish_reason = str(choice.finish_reason)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
    elapsed = time.perf_counter() - started
    raw = complete if complete is not None else "".join(chunks)
    return raw, elapsed, None, finish_reason

def _qwen35_chatml_prompt(system: str, user: str, schema: dict[str, Any]) -> str:
    # The Qwen3.5 GGUF template supplied by the model starts assistant generation
    # with <think>.  In non-thinking mode it inserts an EMPTY closed think block.
    # Building that prefix ourselves via /v1/completions avoids LM Studio builds
    # where enable_thinking/chat_template_kwargs are ignored by /v1/chat/completions.
    schema_text = json.dumps(schema["schema"], ensure_ascii=False)
    system_json = (
        system
        + "\n\nReturn ONLY valid JSON. Do not use Markdown fences or commentary. "
          "Your output must satisfy the JSON schema supplied by the user."
    )
    user_json = user + "\n\nRequired JSON schema:\n" + schema_text
    assistant_prefix = "<think>\n" if THINKING_ENABLED else "<think>\n\n</think>\n\n"
    return (
        "<|im_start|>system\n" + system_json + "<|im_end|>\n"
        "<|im_start|>user\n" + user_json + "<|im_end|>\n"
        "<|im_start|>assistant\n" + assistant_prefix
    )


def chat_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    start_time = time.perf_counter()

    # Qwen3.5-specific robust path. It bypasses LM Studio's chat-template
    # thinking toggle and json_schema/reasoning interaction by constructing the
    # model's ChatML prefix explicitly.
    if _use_qwen35_chatml(model):
        effective_max_tokens = min(max_tokens, QWEN35_MAX_OUTPUT_TOKENS)
        last_error: Exception | None = None
        last_raw = ""
        for attempt in range(QWEN35_LENGTH_RETRIES + 1):
            retry_note = ""
            if attempt:
                retry_note = (
                    "\n\nCRITICAL RETRY: The previous JSON was truncated or invalid. "
                    "Return a substantially more compact JSON response. Remove repetition, keep descriptions concise, "
                    "and close the root JSON object well before the token limit."
                )
            prompt = _qwen35_chatml_prompt(system, user + retry_note, schema)
            try:
                raw, elapsed, completion_tokens, finish_reason = _qwen35_stream_json_completion(
                    client=client, model=model, prompt=prompt,
                    temperature=(min(temperature, 0.15) if attempt else temperature),
                    top_p=(0.8 if attempt else 0.9), max_tokens=effective_max_tokens,
                )
                last_raw = raw
                token_note = f", {completion_tokens} output tokens" if completion_tokens is not None else ""
                print(
                    f"    LLM: qwen35-chatml-stream, thinking={'on' if THINKING_ENABLED else 'off'}, "
                    f"{elapsed:.1f}s{token_note}, stop={finish_reason}, cap={effective_max_tokens}, attempt={attempt + 1}"
                )
                if finish_reason == "length":
                    last_error = RuntimeError(f"generation hit output cap ({effective_max_tokens} tokens)")
                    if attempt < QWEN35_LENGTH_RETRIES:
                        print("    retrying with aggressive JSON compaction...")
                        continue
                try:
                    return parse_json(raw)
                except Exception as parse_error:
                    last_error = parse_error
                    if attempt < QWEN35_LENGTH_RETRIES:
                        print(f"    JSON incomplete/invalid ({parse_error}); retrying compactly...")
                        continue
            except Exception as error:
                if _is_comfy_interrupt(error):
                    raise
                last_error = error
                if attempt < QWEN35_LENGTH_RETRIES:
                    print(f"    Qwen3.5 call failed ({error}); retrying...")
                    continue
        elapsed_total = time.perf_counter() - start_time
        snippet = last_raw[-1200:] if last_raw else "<no output>"
        raise RuntimeError(
            f"Qwen3.5 manual ChatML completion failed after {elapsed_total:.1f}s and "
            f"{QWEN35_LENGTH_RETRIES + 1} attempt(s): {last_error}\nTail of last raw output:\n{snippet}"
        )

    # Generic OpenAI-compatible chat path for non-Qwen3.5 models. We still send
    # both common hints. Servers/models that do not recognize them may ignore them.
    thinking_directive = "/think" if THINKING_ENABLED else "/no_think"
    controlled_system = f"{thinking_directive}\n\n{system}"
    messages = [{"role": "system", "content": controlled_system}, {"role": "user", "content": user}]
    extra_body = {
        "enableThinking": bool(THINKING_ENABLED),
        "chat_template_kwargs": {"enable_thinking": bool(THINKING_ENABLED)},
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": schema},
            extra_body=extra_body,
        )
        raw = response.choices[0].message.content or ""
        if not raw.strip():
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            if reasoning:
                raise RuntimeError(
                    "LM Studio returned reasoning_content but empty content; "
                    "try --chat-backend qwen35-chatml for a Qwen3.5 model."
                )
        elapsed = time.perf_counter() - start_time
        print(f"    LLM: openai-chat structured, {elapsed:.1f}s")
        return parse_json(raw)
    except Exception as first_error:
        if _is_comfy_interrupt(first_error):
            raise
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": controlled_system + "\nReturn ONLY valid JSON with no Markdown."},
                {
                    "role": "user",
                    "content": user + "\n\nRequired JSON schema:\n" + json.dumps(schema["schema"], ensure_ascii=False),
                },
            ],
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        raw = response.choices[0].message.content or ""
        elapsed = time.perf_counter() - start_time
        print(f"    LLM: openai-chat JSON fallback, {elapsed:.1f}s")
        try:
            return parse_json(raw)
        except Exception as second_error:
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            reasoning_note = "\nReasoning stream was present." if reasoning else ""
            raise RuntimeError(
                f"Structured output failed: {first_error}\n"
                f"Fallback JSON failed: {second_error}{reasoning_note}\n"
                f"Raw output:\n{raw[:3000]}"
            ) from second_error


def discover_jsons(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        [p for p in path.glob("*_references.json") if p.is_file() and p.name != "consolidated_references.json"],
        key=lambda p: natural_key(p.name),
    )


def load_chapters(paths: list[Path]) -> list[dict[str, Any]]:
    chapters: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        schema = data.get("schema_version")
        if schema not in {INPUT_SCHEMA, LEGACY_INPUT_SCHEMA}:
            raise ValueError(
                f"{path.name}: unsupported schema {schema!r}; expected {INPUT_SCHEMA!r} "
                f"or legacy {LEGACY_INPUT_SCHEMA!r}."
            )
        if schema == LEGACY_INPUT_SCHEMA:
            print(f"INFO: upgrading legacy v1 chapter JSON in memory: {path.name}")
            for key in ("characters", "locations", "objects"):
                for entity in data.get(key, []):
                    entity.setdefault("reference_view_hints", [])
        data["_json_path"] = str(path.resolve())
        chapters.append(data)
    return chapters


def incoming_entities(chapter: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source_key, entity_type, state_key in (
        ("characters", "character", "chapter_appearance"),
        ("locations", "location", "chapter_state"),
        ("objects", "object", "chapter_state"),
    ):
        for e in chapter.get(source_key, []):
            out.append(
                {
                    "chapter_id": chapter["chapter_id"],
                    "local_id": e["local_id"],
                    "entity_type": entity_type,
                    "canonical_name": e.get("canonical_name", ""),
                    "aliases": e.get("aliases", []),
                    "stable_visual_description": e.get("stable_visual_description", ""),
                    "chapter_visual_state": e.get(state_key, ""),
                    "distinguishing_features": e.get("distinguishing_features", []),
                    "voice_description": e.get("voice_description", "") if entity_type == "character" else "",
                    "speaks": bool(e.get("speaks", False)) if entity_type == "character" else False,
                    "importance": e.get("importance", "minor"),
                    "reference_priority": e.get("reference_priority", "optional"),
                    "reference_view_hints": e.get("reference_view_hints", []),
                }
            )
    return out


def compact_global(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "global_id": entity["global_id"],
        "entity_type": entity["entity_type"],
        "canonical_name": entity["canonical_name"],
        "aliases": entity.get("aliases", []),
        "stable_visual_description": entity.get("stable_visual_description", ""),
        "distinguishing_features": entity.get("distinguishing_features", []),
        "voice_description": entity.get("voice_description", ""),
        "speaks": entity.get("speaks", False),
        "importance": entity.get("importance", "minor"),
        "reference_priority": entity.get("reference_priority", "optional"),
        "reference_view_hints": entity.get("reference_view_hints", []),
    }


def similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    a_names = [norm_name(a.get("canonical_name", "")), *[norm_name(x) for x in a.get("aliases", [])]]
    b_names = [norm_name(b.get("canonical_name", "")), *[norm_name(x) for x in b.get("aliases", [])]]
    best = 0.0
    for x in filter(None, a_names):
        for y in filter(None, b_names):
            if x == y:
                return 1.0
            ratio = difflib.SequenceMatcher(None, x, y).ratio()
            tx, ty = set(x.split()), set(y.split())
            overlap = len(tx & ty) / max(1, len(tx | ty))
            best = max(best, ratio, overlap)
    return best


def candidate_catalog(
    incoming: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    top_k: int,
    include_all_below: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in incoming:
        pool = [e for e in registry if e["entity_type"] == item["entity_type"]]
        if len(pool) <= include_all_below:
            selected = pool
        else:
            scored = sorted(((similarity(item, e), e) for e in pool), key=lambda x: x[0], reverse=True)
            selected = [e for score, e in scored[:top_k] if score >= 0.15] or [e for _, e in scored[:3]]
        out[item["local_id"]] = [compact_global(x) for x in selected]
    return out


def next_global_id(registry: list[dict[str, Any]], entity_type: str) -> str:
    prefix = TYPE_PREFIX[entity_type]
    nums = []
    for e in registry:
        m = re.fullmatch(rf"{prefix}_(\d+)", e["global_id"])
        if m:
            nums.append(int(m.group(1)))
    return f"{prefix}_{(max(nums) + 1 if nums else 1):03d}"


RECONCILE_SYSTEM = """
Reconcile chapter-local fictional entities against an existing cross-novel registry.
For each incoming entity, decide whether it is exactly the same character/location/
object as one candidate global entity.

Rules:
- match_global_id must be one supplied candidate global_id or exactly NEW.
- Never merge different entities merely because descriptions are similar.
- Names, aliases, relationships, distinctive traits and narrative role are stronger
  identity evidence than generic appearance.
- When uncertain, choose NEW.
- Merge only source-supported profile information; never invent missing traits.
- reference_view_hints should be the union of justified useful views and must be
  valid for the entity type.
- variant_reference_recommended is true only when the chapter-specific visible
  state merits an alternate reusable image: substantial disguise/costume, major
  injury/transformation, large time jump, structural damage/change, etc. Ordinary
  lighting/weather or trivial clothing changes should normally be false.
""".strip()


def reconcile_chapter(
    client: OpenAI,
    model: str,
    chapter: dict[str, Any],
    registry: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    incoming = incoming_entities(chapter)
    candidates = candidate_catalog(incoming, registry, args.candidate_count, args.include_all_below)
    result = chat_json(
        client,
        model,
        RECONCILE_SYSTEM,
        f"Chapter: {chapter['chapter_id']}\n\nINCOMING:\n{json.dumps(incoming, ensure_ascii=False, indent=2)}\n\nCANDIDATES:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}",
        RECONCILE_SCHEMA,
        args.temperature,
        args.max_tokens,
    )
    resolutions = {x["local_id"]: x for x in result.get("resolutions", [])}
    current_ids = {e["global_id"] for e in registry}

    for item in incoming:
        r = resolutions.get(item["local_id"], {})
        match = r.get("match_global_id", "NEW")
        if match != "NEW" and match not in current_ids:
            match = "NEW"

        if match == "NEW":
            gid = next_global_id(registry, item["entity_type"])
            e = {
                "global_id": gid,
                "entity_type": item["entity_type"],
                "canonical_name": (r.get("canonical_name") or item["canonical_name"]).strip(),
                "aliases": dedupe(item["aliases"] + r.get("aliases", []), 50),
                "stable_visual_description": (r.get("stable_visual_description") or item["stable_visual_description"]).strip(),
                "distinguishing_features": dedupe(r.get("distinguishing_features", item["distinguishing_features"]), 30),
                "voice_description": (r.get("voice_description") or item["voice_description"]).strip() if item["entity_type"] == "character" else "",
                "speaks": bool(r.get("speaks", item["speaks"])) if item["entity_type"] == "character" else False,
                "importance": r.get("importance", item["importance"]),
                "reference_priority": r.get("reference_priority", item["reference_priority"]),
                "reference_view_hints": dedupe(item["reference_view_hints"] + r.get("reference_view_hints", []), 20),
                "chapters_seen": [item["chapter_id"]],
                "source_entities": [{"chapter_id": item["chapter_id"], "local_id": item["local_id"]}],
                "chapter_variations": [],
            }
            registry.append(e)
            current_ids.add(gid)
        else:
            e = next(x for x in registry if x["global_id"] == match)
            e["canonical_name"] = (r.get("canonical_name") or e["canonical_name"]).strip()
            e["aliases"] = dedupe(e.get("aliases", []) + item["aliases"] + r.get("aliases", []), 50)
            e["stable_visual_description"] = (r.get("stable_visual_description") or e.get("stable_visual_description", "")).strip()
            e["distinguishing_features"] = dedupe(e.get("distinguishing_features", []) + r.get("distinguishing_features", []), 30)
            if e["entity_type"] == "character":
                e["voice_description"] = (r.get("voice_description") or e.get("voice_description", "")).strip()
                e["speaks"] = bool(e.get("speaks") or item["speaks"] or r.get("speaks"))
            e["importance"] = stronger(e.get("importance", "minor"), r.get("importance", item["importance"]), IMPORTANCE_ORDER)
            e["reference_priority"] = stronger(e.get("reference_priority", "optional"), r.get("reference_priority", item["reference_priority"]), PRIORITY_ORDER)
            e["reference_view_hints"] = dedupe(e.get("reference_view_hints", []) + item["reference_view_hints"] + r.get("reference_view_hints", []), 20)
            if item["chapter_id"] not in e["chapters_seen"]:
                e["chapters_seen"].append(item["chapter_id"])
            src = {"chapter_id": item["chapter_id"], "local_id": item["local_id"]}
            if src not in e["source_entities"]:
                e["source_entities"].append(src)

        state = item.get("chapter_visual_state", "").strip()
        if state:
            variant = {
                "chapter_id": item["chapter_id"],
                "visual_state": state,
                "variant_reference_recommended": bool(r.get("variant_reference_recommended", False)),
            }
            old = next((x for x in e["chapter_variations"] if x["chapter_id"] == item["chapter_id"]), None)
            if old:
                old.update(variant)
            else:
                e["chapter_variations"].append(variant)

    return registry


AUDIT_SYSTEM = """
Audit the cross-novel registry for accidental duplicates. Merge IDs only when they
clearly identify the exact same fictional entity. Never merge merely similar
entities and never merge across entity types. Preserve only source-supported facts.
Union useful reference_view_hints. If there are no clear duplicates, return none.
""".strip()


def audit_registry(
    client: OpenAI,
    model: str,
    registry: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.no_audit or len(registry) < 2 or len(registry) > args.audit_max_entities:
        return registry
    result = chat_json(
        client,
        model,
        AUDIT_SYSTEM,
        json.dumps([compact_global(e) for e in registry], ensure_ascii=False, indent=2),
        AUDIT_SCHEMA,
        min(args.temperature, 0.10),
        max(args.max_tokens, 6500),
    )
    by_id = {e["global_id"]: e for e in registry}
    removed: set[str] = set()
    for group in result.get("merge_groups", []):
        keep_id = group.get("keep_global_id")
        merge_ids = [x for x in group.get("merge_global_ids", []) if x in by_id and x != keep_id]
        if keep_id not in by_id or not merge_ids:
            continue
        keep = by_id[keep_id]
        if any(by_id[x]["entity_type"] != keep["entity_type"] for x in merge_ids):
            continue
        keep["canonical_name"] = (group.get("canonical_name") or keep["canonical_name"]).strip()
        keep["aliases"] = dedupe(
            keep.get("aliases", []) + group.get("aliases", []) +
            [by_id[x]["canonical_name"] for x in merge_ids] +
            sum((by_id[x].get("aliases", []) for x in merge_ids), []), 50
        )
        keep["stable_visual_description"] = (group.get("stable_visual_description") or keep.get("stable_visual_description", "")).strip()
        keep["distinguishing_features"] = dedupe(
            keep.get("distinguishing_features", []) + group.get("distinguishing_features", []) +
            sum((by_id[x].get("distinguishing_features", []) for x in merge_ids), []), 30
        )
        keep["reference_view_hints"] = dedupe(
            keep.get("reference_view_hints", []) + group.get("reference_view_hints", []) +
            sum((by_id[x].get("reference_view_hints", []) for x in merge_ids), []), 20
        )
        if keep["entity_type"] == "character":
            keep["voice_description"] = (group.get("voice_description") or keep.get("voice_description", "")).strip()
            keep["speaks"] = any(by_id[x].get("speaks", False) for x in [keep_id] + merge_ids)
        for mid in merge_ids:
            other = by_id[mid]
            keep["importance"] = stronger(keep["importance"], other["importance"], IMPORTANCE_ORDER)
            keep["reference_priority"] = stronger(keep["reference_priority"], other["reference_priority"], PRIORITY_ORDER)
            keep["chapters_seen"] = dedupe(keep["chapters_seen"] + other["chapters_seen"])
            for src in other["source_entities"]:
                if src not in keep["source_entities"]:
                    keep["source_entities"].append(src)
            for var in other["chapter_variations"]:
                if var not in keep["chapter_variations"]:
                    keep["chapter_variations"].append(var)
            removed.add(mid)
    return [e for e in registry if e["global_id"] not in removed]


def threshold(priority: str, minimum: str) -> bool:
    return PRIORITY_ORDER.get(priority, 0) >= PRIORITY_ORDER.get(minimum, 1)


def ordered_valid_views(entity_type: str, views: Iterable[str]) -> list[str]:
    allowed = ALLOWED_VIEWS[entity_type]
    requested = set(views or [])
    return [v for v in allowed if v in requested]


def desired_base_views(entity: dict[str, Any], args: argparse.Namespace) -> list[str]:
    typ = entity["entity_type"]
    importance = entity.get("importance", "minor")
    priority = entity.get("reference_priority", "optional")

    if typ == "character":
        if importance == "major" or priority == "required":
            base = ["face_front", "full_body_front", "three_quarter", "back_view"]
        elif importance == "recurring" or priority == "recommended":
            base = ["face_front", "full_body_front", "three_quarter"]
        else:
            base = ["face_front"]
        limit = args.max_character_base_views
    elif typ == "location":
        if importance == "major" or priority == "required":
            base = ["wide_establishing", "secondary_angle", "key_detail"]
        elif importance == "recurring" or priority == "recommended":
            base = ["wide_establishing", "secondary_angle"]
        else:
            base = ["wide_establishing"]
        limit = args.max_location_base_views
    else:
        if importance == "major" or priority == "required":
            base = ["hero_three_quarter", "detail_closeup"]
        else:
            base = ["hero_three_quarter"]
        limit = args.max_object_base_views

    merged = ordered_valid_views(typ, base + entity.get("reference_view_hints", []))
    return merged[:max(1, limit)]


def variant_views(entity_type: str) -> list[str]:
    if entity_type == "character":
        return ["full_body_front", "face_front"]
    if entity_type == "location":
        return ["wide_establishing"]
    return ["hero_three_quarter"]


def build_picture_specs(registry: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for e in registry:
        if not threshold(e.get("reference_priority", "optional"), args.picture_threshold):
            continue
        for view in desired_base_views(e, args):
            specs.append(
                {
                    "asset_id": f"PIC_{e['global_id']}_{view.upper()}",
                    "linked_global_id": e["global_id"],
                    "entity_type": e["entity_type"],
                    "canonical_name": e["canonical_name"],
                    "variant": "base",
                    "view_type": view,
                    "chapters": e["chapters_seen"],
                    "stable_visual_description": e.get("stable_visual_description", ""),
                    "distinguishing_features": e.get("distinguishing_features", []),
                    "chapter_visual_state": "",
                }
            )

        if args.no_variants:
            continue
        for var in e.get("chapter_variations", []):
            if not var.get("variant_reference_recommended"):
                continue
            for view in variant_views(e["entity_type"]):
                specs.append(
                    {
                        "asset_id": f"PIC_{e['global_id']}_{var['chapter_id'].upper()}_{view.upper()}",
                        "linked_global_id": e["global_id"],
                        "entity_type": e["entity_type"],
                        "canonical_name": e["canonical_name"],
                        "variant": var["chapter_id"],
                        "view_type": view,
                        "chapters": [var["chapter_id"]],
                        "stable_visual_description": e.get("stable_visual_description", ""),
                        "distinguishing_features": e.get("distinguishing_features", []),
                        "chapter_visual_state": var.get("visual_state", ""),
                    }
                )
    return specs


def build_audio_specs(registry: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = []
    for e in registry:
        if e["entity_type"] != "character" or not e.get("speaks"):
            continue
        if not threshold(e.get("reference_priority", "optional"), args.audio_threshold):
            continue
        specs.append(
            {
                "asset_id": f"AUD_{e['global_id']}_VOICE",
                "linked_global_id": e["global_id"],
                "canonical_name": e["canonical_name"],
                "chapters": e["chapters_seen"],
                "voice_description": e.get("voice_description", ""),
            }
        )
    return specs


PICTURE_BRIEF_SYSTEM = """
Create reusable image-reference briefs for a novel-to-video workflow.
You receive explicit asset specs. Return exactly one brief for every asset_id.

Rules:
- Preserve the entity's canonical identity across all of its views.
- Use only source-supported stable visual details. Do NOT invent unspecified age,
  ethnicity, hair/eye color, body shape, clothing, architecture, markings, etc.
- The generation_prompt must explicitly request the supplied view_type.
- References should be neutral and legible rather than action-heavy.
- Character face_front: clear identity portrait/front head-and-shoulders or chest-up.
- Character full_body_front: head-to-toe, front-facing, neutral pose, unobstructed.
- three_quarter/profile/back_view must preserve exactly the same identity, body,
  hair and clothing characteristics as the canonical description.
- Location wide_establishing should show persistent spatial layout; secondary/reverse
  angles should depict the same place from another coherent viewpoint; key_detail
  should isolate a source-supported distinctive feature.
- Object references should clearly preserve shape, materials and distinctive details.
- For chapter variants, preserve canonical identity and apply ONLY chapter_visual_state.
- Keep lighting sufficiently neutral for reference utility unless lighting itself is
  a persistent defining trait.
- Do not include MiniMax <Picture N>/<Subject N> labels in generation_prompt.
""".strip()


AUDIO_BRIEF_SYSTEM = """
Create clean reusable voice-reference briefs for speaking novel characters.
Return exactly one brief per asset_id. Preserve only source-supported voice traits.
If the novel gives no vocal traits, request a neutral, consistent, character-
appropriate delivery without inventing accent, precise pitch, age, ethnicity or
other unsupported vocal characteristics. Prefer a dry recording with no music,
reverb or environmental noise. Do not include MiniMax labels.
""".strip()


def batched(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for i in range(0, len(items), max(1, size)):
        yield items[i:i + max(1, size)]


def generate_picture_assets(
    client: OpenAI,
    model: str,
    specs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    briefs: dict[str, dict[str, str]] = {}
    batches = list(batched(specs, args.asset_batch_size))
    for i, batch in enumerate(batches, start=1):
        print(f"  picture brief batch {i}/{len(batches)} ({len(batch)} assets)")
        result = chat_json(
            client,
            model,
            PICTURE_BRIEF_SYSTEM,
            json.dumps(batch, ensure_ascii=False, indent=2),
            PICTURE_BRIEF_SCHEMA,
            0.22,
            max(args.max_tokens, 7000),
        )
        for item in result.get("assets", []):
            briefs[item["asset_id"]] = item
        if args.delay:
            time.sleep(args.delay)

    assets: list[dict[str, Any]] = []
    for spec in specs:
        brief = briefs.get(spec["asset_id"])
        if not brief:
            raise RuntimeError(f"LLM omitted picture asset brief {spec['asset_id']}")
        asset = {
            **{k: v for k, v in spec.items() if k not in {"stable_visual_description", "distinguishing_features", "chapter_visual_state"}},
            "asset_role": "identity_reference" if spec["entity_type"] == "character" else ("environment_reference" if spec["entity_type"] == "location" else "object_reference"),
            "description": brief["description"].strip(),
            "generation_prompt": brief["generation_prompt"].strip(),
            "suggested_filename": spec["asset_id"].lower() + ".png",
        }
        assets.append(asset)
    for i, asset in enumerate(assets, start=1):
        asset["canonical_label"] = f"<Picture {i}>"
    return assets


def generate_audio_assets(
    client: OpenAI,
    model: str,
    specs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not specs:
        return []
    briefs: dict[str, dict[str, str]] = {}
    batches = list(batched(specs, args.asset_batch_size))
    for i, batch in enumerate(batches, start=1):
        print(f"  audio brief batch {i}/{len(batches)} ({len(batch)} assets)")
        result = chat_json(
            client,
            model,
            AUDIO_BRIEF_SYSTEM,
            json.dumps(batch, ensure_ascii=False, indent=2),
            AUDIO_BRIEF_SCHEMA,
            0.18,
            max(args.max_tokens, 6000),
        )
        for item in result.get("assets", []):
            briefs[item["asset_id"]] = item
        if args.delay:
            time.sleep(args.delay)

    assets = []
    for spec in specs:
        brief = briefs.get(spec["asset_id"])
        if not brief:
            raise RuntimeError(f"LLM omitted audio asset brief {spec['asset_id']}")
        assets.append(
            {
                "asset_id": spec["asset_id"],
                "linked_global_id": spec["linked_global_id"],
                "canonical_name": spec["canonical_name"],
                "role": "voice_timbre_reference",
                "chapters": spec["chapters"],
                "description": brief["description"].strip(),
                "generation_prompt": brief["generation_prompt"].strip(),
                "suggested_filename": spec["asset_id"].lower() + ".wav",
            }
        )
    for i, asset in enumerate(assets, start=1):
        asset["canonical_label"] = f"<Audio {i}>"
    return assets


def build_chapter_map(registry: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for entity in registry:
        for src in entity["source_entities"]:
            out.setdefault(src["chapter_id"], {})[src["local_id"]] = entity["global_id"]
    return out


def build_entity_asset_index(
    registry: list[dict[str, Any]],
    pictures: list[dict[str, Any]],
    audio: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        e["global_id"]: {"picture_asset_ids": [], "audio_asset_ids": []}
        for e in registry
    }
    for p in pictures:
        out.setdefault(p["linked_global_id"], {"picture_asset_ids": [], "audio_asset_ids": []})["picture_asset_ids"].append(p["asset_id"])
    for a in audio:
        out.setdefault(a["linked_global_id"], {"picture_asset_ids": [], "audio_asset_ids": []})["audio_asset_ids"].append(a["asset_id"])
    return out


def write_asset_prompts(path: Path, pictures: list[dict[str, Any]], audio: list[dict[str, Any]]) -> None:
    blocks: list[str] = []
    current_entity = None
    for p in pictures:
        if p["linked_global_id"] != current_entity:
            current_entity = p["linked_global_id"]
            blocks.append(f"######## {current_entity} — {p['canonical_name']} ########")
        blocks.append(
            f"=== {p['asset_id']} | view={p['view_type']} | variant={p['variant']} ===\n"
            f"Suggested file: {p['suggested_filename']}\n"
            f"Description: {p['description']}\n\n"
            f"IMAGE GENERATION PROMPT:\n{p['generation_prompt']}"
        )
    if audio:
        blocks.append("######## VOICE REFERENCES ########")
    for a in audio:
        blocks.append(
            f"=== {a['asset_id']} | {a['canonical_name']} ===\n"
            f"Suggested file: {a['suggested_filename']}\n"
            f"Description: {a['description']}\n\n"
            f"AUDIO GENERATION PROMPT:\n{a['generation_prompt']}"
        )
    path.write_text("\n\n\n".join(blocks) + "\n", encoding="utf-8")



def _cli_quote(value: Any) -> str:
    """Quote one argument for a copy/paste friendly shell command."""
    text = str(value)
    if not text:
        return '""'
    if re.search(r'[\\s"&|<>^()]', text):
        return '"' + text.replace('"', '\\"') + '"'
    return text


def _format_command(parts: list[Any]) -> str:
    return " ".join(_cli_quote(x) for x in parts)



def _recommended_chapter_inputs(chapters: list[dict[str, Any]]) -> list[str]:
    """Prefer one common source directory; otherwise use exact source paths."""
    paths: list[Path] = []
    for chapter in chapters:
        raw = chapter.get("source", {}).get("absolute_path")
        if raw:
            paths.append(Path(raw))
    if not paths:
        return ["chapters"]
    parents = {str(p.parent) for p in paths}
    if len(parents) == 1:
        return [str(paths[0].parent)]
    return [str(p) for p in paths]

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Consolidate v2 chapter refs and generate multi-view asset specs.")
    p.add_argument("input", type=Path, help="Directory of step-1 JSONs or a single JSON.")
    p.add_argument("--out", type=Path, default=Path("consolidated_references.json"))
    p.add_argument("--asset-prompts-out", type=Path, default=Path("reference_asset_prompts.txt"))
    p.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    p.add_argument("--api-key", default="lm-studio")
    p.add_argument("--model", default=None)
    thinking = p.add_mutually_exclusive_group()
    thinking.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        help="Enable Qwen reasoning/thinking mode (/think).",
    )
    thinking.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Disable Qwen reasoning/thinking mode (/no_think, default).",
    )
    p.set_defaults(thinking=False)
    p.add_argument(
        "--chat-backend",
        choices=["auto", "openai-chat", "qwen35-chatml"],
        default="auto",
        help=(
            "LLM transport for JSON calls. auto uses manual Qwen3.5 ChatML via "
            "/v1/completions for qwen3.5/qwen35 model IDs, otherwise OpenAI chat. "
            "qwen35-chatml is the recommended workaround when LM Studio ignores "
            "enable_thinking=false for Qwen3.5."
        ),
    )
    p.add_argument("--temperature", type=float, default=0.12)
    p.add_argument("--max-tokens", type=int, default=8500)
    p.add_argument(
        "--qwen35-max-output-tokens", "--max-output-tokens",
        dest="qwen35_max_output_tokens",
        type=int,
        default=3500,
        help=(
            "Safety cap for manual Qwen3.5 ChatML JSON completions. "
            "Streaming normally stops earlier as soon as a complete JSON object closes."
        ),
    )
    p.add_argument("--candidate-count", type=int, default=12)
    p.add_argument("--include-all-below", type=int, default=35)
    p.add_argument("--picture-threshold", choices=list(PRIORITY_ORDER), default="recommended")
    p.add_argument("--audio-threshold", choices=list(PRIORITY_ORDER), default="recommended")
    p.add_argument("--max-character-base-views", type=int, default=4)
    p.add_argument("--max-location-base-views", type=int, default=3)
    p.add_argument("--max-object-base-views", type=int, default=2)
    p.add_argument("--asset-batch-size", type=int, default=16)
    p.add_argument("--no-variants", action="store_true")
    p.add_argument("--no-audit", action="store_true")
    p.add_argument("--audit-max-entities", type=int, default=120)
    p.add_argument("--qwen35-length-retries", type=int, default=2, help="Compact retries for truncated/invalid Qwen3.5 JSON.")
    p.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    p.add_argument("--delay", type=float, default=0.0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    global THINKING_ENABLED, CHAT_BACKEND, QWEN35_MAX_OUTPUT_TOKENS, QWEN35_LENGTH_RETRIES
    THINKING_ENABLED = bool(args.thinking)
    CHAT_BACKEND = args.chat_backend
    QWEN35_MAX_OUTPUT_TOKENS = max(256, int(args.qwen35_max_output_tokens))
    QWEN35_LENGTH_RETRIES = max(0, int(args.qwen35_length_retries))
    paths = discover_jsons(args.input)
    if not paths:
        print("ERROR: no chapter reference JSONs found.", file=sys.stderr)
        return 2
    try:
        chapters = load_chapters(paths)
        lm = make_client(args.base_url, args.api_key)
        model = select_model(lm, args.model)
    except Exception as exc:
        print(f"ERROR initializing: {exc}", file=sys.stderr)
        return 1

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"LM Studio: {args.base_url}")
    print(f"Model: {model}")
    print(f"Thinking: {'enabled' if THINKING_ENABLED else 'disabled'}")
    resolved_backend = "qwen35-chatml" if _use_qwen35_chatml(model) else "openai-chat"
    print(f"Chat backend: {resolved_backend} (requested: {CHAT_BACKEND})")
    if resolved_backend == "qwen35-chatml":
        print(f"Qwen3.5 output cap: {QWEN35_MAX_OUTPUT_TOKENS} tokens (stream stops at complete JSON)")
    print(f"Chapter JSONs: {len(chapters)}\n")

    registry: list[dict[str, Any]] = []
    for i, chapter in enumerate(chapters, start=1):
        print(f"[{i}/{len(chapters)}] Reconciling {chapter['chapter_id']}")
        try:
            registry = reconcile_chapter(lm, model, chapter, registry, args)
        except Exception as exc:
            print(f"ERROR reconciling {chapter['chapter_id']}: {exc}", file=sys.stderr)
            return 1
        if args.delay:
            time.sleep(args.delay)

    print(f"Registry before audit: {len(registry)} entities")
    try:
        registry = audit_registry(lm, model, registry, args)
    except Exception as exc:
        print(f"WARNING: duplicate audit failed; continuing: {exc}", file=sys.stderr)
    registry.sort(
        key=lambda e: (
            {"character": 0, "location": 1, "object": 2}[e["entity_type"]],
            natural_key(e["global_id"]),
        )
    )
    print(f"Registry after audit: {len(registry)} entities")

    picture_specs = build_picture_specs(registry, args)
    audio_specs = build_audio_specs(registry, args)
    print(f"Planned picture references: {len(picture_specs)}")
    print(f"Planned voice references:   {len(audio_specs)}")

    try:
        pictures = generate_picture_assets(lm, model, picture_specs, args)
        audio = generate_audio_assets(lm, model, audio_specs, args)
    except Exception as exc:
        print(f"ERROR generating asset briefs: {exc}", file=sys.stderr)
        return 1

    digest = hashlib.sha256(
        "\n".join(
            f"{c['chapter_id']}:{c.get('source', {}).get('sha256', '')}"
            for c in chapters
        ).encode()
    ).hexdigest()

    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "source_digest": digest,
        "llm": {"base_url": args.base_url, "model": model, "thinking": THINKING_ENABLED, "chat_backend": CHAT_BACKEND},
        "chapters": [
            {
                "chapter_id": c["chapter_id"],
                "source_file": c.get("source", {}).get("file", ""),
                "source_sha256": c.get("source", {}).get("sha256", ""),
            }
            for c in chapters
        ],
        "entities": registry,
        "picture_assets": pictures,
        "audio_assets": audio,
        "video_assets": [],
        "chapter_entity_map": build_chapter_map(registry),
        "entity_asset_index": build_entity_asset_index(registry, pictures, audio),
        "label_note": (
            "canonical_label is only a convenient full-registry ordering. MiniMax H3 labels are request-local. "
            "Step 3 maps the exact subset used by each clip to <Picture 1>..., <Audio 1>..., while multiple pictures may define the same <Subject N>."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.asset_prompts_out.parent.mkdir(parents=True, exist_ok=True)
    write_asset_prompts(args.asset_prompts_out, pictures, audio)

    print(f"\nSaved: {args.out}")
    print(f"Picture assets: {len(pictures)}")
    print(f"Audio assets:   {len(audio)}")
    print(f"Asset prompts:  {args.asset_prompts_out}")

    next_parts: list[Any] = [
        sys.executable,
        "03_generate_h3_prompts.py",
        *_recommended_chapter_inputs(chapters),
        "--references", args.out,
        "--out-dir", "h3_prompts",
        "--duration", 8,
        "--base-url", args.base_url,
        "--model", model,
        "--thinking" if THINKING_ENABLED else "--no-thinking",
        "--chat-backend", resolved_backend,
        "--max-output-tokens", QWEN35_MAX_OUTPUT_TOKENS,
        "--qwen35-length-retries", QWEN35_LENGTH_RETRIES,
    ]
    if args.api_key != "lm-studio":
        next_parts.extend(["--api-key", "YOUR_LM_STUDIO_API_KEY"])
    print("\nNext recommended command (Step 3/3):")
    print(_format_command(next_parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

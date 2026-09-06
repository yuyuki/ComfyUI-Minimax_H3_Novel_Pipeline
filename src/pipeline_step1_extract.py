"""ComfyUI pipeline step1 extract implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

from . import lmstudio_json as json_backend
from .lmstudio_json import chat_json, select_model as select_model

from .path_access import confined_path
from .util import read_chapter, split_chunks

from .util import CHAPTER_SCHEMA as SCHEMA_VERSION

# Qwen thinking control. Non-thinking is the default for this pipeline.


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

IMPORTANCE = ["major", "recurring", "minor", "background"]
PRIORITY = ["required", "recommended", "optional"]


def entity_schema(kind: str) -> dict[str, Any]:
    if kind == "character":
        props = {
            "canonical_name": {"type": "string", "maxLength": 100},
            "aliases": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 80}},
            "stable_visual_description": {"type": "string", "maxLength": 500},
            "chapter_appearance": {"type": "string", "maxLength": 500},
            "distinguishing_features": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 120}},
            "voice_description": {"type": "string", "maxLength": 300},
            "speaks": {"type": "boolean"},
            "importance": {"type": "string", "enum": IMPORTANCE},
            "reference_priority": {"type": "string", "enum": PRIORITY},
            "reference_view_hints": {
                "type": "array",
                "maxItems": 5,
                "items": {"type": "string", "enum": CHARACTER_VIEWS},
            },
            "evidence": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 120}},
        }
    elif kind == "location":
        props = {
            "canonical_name": {"type": "string", "maxLength": 100},
            "aliases": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 80}},
            "stable_visual_description": {"type": "string", "maxLength": 500},
            "chapter_state": {"type": "string", "maxLength": 500},
            "distinguishing_features": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 120}},
            "importance": {"type": "string", "enum": IMPORTANCE},
            "reference_priority": {"type": "string", "enum": PRIORITY},
            "reference_view_hints": {
                "type": "array",
                "maxItems": 4,
                "items": {"type": "string", "enum": LOCATION_VIEWS},
            },
            "evidence": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 120}},
        }
    else:
        props = {
            "canonical_name": {"type": "string", "maxLength": 100},
            "aliases": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 80}},
            "stable_visual_description": {"type": "string", "maxLength": 500},
            "chapter_state": {"type": "string", "maxLength": 500},
            "distinguishing_features": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 120}},
            "importance": {"type": "string", "enum": IMPORTANCE},
            "reference_priority": {"type": "string", "enum": PRIORITY},
            "reference_view_hints": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string", "enum": OBJECT_VIEWS},
            },
            "evidence": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 120}},
        }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


CHUNK_SCHEMA = {
    "name": "chapter_reference_candidates_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "chunk_summary": {"type": "string", "maxLength": 450},
            # Bounded root arrays are essential for a local model: without them
            # it can keep discovering incidental nouns until max_tokens closes an
            # otherwise valid JSON object mid-array.
            "characters": {"type": "array", "maxItems": 6, "items": entity_schema("character")},
            "locations": {"type": "array", "maxItems": 4, "items": entity_schema("location")},
            "objects": {"type": "array", "maxItems": 6, "items": entity_schema("object")},
        },
        "required": ["chunk_summary", "characters", "locations", "objects"],
        "additionalProperties": False,
    },
}


def merge_entity_schema(kind: str) -> dict[str, Any]:
    s = entity_schema(kind)
    props = {
        "source_candidate_ids": {"type": "array", "maxItems": 24, "items": {"type": "string", "maxLength": 40}},
        **s["properties"],
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


MERGE_SCHEMA = {
    "name": "chapter_reference_catalog_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "chapter_summary": {"type": "string", "maxLength": 600},
            "characters": {"type": "array", "items": merge_entity_schema("character")},
            "locations": {"type": "array", "items": merge_entity_schema("location")},
            "objects": {"type": "array", "items": merge_entity_schema("object")},
        },
        "required": ["chapter_summary", "characters", "locations", "objects"],
        "additionalProperties": False,
    },
}


def natural_key(value: str) -> list[Any]:
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", value)]


def slug(text: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", text.strip(), flags=re.UNICODE).strip("._")
    return value or "chapter"




def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()




def make_client(base_url: str, api_key: str, *, http_client=None) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=300.0, max_retries=2, http_client=http_client)


def compact_strings(values: Iterable[str], max_items: int, max_len: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        value = re.sub(r"\s+", " ", str(raw)).strip()[:max_len]
        key = value.casefold()
        if value and key not in seen:
            out.append(value)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


def clean_entity(entity: dict[str, Any], kind: str) -> dict[str, Any]:
    e = dict(entity)
    e["canonical_name"] = re.sub(r"\s+", " ", e.get("canonical_name", "")).strip()
    e["aliases"] = compact_strings(e.get("aliases", []), 6, 80)
    e["distinguishing_features"] = compact_strings(e.get("distinguishing_features", []), 6, 120)
    e["evidence"] = compact_strings(e.get("evidence", []), 3, 120)
    e["stable_visual_description"] = re.sub(r"\s+", " ", e.get("stable_visual_description", "")).strip()
    e["reference_view_hints"] = list(dict.fromkeys(e.get("reference_view_hints", [])))[:5]
    if kind == "characters":
        e["chapter_appearance"] = re.sub(r"\s+", " ", e.get("chapter_appearance", "")).strip()
        e["voice_description"] = re.sub(r"\s+", " ", e.get("voice_description", "")).strip()
        e["speaks"] = bool(e.get("speaks", False))
    else:
        e["chapter_state"] = re.sub(r"\s+", " ", e.get("chapter_state", "")).strip()
    return e


EXTRACT_SYSTEM = f"""
You are a continuity/reference analyst for a novel-to-video adaptation.

Extract ONLY facts supported by the supplied prose. Never invent age, ethnicity,
hair/eye color, body shape, clothing, architecture, accent, voice pitch, or other
traits that the chapter does not establish.

The goal is to identify reusable visual/audio reference entities for later MiniMax
H3 reference-to-video generation.

Character fields:
- stable_visual_description: identity-level traits likely to remain true.
- chapter_appearance: temporary wardrobe, injuries, dirt/wetness, disguise, carried
  items, age-state or other chapter-specific visible state.
- voice_description: only source-supported voice/delivery traits; empty if unknown.
- speaks: true only when the character actually speaks.
- reference_view_hints: choose views that would materially help preserve identity
  or reproduce likely shots. Allowed: {', '.join(CHARACTER_VIEWS)}.

Location fields:
- stable_visual_description: persistent layout/architecture/environment.
- chapter_state: temporary weather, lighting, damage, crowd, time-of-day, etc.
- reference_view_hints allowed: {', '.join(LOCATION_VIEWS)}.

Object fields:
- include only distinctive, recurring, plot-relevant, or visually important props.
- reference_view_hints allowed: {', '.join(OBJECT_VIEWS)}.

For view hints, be selective. A major character may justify face_front,
full_body_front, three_quarter and back_view; a major location may justify
wide_establishing, secondary_angle and key_detail. Minor entities usually need
only one view.

reference_priority:
- required: continuity would noticeably break without a reference.
- recommended: useful recurring consistency.
- optional: minor/background.

STRICT COMPACTNESS RULES:
- chunk_summary: max 450 characters.
- aliases: max 6 short items.
- distinguishing_features: max 6 items, max 120 characters each.
- reference_view_hints: max 5 for characters, 4 for locations, 3 for objects.
- evidence: MAXIMUM 3 items per entity, MAXIMUM 120 characters each.
- Evidence is only a compact factual anchor. NEVER reproduce long dialogue, long quotations, or whole sentences when a shorter anchor is enough.
- stable_visual_description/chapter_appearance/chapter_state: concise; do not narrate events.
- Keep stable_visual_description strictly persistent. Clothing, wounds, wetness, dirt, restraint state, carried gear, temporary exposure, and other scene-specific conditions belong in chapter_appearance/chapter_state, not the stable identity.
- Return the smallest JSON that fully captures continuity-relevant information.

Evidence must be brief and grounded in the supplied passage. Do not fabricate quotes.
""".strip()


MERGE_SYSTEM = """
Merge duplicate reference candidates extracted from overlapping chunks of ONE chapter.
Merge only candidates that clearly denote the same fictional character, location or
object. Preserve approximate first-appearance order. Do not merge merely similar
entities. Never invent missing visual/voice traits.

Combine reference_view_hints as the union of justified views. Keep the strongest
justified importance and reference_priority. Every output entity must list every
candidate_id it absorbed in source_candidate_ids.

Be compact: preserve at most 3 short evidence anchors per entity, never expand quotations, and keep descriptions concise. The merge output should normally be smaller than the combined input.
""".strip()


def extract_chunk(
    client: OpenAI,
    model: str,
    chapter_id: str,
    text: str,
    index: int,
    total: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    user = f"""Chapter ID: {chapter_id}
Passage chunk: {index}/{total}

--- BEGIN NOVEL PASSAGE ---
{text}
--- END NOVEL PASSAGE ---"""
    return chat_json(client, model, EXTRACT_SYSTEM, user, CHUNK_SCHEMA, args.temperature, args.max_tokens)


def combine_candidates(results: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {"chunk_summaries": [], "characters": [], "locations": [], "objects": []}
    counters = {"characters": 0, "locations": 0, "objects": 0}
    prefixes = {"characters": "C", "locations": "L", "objects": "O"}
    for chunk_no, result in enumerate(results, start=1):
        combined["chunk_summaries"].append(result.get("chunk_summary", ""))
        for kind in ("characters", "locations", "objects"):
            for raw in result.get(kind, []):
                counters[kind] += 1
                item = clean_entity(raw, kind)
                item["candidate_id"] = f"{prefixes[kind]}{counters[kind]:03d}_CHUNK{chunk_no:03d}"
                item["_order"] = counters[kind]
                combined[kind].append(item)
    return combined


def merge_candidates(
    client: OpenAI,
    model: str,
    chapter_id: str,
    combined: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    compact: dict[str, Any] = {"chunk_summaries": combined["chunk_summaries"]}
    for kind in ("characters", "locations", "objects"):
        compact[kind] = [{k: v for k, v in x.items() if k != "_order"} for x in combined[kind]]
    return chat_json(
        client,
        model,
        MERGE_SYSTEM,
        f"Chapter ID: {chapter_id}\n\nMerge this catalog:\n{json.dumps(compact, ensure_ascii=False, indent=2)}",
        MERGE_SCHEMA,
        min(args.temperature, 0.2),
        args.max_tokens,
    )


def assign_local_ids(merged: dict[str, Any], combined: dict[str, Any]) -> dict[str, Any]:
    candidate_order: dict[str, int] = {}
    for kind in ("characters", "locations", "objects"):
        for item in combined[kind]:
            candidate_order[item["candidate_id"]] = item["_order"]

    output: dict[str, Any] = {"chapter_summary": merged.get("chapter_summary", "")}
    for kind, prefix in (("characters", "CHAR"), ("locations", "LOC"), ("objects", "OBJ")):
        items = [clean_entity(x, kind) for x in merged.get(kind, [])]
        items.sort(
            key=lambda item: min(
                (candidate_order.get(x, 10**9) for x in item.get("source_candidate_ids", [])),
                default=10**9,
            )
        )
        final: list[dict[str, Any]] = []
        for i, item in enumerate(items, start=1):
            item.pop("source_candidate_ids", None)
            local_id = f"{prefix}_{i:03d}"
            final.append({"local_id": local_id, **item})
        output[kind] = final
    return output


def _merged_as_partial(merged: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_summary": merged.get("chapter_summary", ""),
        "characters": [{k: v for k, v in x.items() if k != "source_candidate_ids"} for x in merged.get("characters", [])],
        "locations": [{k: v for k, v in x.items() if k != "source_candidate_ids"} for x in merged.get("locations", [])],
        "objects": [{k: v for k, v in x.items() if k != "source_candidate_ids"} for x in merged.get("objects", [])],
    }

def hierarchical_merge_candidates(
    client: OpenAI, model: str, chapter_id: str, chunk_results: list[dict[str, Any]],
    args: argparse.Namespace, cache_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bound every chapter merge prompt by --merge-batch-size.

    Returns the final merged catalog plus the immediate combined catalog used to
    assign local IDs, for stable local ordering.
    """
    batch_size = max(2, int(args.merge_batch_size))
    level = list(chunk_results)
    if not level:
        raise ValueError("No chunk catalogs to merge.")
    round_no = 1
    while True:
        next_level: list[dict[str, Any]] = []
        batch_count = (len(level) + batch_size - 1) // batch_size
        last_merged = None
        last_combined = None
        for batch_no, start in enumerate(range(0, len(level), batch_size), start=1):
            from .lmstudio_pipeline import comfy_interrupt_check
            comfy_interrupt_check()
            batch = level[start:start + batch_size]
            combined = combine_candidates(batch)
            key = hashlib.sha256((
                SCHEMA_VERSION + "\n" + model + "\n" + str(json_backend.THINKING_ENABLED) + "\n" + json_backend.CHAT_BACKEND + "\n" +
                str(args.temperature) + "\n" + str(args.max_tokens) + "\n" + chapter_id + "\n" +
                json.dumps(combined, ensure_ascii=False, sort_keys=True, default=str)
            ).encode()).hexdigest()
            cache_path = confined_path(cache_dir / f"merge_r{round_no:02d}_b{batch_no:03d}.json", cache_dir)
            merged = None
            if cache_path.exists() and not args.force:
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    if cached.get("cache_key") == key:
                        merged = cached["result"]
                except Exception:
                    pass
            if merged is None:
                print(f"  merging round {round_no}, batch {batch_no}/{batch_count} ({len(batch)} partial catalog(s))")
                merged = merge_candidates(client, model, chapter_id, combined, args)
                cache_path.write_text(json.dumps({"cache_key": key, "result": merged}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            else:
                print(f"  merge round {round_no}, batch {batch_no}/{batch_count}: cached")
            last_merged, last_combined = merged, combined
            partial = assign_local_ids(merged, combined)
            for kind in ("characters", "locations", "objects"):
                for entity in partial[kind]:
                    entity.pop("local_id", None)
            next_level.append(_merged_as_partial(partial))
        if len(next_level) == 1:
            assert last_merged is not None and last_combined is not None
            return last_merged, last_combined
        level = next_level
        round_no += 1


def process_chapter(
    path: Path,
    out_dir: Path,
    client: OpenAI,
    model: str,
    args: argparse.Namespace,
) -> Path:
    out_dir = out_dir.resolve()
    chapter_id = slug(path.stem)
    out_path = confined_path(out_dir / f"{chapter_id}_references.json", out_dir)
    source_hash = sha256_file(path)

    if out_path.exists() and not args.force:
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
            if old.get("schema_version") == SCHEMA_VERSION and old.get("source", {}).get("sha256") == source_hash:
                print(f"SKIP {path.name}: unchanged current output exists.")
                return out_path
        except Exception:
            pass

    text = read_chapter(path)
    requested_chunk_chars = max(3000, args.chunk_chars)
    effective_chunk_chars = requested_chunk_chars
    if json_backend._is_qwen35_model(model):
        effective_chunk_chars = min(requested_chunk_chars, max(3000, json_backend.QWEN35_SAFE_CHUNK_CHARS))
        if effective_chunk_chars != requested_chunk_chars:
            print(
                f"  Qwen3.5 safe chunking: {requested_chunk_chars:,} → {effective_chunk_chars:,} chars "
                "to keep each JSON catalog within its output budget"
            )
    chunks = split_chunks(text, effective_chunk_chars, max(0, args.overlap_paragraphs))
    cache_dir = confined_path(out_dir / ".cache" / chapter_id, out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"{path.name}: {len(text):,} chars, {len(chunks)} chunk(s)")
    chunk_results: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks, start=1):
        cache_path = confined_path(cache_dir / f"chunk_{i:03d}.json", out_dir)
        cache_key = hashlib.sha256((SCHEMA_VERSION + "\n" + model + "\nthinking=" + str(json_backend.THINKING_ENABLED) + "\nchat_backend=" + json_backend.CHAT_BACKEND + "\n" + chunk).encode()).hexdigest()
        result = None
        if cache_path.exists() and not args.force:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_key") == cache_key:
                    result = cached["result"]
            except Exception:
                pass
        if result is None:
            print(f"  extracting chunk {i}/{len(chunks)}")
            result = extract_chunk(client, model, chapter_id, chunk, i, len(chunks), args)
            cache_path.write_text(json.dumps({"cache_key": cache_key, "result": result}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            print(f"  chunk {i}/{len(chunks)}: cached")
        chunk_results.append(result)

    merged, combined = hierarchical_merge_candidates(
        client, model, chapter_id, chunk_results, args, cache_dir
    )

    catalog = assign_local_ids(merged, combined)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "chapter_id": chapter_id,
        "source": {
            "file": path.name,
            "absolute_path": str(path.resolve()),
            "sha256": source_hash,
            "character_count": len(text),
        },
        "llm": {"base_url": args.base_url, "model": model, "thinking": json_backend.THINKING_ENABLED, "chat_backend": json_backend.CHAT_BACKEND},
        **catalog,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"  saved {out_path.name}: {len(payload['characters'])} characters, "
        f"{len(payload['locations'])} locations, {len(payload['objects'])} objects"
    )
    return out_path



"""Small bundled subset of step 1 used when the repository script is absent.

This keeps the custom node installable on its own.  The full script is still
used automatically when the plugin is run from the source repository.
"""
from __future__ import annotations

import json
import re
from typing import Any

SCHEMA_VERSION = "minimax-h3-novel-refs.chapter.v2"
CHARACTER_VIEWS = ["face_front", "full_body_front", "three_quarter", "back_view", "profile", "expression_closeup", "costume_detail"]
LOCATION_VIEWS = ["wide_establishing", "secondary_angle", "reverse_angle", "key_detail", "interior_zone", "exterior_approach"]
OBJECT_VIEWS = ["hero_three_quarter", "side_profile", "detail_closeup", "scale_context"]
IMPORTANCE = ["major", "recurring", "minor", "background"]
PRIORITY = ["required", "recommended", "optional"]


def _entity_schema(kind: str) -> dict[str, Any]:
    props: dict[str, Any] = {
        "canonical_name": {"type": "string", "maxLength": 100},
        "aliases": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 80}},
        "stable_visual_description": {"type": "string", "maxLength": 500},
        "distinguishing_features": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 120}},
        "importance": {"type": "string", "enum": IMPORTANCE},
        "reference_priority": {"type": "string", "enum": PRIORITY},
        "evidence": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 120}},
    }
    if kind == "character":
        props.update({
            "chapter_appearance": {"type": "string", "maxLength": 500},
            "voice_description": {"type": "string", "maxLength": 300},
            "speaks": {"type": "boolean"},
            "reference_view_hints": {"type": "array", "maxItems": 5, "items": {"type": "string", "enum": CHARACTER_VIEWS}},
        })
    else:
        props.update({
            "chapter_state": {"type": "string", "maxLength": 500},
            "reference_view_hints": {"type": "array", "maxItems": 4 if kind == "location" else 3, "items": {"type": "string", "enum": LOCATION_VIEWS if kind == "location" else OBJECT_VIEWS}},
        })
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _merge_schema(kind: str) -> dict[str, Any]:
    schema = _entity_schema(kind)
    schema["properties"] = {"source_candidate_ids": {"type": "array", "maxItems": 24, "items": {"type": "string", "maxLength": 40}}, **schema["properties"]}
    schema["required"] = list(schema["properties"])
    return schema


CHUNK_SCHEMA = {"name": "chapter_reference_candidates_v2", "strict": True, "schema": {"type": "object", "properties": {"chunk_summary": {"type": "string", "maxLength": 450}, **{k: {"type": "array", "items": _entity_schema(k[:-1])} for k in ("characters", "locations", "objects")}}, "required": ["chunk_summary", "characters", "locations", "objects"], "additionalProperties": False}}
MERGE_SCHEMA = {"name": "chapter_reference_catalog_v2", "strict": True, "schema": {"type": "object", "properties": {"chapter_summary": {"type": "string", "maxLength": 600}, **{k: {"type": "array", "items": _merge_schema(k[:-1])} for k in ("characters", "locations", "objects")}}, "required": ["chapter_summary", "characters", "locations", "objects"], "additionalProperties": False}}

EXTRACT_SYSTEM = """Extract only continuity-relevant facts supported by the novel passage. Return compact JSON. Never invent traits. Include characters, locations, and distinctive plot-relevant objects."""
MERGE_SYSTEM = """Merge duplicate candidates from overlapping chunks of one chapter. Merge only clearly identical entities, preserve source facts, and keep the response compact."""


def split_chunks(text: str, max_chars: int, overlap_paragraphs: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for paragraph in paragraphs or [text]:
        if current and length + len(paragraph) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            length = sum(map(len, current)) + max(0, len(current) - 1) * 2
        current.append(paragraph)
        length += len(paragraph) + (2 if len(current) > 1 else 0)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def parse_json(value: str) -> dict[str, Any]:
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.S | re.I).strip()
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(value[start:end + 1])


def _clean(value: Any, limit: int) -> list[str]:
    return list(dict.fromkeys(str(x).strip() for x in (value or []) if str(x).strip()))[:limit]


def _clean_entity(raw: dict[str, Any], kind: str) -> dict[str, Any]:
    item = dict(raw)
    item["canonical_name"] = re.sub(r"\s+", " ", str(item.get("canonical_name", ""))).strip()
    item["aliases"] = _clean(item.get("aliases"), 6)
    item["distinguishing_features"] = _clean(item.get("distinguishing_features"), 6)
    item["evidence"] = _clean(item.get("evidence"), 3)
    return item


def combine_candidates(results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"chunk_summaries": [], "characters": [], "locations": [], "objects": []}
    for chunk_no, result in enumerate(results, 1):
        output["chunk_summaries"].append(result.get("chunk_summary", ""))
        for kind, prefix in (("characters", "C"), ("locations", "L"), ("objects", "O")):
            for raw in result.get(kind, []):
                item = _clean_entity(raw, kind)
                item["candidate_id"] = f"{prefix}{len(output[kind]) + 1:03d}_CHUNK{chunk_no:03d}"
                item["_order"] = len(output[kind]) + 1
                output[kind].append(item)
    return output


def assign_local_ids(merged: dict[str, Any], combined: dict[str, Any]) -> dict[str, Any]:
    output = {"chapter_summary": merged.get("chapter_summary", "")}
    for kind, prefix in (("characters", "CHAR"), ("locations", "LOC"), ("objects", "OBJ")):
        output[kind] = [{"local_id": f"{prefix}_{i:03d}", **_clean_entity(item, kind)} for i, item in enumerate(merged.get(kind, []), 1)]
        for item in output[kind]:
            item.pop("source_candidate_ids", None)
    return output

"""ComfyUI pipeline step2 consolidate implementation."""
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

from . import lmstudio_json as json_backend
from .lmstudio_json import chat_json, select_model as select_model, _is_comfy_interrupt

# Qwen thinking control. Non-thinking is the default for this pipeline.


INPUT_SCHEMA = "minimax-h3-novel-refs.chapter.v2"
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


def make_client(base_url: str, api_key: str, *, http_client=None) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=300.0, max_retries=2, http_client=http_client)


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


def _audit_candidate_clusters(registry: list[dict[str, Any]], similarity_threshold: float, max_cluster_size: int) -> list[list[str]]:
    """Build bounded likely-duplicate clusters without ever sending the full registry.

    Blocking by normalized tokens/prefixes keeps candidate-pair growth manageable while
    exact aliases and fuzzy name similarity connect plausible duplicates.
    """
    max_cluster_size = max(2, int(max_cluster_size))
    by_type: dict[str, list[dict[str, Any]]] = {}
    for entity in registry:
        by_type.setdefault(entity["entity_type"], []).append(entity)

    clusters: list[list[str]] = []
    for items in by_type.values():
        buckets: dict[str, set[int]] = {}
        for i, entity in enumerate(items):
            names = [norm_name(entity.get("canonical_name", "")), *[norm_name(x) for x in entity.get("aliases", [])]]
            keys: set[str] = set()
            for name in filter(None, names):
                keys.add("p:" + name[:3])
                keys.update("t:" + token for token in name.split() if len(token) >= 3)
            for key in keys:
                buckets.setdefault(key, set()).add(i)

        parent = list(range(len(items)))
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        seen_pairs: set[tuple[int, int]] = set()
        for members in buckets.values():
            ids = sorted(members)
            for ai in range(len(ids)):
                for bi in range(ai + 1, len(ids)):
                    a, b = ids[ai], ids[bi]
                    pair = (min(a, b), max(a, b))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    if similarity(items[a], items[b]) >= similarity_threshold:
                        union(a, b)

        components: dict[int, list[str]] = {}
        for i, entity in enumerate(items):
            components.setdefault(find(i), []).append(entity["global_id"])
        for ids in components.values():
            if len(ids) < 2:
                continue
            for offset in range(0, len(ids), max_cluster_size):
                part = ids[offset:offset + max_cluster_size]
                if len(part) >= 2:
                    clusters.append(part)
    return clusters


def _apply_audit_result(registry: list[dict[str, Any]], result: dict[str, Any]) -> set[str]:
    by_id = {e["global_id"]: e for e in registry}
    removed: set[str] = set()
    for group in result.get("merge_groups", []):
        keep_id = group.get("keep_global_id")
        merge_ids = [x for x in group.get("merge_global_ids", []) if x in by_id and x != keep_id and x not in removed]
        if keep_id not in by_id or keep_id in removed or not merge_ids:
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
    return removed


def audit_registry(
    client: OpenAI,
    model: str,
    registry: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.no_audit or len(registry) < 2:
        return registry

    # Small registries keep the old whole-registry audit because it is cheap and gives
    # the LLM maximum context. Large registries switch automatically to bounded clusters.
    if len(registry) <= args.audit_max_entities:
        result = chat_json(
            client, model, AUDIT_SYSTEM,
            json.dumps([compact_global(e) for e in registry], ensure_ascii=False, indent=2),
            AUDIT_SCHEMA, min(args.temperature, 0.10), max(args.max_tokens, 6500),
        )
        removed = _apply_audit_result(registry, result)
        return [e for e in registry if e["global_id"] not in removed]

    clusters = _audit_candidate_clusters(registry, float(args.audit_similarity), int(args.audit_cluster_size))
    print(f"  scalable audit: {len(clusters)} candidate cluster(s) from {len(registry)} entities")
    removed_all: set[str] = set()
    by_id = {e["global_id"]: e for e in registry}
    for i, ids in enumerate(clusters, start=1):
        from .lmstudio_pipeline import comfy_interrupt_check
        comfy_interrupt_check()
        active_ids = [x for x in ids if x in by_id and x not in removed_all]
        if len(active_ids) < 2:
            continue
        print(f"  audit cluster {i}/{len(clusters)} ({len(active_ids)} entities)")
        result = chat_json(
            client, model, AUDIT_SYSTEM,
            json.dumps([compact_global(by_id[x]) for x in active_ids], ensure_ascii=False, indent=2),
            AUDIT_SCHEMA, min(args.temperature, 0.10), max(args.max_tokens, 6500),
        )
        removed = _apply_audit_result([by_id[x] for x in active_ids], result)
        removed_all.update(removed)
        if args.delay:
            time.sleep(args.delay)
    if removed_all:
        print(f"  scalable audit merged {len(removed_all)} duplicate ID(s)")
    return [e for e in registry if e["global_id"] not in removed_all]


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



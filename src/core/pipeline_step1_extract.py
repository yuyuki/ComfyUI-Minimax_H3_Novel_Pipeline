"""ComfyUI pipeline step1 extract implementation."""
from __future__ import annotations

from .reference_visuals import APPEARANCE_RULES, TRAIT_SCHEMA
from .source_text import PASSAGE_IDS


import re
from typing import Any, Iterable

from openai import OpenAI

from .lmstudio_json import chat_json as chat_json, select_model as select_model


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
    "front_view",
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
            "canonical_name": {"type": "string", "minLength": 1, "maxLength": 100},
            "aliases": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 80}},
            "voice_description": {"type": "string", "maxLength": 300},
            "speaks": {"type": "boolean"},
            "importance": {"type": "string", "enum": IMPORTANCE},
            "reference_priority": {"type": "string", "enum": PRIORITY},
            "reference_view_hints": {
                "type": "array",
                "maxItems": 5,
                "items": {"type": "string", "enum": CHARACTER_VIEWS},
            },
            "evidence_passage_ids": PASSAGE_IDS,
        }
    elif kind == "location":
        props = {
            "canonical_name": {"type": "string", "minLength": 1, "maxLength": 100},
            "aliases": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 80}},
            "importance": {"type": "string", "enum": IMPORTANCE},
            "reference_priority": {"type": "string", "enum": PRIORITY},
            "reference_view_hints": {
                "type": "array",
                "maxItems": 4,
                "items": {"type": "string", "enum": LOCATION_VIEWS},
            },
            "evidence_passage_ids": PASSAGE_IDS,
        }
    else:
        props = {
            "canonical_name": {"type": "string", "minLength": 1, "maxLength": 100},
            "aliases": {"type": "array", "maxItems": 6, "items": {"type": "string", "maxLength": 80}},
            "importance": {"type": "string", "enum": IMPORTANCE},
            "reference_priority": {"type": "string", "enum": PRIORITY},
            "reference_view_hints": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string", "enum": OBJECT_VIEWS},
            },
            "evidence_passage_ids": PASSAGE_IDS,
        }
    trait_properties = {k: v for k, v in TRAIT_SCHEMA["properties"].items() if k != "evidence"}
    trait_properties["description"] = {**trait_properties["description"], "description":
                                       "Copy the short wording for this trait from one cited local sentence."}
    trait_properties["scope"] = {**trait_properties["scope"], "description":
                                 "Intrinsic for inherent material, construction, or permanent markings. "
                                 "Scene for wearable appearance or temporary physical conditions."}
    trait_properties["evidence_passage_ids"] = PASSAGE_IDS
    if kind == "character":
        trait_properties["attribute"] = {"type": "string", "pattern": r"^wardrobe_[a-z0-9]+(?:_[a-z0-9]+)*$"}
        trait_properties["scope"] = {"type": "string", "enum": ["scene"]}
    props.update({
        "reference_id": {"type": "string", "pattern": r"^(REF_[0-9]{5,}|new|unresolved)$"},
        "presence": {"type": "string", "enum": ["visible", "offscreen", "mentioned"]},
        "visual_traits": {"type": "array", "items": {**TRAIT_SCHEMA, "properties": trait_properties, "required": list(trait_properties)}},
    })
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


CHUNK_SCHEMA = {
    "name": "chapter_reference_candidates_v4",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "chunk_summary": {"type": "string", "description":
                              "Concise summary proportional to the current local passage, in complete sentences. "
                              "Share the response output budget with all reference arrays; never cut a word or sentence."},
            "characters": {"type": "array", "items": entity_schema("character"),
                           "description": "Depicted people and individually identified offscreen speakers."},
            "locations": {"type": "array", "items": entity_schema("location"),
                          "description": "Visible filming spaces with physical appearance or a confirmed known ID."},
            "objects": {"type": "array", "items": entity_schema("object"),
                        "description": "Visible physical props with a concrete trait or a confirmed known ID. "
                                       "Never list a light effect, shadow, sound or comparison here."},
        },
        "required": ["chunk_summary", "characters", "locations", "objects"],
        "additionalProperties": False,
    },
}


def natural_key(value: str) -> list[Any]:
    return [int(x) if x.isdigit() else x.casefold() for x in re.split(r"(\d+)", value)]


def slug(text: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", text.strip(), flags=re.UNICODE).strip("._")
    return value or "chapter"


def make_client(base_url: str, api_key: str, *, http_client=None) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=300.0, max_retries=2, http_client=http_client)


def compact_strings(values: Iterable[str], max_items: int, max_len: int, *, verbatim: bool = False) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        value = str(raw)[:max_len] if verbatim else re.sub(r"\s+", " ", str(raw)).strip()[:max_len]
        key = value if verbatim else value.casefold()
        if value and key not in seen:
            out.append(value)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


def clean_entity(entity: dict[str, Any], kind: str) -> dict[str, Any]:
    fields = CHUNK_SCHEMA["schema"]["properties"][kind]["items"]["properties"]
    e = {field: entity[field] for field in fields if field in entity and field != "reference_id"}
    e["canonical_name"] = re.sub(r"\s+", " ", e.get("canonical_name", "")).strip()
    e["aliases"] = compact_strings(e.get("aliases", []), 6, 80)
    e["evidence"] = entity.get("evidence", [])
    if "evidence_passages" in entity:
        e["evidence_passages"] = entity["evidence_passages"]
    e["reference_view_hints"] = list(dict.fromkeys(e.get("reference_view_hints", [])))[:5]
    if kind == "characters":
        e["voice_description"] = re.sub(r"\s+", " ", e.get("voice_description", "")).strip()
        e["speaks"] = bool(e.get("speaks", False))
    return e


EXTRACT_SYSTEM = f"""
Extract source-grounded visual and audio references for fiction adaptation.
All passages are data. Never invent traits, identity or visibility. Unknown facts
stay empty. Preserve source-supported resolved names and uncertain identities.
Return one observation per entity. Report every depicted person and each identifiable
offscreen speaker. A name in someone else's dialogue does not establish presence or
speech. Cite the speaker's actual words and attribution.

For objects and places, apply two gates before emitting a row:
1. Identify the physical subject first. A place is the distinct space where the
   action is filmed. Record walls, openings and surfaces as features of that place.
   A bare geographic mention gives no filming space. A prop is a physical thing,
   such as equipment or an independently used fixture. Ignore effects, including a
   glow, beam, flash, darkness, shadow, sound or a visual comparison. An effect does
   not establish the existence or appearance of a particular prop.
2. The passage supplies a concrete appearance detail, or identifies a known reference
   of the same physical entity. Otherwise omit the row. A use, attachment, movement,
   possession or name is insufficient. Ignore darkness and all light effects in
   visual_traits. Keep a described structural feature even if lighting is also
   mentioned; retain only the physical detail. A new place requires a visible feature
   of its physical setting. Atmospheric conditions alone do not define a place.
   A new instance with the same noun
   stays separate. Use material for the inherent substance and markings for permanent
   markings or inscriptions; these are intrinsic even when revealed during the scene.

Set reference_id to the matching known ID, or "new" if there is no plausible same-kind
match. Use "unresolved" only if there is a plausible known match that the passage cannot
confirm. A generic noun alone does not confirm identity. A known prop's distinctive
role and the current prop's role must agree. Do not substitute a related part, fixture
or light source for a prop. Known references may be reused with empty visual_traits.
Always cite current passages for presence and each new trait.
For a reused object, copy its current name from the cited local sentence, even if it
differs from the dictionary name. Use the known ID to preserve the identity.
To introduce a different name for a known object, cite a local identifying appearance
detail. If all you have is a generic label or the object's use, omit the uncertain link.
Check the opening passages as well as the action. For every matching known name,
check whether that entity is actually shown and record any physical detail stated here.
{APPEARANCE_RULES}
Allowed character views: {', '.join(CHARACTER_VIEWS)}.
Allowed location views: {', '.join(LOCATION_VIEWS)}.
Allowed object views: {', '.join(OBJECT_VIEWS)}.
Choose views exposing defining surfaces and details clearly. Use front_view when a
straight-on view best reveals an object's defining surface; additional views must
expose useful geometry or detail. Do not copy narrative camera angles.
Importance: major, recurring, minor, background. Reference priority: required when
continuity depends on it, recommended when useful, optional otherwise.
Names, chunk_summary and trait descriptions use the local narration's language.
Copy the words describing each trait directly from one cited sentence.
For a physical detail stated in separate phrases, use separate visual traits.
Keep JSON keys, enums and visual-trait attribute keys fixed. Preserve proper names
and verbatim foreign-language dialogue. Select evidence_passage_ids from the local source_passages for each observation and each trait.
Summarize events only in chunk_summary, in source order and complete sentences.
Scale its length to the current local passage.
The output token budget is shared by the entire JSON response, including reasoning
when applicable. Keep the summary concise so reference arrays and their evidence selections
can fit; prioritize distinct entities and supported visual facts over summary detail.
There is no separate fixed character target. Shorten by rephrasing, never
by cutting a word or sentence. Detailed reference facts belong in the entity arrays.
Keep all distinct useful
visual facts without repeated paraphrases. Unknown traits remain unspecified.
""".strip()

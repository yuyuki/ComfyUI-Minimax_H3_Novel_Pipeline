"""Extract phase-scoped observations without exposing future narrative text."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from shutil import copyfile
from pathlib import Path

from . import lmstudio_pipeline, progress, util
from .editable_schemas import validate_document
from .path_access import confined_path
from .prompt_cache import fingerprint

PHASES = ("initialState", "event", "endingState")
TEMPORAL_VERSION = "cinematic-reference-timeline.v1"
CATALOG_CACHE_VERSION = "cinematic-reference-catalog.v2"
CATALOG_SCHEMA_FILENAME = "chapter_references.schema.json"
CATALOG_SCHEMA = json.loads(
    (Path(__file__).resolve().parent / "schema" / CATALOG_SCHEMA_FILENAME).read_text(encoding="utf-8")
)
PHASE_SYSTEM = """
The passage is ONE phase of one cinematic sequence, the authoritative narrative.
initialState contains only facts already true at START. event contains actions
DURING the sequence, not opening facts. endingState is true only AFTER completion.
Extract only entities and observations supported by this phase. Never infer an
action's success (throwing is not catching), future possession, injuries or changes.
Stable descriptions contain intrinsic identity traits only: no posture, position,
equipment, possession, injuries, relationships or environmental conditions.
Put changing facts in chapter_appearance/chapter_state, retaining action order.
Do not infer omitted facts, carry states forward or resolve uncertain identities.
known_references contains only earlier observations, for identity resolution, not
facts to copy into this phase. Resolve contextual mentions to an existing person,
place or object whenever evidence supports it. Reuse its canonical_name and retain
genuine alternative names in aliases. "L'homme avec la torche" and "l'homme" may
refer to the established person; "la crevasse à Delphes", "la crevasse sombre"
and "la crevasse" may name one location; "la corde" and "le filin" one object.
These are contextual decisions, never universal synonym rules. A different rope,
another man or a newly arriving torch must remain distinct.
Descriptions of parts/aspects ("la paroi rocheuse", "l'abîme" of the crevasse),
actions, sensations and sounds belong in the supported entity's phase observation,
not in separate asset identities or aliases. Attach a cry to its established person
only when supported. Otherwise retain it in the narrative without inventing a person.
Actual unnamed people and explicitly established offscreen speakers remain entities.
Never turn a temporary description or manifestation into a permanent identity trait.
"""

LOCAL_LINK_SYSTEM = """
You are reviewing a single chapter's extracted reference catalog.

For every entity, set link_to_local_id to the local_id of another catalog entry only
when both entries clearly represent the exact same fictional character, physical
location, or object in the story. This includes an incomplete descriptive reference
such as "someone screaming" when the chapter establishes that it is Doriane. Leave
link_to_local_id null when the identity is uncertain, when the entry is a genuinely
different entity, or when there is no better main entry.

Links must be within the same entity type. Select one canonical/main entry as the
target and link secondary descriptive or duplicate entries to it. A main entry must
have null, never link an entry to itself, and never create a chain or cycle. Do not
invent evidence or resolve ambiguity by guesswork.
""".strip()

LOCAL_LINK_SCHEMA = {
    "name": "chapter_reference_local_links_v1",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "local_id": {"type": "string", "minLength": 1},
                        "link_to_local_id": {"type": ["string", "null"]},
                    },
                    "required": ["local_id", "link_to_local_id"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["links"],
        "additionalProperties": False,
    },
}


def parse_chapters(value):
    """Validate in-memory adapter output or its JSON serialization; never read paths."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("cinematic_chapters must contain valid JSON.") from exc
    if not isinstance(value, list) or not value:
        raise ValueError("cinematic_chapters must be a non-empty chapter list from Cinematic Chapter Adapter.")
    for index, chapter in enumerate(value):
        label = f"cinematic_chapters[{index}]"
        if not isinstance(chapter, dict) or not isinstance(chapter.get("chapter_name"), str) or not chapter["chapter_name"].strip():
            raise ValueError(f"{label} requires a non-empty chapter_name.")
        for key in ("source_file", "saved_file"):
            if key in chapter and not isinstance(chapter[key], str):
                raise ValueError(f"{label}.{key} must be a string.")
        sequences = chapter.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise ValueError(f"{label}.sequences must be a non-empty ordered list.")
        previous = 0
        for seq in sequences:
            if not isinstance(seq, dict) or type(seq.get("sequence")) is not int or seq["sequence"] <= previous:
                raise ValueError(f"{label}: sequence IDs must be positive, unique and increasing in input order.")
            previous = seq["sequence"]
            if not isinstance(seq.get("source"), str):
                raise ValueError(f"{label} sequence {previous}: source must be a string.")
            adaptation = seq.get("adaptation")
            if not isinstance(adaptation, dict) or any(not isinstance(adaptation.get(p), str) for p in PHASES):
                raise ValueError(f"{label} sequence {previous}: adaptation requires initialState, event and endingState strings.")
            if not any(adaptation[p].strip() for p in PHASES):
                raise ValueError(f"{label} sequence {previous}: adaptation must not be empty.")
    return deepcopy(value)


def chapter_digest(chapter):
    return hashlib.sha256(json.dumps(chapter, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _catalog_entities(catalog):
    return [entity for kind in ("characters", "locations", "objects") for entity in catalog[kind]]


def assign_local_links(catalog, client, model, args):
    """Ask the model to connect only high-confidence duplicate manifestations."""
    entities = _catalog_entities(catalog)
    for entity in entities:
        entity["link_to_local_id"] = None
    if len(entities) < 2:
        return

    review = {
        kind: [
            {
                "local_id": entity["local_id"],
                "canonical_name": entity["canonical_name"],
                "aliases": entity["aliases"],
                "stable_visual_description": entity["stable_visual_description"],
                "evidence": entity["evidence"],
                "state_by_sequence": entity["state_by_sequence"],
            }
            for entity in catalog[kind]
        ]
        for kind in ("characters", "locations", "objects")
    }
    result = lmstudio_pipeline.load("extract").chat_json(
        client, model, LOCAL_LINK_SYSTEM, json.dumps(review, ensure_ascii=False),
        LOCAL_LINK_SCHEMA, min(args.temperature, 0.2), args.max_tokens,
    )
    by_id = {entity["local_id"]: entity for entity in entities}
    proposed = result.get("links", [])
    seen_ids = set()
    for item in proposed:
        source_id = item.get("local_id")
        if source_id not in by_id or source_id in seen_ids:
            raise ValueError("Local-link review returned an invalid or duplicate local_id.")
        seen_ids.add(source_id)
        source = by_id[source_id]
        target_id = item["link_to_local_id"]
        if target_id is None:
            continue
        target = by_id.get(target_id)
        if target is None or target is source:
            raise ValueError("Local-link review returned an invalid target local_id.")
        source_kind = next(kind for kind in ("characters", "locations", "objects") if source in catalog[kind])
        target_kind = next(kind for kind in ("characters", "locations", "objects") if target in catalog[kind])
        if source_kind != target_kind:
            raise ValueError("Local-link review may only link entities of the same type.")
        source["link_to_local_id"] = target_id
    if any(entity["link_to_local_id"] and by_id[entity["link_to_local_id"]]["link_to_local_id"]
           for entity in entities):
        raise ValueError("Local-link review must point directly to a main local_id, without chains.")


def copy_catalog_schema(output):
    source = confined_path(Path(__file__).resolve().parent / "schema" / CATALOG_SCHEMA_FILENAME,
                           Path(__file__).resolve().parent)
    copyfile(source, confined_path(output / CATALOG_SCHEMA_FILENAME, output))


def process_chapter(chapter, output, client, model, args, chapter_id):
    """Persist identity catalogs plus lossless ordered, phase-specific observations."""
    step = lmstudio_pipeline.load("extract")
    system = step.EXTRACT_SYSTEM + "\n" + PHASE_SYSTEM
    key = fingerprint(model, args, TEMPORAL_VERSION, CATALOG_CACHE_VERSION,
                      system, LOCAL_LINK_SYSTEM, step.CHUNK_SCHEMA, LOCAL_LINK_SCHEMA,
                      chapter, client=client)
    saved = confined_path(output / f"{chapter_id}_references.json", output)
    if saved.exists() and not args.force:
        try:
            cached = util.load_json(saved)
            if (cached.get("cache_key") == key and cached.get("timeline_version") == TEMPORAL_VERSION
                    and cached.get("schema_version") == util.CHAPTER_SCHEMA):
                copy_catalog_schema(output)
                return saved
        except (ValueError, OSError):
            pass
    cache_dir = confined_path(output / ".cache" / chapter_id, output)
    cache_dir.mkdir(parents=True, exist_ok=True)
    catalog = {kind: [] for kind in ("characters", "locations", "objects")}
    jobs = [(seq, phase, part) for seq in chapter["sequences"] for phase in PHASES
            for part in util.split_chunks(seq["adaptation"][phase], args.chunk_chars, 0)
            if part.strip()]
    for seq, phase, text in progress.steps(jobs):
        lmstudio_pipeline.comfy_interrupt_check()
        # No original source, other phases, or later sequences enter this request.
        known = {kind: [{"canonical_name": e["canonical_name"], "aliases": e["aliases"],
                         "last_observations": list(list(e["state_by_sequence"].values())[-1].values())[-1]}
                        for e in entities] for kind, entities in catalog.items()}
        user = json.dumps({"sequence": seq["sequence"], "phase": phase, "text": text,
                           "known_references": known}, ensure_ascii=False)
        part_key = fingerprint(model, args, TEMPORAL_VERSION, system, step.CHUNK_SCHEMA, user, client=client)
        cache = confined_path(cache_dir / f"{part_key}.json", output)
        if cache.exists() and not args.force:
            result = util.load_json(cache)
        else:
            result = step.chat_json(client, model, system, user, step.CHUNK_SCHEMA, args.temperature, args.max_tokens)
            util.save_json(cache, result)
        for kind, prefix in (("characters", "CHAR"), ("locations", "LOC"), ("objects", "OBJ")):
            for raw in result.get(kind, []):
                observation = step.clean_entity(raw, kind)
                names = {n.casefold() for n in [observation["canonical_name"], *observation["aliases"]] if n}
                if not observation["canonical_name"]:
                    raise ValueError("Cinematic reference extraction returned an empty canonical_name.")
                matches = [e for e in catalog[kind] if names & {
                    n.casefold() for n in [e["canonical_name"], *e["aliases"]]}]
                # Ambiguous names never justify merging distinct identities.
                entity = matches[0] if len(matches) == 1 else None
                if entity is None:
                    entity = {**deepcopy(observation), "local_id": f"{prefix}_{len(catalog[kind]) + 1:03d}",
                              "first_sequence": seq["sequence"], "first_phase": phase, "state_by_sequence": {}}
                    entity["chapter_appearance" if kind == "characters" else "chapter_state"] = ""
                    catalog[kind].append(entity)
                entity["aliases"] = list(dict.fromkeys([*entity["aliases"], *observation["aliases"],
                                                       *([observation["canonical_name"]]
                                                         if observation["canonical_name"] != entity["canonical_name"] else [])]))
                phases = entity["state_by_sequence"].setdefault(str(seq["sequence"]), {})
                phases.setdefault(phase, []).append(observation)
    payload = {"schema_version": util.CHAPTER_SCHEMA, "timeline_version": TEMPORAL_VERSION,
               "cache_key": key, "chapter_id": chapter_id, "chapter_name": chapter["chapter_name"],
               "source": {"file": chapter.get("source_file", ""), "sha256": chapter_digest(chapter)},
               "sequences": deepcopy(chapter["sequences"]),
               **catalog}
    assign_local_links(payload, client, model, args)
    validate_document(payload, CATALOG_SCHEMA, saved.name)
    copy_catalog_schema(output)
    util.save_json(saved, payload)
    return saved

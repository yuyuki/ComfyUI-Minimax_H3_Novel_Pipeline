"""Extract phase-scoped observations without exposing future narrative text."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from . import lmstudio_pipeline, progress, util
from .path_access import confined_path
from .prompt_cache import fingerprint

PHASES = ("initialState", "event", "endingState")
TEMPORAL_VERSION = "cinematic-reference-timeline.v1"
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


def process_chapter(chapter, output, client, model, args, chapter_id):
    """Persist identity catalogs plus lossless ordered, phase-specific observations."""
    step = lmstudio_pipeline.load("extract")
    system = step.EXTRACT_SYSTEM + "\n" + PHASE_SYSTEM
    key = fingerprint(model, args, TEMPORAL_VERSION, system, step.CHUNK_SCHEMA, chapter, client=client)
    saved = confined_path(output / f"{chapter_id}_references.json", output)
    if saved.exists() and not args.force:
        try:
            cached = util.load_json(saved)
            if (cached.get("cache_key") == key and cached.get("timeline_version") == TEMPORAL_VERSION
                    and cached.get("schema_version") == util.CHAPTER_SCHEMA):
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
               "sequences": deepcopy(chapter["sequences"]), "chapter_summary": "",
               **catalog}
    util.save_json(saved, payload)
    return saved

"""Extract sequence-source observations and consolidate chapter-local identities."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from shutil import copyfile
from pathlib import Path

from openai import APIConnectionError, InternalServerError, RateLimitError

from . import chapter_reference_identity, lmstudio_json, lmstudio_pipeline, narrative_choices, narrative_validation, progress, temporal_context, util
from .editable_schemas import validate_document
from .cinematic_diagnostics import downstream_chapter
from .path_access import confined_path
from .prompt_cache import fingerprint

PHASES = ("initialState", "event", "endingState")
TEMPORAL_VERSION = "cinematic-reference-source.v1"
CATALOG_CACHE_VERSION = "cinematic-reference-source-catalog.v6"
CATALOG_SCHEMA_FILENAME = "chapter_references.schema.json"
CATALOG_SCHEMA = json.loads(
    (Path(__file__).resolve().parent / "schema" / CATALOG_SCHEMA_FILENAME).read_text(encoding="utf-8")
)
SOURCE_SYSTEM = """
The passage is the source text of ONE cinematic sequence, the only extraction evidence.
Extract only entities and observations supported by this source. Never infer an
action's success (throwing is not catching), future possession, injuries or changes.
Stable descriptions contain intrinsic identity traits only: no posture, position,
equipment, possession, injuries, relationships or environmental conditions.
Put changing facts in chapter_appearance/chapter_state, retaining action order.
Do not infer omitted facts, carry states forward or resolve uncertain identities.
Identity reconciliation between alternative names and aliases belongs to consolidation.
Do not match entries against earlier sequences or rewrite names to force a match.
Retain source-supported aliases without using them to merge catalog entries.
Descriptions of parts/aspects ("la paroi rocheuse", "l'abîme" of the crevasse),
actions, sensations and sounds belong in the supported entity's source observation,
not in separate asset identities or aliases. Attach a cry to its established person
only when supported. Otherwise retain it in the narrative without inventing a person.
Actual unnamed people and explicitly established offscreen speakers remain entities.
Never turn a temporary description or manifestation into a permanent identity trait.
Use distinct canonical names for distinct physical objects, qualified by supported
function or attachment (for example a climber's rope versus a basket's rope).
Shared generic nouns are not identity aliases between distinct objects.
Capture established personal aliases from dialogue. speaks concerns speech in this source.
Locations are actual places or spaces where action occurs, never an entity's position,
orientation, posture, attachment or possession. Keep these spatial facts on the relevant
object/person in chapter_state/chapter_appearance. An inclined object is not a location
because it is inclined. Classify a natural ledge by its source-supported role as a place
or object, consistently; never duplicate it across categories for a spatial property.
Write chunk_summary and ALL generated descriptive prose exclusively in the language of
the supplied source narration. Never translate French source into English. Preserve
proper names and verbatim foreign-language quotations. JSON keys and enum values stay fixed.
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
        temporal_context.normalize(chapter.get("default_temporal_context", ""))
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
    return [narrative_choices.resolve_chapter(chapter) for chapter in value]


def chapter_digest(chapter):
    return hashlib.sha256(json.dumps(chapter, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def copy_catalog_schema(output):
    source = confined_path(Path(__file__).resolve().parent / "schema" / CATALOG_SCHEMA_FILENAME,
                           Path(__file__).resolve().parent)
    copyfile(source, confined_path(output / CATALOG_SCHEMA_FILENAME, output))


# Retain valid extraction data after advisory model failures only.
_REVIEW_ERRORS = (ValueError, RuntimeError, APIConnectionError, InternalServerError, RateLimitError)


def _extract_source(step, client, model, system, user, args):
    result = step.chat_json(client, model, system, user, step.CHUNK_SCHEMA,
                            args.temperature, args.max_tokens)
    history = []
    for attempt in range(3):
        try:
            issues = narrative_validation.review(
                step.chat_json, client, model,
                {"stage": "extraction", "source_input": json.loads(user), "extraction": result},
                1, args.max_tokens,
            )
        except _REVIEW_ERRORS as exc:
            if lmstudio_json._is_comfy_interrupt(exc):
                raise
            lmstudio_pipeline.comfy_interrupt_check()
            history.append({"attempt": attempt + 1, "status": "incomplete",
                            "reason": type(exc).__name__,
                            "message": "Source review failed; valid extraction retained."})
            return result, history, "incomplete"
        messages = [message for values in issues.values() for message in values]
        incomplete = any("could not be interpreted reliably" in m for m in messages)
        status = "incomplete" if incomplete else "issues" if issues else "passed"
        history.append({"attempt": attempt + 1, "status": status, "messages": messages})
        if not issues or incomplete or attempt == 2:
            return result, history, status
        lmstudio_pipeline.comfy_interrupt_check()
        correction = json.dumps({"source_input": json.loads(user), "rejected_extraction": result,
                                 "defects_to_correct": messages}, ensure_ascii=False)
        try:
            result = step.chat_json(client, model, system, correction, step.CHUNK_SCHEMA,
                                    args.temperature, args.max_tokens)
        except _REVIEW_ERRORS as exc:
            if lmstudio_json._is_comfy_interrupt(exc):
                raise
            lmstudio_pipeline.comfy_interrupt_check()
            history.append({"attempt": attempt + 1, "status": "incomplete",
                            "reason": type(exc).__name__,
                            "message": "Source correction failed; previous valid extraction retained."})
            return result, history, "incomplete"


def _catalog_digest(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def process_chapter(chapter, output, client, model, args, chapter_id):
    """Persist source observations by sequence, with separate review diagnostics."""
    chapter = parse_chapters([chapter])[0]
    step = lmstudio_pipeline.load("extract")
    period = temporal_context.normalize(getattr(args, "default_temporal_context", ""))
    system = step.EXTRACT_SYSTEM + "\n" + SOURCE_SYSTEM + "\n" + temporal_context.RULES
    key = fingerprint(model, args, TEMPORAL_VERSION, CATALOG_CACHE_VERSION,
                      system, narrative_validation.SYSTEM, narrative_validation.SCHEMA,
                      step.CHUNK_SCHEMA, chapter_reference_identity.VERSION,
                      chapter_reference_identity.SYSTEM, chapter_reference_identity.SCHEMA, chapter, client=client)
    saved = confined_path(output / f"{chapter_id}_references.json", output)
    diagnostic_path = confined_path(output / f"{chapter_id}_references.diagnostics.json", output)
    if saved.exists() and diagnostic_path.exists() and not args.force:
        cached = util.load_json(saved)
        diagnostics = util.load_json(diagnostic_path)
        if (diagnostics.get("cache_key") == key and diagnostics.get("status") in {"passed", "issues"}
                and diagnostics.get("catalog_digest") == _catalog_digest(cached)):
            validate_document(cached, CATALOG_SCHEMA, saved.name)
            copy_catalog_schema(output)
            return saved
    cache_dir = confined_path(output / ".cache" / chapter_id, output)
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    diagnostics = {"chapter_id": chapter_id, "cache_key": key,
                   "input_narrative_review": deepcopy(chapter.get("narrative_review", {})),
                   "sequence_reviews": [{"sequence": s["sequence"], "review": deepcopy(s["review"])}
                                        for s in chapter["sequences"] if "review" in s],
                   "checks": [], "status": "in_progress"}
    util.save_json(diagnostic_path, diagnostics)
    jobs = [(seq, number, part) for seq in chapter["sequences"]
            for number, part in enumerate(util.split_chunks(seq["source"], args.chunk_chars, 0), 1)
            if part.strip()]
    if not jobs:
        raise ValueError("Chapter reference extraction requires non-empty sequence source text.")
    for seq, number, text in progress.steps(jobs, 0, 0.75):
        lmstudio_pipeline.comfy_interrupt_check()
        user = json.dumps({"sequence": seq["sequence"], "text": text,
                           "default_temporal_context": period}, ensure_ascii=False)
        part_key = fingerprint(model, args, CATALOG_CACHE_VERSION, system, step.CHUNK_SCHEMA,
                               narrative_validation.SYSTEM, narrative_validation.SCHEMA, user, client=client)
        cache = confined_path(cache_dir / f"{part_key}.json", output)
        if cache.exists() and not args.force:
            record = util.load_json(cache)
            result, history, status = record["extraction"], record["history"], record["status"]
        else:
            result, history, status = _extract_source(step, client, model, system, user, args)
            if status != "incomplete":
                util.save_json(cache, {"extraction": result, "history": history, "status": status})
        diagnostics["checks"].append({"sequence": seq["sequence"], "chunk": number,
                                      "stage": "source_extraction_review", "status": status,
                                      "source_excerpt": text, "history": history})
        util.save_json(diagnostic_path, diagnostics)
        if status != "passed":
            print(f"Sequence {seq['sequence']} source: review {status}; extraction retained.", flush=True)
        extracted.append({"sequence": seq["sequence"], "chunk": number, "text": text, "result": result})
    with progress.scope(0.75, 0.98):
        catalog = chapter_reference_identity.consolidate(
            extracted, step, client, model, args, cache_dir, diagnostics, diagnostic_path)
    functional = downstream_chapter(chapter)
    payload = {"schema_version": util.CHAPTER_SCHEMA, "timeline_version": TEMPORAL_VERSION,
               "default_temporal_context": period,
               "chapter_id": chapter_id, "chapter_name": chapter["chapter_name"],
               "source": {"file": chapter.get("source_file", ""), "sha256": chapter_digest(chapter)},
               "narrative_review": functional.get("narrative_review", {"issues": []}),
               "sequences": functional["sequences"], **catalog}
    validate_document(payload, CATALOG_SCHEMA, saved.name)
    copy_catalog_schema(output)
    util.save_json(saved, payload)
    diagnostics["status"] = ("incomplete" if any(c["status"] == "incomplete" for c in
                                              diagnostics["checks"] + diagnostics.get("identity_consolidation", []))
                             else "issues" if any(c["status"] == "issues" for c in diagnostics["checks"]) else "passed")
    diagnostics["catalog_digest"] = _catalog_digest(payload)
    util.save_json(diagnostic_path, diagnostics)
    return saved

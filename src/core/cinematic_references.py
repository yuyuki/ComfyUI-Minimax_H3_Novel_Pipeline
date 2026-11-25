"""Extract sequence-source observations and consolidate chapter-local identities."""
from __future__ import annotations

from copy import deepcopy, copy
import hashlib
import json
import re
from shutil import copyfile
from pathlib import Path

from openai import APIConnectionError, InternalServerError, RateLimitError

from . import chapter_reference_identity, continuity_context, lmstudio_json, lmstudio_pipeline, narrative_choices, progress, temporal_context, util
from .editable_schemas import validate_document
from .cinematic_diagnostics import downstream_chapter
from .path_access import confined_path
from .prompt_cache import fingerprint
from . import reference_visuals
from .reference_requests import InvalidResponse, validated_request
from .source_text import comparison_text, normalize_data
from .util import REFERENCE_TIMELINE_VERSION as TEMPORAL_VERSION

PHASES = ("initialState", "event", "endingState")
from .source_text import PASSAGE_VERSION, paragraph_chunks

CATALOG_CACHE_VERSION = ("cinematic-neutral-catalog.v4", PASSAGE_VERSION)
CATALOG_SCHEMA_FILENAME = "chapter_references.schema.json"
CATALOG_SCHEMA = json.loads(
    (Path(__file__).resolve().parent / "schema" / CATALOG_SCHEMA_FILENAME).read_text(encoding="utf-8")
)
SOURCE_SYSTEM = """
Analyze the numbered source_passages. opening_passage_ids mark what is already
established before the action. Distinguish existing appearance from changes caused
by the current action. Limit visual traits to appearance supported by the passage.
Temporary lighting, poses, sensations and the use or ownership of a prop do not
define its appearance. Actual visual features can be revealed during an action.
The dictionary contains earlier names and appearance solely for matching identity.
Return a known ID only when the current sentences show that same entity or attribute
speech to that same person. Never copy dictionary traits into the result.
"""
_REFERENCE_KINDS = ("characters", "locations", "objects")
_ATMOSPHERE_ATTRIBUTES = {"condition", "location_condition", "environment_condition", "atmosphere",
                          "lighting", "visibility", "weather", "fog"}
_MODEL_ERRORS = (InvalidResponse, lmstudio_json.StructuredOutputError,
                 APIConnectionError, InternalServerError, RateLimitError)


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
    normalized = []
    for chapter in value:
        chapter = {**chapter, "sequences": normalize_data(chapter["sequences"])}
        if "narrative_review" in chapter:
            chapter["narrative_review"] = normalize_data(chapter["narrative_review"])
        normalized.append(narrative_choices.resolve_chapter(chapter))
    return normalized


def chapter_digest(chapter):
    return hashlib.sha256(json.dumps(chapter, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def copy_catalog_schema(output):
    source = confined_path(Path(__file__).resolve().parent / "schema" / CATALOG_SCHEMA_FILENAME,
                           Path(__file__).resolve().parent)
    copyfile(source, confined_path(output / CATALOG_SCHEMA_FILENAME, output))


def _candidate_references(entities, text):
    """Prefer names mentioned now, then the most recently attested identities."""
    text = comparison_text(text).casefold()
    candidates = []
    for entity in entities.values():
        names = [entity["canonical_name"], *entity["aliases"]]
        named = any(re.search(r"(?<!\w)" + re.escape(name.casefold()) + r"(?!\w)", text) for name in names)
        candidates.append((named, entity["members"][-1]["order"], entity))
    candidates.sort(key=lambda item: item[:2], reverse=True)
    shortlisted = []
    for _, _, entity in candidates[:24]:
        traits = dict.fromkeys(trait["description"] for member in entity["members"]
                              for trait in member["observation"]["visual_traits"] if trait["scope"] == "intrinsic")
        shortlisted.append({"reference_id": entity["id"], "kind": entity["kind"], "name": entity["canonical_name"],
                            "aliases": entity["aliases"][:6], "defining_traits": list(traits)[:4]})
    return shortlisted


def _quoted_detail(detail, passages):
    pattern = r"(?<!\w)" + re.escape(detail) + r"(?!\w)"
    return bool(detail) and any(re.search(pattern, comparison_text(p["text"]).casefold()) for p in passages)


def _eligible_observations(result, offered, rejected):
    offered_kinds = {entry["kind"] for entry in offered.values()}
    for kind in _REFERENCE_KINDS:
        seen = set()
        retained = []
        for index, row in enumerate(result[kind]):
            name = comparison_text(row["canonical_name"]).casefold()
            traits = []
            for trait_index, trait in enumerate(row["visual_traits"]):
                detail = comparison_text(trait["description"]).casefold().strip(" .,!?:;\"'")
                path = f"{kind}[{index}].visual_traits[{trait_index}]"
                if kind != "characters" and trait["attribute"].startswith("wardrobe_"):
                    rejected.append(path + ".attribute")
                elif detail == name:
                    rejected.append(path + ".description")
                elif _quoted_detail(detail, trait["evidence_passages"]):
                    traits.append(trait)
                else:
                    rejected.append(path + ".description")
            row["visual_traits"] = traits
            ref = row["reference_id"]
            known = ref not in ("new", "unresolved")
            if ref == "unresolved" and kind not in offered_kinds:
                row["reference_id"] = "new"
            entry = offered.get(ref)
            if known and (entry is None or entry["kind"] != kind or ref in seen):
                rejected.append(f"{kind}[{index}].reference_id")
                continue
            if kind == "objects" and entry is not None:
                names = {comparison_text(item).casefold() for item in [entry["name"], *entry["aliases"]]}
                if not _quoted_detail(name, row["evidence_passages"]) or (name not in names and not traits):
                    rejected.append(f"{kind}[{index}].canonical_name")
                    continue
            if kind == "characters":
                if row["presence"] != "visible" and not row["speaks"]:
                    continue
            else:
                if row["presence"] != "visible" or (not known and not traits):
                    continue
                if kind == "locations" and not known and all(
                    trait["scope"] == "scene"
                    and reference_visuals.attribute_key(trait["attribute"]) in _ATMOSPHERE_ATTRIBUTES
                    for trait in traits
                ):
                    continue
            retained.append(row)
            if known:
                seen.add(ref)
        result[kind] = retained
    return result


def _extract_source(step, client, model, system, source_input, args):
    index = reference_visuals.evidence_index(source_input)
    passages = index.passages()
    payload = {
        "sequence": source_input["sequence"],
        "source_passages": passages,
        "opening_passage_ids": [p["id"] for p in passages
                                if source_input["offset"] + index.ranges[p["id"]][0] < source_input["opening_end"]],
        "default_temporal_context": source_input["default_temporal_context"],
        "known_references": list(source_input["known_references"]),
    }
    while payload["known_references"] and not continuity_context.fits(system, payload, step.CHUNK_SCHEMA):
        payload["known_references"].pop()
    offered = {row["reference_id"]: row for row in payload["known_references"]}

    def resolve(data, rejected):
        validate_document(data, step.CHUNK_SCHEMA["schema"], step.CHUNK_SCHEMA["name"])
        result = reference_visuals.resolve_evidence(data, source_input, rejected=rejected)
        return _eligible_observations(result, offered, rejected)

    def validate(data, *, cached=False):
        rejected = []
        resolve(data, rejected)
        if rejected:
            raise ValueError("Use offered local passage IDs and each known ID at most once. Quote a locally "
                             "stated known object name. An unfamiliar name needs a local identifying "
                             "appearance detail to reuse the ID. Traits must add detail beyond a name. "
                             "Reserve wardrobe attributes for characters. Check: "
                             + ", ".join(rejected))

    try:
        raw = validated_request(step.chat_json, client, model, system, payload, step.CHUNK_SCHEMA,
                                args, validate, validation_version=CATALOG_CACHE_VERSION, bounded=True)
    except InvalidResponse as exc:
        if lmstudio_json._is_comfy_interrupt(exc) or exc.rejected is None:
            raise
        rejected = []
        try:
            result = resolve(exc.rejected, rejected)
        except ValueError:
            raise exc
        if not any(result[kind] for kind in _REFERENCE_KINDS):
            raise
        return result, rejected
    return resolve(raw, []), []


def _catalog_digest(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def process_chapter(chapter, output, client, model, args, chapter_id):
    """Extract initialState/event observations and consolidate them by sequence."""
    chapter = parse_chapters([chapter])[0]
    step = lmstudio_pipeline.load("extract")
    period = temporal_context.normalize(getattr(args, "default_temporal_context", ""))
    system = step.EXTRACT_SYSTEM + "\n" + SOURCE_SYSTEM + "\n" + temporal_context.RULES
    key = fingerprint(model, args, TEMPORAL_VERSION, CATALOG_CACHE_VERSION,
                      system, step.CHUNK_SCHEMA, chapter_reference_identity.VERSION,
                      chapter_reference_identity.SYSTEM, chapter_reference_identity.SCHEMA,
                      reference_visuals.VERSION, reference_visuals.EQUIVALENCE_SYSTEM,
                      reference_visuals.EQUIVALENCE_SCHEMA, chapter, client=client)
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
    args = copy(args)
    if getattr(args, "resume_cache_dir", None) is not None:
        args.resume_cache_dir = confined_path(chapter_id, args.resume_cache_dir)
    extracted = []
    entities, order = {}, 0
    args.cache_dir = cache_dir
    diagnostics = {"chapter_id": chapter_id, "cache_key": key,
                   "input_narrative_review": deepcopy(chapter.get("narrative_review", {})),
                   "sequence_reviews": [{"sequence": s["sequence"], "review": deepcopy(s["review"])}
                                        for s in chapter["sequences"] if "review" in s],
                   "checks": [], "status": "in_progress"}
    util.save_json(diagnostic_path, diagnostics)
    jobs = [(seq, number, offset, part) for seq in chapter["sequences"]
            for number, (offset, part) in enumerate(paragraph_chunks(
                "\n\n".join(seq["adaptation"][p] for p in ("initialState", "event")
                            if seq["adaptation"][p].strip()), args.chunk_chars), 1)
            if part.strip()]
    if not jobs:
        raise ValueError("Chapter reference extraction requires non-empty initialState or event text.")
    for seq, number, offset, text in progress.steps(jobs, 0, 0.75):
        lmstudio_pipeline.comfy_interrupt_check()
        opening = seq["adaptation"]["initialState"]
        source_input = {"sequence": seq["sequence"], "chunk": number, "offset": offset, "text": text,
                        "opening_end": len(opening) if opening.strip() else 0,
                        "default_temporal_context": period,
                        "known_references": _candidate_references(entities, text)}
        check = {"sequence": seq["sequence"], "chunk": number, "stage": "source_evidence_validation",
                 "source_excerpt": text, "status": "passed"}
        try:
            result, rejected = _extract_source(step, client, model, system, source_input, args)
        except _MODEL_ERRORS as exc:
            if lmstudio_json._is_comfy_interrupt(exc):
                raise
            check.update(status="incomplete", reason=type(exc).__name__,
                         message="No valid extraction response was available for this passage.")
        else:
            if rejected:
                check.update(status="incomplete", rejected_fields=rejected,
                             message="Invalid observations or traits were excluded after correction.")
            job = {"sequence": seq["sequence"], "chunk": number, "offset": offset, "text": text, "result": result}
            extracted.append(job)
            order = chapter_reference_identity.append_occurrences(entities, job, step, order)
        diagnostics["checks"].append(check)
        util.save_json(diagnostic_path, diagnostics)
    if not extracted:
        diagnostics["status"] = "incomplete"
        util.save_json(diagnostic_path, diagnostics)
        raise InvalidResponse("No sequence retained valid reference extraction. Inspect the chapter diagnostic file.")
    with progress.scope(0.75, 0.98):
        catalog = chapter_reference_identity.finish_dictionary(
            entities, extracted, step, client, model, args, cache_dir, diagnostics)
    source_by_scene = reference_visuals.scene_sources({chapter_id: chapter["sequences"]})
    for kind in ("characters", "locations", "objects"):
        for entity in catalog[kind]:
            visual_args = copy(args)
            visual_args.cache_dir = cache_dir
            reference = reference_visuals.build_reference(
                {chapter_id: entity["state_by_sequence"]}, diagnostics["checks"], entity["local_id"],
                step.chat_json, client, model, visual_args, source_by_scene=source_by_scene)
            entity["visual_reference"] = reference
            entity["stable_visual_description"] = reference_visuals.descriptions(reference["common_traits"])
            entity["distinguishing_features"] = []
            entity.pop("presence", None)
            entity.pop("visual_traits", None)
            if kind != "characters" and not any(
                reference_visuals.appearance_at(entity, chapter_id, sequence)
                for sequence in reference["scenes"][chapter_id]
            ):
                diagnostics["checks"].append({"stage": "neutral_visual_selection", "entity_id": entity["local_id"],
                                              "status": "passed", "image_eligible": False,
                                              "message": "No visible, source-described image reference; observations retained."})
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

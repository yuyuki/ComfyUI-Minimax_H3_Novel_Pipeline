"""Validation of editable profiles, references and direct identity links."""
from copy import deepcopy
import json

from . import character_profiles as cp
from .source_text import contains
from .editable_schemas import SCHEMA_DIR, validate_document as validate_schema

SCHEMA = json.loads((SCHEMA_DIR / "character_profiles.schema.json").read_text(encoding="utf-8"))


def make_document(data, records, profiles):
    retained = {item["_source_index"] for item in cp._usable(records).values()}
    source = deepcopy(data)
    for chapter in source["chapters"]:
        chapter["evidence"] = [row for index, row in enumerate(chapter["evidence"])
                               if (chapter["chapter"], index) in retained]
    return {"$schema": "./character_profiles.schema.json", "format_version": cp.VERSION,
            "source_digest": cp._digest(source), "source_document": source, "profiles": profiles}


def validate_document(data, *, texts=None):
    """Validate edits without rebuilding model decisions or normalizing selections."""
    if not isinstance(data, dict) or type(data.get("format_version")) is not int or data["format_version"] != cp.VERSION:
        raise ValueError("Unsupported character profiles format. Regenerate with Consolidate Character Profiles.")
    validate_schema(data, SCHEMA, "character_profiles.json")
    if data["source_digest"] != cp._digest(data["source_document"]):
        raise ValueError("source_digest: source_document was altered; reconsolidate the edited evidence file.")
    records = cp.prepare_evidence(data["source_document"], texts=texts)
    if len(records) != len(cp._usable(records)):
        raise ValueError("source_document contains invalid evidence. Regenerate with Consolidate Character Profiles.")
    profiles = {p["local_id"]: p for p in data["profiles"]}
    if len(profiles) != len(data["profiles"]):
        raise ValueError("profiles.local_id: duplicate identifier.")
    assigned = set()
    for index, profile in enumerate(data["profiles"]):
        where = f"profiles[{index}]"
        target = profile["link_to_local_id"]
        if target is not None and (target == profile["local_id"] or target not in profiles
                                   or profiles[target]["link_to_local_id"] is not None):
            raise ValueError(f"{where}.link_to_local_id: expected another principal in this file, without chain or cycle.")
        own = {}
        for row in profile["supporting_evidence"]:
            key = row["evidence_id"]
            if key in assigned or key not in records or records[key]["validation_confidence"] == "low":
                raise ValueError(f"{where}.supporting_evidence: duplicate, excluded or unknown evidence_id {key}.")
            if row["observation"] != cp._source_observation(records[key]):
                raise ValueError(f"{where}.supporting_evidence.{key}: observation differs from its validated source.")
            assigned.add(key)
            own[key] = row["observation"]
        def references(ids, field):
            if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(own):
                raise ValueError(f"{where}.{field}: nonempty unique references to this profile's evidence required.")
        if profile["identity_support"]:
            references(profile["identity_support"], "identity_support")
        for bucket, rows in profile["attributes"].items():
            for row in rows:
                references([row["evidence_id"]], f"attributes.{bucket}")
                if any(row[k] != own[row["evidence_id"]][k] for k in
                       ("attribute", "value", "classification", "persistence")):
                    raise ValueError(f"{where}.attributes.{bucket}: source observation mismatch.")
        for field in ("global_traits", "temporary_states"):
            for row in profile[field]:
                references(row["evidence_ids"], field)
                if field == "global_traits" and any(own[key]["classification"] != "explicit"
                        or own[key]["persistence"] != "persistent" or own[key]["attribute"] not in
                        {"age", "physical_appearance", "distinguishing_feature"} for key in row["evidence_ids"]):
                    raise ValueError(f"{where}.global_traits: explicit persistent physical evidence required.")
        conflicts = set()
        for conflict in profile["unresolved_conflicts"]:
            if conflict["conflict_id"] in conflicts:
                raise ValueError(f"{where}.unresolved_conflicts.conflict_id: duplicate identifier.")
            conflicts.add(conflict["conflict_id"])
            references(conflict["evidence_ids"], "unresolved_conflicts.evidence_ids")
            choices = set()
            covered = set()
            for choice in conflict["choices"]:
                if choice["id"] in choices:
                    raise ValueError(f"{where}.unresolved_conflicts.choices.id: duplicate identifier.")
                choices.add(choice["id"])
                references(choice["evidence_ids"], "unresolved_conflicts.choices.evidence_ids")
                covered.update(choice["evidence_ids"])
            if covered != set(conflict["evidence_ids"]):
                raise ValueError(f"{where}.unresolved_conflicts.choices: evidence coverage mismatch.")
        for row in profile["chronological_changes"]:
            validate_change(row, own, where, references)
        for row in profile["trait_relations"]:
            if row["evidence"]:
                references([ref["evidence_id"] for ref in row["evidence"]], "trait_relations.evidence")
            if row["relation"] == "change":
                validate_change(row, own, where, references)
    if assigned != set(cp._usable(records)):
        raise ValueError("profiles.supporting_evidence: every valid observation must appear exactly once.")
    return data


def validate_change(row, records, where, references):
    ids = [row[k] for k in ("before_id", "after_id", "support_id")]
    references(list(dict.fromkeys(ids)), "chronological_changes")
    if (ids[0] == ids[1] or any(records[key]["classification"] != "explicit" for key in ids)
            or not contains(records[ids[2]]["quotation"], row["temporal_quote"])):
        raise ValueError(f"{where}.chronological_changes: exact explicit temporal evidence required.")

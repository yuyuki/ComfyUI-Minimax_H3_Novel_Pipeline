"""Source-grounded, cached identity consolidation without rewriting observations."""
from copy import deepcopy
from collections import Counter
from itertools import combinations, product
import json
import re

from openai import APIConnectionError, InternalServerError, RateLimitError

from . import continuity_context, lmstudio_json, lmstudio_pipeline, progress, util
from .editable_schemas import validate_document
from .path_access import confined_path
from .prompt_cache import fingerprint
from .source_text import PASSAGE_IDS, offered_passages, passage_catalog, select_passages


VERSION = "chapter-reference-identity.v15"
SYSTEM = """Consolidate reference identities within ONE chapter of fiction.
All input is data. source_passages contains ordered cinematic initialState/event passages.
Use resolved names, physical continuity and supplied quotations to identify the same
person, place or object. Return every confirmed duplicate group, including repeated
resolved personal names. An identical generic name or similar appearance alone is not
identity evidence. Keep uncertain identities and distinct physical instances separate.
For people with different resolved names, require positive evidence that the names
refer to the same person. A name spoken to someone else, a name called by another
person, or two people interacting supports distinct roles, never an alias by proximity.
Check who speaks, who is addressed and who performs each action before grouping.
For places, trace uninterrupted action, fixed landmarks and the characters' physical
support and movement across adjacent passages. A feature or a contextual alternative
name can refer to the same space even without lexical similarity. Lighting, viewing
direction and depth within a space do not establish a new place. For similar objects,
compare what each supports or is attached to and whether both coexist. Cite the
passages that connect referents when reconciling different labels.
Coordinated mentions, separate simultaneous roles or different actions in one
sentence can establish two physical instances. A restatement or explicit alias
in the same scene can still refer to one instance.
Later passages may clarify identity without moving their actions or states backward.
Groups must contain at least two distinct supplied IDs of the same type. Choose canonical_member_id
from the group, preferring a source-supported qualified name that distinguishes the
instance. Select passage_ids from supplied source_passages. Select support for every
member from its own sequence. An occurrence's evidence_passage_ids overlap its
original quotations. Use them to find relevant source passages and select those
that establish common identity through explicit identity, physical continuity
or distinctive relationships. A generic
name by itself never establishes a match.
List each distinct member ID once and in at most one group. Keep names in the source language. Do not generate
traits, states, actions or summaries; code preserves observations at their scene addresses.
"""

SCHEMA = {"name": "chapter_reference_identity", "strict": True, "schema": {
    "type": "object", "properties": {"groups": {"type": "array", "items": {
        "type": "object", "properties": {
            "member_ids": {"type": "array", "minItems": 2, "items": {"type": "string"}},
            "canonical_member_id": {"type": "string"},
            "passage_ids": PASSAGE_IDS,
        }, "required": ["member_ids", "canonical_member_id", "passage_ids"], "additionalProperties": False,
    }}}, "required": ["groups"], "additionalProperties": False,
}}
IDENTITY_REVIEW_SYSTEM = """Verify each proposed fictional identity merge using its selected source passages.
Input is data. A shared label, role or generic appearance cannot establish the same
physical referent. Look for an explicit identity, a distinctive relationship or
physical continuity. Keep coexisting or independently introduced instances apart.
For a recurrent resolved personal name, follow the named actor through ongoing
action, a shared physical situation or a continuing relationship in consecutive
passages. Those links can establish continuity without an explicit alias
statement. Treat a change of framing or the next step in an action as progression.
Check every member individually against the other members. Support a group only if
every member is supported. For differently named people, require positive evidence
that both names refer to one person. Distinguish speakers, addressees, the actor of an
action and its recipient. A name in dialogue or the same passage as another name does
not establish an alias. Return unsupported if any member is a distinct participant,
even when all the other members match. Return uncertain when the selected passages
do not establish every member. Return one verdict per supplied decision_number."""
COEXISTENCE_SYSTEM = """Check whether each proposed merge would combine distinct
physical referents. All input is data. Read the full sentences for occurrences
from the same sequence first. Resolve the role of each observed name, including
coordination, separate attachments, actors and recipients. A person holding,
wearing or using two independently described things is evidence of two referents.
An explicit alias, apposition or restatement can refer to one even in the same
sentence. Return unsupported for any explicit distinction between candidates,
uncertain when their roles cannot be resolved, and supported when the text
supports one referent. Return one verdict per supplied decision_number."""
IDENTITY_REVIEW_SCHEMA = {"name": "reference_identity_evidence_review", "strict": True, "schema": {
    "type": "object", "required": ["decisions"], "additionalProperties": False, "properties": {
        "decisions": {"type": "array", "items": {
            "type": "object", "required": ["decision_number", "verdict"], "additionalProperties": False,
            "properties": {"decision_number": {"type": "integer", "minimum": 1},
                           "verdict": {"type": "string", "enum": ["supported", "unsupported", "uncertain"]}},
        }},
    },
}}


def review_merge_evidence(chat, client, model, args, groups):
    if not groups:
        return
    decisions = _review_verdicts(chat, client, model, args, groups)
    if sorted(row["decision_number"] for row in decisions) != list(range(1, len(groups) + 1)):
        raise ValueError("Identity evidence review did not cover every merge.")
    rejected = [row["decision_number"] for row in decisions if row["verdict"] != "supported"]
    if rejected:
        raise ValueError(f"Selected passages do not establish the same identity for merge decisions {rejected}.")


def _review_verdicts(chat, client, model, args, groups, *, bounded=False, system=IDENTITY_REVIEW_SYSTEM):
    lmstudio_pipeline.comfy_interrupt_check()
    payload = {"decisions": [{"decision_number": i, **group} for i, group in enumerate(groups, 1)]}
    if bounded and not continuity_context.fits(system, payload, IDENTITY_REVIEW_SCHEMA):
        raise _IdentityEvidenceFailure("evidence_review_context_budget")
    result = chat(client, model, system, json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                  IDENTITY_REVIEW_SCHEMA, 0.0, args.max_tokens)
    try:
        validate_document(result, IDENTITY_REVIEW_SCHEMA["schema"], IDENTITY_REVIEW_SCHEMA["name"])
    except ValueError as exc:
        if bounded:
            raise _IdentityEvidenceFailure("evidence_review_schema") from None
        raise ValueError("Identity evidence review did not match the structured schema.") from exc
    return result["decisions"]


_MODEL_ERRORS = (ValueError, RuntimeError, APIConnectionError, InternalServerError, RateLimitError)
_INDEFINITE_ARTICLE = r"(?:an?|une?)\s+"
_PHYSICAL_VERB = (r"(?:holds?|holding|carries|carrying|grips?|gripping|picks up|picked up|"
                  r"tient|tenant|porte|portant|prend|prenant|saisit|saisissant)")
_COORDINATION = r"(?:\s+and\s+|\s+et\s+|\s+ainsi qu(?:e\s+|'))"
_ALIAS_CUE = re.compile(r"\b(?:alias|aka|(?:also )?called|same|mêmes?|appel\w+|nommé\w+)\b")


class _IdentityBudgetExceeded(Exception):
    """The full request and its bounded correction do not fit."""


class _IdentityEvidenceFailure(ValueError):
    """A known evidence review failure with a safe diagnostic code."""


def _unique(values):
    return list(dict.fromkeys(values))


def _names(entity):
    return _unique([entity["canonical_name"], *entity["aliases"], *[
        name for member in entity["members"]
        for name in [member["observation"]["canonical_name"], *member["observation"]["aliases"]]
    ]])


def _compact(entity):
    return {"id": entity["id"], "entity_type": entity["kind"],
            "canonical_name": entity["canonical_name"], "names": _names(entity),
            "sequences": _unique([m["sequence"] for m in entity["members"]]),
            "intrinsic_descriptions": _unique([trait["description"] for m in entity["members"]
                                               for trait in m["observation"]["visual_traits"]
                                               if trait["scope"] == "intrinsic"]),
            "occurrences": [{"sequence": m["sequence"], "chunk": m["chunk"],
                             "observed_name": m["observation"]["canonical_name"],
                             "evidence": m["observation"].get("evidence_passages") or m["observation"]["evidence"]}
                            for m in entity["members"]]}


def _validate(result, entities, catalog, proposals_by_id, *, include_member_context=False):
    if not isinstance(result, dict) or set(result) != {"groups"} or not isinstance(result["groups"], list):
        return [], [{"reason": "response_schema"}]
    by_id = {entity["id"]: entity for entity in entities}
    claims = []
    for group in result["groups"]:
        ids = group.get("member_ids") if isinstance(group, dict) else None
        claims.append({i for i in ids if isinstance(i, str) and i in by_id} if isinstance(ids, list) else set())
    valid, rejected = [], []
    group_schema = SCHEMA["schema"]["properties"]["groups"]["items"]
    for number, (group, claim) in enumerate(zip(result["groups"], claims), 1):
        problem = {"group": number, "candidate_ids": sorted(claim)}
        passage_ids = group.get("passage_ids") if isinstance(group, dict) else None
        if isinstance(passage_ids, list) and all(isinstance(value, str) for value in passage_ids):
            group = {**group, "passage_ids": _unique(passage_ids)}
        try:
            validate_document(group, group_schema, "identity group")
        except ValueError:
            rejected.append({**problem, "reason": "group_schema"})
            continue
        ids = _unique(group["member_ids"])
        if len(claim) != len(ids):
            missing = [value for value in ids if value not in by_id]
            rejected.append({**problem, "reason": "unknown_member_id",
                             "member_positions": [i for i, value in enumerate(ids, 1) if value not in by_id],
                             "unknown_ids": [value for value in missing if re.fullmatch(r"REF_[0-9]{5,10}", value)]})
            continue
        if len(ids) < 2:
            rejected.append({**problem, "reason": "insufficient_distinct_members"})
            continue
        if group["canonical_member_id"] not in claim:
            rejected.append({**problem, "reason": "invalid_canonical_member_id"})
            continue
        members = [by_id[i] for i in ids]
        if len({e["kind"] for e in members}) != 1:
            rejected.append({**problem, "reason": "mixed_entity_types"})
            continue
        try:
            select_passages(catalog, group["passage_ids"])
        except ValueError:
            rejected.append({**problem, "reason": "invalid_passage_selection"})
            continue
        passage_ids = list(group["passage_ids"])
        missing = {}
        for entity in members:
            addresses = {(m["sequence"], m["chunk"]) for m in entity["members"]}
            occurrences = proposals_by_id[entity["id"]]["occurrences"]
            anchors = _unique(identifier for row in occurrences for identifier in row.get("evidence_passage_ids", []))
            if anchors:
                if not set(anchors).intersection(passage_ids):
                    # Reuse an exact occurrence citation already validated during extraction.
                    # The semantic reviewer still decides whether it establishes continuity.
                    passage_ids.append(anchors[0])
            elif not any((catalog[i]["sequence"], catalog[i]["chunk"]) in addresses for i in passage_ids):
                missing[entity["id"]] = [{"sequence": sequence, "chunk": chunk}
                                         for sequence, chunk in sorted(addresses)]
            if include_member_context:
                passage_ids.extend(identifier for identifier, row in catalog.items()
                                   if (row["sequence"], row["chunk"]) in addresses)
        if missing:
            rejected.append({**problem, "reason": "missing_member_evidence", "uncovered_occurrences": missing})
            continue
        passage_ids = sorted(set(passage_ids), key=lambda identifier: int(identifier[1:]))
        evidence = select_passages(catalog, passage_ids)
        group = {**group, "member_ids": ids, "passage_ids": passage_ids}
        name = by_id[group["canonical_member_id"]]["canonical_name"]
        valid.append((group, {**group, "canonical_name": name,
                              "aliases": [a for a in _unique(n for e in members for n in _names(e)) if a != name],
                              "source_passages": evidence,
                              "candidates": [proposals_by_id[i] for i in ids]}))
    return valid, rejected


def _apply(result, entities):
    by_id = {e["id"]: e for e in entities}
    removed = set()
    for group in result["groups"]:
        members = sorted((by_id[i] for i in group["member_ids"]), key=lambda e: min(m["order"] for m in e["members"]))
        keep = by_id[members[0]["id"]] = members[0].copy()
        names = [a for e in members for a in _names(e)]
        keep["members"] = sorted([m for e in members for m in e["members"]], key=lambda m: m["order"])
        keep["canonical_name"] = group["canonical_name"]
        keep["aliases"] = [name for name in _unique(names) if name.casefold() != keep["canonical_name"].casefold()]
        keep["unresolved"] = False
        removed.update(e["id"] for e in members[1:])
    return [by_id[e["id"]] for e in entities if e["id"] not in removed]


def _candidate_schema(entities, catalog):
    schema = deepcopy(SCHEMA)
    groups = schema["schema"]["properties"]["groups"]
    fields = groups["items"]["properties"]
    candidate_ids = [entity["id"] for entity in entities]
    groups["maxItems"] = len(candidate_ids) // 2
    fields["member_ids"]["maxItems"] = len(candidate_ids)
    fields["member_ids"]["items"]["enum"] = candidate_ids
    fields["canonical_member_id"]["enum"] = candidate_ids
    fields["passage_ids"]["maxItems"] = len(catalog)
    fields["passage_ids"]["items"]["enum"] = list(catalog)
    return schema


def _proposal_entity(entity, passages_by_address):
    payload = _compact(entity)
    for row in payload["occurrences"]:
        quoted_spans = [quote for quote in row["evidence"] if isinstance(quote, dict)]
        passages = passages_by_address.get((row["sequence"], row["chunk"]), [])
        ids = [passage["id"] for passage in passages if any(
            quote["start"] < passage["end"] and passage["start"] < quote["end"] for quote in quoted_spans
        )]
        if ids:
            row["evidence_passage_ids"] = ids
    return payload


def _request(step, client, model, args, entities, passages, cache_dir, *, all_entities, jobs,
             include_member_context=False):
    sources = [({"sequence": p["sequence"], "chunk": p["chunk"], "offset": p.get("offset", 0)}, p["text"])
               for p in passages]
    indexed = passage_catalog(sources, "chapter-identity", whole_sentences=True)
    catalog = {f"P{number}": passage for number, passage in enumerate(indexed.values(), 1)}
    offered = offered_passages(catalog)
    passages_by_address = {}
    for passage in offered:
        passages_by_address.setdefault((passage["sequence"], passage["chunk"]), []).append(passage)
    proposals_by_id = {entity["id"]: _proposal_entity(entity, passages_by_address) for entity in entities}
    payload = {"source_passages": offered, "entities": list(proposals_by_id.values())}
    schema = _candidate_schema(entities, catalog)
    if not continuity_context.fits(SYSTEM, payload, schema, reserve=128):
        raise _IdentityBudgetExceeded
    full_members = {entity["id"]: entity["members"] for entity in all_entities
                    if entity["id"] in proposals_by_id}
    addresses = {(m["sequence"], m["chunk"]) for members in full_members.values() for m in members}
    source_by_address = {(job["sequence"], job["chunk"]): {"text": job["text"], "offset": job.get("offset", 0)}
                         for job in jobs
                         if (job["sequence"], job["chunk"]) in addresses}
    key = fingerprint(model, args, VERSION, SYSTEM, schema, IDENTITY_REVIEW_SYSTEM,
                      COEXISTENCE_SYSTEM, IDENTITY_REVIEW_SCHEMA, payload, full_members,
                      list(source_by_address.items()), include_member_context, client=client)

    def validate(result, candidates):
        return _validate(result, candidates, catalog, proposals_by_id, include_member_context=include_member_context)

    path = confined_path(cache_dir / f"identity_{key}.json", cache_dir)
    pending = confined_path(cache_dir / f"pending_identity_{key}.json", cache_dir)
    resume = getattr(args, "resume_cache_dir", None)
    folders = [cache_dir, resume] if resume is not None else [cache_dir]
    history = []
    errors = []
    accepted = []
    unresolved = set()
    if not args.force:
        for folder in folders:
            candidate = confined_path(path.name, folder)
            try:
                cached = util.load_json(candidate)
            except FileNotFoundError:
                continue
            resolved, errors = validate(cached, entities)
            errors.extend(_overlap_errors(resolved))
            if not errors:
                if candidate != path:
                    util.save_json(path, cached)
                return {"groups": [group for _, group in resolved], "attempts": history, "status": "passed"}
            history.append({"attempt": 0, "failures": errors})
        for folder in folders:
            try:
                checkpoint = util.load_json(confined_path(pending.name, folder))
            except FileNotFoundError:
                continue
            if checkpoint["status"] != "incomplete":
                raise ValueError("Invalid pending identity checkpoint status.")
            accepted, invalid = validate({"groups": checkpoint["confirmed_groups"]}, entities)
            invalid.extend(_overlap_errors(accepted))
            pending_ids = set(checkpoint["unresolved_candidate_ids"])
            if invalid or not pending_ids <= {entity["id"] for entity in entities}:
                raise ValueError("Invalid confirmed groups or candidate IDs in pending identity checkpoint.")
            unresolved = pending_ids
            errors = [{"reason": "pending_identity_decisions", "candidate_ids": sorted(unresolved)}]
            history.append({"attempt": 0, "failures": errors})
            break
    used = {identifier for proposal, _ in accepted for identifier in proposal["member_ids"]}
    remaining = [entity for entity in entities if entity["id"] not in used]
    for attempt in (1, 2):
        if len(remaining) < 2:
            break
        lmstudio_pipeline.comfy_interrupt_check()
        request = {**payload, "entities": [proposals_by_id[e["id"]] for e in remaining]}
        schema = _candidate_schema(remaining, catalog)
        if errors:
            correction = {"reason": errors[0]["reason"],
                          "instruction": "Recheck candidates using only supplied IDs. For each uncovered "
                                         "member, select source_passages by its sequence and chunk. "
                                         "For unsupported groups, reconsider which members refer to "
                                         "distinct entities. Keep any smaller groups with evidence for "
                                         "every member. Omit a group without support."}
            detailed = {**request, "correction": {**correction, "failures": errors}}
            compact = {**request, "correction": correction}
            if continuity_context.fits(SYSTEM, detailed, schema):
                request = detailed
            elif continuity_context.fits(SYSTEM, compact, schema):
                request = compact
        if not continuity_context.fits(SYSTEM, request, schema):
            raise _IdentityBudgetExceeded
        try:
            result = step.chat_json(client, model, SYSTEM,
                                    json.dumps(request, ensure_ascii=False, separators=(",", ":")),
                                    schema, min(args.temperature, 0.1), args.max_tokens)
        except _MODEL_ERRORS as exc:
            if lmstudio_json._is_comfy_interrupt(exc):
                raise
            resolved, errors = [], [{"reason": type(exc).__name__}]
        else:
            resolved, errors = validate(result, remaining)
        confirmed, review_errors = _verified(step, client, model, args, resolved)
        errors.extend(review_errors)
        confirmed, coexistence_errors = _verified_coexistence(
            step, client, model, args, confirmed, full_members, source_by_address)
        errors.extend(coexistence_errors)
        overlaps = _overlap_errors(confirmed)
        errors.extend(overlaps)
        overlapping = {tuple(error["candidate_ids"]) for error in overlaps}
        confirmed = [pair for pair in confirmed if tuple(pair[0]["member_ids"]) not in overlapping]
        accepted.extend(confirmed)
        used = {identifier for proposal, _ in confirmed for identifier in proposal["member_ids"]}
        unresolved.update(identifier for error in errors for identifier in error.get("candidate_ids", []))
        unresolved.difference_update(used)
        history.append({"attempt": attempt, "failures": errors})
        if not errors:
            break
        remaining = [entity for entity in remaining if entity["id"] not in used]
        if all(error["reason"] == "missing_member_evidence" for error in errors):
            remaining = [entity for entity in remaining if entity["id"] in unresolved]
    status = "incomplete" if errors or unresolved else "passed"
    proposals = [proposal for proposal, _ in accepted]
    if status == "passed":
        util.save_json(path, {"groups": proposals})
        pending.unlink(missing_ok=True)
    elif proposals:
        util.save_json(pending, {"status": "incomplete", "confirmed_groups": proposals,
                                 "unresolved_candidate_ids": sorted(unresolved)})
    return {"groups": [group for _, group in accepted], "attempts": history, "status": status,
            **({"unresolved_candidate_ids": sorted(unresolved)} if unresolved else {})}


def _overlap_errors(groups):
    counts = Counter(identifier for proposal, _ in groups for identifier in proposal["member_ids"])
    return [{"reason": "overlapping_groups", "candidate_ids": proposal["member_ids"],
             "conflicting_ids": [identifier for identifier in proposal["member_ids"] if counts[identifier] > 1]}
            for proposal, _ in groups if any(counts[identifier] > 1 for identifier in proposal["member_ids"])]


def _verified(step, client, model, args, groups, *, system=IDENTITY_REVIEW_SYSTEM):
    if not groups:
        return [], []
    try:
        verdicts = _review_verdicts(step.chat_json, client, model, args, [
            {"members": group["candidates"],
             "source_passages": group["source_passages"]}
            for _, group in groups
        ], bounded=True, system=system)
    except _MODEL_ERRORS as exc:
        if lmstudio_json._is_comfy_interrupt(exc):
            raise
        if len(groups) > 1:
            middle = len(groups) // 2
            left, left_errors = _verified(step, client, model, args, groups[:middle], system=system)
            right, right_errors = _verified(step, client, model, args, groups[middle:], system=system)
            return left + right, left_errors + right_errors
        reason = str(exc) if isinstance(exc, _IdentityEvidenceFailure) else type(exc).__name__
        return [], [{"reason": reason, "candidate_ids": groups[0][0]["member_ids"]}]
    numbers = Counter(row["decision_number"] for row in verdicts)
    by_number = {row["decision_number"]: row["verdict"] for row in verdicts}
    accepted, errors = [], []
    for number, pair in enumerate(groups, 1):
        verdict = by_number[number] if numbers[number] == 1 else "review_coverage"
        if verdict == "supported":
            accepted.append(pair)
        else:
            errors.append({"reason": "evidence_" + verdict, "candidate_ids": pair[0]["member_ids"],
                           **({"sequence": pair[1]["sequence"]} if "sequence" in pair[1] else {})})
    if any(number > len(groups) for number in numbers):
        errors.append({"reason": "unexpected_evidence_review_target"})
    return accepted, errors


def _coordinated_objects(members, passages):
    """Recognize independent direct objects in an explicit handling clause."""
    sentences = passage_catalog([
        ({"chunk": passage["chunk"], "offset": passage["offset"]}, passage["text"])
        for passage in passages
    ], "object-coexistence", whole_sentences=True)
    for sentence in sentences.values():
        text = sentence["text"].casefold()
        if _ALIAS_CUE.search(text):
            continue
        names = []
        for occurrences in members.values():
            cited_names = []
            for member in occurrences:
                if member["chunk"] != sentence["chunk"]:
                    continue
                observation = member["observation"]
                for evidence in observation.get("evidence_passages", []):
                    if (evidence["chunk"] == sentence["chunk"]
                            and sentence["start"] <= evidence["start"] < evidence["end"] <= sentence["end"]
                            and sentence["text"][evidence["start"] - sentence["start"]:
                                                 evidence["end"] - sentence["start"]] == evidence["text"]):
                        cited_names.append(re.sub(r"^" + _INDEFINITE_ARTICLE, "",
                                                  observation["canonical_name"].casefold()))
                        break
            names.append(_unique(cited_names))
        for left, right in combinations(names, 2):
            for a, b in product(left, right):
                if a == b:
                    continue
                phrases = [_INDEFINITE_ARTICLE + re.escape(name) for name in (a, b)]
                both_orders = "|".join(_COORDINATION.join(order) for order in (phrases, phrases[::-1]))
                if re.search(r"\b" + _PHYSICAL_VERB + r"\s+(?:" + both_orders + r")(?!\w)", text):
                    return sentence["text"]
    return None


def _verified_coexistence(step, client, model, args, groups, full_members, source_by_address):
    checks = []
    proven_distinct = []
    for proposal, decision in groups:
        owners = {}
        for identifier in proposal["member_ids"]:
            for member in full_members[identifier]:
                owners.setdefault(member["sequence"], {}).setdefault(identifier, []).append(member)
        # Include previously merged occurrences even when the comparison window
        # projects them out. A later bridge must not hide a known co-occurrence.
        for sequence, members in owners.items():
            if len(members) < 2:
                continue
            candidates = []
            chunks = set()
            for identifier, occurrences in members.items():
                names = _unique(m["observation"]["canonical_name"] for m in occurrences)
                candidates.append({"id": identifier, "observed_names": names,
                                   "occurrence_evidence": _unique(q for m in occurrences
                                                                  for q in m["observation"]["evidence"])})
                chunks.update(m["chunk"] for m in occurrences)
            passages = [{"sequence": sequence, "chunk": chunk, **source_by_address[sequence, chunk]}
                        for chunk in sorted(chunks)]
            if decision["candidates"][0]["entity_type"] == "objects":
                evidence = _coordinated_objects(members, passages)
                if evidence:
                    proven_distinct.append({"reason": "evidence_distinct_instances",
                                            "candidate_ids": proposal["member_ids"], "sequence": sequence,
                                            "evidence": evidence})
                    break
            checks.append((proposal, {"sequence": sequence, "candidates": candidates, "source_passages": passages}))
    blocked = {tuple(error["candidate_ids"]) for error in proven_distinct}
    checks = [pair for pair in checks if tuple(pair[0]["member_ids"]) not in blocked]
    _, errors = _verified(step, client, model, args, checks, system=COEXISTENCE_SYSTEM)
    errors = proven_distinct + errors
    blocked.update(tuple(error["candidate_ids"]) for error in errors if "candidate_ids" in error)
    return ([pair for pair in groups if tuple(pair[0]["member_ids"]) not in blocked],
            [{**error, "stage": "coexistence_review"} for error in errors])


def _catalog(entities, step):
    catalog = {kind: [] for kind in ("characters", "locations", "objects")}
    for entity in sorted(entities, key=lambda e: min(m["order"] for m in e["members"])):
        kind = entity["kind"]
        observations = [m["observation"] for m in entity["members"]]
        item = deepcopy(observations[0])
        item.update(canonical_name=entity["canonical_name"], aliases=entity["aliases"], state_by_sequence={})
        prefix = {"characters": "CHAR", "locations": "LOC", "objects": "OBJ"}[kind]
        item["local_id"] = f"{prefix}_{len(catalog[kind]) + 1:03d}"
        item["first_sequence"] = min(m["sequence"] for m in entity["members"])
        for field in ("evidence", "reference_view_hints"):
            item[field] = _unique([v for o in observations for v in o[field]])
        for field, order in (("importance", step.IMPORTANCE), ("reference_priority", step.PRIORITY)):
            values = [o[field] for o in observations if o.get(field) in order]
            if values:
                item[field] = min(values, key=order.index)
        if kind == "characters":
            item["speaks"] = any(o["speaks"] for o in observations)
            item["voice_description"] = "\n".join(_unique([o["voice_description"] for o in observations if o["voice_description"]]))
        for member in entity["members"]:
            item["state_by_sequence"].setdefault(str(member["sequence"]), {"source": []})["source"].append(deepcopy(member["observation"]))
        catalog[kind].append(item)
    # Shared labels cannot be global identity aliases for distinct physical instances.
    for entries in catalog.values():
        names_by_id = {e["local_id"]: {n.casefold() for n in [e["canonical_name"], *e["aliases"]]} for e in entries}
        for item in entries:
            other_names = {n for lid, names in names_by_id.items() if lid != item["local_id"] for n in names}
            item["aliases"] = [a for a in item["aliases"] if a.casefold() not in other_names]
    return catalog


def _reconcile_windows(entities, jobs, step, client, model, args, cache_dir, orders):
    """Split over-budget work into within-window and cross-window comparisons."""
    windows = []
    members = {member["order"]: member for entity in entities for member in entity["members"]}
    owners = {member["order"]: entity["id"] for entity in entities for member in entity["members"]}

    def reconcile(left, right=()):
        nonlocal entities, owners
        lmstudio_pipeline.comfy_interrupt_check()
        selected = {}
        for order in [*left, *right]:
            selected.setdefault(owners[order], []).append(members[order])
        ids = set(selected)
        if len(ids) < 2:
            return
        # Offer only selected occurrences. Verification can see the full identities.
        candidates = [{**e, "members": selected[e["id"]]} for e in entities if e["id"] in ids]
        addresses = {(m["sequence"], m["chunk"]) for e in candidates for m in e["members"]}
        passages = [job for job in jobs if (job["sequence"], job["chunk"]) in addresses]
        try:
            result = _request(step, client, model, args, candidates, passages, cache_dir,
                              all_entities=entities, jobs=jobs, include_member_context=True)
        except _IdentityBudgetExceeded:
            if not right and len(left) > 2:
                middle = len(left) // 2
                first, second = left[:middle], left[middle:]
                reconcile(first)
                reconcile(second)
                reconcile(first, second)
                return
            if right and max(len(left), len(right)) > 1:
                if len(right) > len(left):
                    left, right = right, left
                middle = len(left) // 2
                reconcile(left[:middle], right)
                reconcile(left[middle:], right)
                return
            result = {"status": "incomplete", "reason": "context_budget", "groups": []}
        windows.append({"candidate_ids": sorted(ids), **result})
        if result["groups"]:
            entities = _apply(result, entities)
            owners = {m["order"]: e["id"] for e in entities for m in e["members"]}

    reconcile(sorted(orders))
    return entities, windows


def _record_windows(check, windows):
    check["groups"] = [{key: value for key, value in group.items() if key != "candidates"}
                       for window in windows for group in window.pop("groups")]
    check["windows"] = windows
    if any(window["status"] == "incomplete" for window in windows):
        check["status"] = "incomplete"


def append_occurrences(entities, job, step, order):
    """Replay a validated answer into the ordered chapter dictionary."""
    for kind in ("characters", "locations", "objects"):
        for row in job["result"][kind]:
            order += 1
            observation = step.clean_entity(row, kind)
            member = {"sequence": job["sequence"], "chunk": job["chunk"],
                      "order": order, "observation": observation}
            ref = row["reference_id"]
            if ref in entities:
                entity = entities[ref]
                entity["members"].append(member)
                names = [*entity["aliases"], observation["canonical_name"], *observation["aliases"]]
                entity["aliases"] = [name for name in _unique(names)
                                     if name.casefold() != entity["canonical_name"].casefold()]
            else:
                ref_id = f"REF_{order:05d}"
                entities[ref_id] = {"id": ref_id, "kind": kind, "members": [member],
                                    "canonical_name": observation["canonical_name"],
                                    "aliases": observation["aliases"], "unresolved": ref == "unresolved"}
    return order


def finish_dictionary(entities, jobs, step, client, model, args, cache_dir, diagnostics):
    """Send only provisional identities and possible duplicates to the merge audit."""
    finished = []
    checks = diagnostics.setdefault("identity_consolidation", [])
    for kind in progress.steps(("characters", "locations", "objects")):
        lmstudio_pipeline.comfy_interrupt_check()
        entries = [entity for entity in entities.values() if entity["kind"] == kind]
        name_counts = Counter(name for entity in entries for name in {name.casefold() for name in _names(entity)})
        unresolved = any(entity["unresolved"] for entity in entries)
        suspects = [entity for entity in entries if unresolved or any(
            name_counts[name.casefold()] > 1 for name in _names(entity))]
        if not suspects:
            finished.extend(entries)
            continue
        check = {"stage": "chapter_identity_consolidation", "kind": kind, "status": "passed"}
        if len(suspects) > 1:
            orders = [m["order"] for entry in suspects for m in (entry["members"][0], entry["members"][-1])]
            entries, windows = _reconcile_windows(entries, jobs, step, client, model, args, cache_dir, set(orders))
            _record_windows(check, windows)
        pending = [entry["id"] for entry in entries if entry["unresolved"]]
        if pending:
            check.update(status="incomplete", unresolved_ids=pending)
        checks.append(check)
        finished.extend(entries)
    catalog = _catalog(finished, step)
    by_kind = {kind: iter(catalog[kind]) for kind in catalog}
    diagnostics["identity_occurrences"] = []
    for entry in sorted(finished, key=lambda e: e["members"][0]["order"]):
        item = next(by_kind[entry["kind"]])
        diagnostics["identity_occurrences"].extend(
            {"occurrence_id": f"REF_{m['order']:05d}", "local_id": item["local_id"], "sequence": m["sequence"]}
            for m in entry["members"])
    return catalog

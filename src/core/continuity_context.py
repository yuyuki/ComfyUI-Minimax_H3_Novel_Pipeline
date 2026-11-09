"""Lossless evidence packing and conservative request budgets for H3 continuity."""
from __future__ import annotations

import json


# UTF-8 bytes, deliberately not the optimistic characters/4 token estimate.
# Leaves ample room for templates and completion in a 65,536-token context.
REQUEST_BYTES = 48000
EVIDENCE_MARKER = "\nAuthoritative sequence phases: "


def pack(payload):
    """Intern repeated sequence evidence without changing cached scene records."""
    evidence = {}
    ids = {}

    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: visit(item) for key, item in value.items()}
        notes = result.get("adaptation_notes")
        if isinstance(notes, str) and EVIDENCE_MARKER in notes:
            local, suffix = notes.split(EVIDENCE_MARKER, 1)
            shared = EVIDENCE_MARKER + suffix
            if shared not in ids:
                key = f"sequence_evidence_{len(ids) + 1}"
                ids[shared] = key
                evidence[key] = shared
            result["adaptation_notes"] = local
            result["sequence_evidence_ref"] = ids[shared]
        return result

    result = visit(payload)
    if evidence:
        result["shared_sequence_evidence"] = evidence
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def fits(system, payload, schema, reserve=0):
    return (len(system.encode("utf-8")) + len(pack(payload).encode("utf-8"))
            + len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
            + reserve <= REQUEST_BYTES)


def batches(system, base, groups, schema, reserve=0):
    """Keep every observation, in order; never truncate independent evidence."""
    empty = {key: [] for key in groups}
    current = {**base, **empty}
    if not fits(system, current, schema, reserve):
        raise ValueError("H3 continuity current-scene context exceeds the safe request budget. "
                         "Reduce the size of this cinematic sequence; no evidence was discarded.")
    count = 0
    for key, items in groups.items():
        for item in items:
            candidate = {**current, key: [*current[key], item]}
            if not fits(system, candidate, schema, reserve):
                if count:
                    yield current
                current = {**base, **empty}
                candidate = {**current, key: [item]}
                if not fits(system, candidate, schema, reserve):
                    raise ValueError("One H3 continuity evidence record exceeds the safe request budget. "
                                     "Reduce the size of its cinematic sequence; no evidence was discarded.")
                count = 0
            current = candidate
            count += 1
    if count or not any(groups.values()):
        yield current


INSTRUCTIONS = (
    "\nResolve each sequence_evidence_ref using shared_sequence_evidence; its full text "
    "belongs to that scene. Evidence may arrive in batches. Judge only supplied evidence; "
    "absence from this batch is not evidence of absence. Preserve explicit temporal changes."
)

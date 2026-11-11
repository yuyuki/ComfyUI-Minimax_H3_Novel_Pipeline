"""Lossless evidence packing and conservative request budgets for H3 continuity."""
from __future__ import annotations

import json
from collections import Counter


# UTF-8 bytes, deliberately not the optimistic characters/4 token estimate.
# Leaves ample room for templates and completion in a 65,536-token context.
REQUEST_BYTES = 48000
EVIDENCE_MARKER = "\nAuthoritative sequence phases: "


def pack(payload):
    """Intern exact shared evidence/excerpts; retain every scene occurrence."""
    evidence = {}
    ids = {}
    excerpts = Counter()
    def count(value):
        if isinstance(value, list):
            for item in value:
                count(item)
        elif isinstance(value, dict):
            excerpt = value.get("source_excerpt")
            if isinstance(excerpt, str) and len(excerpt) >= 256:
                excerpts[excerpt] += 1
            for item in value.values():
                count(item)
    count(payload)
    shared_excerpts, excerpt_ids = {}, {}

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
        excerpt = result.get("source_excerpt")
        if isinstance(excerpt, str) and excerpts[excerpt] > 1:
            if excerpt not in excerpt_ids:
                key = f"source_excerpt_{len(excerpt_ids) + 1}"
                excerpt_ids[excerpt] = key
                shared_excerpts[key] = excerpt
            del result["source_excerpt"]
            result["source_excerpt_ref"] = excerpt_ids[excerpt]
        return result

    result = visit(payload)
    if evidence:
        result["shared_sequence_evidence"] = evidence
    if shared_excerpts:
        result["shared_source_excerpts"] = shared_excerpts
    # Intern repeated citation snapshots only. Keep their paths and occurrences;
    # a citation may describe an older prompt, so never resolve it against a
    # newer scene/prompt merely because its original path matches.
    citations = []

    def collect(value):
        if isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            if set(value) == {"path", "value"} and isinstance(value["path"], str):
                citations.append(value)
            else:
                for item in value.values():
                    collect(item)

    collect(result)
    fingerprints = [json.dumps(item["value"], ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")) for item in citations]
    counts = Counter(fingerprints)
    shared_citations, citation_ids = {}, {}
    for citation, fingerprint in zip(citations, fingerprints):
        if counts[fingerprint] < 2 or len(fingerprint.encode("utf-8")) < 256:
            continue
        if fingerprint not in citation_ids:
            key = f"cited_value_{len(citation_ids) + 1}"
            citation_ids[fingerprint] = key
            shared_citations[key] = citation["value"]
        citation["value_ref"] = citation_ids[fingerprint]
        del citation["value"]
    if shared_citations:
        result["shared_cited_values"] = shared_citations
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def fits(system, payload, schema, reserve=0):
    return (len(system.encode("utf-8")) + len(pack(payload).encode("utf-8"))
            + len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
            + reserve <= REQUEST_BYTES)


def budget_details(system, payload, schema):
    """Describe packed byte costs without logging source text or cited values."""
    packed = pack(payload)
    system_bytes = len(system.encode("utf-8"))
    payload_bytes = len(packed.encode("utf-8"))
    schema_bytes = len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    fields = {key: len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
              for key, value in json.loads(packed).items()}
    largest = sorted(fields.items(), key=lambda item: item[1], reverse=True)[:5]
    return (f"request_bytes={system_bytes + payload_bytes + schema_bytes}, limit_bytes={REQUEST_BYTES}, "
            f"system_bytes={system_bytes}, payload_bytes={payload_bytes}, schema_bytes={schema_bytes}, "
            f"largest_field_value_bytes={dict(largest)}")


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
    " Resolve each source_excerpt_ref using shared_source_excerpts: it replaces that scene's "
    "source_excerpt with the EXACT original prose, not a summary. Shared text does not merge "
    "scene occurrences or their separately assigned beats."
    " Resolve citation value_ref using shared_cited_values: it replaces the citation's value "
    "with its exact captured evidence. Keep every citation and issue distinct; these snapshots "
    "may refer to earlier wording, not the current prompt."
)

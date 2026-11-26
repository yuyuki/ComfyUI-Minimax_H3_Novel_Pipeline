"""Build structured wire selections from readable scenario fixtures."""
from copy import deepcopy
from functools import wraps
import json

from minimax_h3_novel_pipeline.core.source_text import comparison_text


def _select(offers, quote, *, all_matches=False):
    if not isinstance(quote, str) or not quote:
        return []
    exact = [row["id"] for row in offers if comparison_text(row["text"]) == comparison_text(quote)]
    if exact:
        return exact if all_matches else exact[:1]
    bits = [row for row in offers if comparison_text(row["text"]) in comparison_text(quote)]
    if comparison_text(" ".join(row["text"] for row in bits)) == comparison_text(quote):
        return [row["id"] for row in bits]
    enclosing = [row["id"] for row in offers if comparison_text(quote) in comparison_text(row["text"])]
    return enclosing[:1] or ["unoffered-fixture-passage"]


def adapt_fixture_reply(reply, payload, schema):
    if not isinstance(reply, dict):
        return reply
    result = deepcopy(reply)
    fields = schema["schema"].get("properties", {})
    if "source_segments" in payload and "sequences" in result:
        segments = payload["source_segments"]
        cursor = 0
        result.pop("unprocessed_source", None)
        for row in result["sequences"]:
            if not isinstance(row, dict) or "source" not in row:
                continue
            excerpt = comparison_text(row.pop("source"))
            end = next((end for end in range(cursor + 1, len(segments) + 1)
                        if comparison_text("".join(p["text"] for p in segments[cursor:end])) == excerpt), 0)
            row["source_end_id"] = end
            if end:
                cursor = end
    if "requested_sections" in payload:
        for key in ("timing", "continuity"):
            if key not in fields:
                result.pop(key, None)
        for beat in result.get("timing", {}).get("beats", []):
            for row in beat.get("events", []):
                if isinstance(row, dict) and "text" in row:
                    excerpt = row.pop("text")
                    row["fragment"] = None
                    if excerpt is not None:
                        index = row["event_index"] - 1
                        words = payload["event_words"][index] if 0 <= index < len(payload["event_words"]) else []
                        needle = comparison_text(excerpt)
                        valid = next(((left + 1, right) for left in range(len(words))
                                      for right in range(left + 1, len(words) + 1)
                                      if comparison_text(" ".join(w["text"] for w in words[left:right])) == needle), (0, 0))
                        row["fragment"] = dict(zip(("first_word", "last_word"), valid))
        if isinstance(result.get("continuity"), dict):
            section = result["continuity"]
            offers = payload["current_source_passages"]
            if "reset_evidence" in section:
                section["reset_passage_ids"] = _select(offers, section.pop("reset_evidence"))
            section.pop("reset_passages", None)
            for row in section.get("updates", []):
                if "source_quote" in row:
                    row["passage_ids"] = _select(offers, row.pop("source_quote"))
                row.pop("source_passages", None)
    if "review_passages" in payload and isinstance(result.get("issues"), list):
        offers = payload["review_passages"]
        issue_fields = fields["issues"]["items"]["properties"]
        if "narration_language" in fields:
            result.setdefault("narration_language", {
                field: "fr" if field == "source" else ["fr"] for field in fields["narration_language"]["required"]
            })
        for row in result["issues"]:
            if isinstance(row, dict) and set(row) == {"target", "message"} and offers:
                target = row["target"]
                eligible = [item["id"] for item in offers if item["sequence_index"] == target]
                row["passage_ids"] = eligible[:1] or ["unoffered-fixture-passage"]
                if "check" in issue_fields:
                    row["check"] = "adaptation"
            if isinstance(row, dict) and "check" in row and "field" in issue_fields:
                row.setdefault("field", "sequence")
                row.setdefault("disposition", "repair")
    if "evidence" in result and "passages" in payload and "source_passages" in payload:
        offers = payload["source_passages"]
        rows = []
        for row in result["evidence"]:
            if not isinstance(row, dict) or "quotation" not in row:
                rows.append(row)
                continue
            quote = row.pop("quotation")
            if not isinstance(quote, str):
                rows.append({**row, "quotation_passage_ids": ["unoffered-fixture-passage"]})
                continue
            exact = [item["id"] for item in offers if comparison_text(item["text"]) == comparison_text(quote)]
            if exact:
                rows.extend({**row, "quotation_passage_ids": [identifier]} for identifier in exact)
            else:
                rows.append({**row, "quotation_passage_ids": _select(offers, quote)})
        result["evidence"] = rows
    if "known_references" in payload and "chunk_summary" in result:
        offers = payload["source_passages"]
        for kind in ("characters", "locations", "objects"):
            for row in result.get(kind, []):
                row.setdefault("reference_id", "new")
                if "evidence_passage_ids" not in row:
                    row.pop("evidence", None)
                    row["evidence_passage_ids"] = [offer["id"] for offer in offers]
                for trait in row.get("visual_traits", []):
                    if "evidence" in trait:
                        trait["evidence_passage_ids"] = _select(offers, trait.pop("evidence"))
    if "source_passages" in payload and "groups" in result and schema["name"] == "chapter_reference_identity":
        for group in result["groups"]:
            if "source_quotes" in group:
                group.pop("source_quotes")
                name = group.pop("canonical_name")
                group.pop("aliases", None)
                group["canonical_member_id"] = next(
                    row["id"] for row in payload["entities"]
                    if row["id"] in group["member_ids"] and row["canonical_name"] == name)
                owners = {member for row in payload["entities"] if row["id"] in group["member_ids"]
                          for member in row["sequences"]}
                group["passage_ids"] = [offer["id"] for offer in payload["source_passages"] if offer["sequence"] in owners]
    if "jobs" in payload and schema["name"] == "character_trait_relations":
        jobs = {job["job_id"]: job for job in payload["jobs"]}
        for row in result.get("decisions", []):
            if "temporal_quote" in row:
                options = jobs.get(row["job_id"], {}).get("temporal_passages", {}).get(row["support_id"], [])
                row["temporal_passage_ids"] = _select(options, row.pop("temporal_quote"))
    if "source_passages" in payload and "sequence" in payload and "character" in schema["name"]:
        offers = payload["source_passages"]
        for field in ("characters", "unresolved", "states", "observations"):
            for row in result.get(field, []):
                if "source_quote" in row:
                    row["passage_ids"] = _select(offers, row.pop("source_quote"))
        if schema["name"] == "character_local_observation_review":
            result["issues"] = [{"message": issue, "passage_ids": [offers[0]["id"]]}
                                if isinstance(issue, str) else issue for issue in result.get("issues", [])]
    return result


def with_passage_ids(chat):
    @wraps(chat)
    def wrapped(client, model, system, prompt, schema, *args, **kwargs):
        if isinstance(schema, dict) and schema.get("name") == "reference_identity_evidence_review":
            decisions = json.loads(prompt)["decisions"]
            return {"decisions": [{"decision_number": row["decision_number"], "verdict": "supported"} for row in decisions]}
        reply = chat(client, model, system, prompt, schema, *args, **kwargs)
        if not isinstance(schema, dict) or "schema" not in schema:
            return reply
        try:
            payload, _ = json.JSONDecoder().raw_decode(prompt)
        except (TypeError, ValueError):
            return reply
        if not isinstance(payload, dict):
            return reply
        return adapt_fixture_reply(reply, payload, schema)
    return wrapped

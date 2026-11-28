"""Exercise source coverage, carried state and bounded semantic review."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca
from minimax_h3_novel_pipeline.core import cinematic_diagnostics, narrative_choices

DELPHES = Path(__file__).with_name("fixtures") / "delphes.txt"


def state(id_, description):
    return {"id": id_, "description": description}


def event(description, *changes):
    return {"description": description, "changes": list(changes)}


def sequence(end, seconds, *events):
    return {"source_end_id": end, "estimated_seconds": seconds, "events": list(events)}


def approved(payload, findings=(), choices=()):
    return {"source_language": "fr", "language_checks": [
                {"sequence": row["sequence"], **dict.fromkeys(narrative_choices.PHASES, "fr")}
                for row in payload["sequences"]],
            "descriptive_checks": [
                {"sequence": row["sequence"], "unsupported_transfers": [], "lost_visual_or_spatial_meaning": []}
                for row in payload["sequences"]],
            "findings": list(findings), "choices": list(choices)}


def run(source, checkpoints=None, **kwargs):
    settings = dict(chunk_chars=4000, temperature=0.15, max_tokens=4096, correction_attempts=1)
    settings.update(kwargs)
    return ca.adapt_chapter(None, "gemma-4-26b-it", source, chapter_name="delphes",
                            on_checkpoint=None if checkpoints is None else checkpoints.append, **settings)


def delphes_draft(payload, *, short=False, seconds=(3, 2)):
    segments = payload["source_segments"]
    first = next(row["id"] for row in segments if row["text"].endswith("aisselles."))
    return {
        "opening_state": [
            state("period", "Delphes, Grèce, en 1922."),
            state("space", "Dans l'obscurité, une ouverture est loin au-dessus d'Indy et laisse passer très peu de lumière."),
            state("indy", "Indy se balance, suspendu à une corde qui comprime son torse et ses aisselles."),
        ],
        "sequences": [
            sequence(first, seconds[0], event("Indy continue de se balancer au bout de la corde.")),
            sequence(len(segments), 9 if short else seconds[1],
                     event("Des cris indistincts retentissent au-dessus d'Indy, hors champ."),
                     event("Indy penche la tête en arrière vers l'ouverture.",
                           state("indy", "Indy se balance toujours au bout de la corde qui comprime son torse et ses aisselles. "
                                         "Sa tête est penchée en arrière."))),
        ],
    }


@pytest.mark.parametrize(("seconds", "duration", "expected"), [
    ((3, 2), 8, [5]),
    ((3.1, 4.9), 8, [8]),
    ((3, 6), 8, [3, 6]),
    ((3, 2), 4, [3, 2]),
])
def test_delphes_validates_story_context_and_keeps_source_verbatim(monkeypatch, seconds, duration, expected):
    source = DELPHES.read_bytes().decode("utf-8")
    calls = []

    def chat(client, model, system, body, schema, temp, tokens):
        payload = json.loads(body)
        calls.append(schema["name"])
        assert schema["strict"] and schema["schema"]["additionalProperties"] is False
        assert tokens == ca.lmstudio_json.REQUEST_OUTPUT_LIMIT.get() == 4096
        if schema == ca.SCHEMA:
            assert "".join(row["text"] for row in payload["source_segments"]) == source
            assert payload["current_passage"] == source
            assert ca._DESCRIPTIVE_INTENT in system
            assert "Properties of the figurative image" in system
            return delphes_draft(payload, seconds=seconds)
        assert schema == ca.REVIEW_SCHEMA
        assert "offscreen sound" in system and "spatial relationships" in system
        assert ca._DESCRIPTIVE_INTENT in system
        assert "Leave both arrays empty for faithful simplification" in system
        assert [row["estimated_seconds"] for row in payload["sequences"]] == expected
        return approved(payload)

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    checkpoints = []
    chapter = run(source, checkpoints, duration=duration)
    assert len(calls) == 2
    assert json.loads(json.dumps(chapter)) == chapter
    assert "".join(row["source"] for row in chapter["sequences"]) == source
    scenes = chapter["sequences"]
    assert [row["sequence"] for row in scenes] == list(range(1, len(scenes) + 1))
    assert [row["estimated_seconds"] for row in scenes] == expected
    initial = scenes[0]["adaptation"]["initialState"]
    ending = scenes[-1]["adaptation"]["endingState"]
    assert "Sa tête est penchée" not in initial
    assert "Sa tête est penchée" in ending
    assert "1922" in ending and "loin au-dessus d'Indy" in ending and "corde qui comprime" in ending
    event_lines = [line for scene in scenes for line in scene["adaptation"]["event"].splitlines()]
    assert "cris indistincts" in event_lines[1] and "hors champ" in event_lines[1]
    assert "penche la tête" in event_lines[2]
    if len(scenes) == 1:
        assert [line.split(". ", 1)[0] for line in event_lines] == ["1", "2", "3"]
    else:
        assert scenes[0]["adaptation"]["endingState"] == scenes[1]["adaptation"]["initialState"]
    assert all(word not in ending for word in ("étoile", "croissant", "clignote", "projecteur"))
    assert "performance" not in chapter and "narrative_review" not in chapter
    assert ca.lmstudio_json.REQUEST_OUTPUT_LIMIT.get() is None


def test_semantic_repair_uses_exact_evidence_and_retains_editable_event_choice(monkeypatch):
    source = "Ana est près de l’ouverture. Elle entend un cri indistinct."
    calls = []

    def chat(client, model, system, body, schema, *args):
        payload = json.loads(body)
        calls.append(schema["name"])
        if schema == ca.SCHEMA:
            correction = payload.get("correction")
            if correction:
                assert "verify each finding against the source" in system
                assert "Preserve all valid facts and paraphrases" in system
                assert correction["findings"][0]["source_quote"] == "Ana est près de l’ouverture."
            fact = "Ana est près de l'ouverture." if correction else "Ana est près d'un projecteur."
            return {"opening_state": [state("ana", fact)],
                    "sequences": [sequence(len(payload["source_segments"]), 3, event("Ana entend un cri indistinct."))]}
        row = payload["sequences"][0]
        evidence = row["source_passages"][0]["id"]
        if len(calls) == 2:
            review = approved(payload)
            review["descriptive_checks"][0]["unsupported_transfers"].append(
                {"passage_id": evidence, "message": "The opening was replaced with an invented lamp."})
            return review
        return approved(payload, choices=[{"sequence": 1, "passage_id": evidence,
                                           "reason": "The direction of the cry remains unknown.",
                                           "alternative_event": "Un cri indistinct retentit hors champ."}])

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run(source)
    assert len(calls) == 4
    issue = result["narrative_review"]["issues"][0]
    assert issue["source_excerpt"] == "Ana est près de l’ouverture."
    issue["proposals"][0]["selected"] = False
    issue["proposals"][1]["selected"] = True
    resolved = narrative_choices.resolve_chapter(result)["sequences"][0]
    assert resolved["adaptation"]["event"] == "Un cri indistinct retentit hors champ."
    assert resolved["adaptation"]["initialState"] == resolved["adaptation"]["endingState"]


@pytest.mark.parametrize("error", ["duration", "overlapping_end", "missing_tail", "unknown_removal", "duplicate_change"])
def test_invalid_generation_never_commits(monkeypatch, error):
    source = DELPHES.read_bytes().decode("utf-8")

    def chat(client, model, system, body, schema, *args):
        assert schema == ca.SCHEMA
        payload = json.loads(body)
        draft = delphes_draft(payload, short=error == "duration")
        first, last = draft["sequences"]
        if error == "overlapping_end":
            last["source_end_id"] = first["source_end_id"]
        if error == "missing_tail":
            last["source_end_id"] -= 1
        if error == "unknown_removal":
            last["events"][0]["changes"] = [state("absent", "")]
        if error == "duplicate_change":
            last["events"][0]["changes"] = [state("indy", "Indy se balance."), state("indy", "Indy attend.")]
        return draft

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    checkpoints = []
    with pytest.raises(ValueError, match="invalid_structured_generation"):
        run(source, checkpoints)
    assert not checkpoints


@pytest.mark.parametrize("error", ["network", "missing_coverage", "foreign_evidence"])
def test_failed_review_retains_valid_data_with_incomplete_diagnostics(monkeypatch, error):
    def chat(client, model, system, body, schema, *args):
        payload = json.loads(body)
        if schema == ca.SCHEMA:
            return delphes_draft(payload)
        if error == "network":
            raise ConnectionError("secret backend detail")
        if error == "missing_coverage":
            review = approved(payload)
            review["language_checks"].pop()
            return review
        return approved(payload, findings=[{"sequence": 1, "passage_id": "wrong", "check": "continuity",
                                            "message": "Unknown citation."}])

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    checkpoints = []
    chapter = run(DELPHES.read_bytes().decode("utf-8"), checkpoints)
    report = cinematic_diagnostics.report(
        checkpoints[-1], chapter_name="delphes", source_file=str(DELPHES), maximum_seconds=8, completed=True)
    assert len(chapter["sequences"]) == 1
    assert report["status"] == "incomplete" and report["generation_status"] == "completed"
    assert report["summary"]["incomplete_scenes"] == 1
    assert "secret backend detail" not in json.dumps(report)


def test_resume_carries_state_uses_only_unread_source_and_rejects_modified_checkpoint(monkeypatch):
    source = "Ana prend la clé. Ana ouvre la porte. Ana appelle depuis la salle."
    chunks = []
    checkpoints = []

    def chat(client, model, system, body, schema, *args):
        payload = json.loads(body)
        if schema == ca.REVIEW_SCHEMA:
            return approved(payload)
        segments = payload["source_segments"]
        chunks.append("".join(row["text"] for row in segments))
        if not payload["previous_state"]:
            return {"opening_state": [state("door", "La porte se trouve face à Ana.")],
                    "sequences": [sequence(len(segments), 2, event("Ana prend la clé.", state("key", "Ana tient la clé.")))]}
        assert state("key", "Ana tient la clé.") in payload["previous_state"]
        return {"opening_state": [], "sequences": [sequence(len(segments), 3, event(chunks[-1].strip()))]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    full = run(source, checkpoints, chunk_chars=25)
    first = checkpoints[0]
    assert first["consumed_chars"] < len(source)
    chunks.clear()
    assert full == run(source, chunk_chars=25, resume_state=first)
    assert "".join(chunks) == source[first["consumed_chars"]:]
    assert all("Ana tient la clé." in row["adaptation"]["endingState"] for row in full["sequences"])
    broken = deepcopy(first)
    broken["sequences"][0]["source"] += "!"
    chunks.clear()
    with pytest.raises(ValueError, match="resume checkpoint"):
        run(source, chunk_chars=25, resume_state=broken)
    assert chunks == []



def test_cited_timing_overrun_requires_real_split(monkeypatch):
    source = "Ana ferme la première porte. Ana descend les marches."
    opening = [state("door", "Ana est devant la première porte.")]
    calls = []

    def chat(client, model, system, body, schema, *args):
        payload = json.loads(body)
        calls.append(schema["name"])
        if schema == ca.SCHEMA:
            assert ca.lmstudio_json.REQUEST_THINKING.get() is None
            segments = payload["source_segments"]
            end = len(segments)
            if "correction" not in payload:
                return {"opening_state": opening,
                        "sequences": [sequence(end, 8, event("Ana ferme la porte puis descend les marches."))]}
            first = next(row["id"] for row in segments if row["text"].endswith("porte."))
            return {"opening_state": opening, "sequences": [
                sequence(first, 3, event("Ana ferme la première porte.", state("door", "La porte est fermée."))),
                sequence(end, 6, event("Ana descend les marches.")),
            ]}
        assert ca.lmstudio_json.REQUEST_THINKING.get() is False
        if len(calls) == 2:
            first = payload["sequences"][0]
            return approved(payload, findings=[{"sequence": 1, "passage_id": first["source_passages"][0]["id"],
                                                "check": "duration", "message": "These actions take more than eight seconds."}])
        return approved(payload)

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(ValueError, match="duration overrun"):
        run(source, correction_attempts=0)
    calls.clear()
    first, second = run(source)["sequences"]
    assert [first["estimated_seconds"], second["estimated_seconds"]] == [3, 6]
    assert first["adaptation"]["endingState"] == second["adaptation"]["initialState"] == "La porte est fermée."
    assert first["source"] + second["source"] == source
    assert ca.lmstudio_json.REQUEST_THINKING.get() is None


def test_review_does_not_swallow_comfy_cancellation(monkeypatch):
    class InterruptProcessingException(RuntimeError):
        pass

    def chat(client, model, system, body, schema, *args):
        if schema == ca.REVIEW_SCHEMA:
            raise InterruptProcessingException()
        return delphes_draft(json.loads(body))

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    checkpoints = []
    with pytest.raises(InterruptProcessingException):
        run(DELPHES.read_bytes().decode("utf-8"), checkpoints)
    assert not checkpoints
    assert ca.lmstudio_json.REQUEST_METRICS.get() is None
    assert not ca.lmstudio_json.REQUEST_DRAIN_USAGE.get()


def test_language_verdict_alone_triggers_repair(monkeypatch):
    source = "Ana attend près de la porte."

    def chat(client, model, system, body, schema, *args):
        payload = json.loads(body)
        if schema == ca.REVIEW_SCHEMA:
            review = approved(payload)
            if "waits" in payload["sequences"][0]["adaptation"]["event"]:
                review["language_checks"][0]["event"] = "en"
            return review
        if "correction" in payload:
            defect = payload["correction"]["findings"][0]
            assert defect["check"] == "language"
            assert defect["source_quote"] == source
            action = "Ana attend."
        else:
            action = "Ana waits."
        return {"opening_state": [state("ana", "Ana se tient près de la porte.")],
                "sequences": [sequence(len(payload["source_segments"]), 2, event(action))]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    assert run(source)["sequences"][0]["adaptation"]["event"] == "1. Ana attend."

"""Compound events decompose losslessly into independently timed action fragments."""
from copy import deepcopy
import json

import pytest

from tests.passages_helpers import with_passage_ids

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca, cinematic_scene_planning as planning
from minimax_h3_novel_pipeline.core import narrative_validation as nv


def scene(source, events):
    return {"source": source, "initial_state": "Avant.", "events": events, "final_state": "Après."}


def plan(fragments, seconds):
    return {"timing": {"beats": [
        {"events": [{"event_index": index, "text": text}], "seconds": duration, "reason": "Action distincte."}
        for (index, text), duration in zip(fragments, seconds)],
        "pause_seconds": 0, "estimated_seconds": sum(seconds), "reason": "Actions successives."},
        "continuity": {"reset": False, "reset_evidence": "", "updates": []}}


FRAGMENTS = [(1, "Il sort le filet"), (1, "et recouvre la tablette."), (2, "Il noue les lacets.")]


@pytest.mark.parametrize("stage", ["scene_planned", "scene_committed"])
def test_whole_event_references_survive_review_and_resume(monkeypatch, stage):
    item = scene("Sous ses bras, la corde se resserra.", ["la corde se resserre sous ses bras"])
    snapshots, planning_calls, reviews = [], [], []
    options = dict(chunk_chars=4000, temperature=0.15, max_tokens=16384, correction_attempts=0)

    def chat(client, model, system, user, schema, *args):
        if schema["name"] == planning.SCHEMA["name"]:
            planning_calls.append(user)
            return plan([(1, None)], [2])
        if schema["name"] == nv.SCHEMA["name"]:
            payload = json.loads(user)
            reviews.append(payload)
            assert payload["scene_plan"]["timing"]["beats"][0]["events"][0]["text"] == item["events"][0]
            return {"issues": []}
        return {"sequences": [deepcopy(item)]}

    def checkpoint(state):
        snapshots.append(deepcopy(state))
        if state["stage"] == stage:
            raise KeyboardInterrupt()

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    with pytest.raises(KeyboardInterrupt):
        ca.adapt_chapter(None, "mock", item["source"], on_checkpoint=checkpoint, **options)
    saved = snapshots[-1]
    cached = saved["pending_sequences"][0]["planning"] if stage == "scene_planned" else saved["scene_plans"][0]["plan"]
    assert cached["timing"]["beats"][0]["events"] == [{"event_index": 1, "text": None}]
    result = ca.adapt_chapter(None, "mock", item["source"], resume_state=saved, **options)
    assert result["sequences"][0]["adaptation"]["event"] == "1. " + item["events"][0]
    assert len(planning_calls) == 1 and len(reviews) == 1


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("spans,events,fragments,seconds", [
    (["\n\nIl parcourut les signes du regard. L'excitation lui noua les tripes. "
      "Insérant la torche dans son support,",
      " il tira un carnet d'une poche latérale de son sac et y griffonna rapidement la traduction. "
      "Il ne parvenait pas à y croire. Ils avaient raison. Ces dingues savaient de quoi ils parlaient."],
     ["Indy parcourt les signes du regard avec intensité.", "Il insère la torche dans son support.",
      "Il sort un carnet de sa poche latérale et écrit rapidement dedans."],
     [(1, "Indy parcourt les signes du regard avec intensité."), (2, "Il insère la torche dans son support."),
      (3, "Il sort un carnet de sa poche latérale"), (3, "et écrit rapidement dedans.")], [2, 2, 2, 4]),
    (["\n\nIl brûlait d'envie d'avertir ses compagnons mais décida de conserver ses forces. "
      "Il fourra de nouveau le carnet dans son sac et en tira un filet",
      " dont il recouvrit soigneusement la tablette, avant d'en nouer les lacets à un crochet au bout de la corde."],
     ["Indy range le carnet dans son sac.", "Il sort un filet et recouvre la tablette avec précaution.",
      "Il noue les lacets du filet à un crochet fixé au bout de la corde."],
     [(1, "Indy range le carnet dans son sac."), (2, "Il sort un filet"),
      (2, "et recouvre la tablette avec précaution."),
      (3, "Il noue les lacets du filet à un crochet fixé au bout de la corde.")], [2, 1, 2, 3]),
], ids=["prologue_scene_13", "prologue_scene_14"])
def test_prologue_compound_events_trigger_timing_split_and_resume_with_partial_reviews(
        monkeypatch, partial, spans, events, fragments, seconds):
    original = scene("".join(spans), events)
    planned, reviewed, snapshots = [], [], []
    options = dict(chunk_chars=4000, temperature=0.15, max_tokens=16384, correction_attempts=1)

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema["name"] == planning.SCHEMA["name"]:
            item = payload["current_scene"]
            planned.append(item["source"])
            if item["source"] == original["source"]:
                return plan(fragments, seconds)
            return plan([(1, item["events"][0])], [3])
        if schema["name"] == nv.SCHEMA["name"]:
            source = payload["sequences"][0]["source"]
            reviewed.append(source)
            if partial and source == spans[0]:
                raise RuntimeError("Review unavailable.")
            return {"issues": []}
        if "rejected_adaptation" in payload:
            assert payload["minimum_replacement_scenes"] == 2
            assert planning.duration_issues(payload["rejected_scene_plan"], 8)
            return {"sequences": [scene(span, [f"Action {i}."]) for i, span in enumerate(spans, 1)]}
        return {"sequences": [deepcopy(original)]}

    def checkpoint(state):
        snapshots.append(deepcopy(state))
        if state["stage"] == "scene_planned" and not state["sequences"]:
            raise KeyboardInterrupt()

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    with pytest.raises(KeyboardInterrupt):
        ca.adapt_chapter(None, "mock", original["source"], on_checkpoint=checkpoint, **options)
    saved = snapshots[-1]
    assert saved["approved_reviews"] == []
    saved_plan = saved["pending_sequences"][0]["planning"]
    assert len(saved_plan["timing"]["beats"]) == 4
    assert saved_plan["timing"]["estimated_seconds"] == sum(seconds)
    assert planning.duration_issues(saved_plan, 8)
    result = ca.adapt_chapter(None, "mock", original["source"], resume_state=saved,
                             on_checkpoint=lambda state: snapshots.append(deepcopy(state)), **options)
    assert planned == [original["source"], *spans]  # Cached compound plan is validated and reused.
    assert reviewed == [original["source"], *spans]
    assert [item["source"] for item in result["sequences"]] == spans
    assert "".join(item["source"] for item in result["sequences"]) == original["source"]
    assert all(not planning.duration_issues(record["plan"], 8) for record in snapshots[-1]["scene_plans"])
    assert len(snapshots[-1]["approved_reviews"]) == (1 if partial else 2)
    if partial:
        assert result["sequences"][0]["review"]["status"] == "incomplete"
        assert "review" not in result["sequences"][1]
    else:
        assert all("review" not in item for item in result["sequences"])

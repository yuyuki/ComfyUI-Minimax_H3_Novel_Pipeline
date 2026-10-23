"""Semantic review orchestration with mocked LM Studio responses."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from minimax_h3_novel_pipeline import cinematic_adaptation as ca
from minimax_h3_novel_pipeline import cinematic_references as cr
from minimax_h3_novel_pipeline import narrative_validation as nv, narrative_choices as nc, path_access, util


@pytest.mark.parametrize("source,bad,good", [
    ("Mara catches the lamp. The wall is lit.", "The lamp falls; total darkness.",
     "Mara holds the lamp; the wall is lit, the distance dark."),
    ("Luc opens the shutter and looks up at the pale sky.", "Luc opens the shutter.",
     "The shutter is open; Luc looks up at the pale sky."),
])
def test_targeted_repair_preserves_other_sequences_and_rechecks(monkeypatch, source, bad, good):
    first = {"source": "They wait. ", "initial_state": "They stand.",
             "events": ["They wait."], "final_state": "They stand waiting."}
    second = {"source": source, "initial_state": "They stand waiting.",
              "events": [source], "final_state": bad}
    reviews, repairs = [], []

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema == nc.PROPOSAL_SCHEMA:
            return proposal_response()
        if schema == nv.SCHEMA:
            reviews.append(deepcopy(data))
            return {"issues": [{"target": 2, "message": "Final state contradicts or omits source facts."}]
                    if data["sequences"][1]["final_state"] == bad else []}
        if schema == ca._SINGLE_SCHEMA:
            repairs.append(data)
            return {"initial_state": second["initial_state"], "events": [source], "final_state": good}
        return {"sequences": deepcopy([first, second])}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = ca.adapt_chapter(None, "mock", first["source"] + source, chunk_chars=4000,
                             temperature=0.15, max_tokens=1000, correction_attempts=1)
    assert len(repairs) == 1 and repairs[0]["current_passage"] == " " + source
    assert repairs[0]["previous_final_state"] == first["final_state"]
    assert repairs[0]["source_context"] == first["source"] + source
    assert repairs[0]["target_sequence"] == 2
    assert repairs[0]["context_sequences"] == reviews[0]["sequences"]
    assert len(reviews) == 2 and reviews[0]["sequences"][0] == reviews[1]["sequences"][0]
    assert result["sequences"][1]["adaptation"]["endingState"] == good
    assert "".join(s["source"] for s in result["sequences"]) == first["source"] + source


def test_semantic_failure_is_bounded_even_for_small_fallback(monkeypatch):
    calls = []

    def chat(client, model, system, prompt, schema, *args):
        calls.append(schema["name"])
        if schema == nc.PROPOSAL_SCHEMA:
            return proposal_response()
        if schema == nv.SCHEMA:
            return {"issues": [{"target": 1, "message": "Missing explicit action."}]}
        fields = {"initial_state": "Before.", "events": [], "final_state": "After."}
        if schema == ca._SINGLE_SCHEMA:
            return fields
        return {"sequences": [{"source": "Wrong copy.", **fields}]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = ca.adapt_chapter(None, "mock", "A bell rings.", chunk_chars=4000,
                              temperature=0.1, max_tokens=1000, correction_attempts=1)
    issue = result["narrative_review"]["issues"][0]
    assert issue["sequence"] == 1
    assert [p["selected"] for p in issue["proposals"]] == [True, False]
    assert result["sequences"][0]["adaptation"] == issue["proposals"][0]["adaptation"]
    assert calls.count(nv.SCHEMA["name"]) == 3
    assert calls.count(ca._SINGLE_SCHEMA["name"]) == 3


@pytest.mark.parametrize("attempts", [0, 1, 2])
def test_semantic_reconstruction_uses_full_source_and_is_reviewed(monkeypatch, attempts):
    source = "Indy se balance. Il lève les yeux."
    following = " Il attrape la torche."
    bad = {"initial_state": "Il reste immobile.", "events": ["Il lève les yeux."],
           "final_state": "Il regarde en haut."}
    good = {"initial_state": "Il se balance.",
            "events": ["Il se balance.", "Il lève les yeux."],
            "final_state": "Il se balance, le regard vers le haut."}
    repairs, reviews = [], []

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema == nc.PROPOSAL_SCHEMA:
            return proposal_response()
        if schema == nv.SCHEMA:
            reviews.append(deepcopy(data))
            return {"issues": [{"target": 1, "message": "Preserve the swinging motion."}]
                    if data["sequences"][0]["initial_state"] == bad["initial_state"] else []}
        if schema == ca._SINGLE_SCHEMA:
            repairs.append(data)
            assert "Review\nmessages and generated states are fallible" in system
            if "target_sequence" in data:
                return deepcopy(bad)  # Targeted repairs fail; reconstruction succeeds.
            assert data["current_passage"] == source
            assert data["previous_final_state"] == ""
            assert data["defects_to_correct"] == {"1": ["Preserve the swinging motion."]}
            assert len(data["rejected_adaptation"]) == 2
            return deepcopy(good)
        if data["current_passage"] == following:
            assert data["previous_final_state"] == good["final_state"]
            return {"sequences": [{"source": following, **good}]}
        return {"sequences": [{"source": "Indy se balance.", **bad},
                              {"source": "Il lève les yeux.", **bad}]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    kwargs = dict(chunk_chars=len(source), temperature=0.15, max_tokens=1000,
                  correction_attempts=attempts)
    if not attempts:
        result = ca.adapt_chapter(None, "mock", source, **kwargs)
        assert result["narrative_review"]["issues"]
        assert not repairs and len(reviews) == 1
        return
    monkeypatch.setattr(ca, "source_chunks", lambda *a: [source, following])
    result = ca.adapt_chapter(None, "mock", source + following, **kwargs)
    assert len(repairs) == attempts + 1
    assert len(reviews) == attempts + 3  # Initial, targeted, rebuild, following passage.
    assert len(reviews[-2]["sequences"]) == 1
    assert result["sequences"][0]["adaptation"]["initialState"] == good["initial_state"]
    assert "".join(s["source"] for s in result["sequences"]) == source + following


def test_targeted_repairs_use_updated_neighbor_state(monkeypatch):
    sequences = [{"source": source, "initial_state": "Wrong.", "events": [source],
                  "final_state": "Wrong."} for source in ["Il se balance.", " Il lève les yeux."]]
    repairs = []

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema == nc.PROPOSAL_SCHEMA:
            return proposal_response()
        if schema == nv.SCHEMA:
            return {"issues": [{"target": i + 1, "message": "Incorrect posture."}
                               for i, s in enumerate(data["sequences"]) if s["final_state"] == "Wrong."]}
        if schema == ca._SINGLE_SCHEMA:
            repairs.append(data)
            return {"initial_state": "Il se balance.", "events": [data["current_passage"]],
                    "final_state": "Il se balance."}
        return {"sequences": deepcopy(sequences)}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    ca.adapt_chapter(None, "mock", "".join(s["source"] for s in sequences), chunk_chars=4000,
                     temperature=0.15, max_tokens=1000, correction_attempts=1)
    assert len(repairs) == 2
    assert repairs[1]["previous_final_state"] == "Il se balance."
    assert repairs[1]["context_sequences"][0]["final_state"] == "Il se balance."


@pytest.mark.parametrize("response", [{}, {"issues": None}, {"issues": [{"target": True, "message": "x"}]},
                                      {"issues": [{"target": 2, "message": "x"}]}])
def test_invalid_review_never_approves(response):
    with pytest.raises(ValueError, match="Invalid narrative"):
        nv.review(lambda *a: response, None, "mock", {}, 1, 1000)


def test_cancellation_before_review_prevents_request(monkeypatch):
    def stop():
        raise KeyboardInterrupt()
    monkeypatch.setattr(nv.lmstudio_pipeline, "comfy_interrupt_check", stop)
    with pytest.raises(KeyboardInterrupt):
        nv.review(lambda *a: pytest.fail("Network call after cancellation"), None, "mock", {}, 1, 1000)


@pytest.mark.parametrize("persistent_defect", [False, True])
def test_phase_repair_identity_speech_and_cache(tmp_path, monkeypatch, persistent_defect):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    output = tmp_path / "catalogs"
    output.mkdir()
    chapter = {"chapter_name": "scene", "sequences": [{"sequence": 1, "source": "Not sent to extraction.",
               "adaptation": {"initialState": "Mara hangs from her rope.",
                              "event": "Mara, called Vale, shouts. A basket descends on another rope.",
                              "endingState": "Mara holds the basket beside her own rope."}}]}
    args = SimpleNamespace(chunk_chars=4000, temperature=0.1, max_tokens=2000, force=False)
    calls, reviews = [], []

    def chat(client, model, system, prompt, schema, *rest):
        data = json.loads(prompt)
        if schema == nc.PROPOSAL_SCHEMA:
            return proposal_response()
        if schema == nv.SCHEMA:
            reviews.append(data)
            assert "Not sent" not in prompt
            rejected = data["extraction"]["chunk_summary"] == "bad"
            return {"issues": [{"target": 1, "message": "Separate the basket rope from the support rope."}]
                    if rejected else []}
        if schema == cr.LOCAL_LINK_SCHEMA:
            return {"links": []}
        calls.append(data)
        phase_input = data.get("phase_input", data)
        phase = phase_input["phase"]
        bad = phase == "event" and (persistent_defect or "defects_to_correct" not in data)
        return {"chunk_summary": "bad" if bad else "", "locations": [],
                "characters": [{"canonical_name": "Mara", "aliases": ["Vale"] if phase == "event" else [],
                                "speaks": phase == "event"}],
                "objects": [{"canonical_name": "support rope" if phase == "initialState" else "basket rope",
                             "aliases": ["rope"], "chapter_state": phase_input["text"]}]}

    monkeypatch.setattr(cr.lmstudio_pipeline.load("extract"), "chat_json", chat)
    saved = cr.process_chapter(chapter, output, None, "mock", args, "scene")
    catalog = util.load_json(saved)
    assert len(catalog["objects"]) == 2  # Shared alias must not merge physical objects.
    person = catalog["characters"][0]
    assert person["speaks"] is True and person["aliases"] == ["Vale"]
    states = person["state_by_sequence"]["1"]
    assert states["initialState"][0]["speaks"] is False
    assert states["endingState"][0]["speaks"] is False
    assert bool(catalog["narrative_review"]["extraction_issues"]) == persistent_defect
    assert len(calls) == (5 if persistent_defect else 4)
    assert len(reviews) == (5 if persistent_defect else 4)
    assert "basket" not in json.dumps(reviews[0])
    cr.process_chapter(chapter, output, None, "mock", args, "scene")
    assert bool(catalog["narrative_review"]["extraction_issues"]) == persistent_defect
    assert len(calls) == (5 if persistent_defect else 4)
    assert len(reviews) == (5 if persistent_defect else 4)


def proposal_response():
    return {"proposals": [
        {"reason": "Clear staging.", "adaptation": {"initialState": "Before.", "event": "1. A bell rings.", "endingState": "After."}},
        {"reason": "Closer framing.", "adaptation": {"initialState": "Close view.", "event": "1. A bell rings in close view.", "endingState": "After, close view."}},
    ]}


def test_proposals_keep_global_sequence_ids_and_selected_continuity(monkeypatch):
    sources = ["First scene.", "Second scene."]
    calls = []
    monkeypatch.setattr(ca, "source_chunks", lambda *a: sources)

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema == nv.SCHEMA:
            return {"issues": [{"target": 1, "message": "Uncertain staging."}]}
        if schema == nc.PROPOSAL_SCHEMA:
            return proposal_response()
        calls.append(data)
        return {"sequences": [{"source": data["current_passage"],
                               "initial_state": "Before.", "events": ["Wait."], "final_state": "Wrong."}]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = ca.adapt_chapter(None, "mock", "".join(sources), chunk_chars=1000,
                             temperature=0.1, max_tokens=1000, correction_attempts=0)
    assert [i["sequence"] for i in result["narrative_review"]["issues"]] == [1, 2]
    assert calls[1]["previous_final_state"] == "After."
    assert "".join(s["source"] for s in result["sequences"]) == "".join(sources)

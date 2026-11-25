from tests.neutral_reference_helpers import candidates as neutral_candidates
"""Semantic review orchestration with mocked LM Studio responses."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

pytestmark = pytest.mark.usefixtures("mock_scene_planning")

from tests.passages_helpers import with_passage_ids

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca
from minimax_h3_novel_pipeline.core import cinematic_references as cr
from minimax_h3_novel_pipeline.core import narrative_validation as nv, narrative_choices as nc, path_access, util


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
        if schema["name"] == nv.SCHEMA["name"]:
            reviews.append(deepcopy(data))
            return {"issues": [{"target": 1, "message": "Final state contradicts or omits source facts."}]
                    if data["sequences"][0]["final_state"] == bad else []}
        if "rejected_adaptation" in data:
            repairs.append(data)
            return {"sequences": [{"source": data["current_passage"], "initial_state": second["initial_state"], "events": [source], "final_state": good}]}
        return {"sequences": deepcopy([first, second])}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    result = ca.adapt_chapter(None, "mock", first["source"] + source, chunk_chars=4000,
                             temperature=0.15, max_tokens=1000, correction_attempts=1)
    assert len(repairs) == 1 and repairs[0]["current_passage"] == " " + source
    assert repairs[0]["previous_final_state"] == first["final_state"]
    assert repairs[0]["target_sequence"] == 2
    assert repairs[0]["previous_sequence"] == {
        **reviews[0]["sequences"][0], "events": [result["sequences"][0]["adaptation"]["event"]],
    }
    assert "source" not in repairs[0]["rejected_adaptation"]
    assert "source_context" not in repairs[0] and "context_sequences" not in repairs[0]
    assert len(reviews) == 3
    assert reviews[1]["previous_sequence"] == reviews[2]["previous_sequence"]
    assert result["sequences"][1]["adaptation"]["endingState"] == good
    assert "".join(s["source"] for s in result["sequences"]) == first["source"] + source


def test_semantic_failure_is_bounded_even_for_small_fallback(monkeypatch):
    calls = []

    def chat(client, model, system, prompt, schema, *args):
        data, _ = json.JSONDecoder().raw_decode(prompt)
        calls.append(schema["name"])
        if schema == nc.PROPOSAL_SCHEMA:
            return proposal_response()
        if schema["name"] == nv.SCHEMA["name"]:
            return {"issues": [{"target": 1, "message": "Missing explicit action."}]}
        fields = {"initial_state": "Before.", "events": [], "final_state": "After."}
        return {"sequences": [{"source": "Wrong boundary." if "rejected_adaptation" in data else data["current_passage"], **fields}]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    result = ca.adapt_chapter(None, "mock", "A bell rings.", chunk_chars=4000,
                              temperature=0.1, max_tokens=1000, correction_attempts=1)
    issue = result["narrative_review"]["issues"][0]
    assert issue["sequence"] == 1
    assert [p["selected"] for p in issue["proposals"]] == [True, False]
    assert result["sequences"][0]["adaptation"] == issue["proposals"][0]["adaptation"]
    assert calls.count(nv.SCHEMA["name"]) == 4
    assert calls.count(ca.SCHEMA["name"]) == 5


@pytest.mark.parametrize("attempts", [0, 1, 2])
def test_final_semantic_recovery_stays_local_and_is_reviewed(monkeypatch, attempts):
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
        if schema["name"] == nv.SCHEMA["name"]:
            reviews.append(deepcopy(data))
            return {"issues": [{"target": 1, "message": "Preserve the swinging motion."}]
                    if data["sequences"][0]["source"].strip() == "Indy se balance."
                    and data["sequences"][0]["initial_state"] == bad["initial_state"] else []}
        if "rejected_adaptation" in data:
            repairs.append(data)
            assert "Review\nmessages and generated states are fallible" in system
            assert data["target_sequence"] == 1
            assert data["current_passage"] == "Indy se balance."
            assert data["previous_final_state"] == ""
            assert data["defects_to_correct"] == ["Preserve the swinging motion."]
            return {"sequences": [{"source": data["current_passage"], **(good if len(repairs) > attempts else bad)}]}
        if data["current_passage"] == following:
            assert data["previous_final_state"] == bad["final_state"]
            return {"sequences": [{"source": following, **good}]}
        result = {"sequences": [{"source": "Indy se balance.", **bad},
                                {"source": "Il lève les yeux.", **bad}]}
        if schema == ca.WINDOW_SCHEMA:
            result["unprocessed_source"] = ""
        return result

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    kwargs = dict(chunk_chars=len(source), temperature=0.15, max_tokens=1000,
                  correction_attempts=attempts)
    if not attempts:
        result = ca.adapt_chapter(None, "mock", source, **kwargs)
        assert result["narrative_review"]["issues"]
        assert not repairs and len(reviews) == 3
        return
    result = ca.adapt_chapter(None, "mock", source + following, **kwargs)
    assert len(repairs) == attempts + 1
    assert len(reviews) == attempts + 4  # Current scene, repairs, then successor and following passage once each.
    assert all(len(review["sequences"]) == 1 for review in reviews)
    assert result["sequences"][1]["adaptation"]["endingState"] == bad["final_state"]
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
        if schema["name"] == nv.SCHEMA["name"]:
            return {"issues": [{"target": i + 1, "message": "Incorrect posture."}
                               for i, s in enumerate(data["sequences"]) if s["final_state"] == "Wrong."]}
        if "rejected_adaptation" in data:
            repairs.append(data)
            return {"sequences": [{"source": data["current_passage"], "initial_state": "Il se balance.", "events": [data["current_passage"]],
                    "final_state": "Il se balance."}]}
        return {"sequences": deepcopy(sequences)}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    ca.adapt_chapter(None, "mock", "".join(s["source"] for s in sequences), chunk_chars=4000,
                     temperature=0.15, max_tokens=1000, correction_attempts=1)
    assert len(repairs) == 2
    assert repairs[1]["previous_final_state"] == "Il se balance."
    assert repairs[1]["previous_sequence"]["final_state"] == "Il se balance."


def test_duration_repair_splits_scene_and_keeps_following_target_and_state(monkeypatch):
    sources = ["First. Second.", " Third."]
    fields = {"initial_state": "Before.", "events": ["Action."], "final_state": "Wrong."}
    repairs = []
    def chat(client, model, system, prompt, schema, *args):
        assert "6" in system and "duration" in system
        data = json.loads(prompt)
        if schema["name"] == nv.SCHEMA["name"]:
            return {"issues": [{"target": i + 1, "message": "Split excessive action or correct state."}
                               for i, s in enumerate(data["sequences"]) if s["final_state"] == "Wrong."]}
        assert schema == ca.SCHEMA and schema["strict"]
        if "target_sequence" in data:
            repairs.append(data)
            parts = ["First.", " Second."] if len(repairs) == 1 else [sources[1]]
            return {"sequences": [{"source": s, **fields, "final_state": "Correct."} for s in parts]}
        return {"sequences": [{"source": s, **fields} for s in sources]}
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    result = ca.adapt_chapter(None, "mock", "".join(sources), chunk_chars=4000,
                             temperature=.1, max_tokens=2000, correction_attempts=1, duration=6)
    assert result["duration"] == 6
    assert len(result["sequences"]) == 3 and len(repairs) == 2
    assert repairs[1]["target_sequence"] == 3
    assert repairs[1]["previous_final_state"] == "Correct."
    assert repairs[1]["current_passage"].strip() == "Third."
    assert "".join(s["source"] for s in result["sequences"]) == "".join(sources)


@pytest.mark.parametrize("response", [{}, {"issues": None}, {"issues": [{"target": True, "message": "x"}]},
                                      {"issues": [{"target": 2, "message": "x"}]},
                                      {"issues": [None]}, {"issues": [{"message": "x"}]},
                                      {"issues": [{"target": 1, "message": " "}]},
                                      {"issues": [], "doubt": "Missing action."}])
def test_invalid_review_never_approves(response):
    issues = nv.review(lambda *a: response, None, "mock", {}, 1, 1000)
    assert set(issues) == {1}
    assert "not an established defect" in issues[1][0]


def test_invalid_review_preserves_valid_doubts_and_rejects_foreign_passages():
    def chat(client, model, system, prompt, schema, *args):
        offers = json.loads(prompt)["review_passages"]
        return {"issues": [
            {"target": 2, "message": "Missing dialogue.", "passage_ids": [offers[1]["id"]]},
            {"target": 2, "message": "Wrong gaze.", "passage_ids": [offers[0]["id"]]},
        ]}

    issues = nv.review(chat, None, "mock", {"sequences": [{"source": "Before."}, {"source": "She says hello."}]}, 2, 1000)
    assert list(issues) == [2]
    assert issues[2][0] == "Missing dialogue."
    assert "not an established defect" in issues[2][1]


@pytest.mark.parametrize("attempts", [0, 1])
def test_malformed_reviews_remain_incomplete_without_editable_defects(monkeypatch, attempts):
    source = "A bell rings."
    calls = []

    def chat(client, model, system, prompt, schema, *args):
        calls.append(schema["name"])
        if schema["name"] == nv.SCHEMA["name"]:
            return {"issues": [{"target": 99, "message": "Missing bell."}]}
        assert schema != nc.PROPOSAL_SCHEMA
        return {"sequences": [{"source": source, "initial_state": "Before.", "events": [source], "final_state": "After."}]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    result = ca.adapt_chapter(None, "mock", source, chunk_chars=4000,
                             temperature=0.1, max_tokens=1000, correction_attempts=attempts)
    assert result["narrative_review"]["issues"] == []
    assert result["sequences"][0]["review"]["status"] == "incomplete"
    assert result["sequences"][0]["source"] == source
    assert calls.count(nv.SCHEMA["name"]) == (2 if attempts else 1)


@pytest.mark.parametrize("error", [ConnectionError("Offline"), RuntimeError("Invalid structured JSON"),
                                  ValueError("Invalid client configuration"), KeyboardInterrupt()])
def test_review_technical_failures_and_interruptions_propagate(error):
    def fail(*args):
        raise error
    with pytest.raises(type(error)) as caught:
        nv.review(fail, None, "mock", {}, 1, 1000)
    assert caught.value is error


def test_cancellation_before_review_prevents_request(monkeypatch):
    def stop():
        raise KeyboardInterrupt()
    monkeypatch.setattr(nv.lmstudio_pipeline, "comfy_interrupt_check", stop)
    with pytest.raises(KeyboardInterrupt):
        nv.review(lambda *a: pytest.fail("Network call after cancellation"), None, "mock", {}, 1, 1000)


@pytest.mark.parametrize("persistent_defect", [False, True])
def test_source_repair_identity_speech_and_cache(tmp_path, monkeypatch, persistent_defect):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    output = tmp_path / "catalogs"
    output.mkdir()
    chapter = {"chapter_name": "scene", "sequences": [{"sequence": 1,
               "source": "NOT EXTRACTION INPUT",
               "adaptation": {"initialState": "", "event": "Mara, called Vale, shouts. A basket descends on another rope.",
                              "endingState": "NOT EXTRACTION INPUT"}}]}
    args = SimpleNamespace(chunk_chars=4000, temperature=0.1, max_tokens=2000, force=False)
    calls, reviews = [], []

    def chat(client, model, system, prompt, schema, *rest):
        data = json.loads(prompt)
        assert "NOT EXTRACTION INPUT" not in prompt
        if schema["name"] == nv.SCHEMA["name"]:
            reviews.append(data)
            assert data["source_input"]["text"] == chapter["sequences"][0]["adaptation"]["event"]
            rejected = data["extraction"]["chunk_summary"] == "bad"
            return {"issues": [{"target": 1, "message": "The summary omits the descent of the basket."}]
                    if rejected else []}
        if schema["name"] == cr.chapter_reference_identity.SCHEMA["name"]:
            assert all(row["text"] in chapter["sequences"][0]["adaptation"]["event"] for row in data["source_passages"])
            return {"groups": []}
        assert schema == cr.lmstudio_pipeline.load("extract").CHUNK_SCHEMA
        calls.append(data)
        bad = persistent_defect or "defects_to_correct" not in data
        return neutral_candidates({"chunk_summary": "bad" if bad else "A basket descends on a rope.", "locations": [],
                "characters": [{"canonical_name": "Mara", "aliases": ["Vale"], "speaks": True}],
                "objects": [{"canonical_name": name, "aliases": ["rope"]}
                            for name in ("support rope", "basket rope")]}, data.get("source_input", data)["text"])

    monkeypatch.setattr(cr.lmstudio_pipeline.load("extract"), "chat_json",
                        with_passage_ids(chat, extraction_edit="chunk_summary"))
    saved = cr.process_chapter(chapter, output, None, "mock", args, "scene")
    catalog = util.load_json(saved)
    assert len(catalog["objects"]) == 2
    person = catalog["characters"][0]
    assert person["speaks"] is True and person["aliases"] == ["Vale"]
    assert list(person["state_by_sequence"]["1"]) == ["source"]
    assert "extraction_issues" not in catalog["narrative_review"]
    diagnostics = util.load_json(output / "scene_references.diagnostics.json")
    assert diagnostics["status"] == ("incomplete" if persistent_defect else "passed")
    assert len(calls) == (3 if persistent_defect else 2)
    assert len(reviews) == 3
    cr.process_chapter(chapter, output, None, "mock", args, "scene")
    assert (len(calls), len(reviews)) == ((5, 6) if persistent_defect else (2, 3))


def proposal_response():
    return {"proposals": [
        {"reason": "Clear staging.", "adaptation": {"initialState": "Before.", "event": "1. A bell rings.", "endingState": "After."}},
        {"reason": "Closer framing.", "adaptation": {"initialState": "Close view.", "event": "1. A bell rings in close view.", "endingState": "After, close view."}},
    ]}


@pytest.mark.parametrize("remaining", [[], [{"target": 1, "message": "Motion still omitted."}]])
def test_selected_proposals_are_reviewed_once_with_previous_scene(monkeypatch, remaining):
    checked = []
    snapshots = []

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema == nc.PROPOSAL_SCHEMA:
            return proposal_response()
        if schema["name"] == nv.SCHEMA["name"]:
            if "defects_to_recheck" in data:
                checked.append(data)
                assert data["sequences"][0]["final_state"] == "After."
                assert data["defects_to_recheck"] == [
                    "Motion omitted.",
                    "events is empty: describe supported ongoing activity or regroup this held moment.",
                ]
                return {"issues": remaining}
            return {"issues": [{"target": 1, "message": "Motion omitted."}]}
        return {"sequences": [
            {"source": source, "initial_state": "Before.", "events": [], "final_state": "Wrong."}
            for source in ("He swings.", " He looks up.")
        ]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    result = ca.adapt_chapter(None, "mock", "He swings. He looks up.", chunk_chars=4000,
                             temperature=0.1, max_tokens=1000, correction_attempts=0,
                             on_checkpoint=lambda state: snapshots.append(deepcopy(state)))
    assert len(checked) == 2
    assert all("next_sequence" not in review for review in checked)
    assert checked[1]["previous_sequence"]["final_state"] == "After."
    assert checked[1]["previous_final_state"] == "After."
    for issue in result["narrative_review"]["issues"]:
        assert issue["proposals"][0]["review"] == {
            "status": "unresolved" if remaining else "approved",
            "issues": ["Motion still omitted."] if remaining else [],
        }
    assert sum(s["stage"] == "proposal_reviewed" for s in snapshots) == 2


def test_proposals_keep_global_sequence_ids_and_selected_continuity(monkeypatch):
    sources = ["First scene.", "Next scene."]
    calls = []

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema["name"] == nv.SCHEMA["name"]:
            return {"issues": [{"target": 1, "message": "Uncertain staging."}]}
        if schema == nc.PROPOSAL_SCHEMA:
            return proposal_response()
        calls.append(data)
        response = {"sequences": [{"source": data["current_passage"],
                                   "initial_state": "Before.", "events": ["Wait."], "final_state": "Wrong."}]}
        if schema == ca.WINDOW_SCHEMA:
            response["unprocessed_source"] = ""
        return response

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    result = ca.adapt_chapter(None, "mock", "".join(sources), chunk_chars=len(sources[0]),
                             temperature=0.1, max_tokens=1000, correction_attempts=0)
    assert [i["sequence"] for i in result["narrative_review"]["issues"]] == [1, 2]
    assert calls[1]["previous_final_state"] == "After."
    assert "".join(s["source"] for s in result["sequences"]) == "".join(sources)

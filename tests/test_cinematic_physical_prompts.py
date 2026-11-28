"""Physical fidelity instructions reach generation, retries, repair and focused reviews.

These mocked checks verify request contracts, not a live model's semantic accuracy.
"""
from copy import deepcopy
import json

import pytest

pytestmark = pytest.mark.usefixtures("mock_scene_planning")

from tests.passages_helpers import with_passage_ids

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca, narrative_validation as nv


@pytest.mark.parametrize("focused", [False, True])
def test_physical_transition_repair_preserves_source_and_budget(monkeypatch, focused):
    source = "Il attache le filet à la tablette encore enchâssée dans la paroi."
    draft = {"source": source, "initial_state": "La tablette est enchâssée dans la paroi.",
             "events": ["Il attache le filet à la tablette."],
             "final_state": "La tablette est suspendue à la corde."}
    corrected = {**draft, "final_state": "Le filet est attaché à la tablette encore enchâssée."}
    generations, reviews, repairs = [], [], []
    defect = "final_state: attacher le filet ne dégage pas la tablette de la paroi."

    def chat(client, model, system, user, schema, temperature, tokens):
        payload, _ = json.JSONDecoder().raw_decode(user)
        assert schema["strict"] and schema["schema"]
        assert tokens == 16384 and ca.lmstudio_json.REQUEST_OUTPUT_LIMIT.get() == 16384
        if schema["name"] == nv.SCHEMA["name"]:
            focus = payload.get("review_focus")
            reviews.append(focus)
            if repairs:
                return {"issues": []}
            if focused and focus is None:
                raise ca.lmstudio_json.StructuredOutputLengthError("length")
            if focus in (None, "local_transition"):
                assert "source-supported transition" in system
                assert "without performing an additional action" in system
                return {"issues": [{"target": 1, "message": defect}]}
            if focus == "source_fidelity":
                assert "observable" in system and "unsupported adaptation claims" in system
            if focus == "cinematic_pacing":
                assert "duration is a maximum" in system
            return {"issues": []}

        generations.append(payload)
        assert "changing one does not automatically change the others" in system
        assert "without numbering or bullet prefixes" in system
        assert "one continuous, physically plausible interaction" in system
        assert "merge successive beats" not in system
        assert payload["current_passage"] == source
        if "rejected_adaptation" in payload:
            repairs.append(payload)
            assert payload["defects_to_correct"] == [defect]
            assert "Do not invent intermediate handling actions" in system
            return {"sequences": [deepcopy(corrected)]}
        # Exercise source-copy retry before semantic review and physical repair.
        return {"sequences": [{**draft, "source": "Incorrect." if len(generations) == 1 else source}]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    result = ca.adapt_chapter(None, "mock", source, chunk_chars=4000, temperature=0.15,
                             max_tokens=16384, correction_attempts=1)
    assert len(generations) == 3 and len(repairs) == 1
    assert reviews == ([None, *nv.ADAPTATION_REVIEW_FOCUSES, None] if focused else [None, None])
    assert result["sequences"][0]["source"] == source
    assert result["sequences"][0]["adaptation"]["endingState"] == corrected["final_state"]
    assert result["sequences"][0]["adaptation"]["event"] == "1. Il attache le filet à la tablette."
    assert result["narrative_review"]["issues"] == []
    assert ca.lmstudio_json.REQUEST_OUTPUT_LIMIT.get() is None


@pytest.mark.parametrize("review_focus", [None, "source_fidelity", "source_fidelity_event_group"])
def test_light_comparison_is_reviewed_and_repaired(monkeypatch, review_focus):
    source = ca._normalize_text(
        "La porte laisse entrer à peine autant de lumière du jour qu'une bougie vacillante. Ana s'arrête."
    )
    draft = {"source": source, "initial_state": "Une lueur vacille dans la porte.",
             "events": ["Ana s'arrête."], "final_state": "Ana est immobile."}
    corrected_state = "On distingue la porte à la faible lumière du jour. La pièce reste sombre."
    repairs = []

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema["name"] == nv.SCHEMA["name"]:
            if repairs:
                assert payload["exported_adaptation"]["initialState"] == corrected_state
                return {"issues": []}
            focus = payload.get("review_focus")
            if focus in (None, "source_fidelity") and focus != review_focus:
                raise ca.lmstudio_json.StructuredOutputLengthError("length")
            if focus != review_focus:
                return {"issues": []}
            assert "when it comes only from the comparison" in system
            assert "exterior light seen through an opening" in system
            result = {"issues": [{
                "target": 1, "message": "The comparison conveys intensity and does not establish flicker.",
                "field": "initialState", "disposition": "repair", "check": "adaptation",
                "passage_ids": [payload["review_passages"][0]["id"]],
            }]}
            if "assigned_events" in payload:
                result["covered_events"] = payload["assigned_events"]
            return result
        assert "light seen through an opening to the exterior" in system
        assert "flicker or other lighting effect only if the prose establishes it" in system
        if "rejected_adaptation" in payload:
            repairs.append(payload)
            assert payload["fields_to_correct"] == ["initialState"]
            return {"sequences": [{**draft, "initial_state": corrected_state}]}
        return {"sequences": [draft]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    chapter = ca.adapt_chapter(None, "mock", source, chunk_chars=4000, temperature=0.15,
                               max_tokens=16384, correction_attempts=1)
    assert len(repairs) == 1
    sequence = chapter["sequences"][0]
    assert sequence["source"] == source
    assert sequence["adaptation"] == {
        "initialState": corrected_state, "event": "1. Ana s'arrête.", "endingState": "Ana est immobile.",
    }

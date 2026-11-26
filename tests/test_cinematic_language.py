"""Language instructions and correction routing, with no live model dependency."""
from copy import deepcopy
import json

import pytest

pytestmark = pytest.mark.usefixtures("mock_scene_planning")

from tests.passages_helpers import with_passage_ids

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca
from minimax_h3_novel_pipeline.core import narrative_validation as nv


def test_compact_review_retries_preserve_all_language_verdicts_and_event_evidence():
    from tests.test_structured_json import Stream, client_for

    scene = {"source": "Elle écoute. Elle crie : « Help! »", "initial_state": "Elle écoute.",
             "events": ["Elle tend l'oreille.", "She shouts: \"Help!\""], "final_state": "Elle attend."}
    language = {"source": "fr", "initialState": ["fr"], "event": ["fr", "en"], "endingState": ["fr"]}
    wire = {"issues": [], "narration_language": language}
    client, create = client_for(Stream(['{"issues":']), Stream([json.dumps(wire)]))
    issues = nv.review(ca.lmstudio_json.chat_json, client, "qwen3.5-9b",
                       {"stage": "adaptation", "sequences": [scene]}, 1, 8192)
    assert issues == {1: [nv.LANGUAGE_MISMATCH + "event (source fr, field en, fr)."]}
    assert create.call_count == 2
    for call in create.call_args_list:
        response = call.kwargs["response_format"]
        assert response["type"] == "json_schema"
        schema = response["json_schema"]
        assert schema["strict"]
        assert schema["schema"]["properties"]["narration_language"]["required"] == list(language)
        request = call.kwargs["messages"][-1]["content"]
        assert all(json.dumps(event, ensure_ascii=False) in request for event in scene["events"])
        passages = json.JSONDecoder().raw_decode(request)[0]["review_passages"]
        evidence_schema = schema["schema"]["properties"]["issues"]["items"]["properties"]["passage_ids"]
        assert evidence_schema["items"]["enum"] == [passage["id"] for passage in passages]


def test_repair_cannot_reintroduce_translated_narration(monkeypatch):
    source = "Elle saisit la rampe. La main reste sur la rampe."
    french = {"source": source, "initial_state": "Elle longe le mur.",
              "events": ["Elle avance."], "final_state": "Elle reste près du mur."}
    wrong = {"source": source, "initial_state": "She walks along the wall.",
             "events": ["She grabs the railing."], "final_state": "Her hand stays on the railing."}
    fixed = {**french, "events": ["Elle saisit la rampe."], "final_state": "La main reste sur la rampe."}
    reviewed, repairs = [], []

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema["name"] == nv.SCHEMA["name"]:
            rendered = data["exported_adaptation"]
            reviewed.append(rendered)
            language = {"source": "fr", **dict.fromkeys(nv.LANGUAGE_FIELDS, ["fr"])}
            if rendered["initialState"] == wrong["initial_state"]:
                return {"issues": [], "narration_language": {"source": "fr", **dict.fromkeys(nv.LANGUAGE_FIELDS, ["en"])}}
            return {"narration_language": language, "issues": [] if rendered["event"] == "1. Elle saisit la rampe."
                    else [{"target": 1, "message": "The grasp and its resulting contact are missing."}]}
        if "rejected_adaptation" not in data:
            return {"sequences": [french]}
        repairs.append(data)
        return {"sequences": [wrong if len(repairs) == 1 else fixed]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    chapter = ca.adapt_chapter(None, "mock", source, chunk_chars=4000,
                               temperature=0.15, max_tokens=8192, correction_attempts=1)
    assert len(reviewed) == 3
    assert reviewed[1]["event"] == "1. She grabs the railing."
    assert repairs[1]["rejected_adaptation"] == repairs[0]["rejected_adaptation"]
    assert chapter["sequences"][0]["adaptation"] == reviewed[-1]
    assert "review" not in chapter["sequences"][0]
    findings = chapter["narrative_review"]["cited_findings"]
    assert {row["field"] for row in findings if row["message"].startswith(nv.LANGUAGE_MISMATCH)} == set(nv.LANGUAGE_FIELDS)


def test_translated_editable_default_cannot_replace_valid_narration(monkeypatch):
    source = "Elle saisit la rampe."
    draft = {"source": source, "initial_state": "Elle longe le mur.",
             "events": [source], "final_state": "Elle tient la rampe."}
    translated = {"initialState": "She walks along the wall.", "event": "1. She grabs the railing.",
                  "endingState": "She holds the railing."}

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema["name"] == nv.SCHEMA["name"]:
            if "defects_to_recheck" in data:
                assert data["exported_adaptation"] == translated
                return {"issues": [], "narration_language": {"source": "fr", **dict.fromkeys(nv.LANGUAGE_FIELDS, ["en"])}}
            return {"issues": [{"target": 1, "message": "The staging needs clarification."}]}
        if schema["name"] == ca.narrative_choices.PROPOSAL_SCHEMA["name"]:
            return {"proposals": [{"reason": reason, "adaptation": translated} for reason in ("A", "B")]}
        return {"sequences": [draft]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    chapter = ca.adapt_chapter(None, "mock", source, chunk_chars=4000,
                               temperature=0.15, max_tokens=8192, correction_attempts=0)
    resolved = ca.narrative_choices.resolve_chapter(chapter)
    assert resolved["sequences"][0]["adaptation"]["event"] == f"1. {source}"
    assert resolved["sequences"][0]["review"]["status"] == "incomplete"
    assert any(row["check"] == "proposal_language" for row in chapter["narrative_review"]["warnings"])


@pytest.mark.parametrize("recovery", [False, True])
def test_translated_adaptation_is_reviewed_and_repaired(monkeypatch, recovery):
    source = 'Indy tire une torche de son sac. Il crie : « Help! »'
    normalized_source = ca._normalize_text(source)
    english = {"initial_state": "Indy carries a bag.",
               "events": ['Indy pulls out a torch and shouts: "Help!"'],
               "final_state": "Indy holds the torch."}
    french = {"initial_state": "Indy porte un sac.",
              "events": ["Indy tire une torche de son sac.", 'Il crie : « Help! »'],
              "final_state": "Indy tient la torche."}
    calls = []

    def chat(client, model, system, prompt, schema, *args):
        data, _ = json.JSONDecoder().raw_decode(prompt)
        calls.append(schema["name"])
        assert schema["strict"] is True
        if schema["name"] == nv.SCHEMA["name"]:
            assert "Report a language mismatch as a concrete" in system
            assert data["sequences"][0]["source"] == normalized_source
            return {"issues": [{"target": 1, "message": "Descriptions must remain in French."}]
                    if data["sequences"][0]["initial_state"] == english["initial_state"] else []}
        assert "LANGUAGE REQUIREMENT" in system
        assert "Never fall back to English" in system
        assert "when you support it" not in system
        assert data["current_passage"] == normalized_source
        if "rejected_adaptation" in data or calls.count(ca.SCHEMA["name"]) > 1:
            if "rejected_adaptation" in data:
                assert data["rejected_adaptation"]["events"] == english["events"]
                return {"sequences": [{"source": source, **deepcopy(french)}]}
            return {"sequences": [{"source": source, **deepcopy(english)}]}
        return {"sequences": [{"source": "Wrong copy." if recovery else source, **english}]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    result = ca.adapt_chapter(None, "gemma-4-26B_q4_0-it.gguf", source,
                             chunk_chars=4000, temperature=0.15, max_tokens=1000,
                             correction_attempts=1)
    assert result["sequences"][0]["source"] == normalized_source
    assert result["sequences"][0]["adaptation"] == {
        "initialState": french["initial_state"],
        "event": '1. Indy tire une torche de son sac.\n2. Il crie : " Help! "',
        "endingState": french["final_state"],
    }
    assert result["narrative_review"]["issues"] == []
    assert calls.count(nv.SCHEMA["name"]) == 2
    assert calls.count(ca.SCHEMA["name"]) == (3 if recovery else 2)

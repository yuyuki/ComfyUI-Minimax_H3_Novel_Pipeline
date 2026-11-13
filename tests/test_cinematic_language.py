"""Language instructions and correction routing, with no live model dependency."""
from copy import deepcopy
import json

import pytest

pytestmark = pytest.mark.usefixtures("mock_scene_planning")

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca
from minimax_h3_novel_pipeline.core import narrative_validation as nv


@pytest.mark.parametrize("recovery", [False, True])
def test_translated_adaptation_is_reviewed_and_repaired(monkeypatch, recovery):
    source = 'Indy tire une torche de son sac. Il crie : « Help! »'
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
        if schema == nv.SCHEMA:
            assert "Report a language mismatch as a concrete" in system
            assert data["sequences"][0]["source"] == source
            return {"issues": [{"target": 1, "message": "Descriptions must remain in French."}]
                    if data["sequences"][0]["initial_state"] == english["initial_state"] else []}
        assert "LANGUAGE REQUIREMENT" in system
        assert "Never fall back to English" in system
        assert "when you support it" not in system
        assert data["current_passage"] == source
        if "rejected_adaptation" in data or calls.count(ca.SCHEMA["name"]) > 2:
            if "rejected_adaptation" in data:
                assert data["rejected_adaptation"]["events"] == english["events"]
                return {"sequences": [{"source": source, **deepcopy(french)}]}
            return {"sequences": [{"source": source, **deepcopy(english)}]}
        return {"sequences": [{"source": "Wrong copy." if recovery else source, **english}]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = ca.adapt_chapter(None, "gemma-4-26B_q4_0-it.gguf", source,
                             chunk_chars=4000, temperature=0.15, max_tokens=1000,
                             correction_attempts=1)
    assert result["sequences"][0]["source"] == source
    assert result["sequences"][0]["adaptation"] == {
        "initialState": french["initial_state"],
        "event": '1. Indy tire une torche de son sac.\n2. Il crie : « Help! »',
        "endingState": french["final_state"],
    }
    assert result["narrative_review"] == {"issues": []}
    assert calls.count(nv.SCHEMA["name"]) == 2
    assert calls.count(ca.SCHEMA["name"]) == (4 if recovery else 2)

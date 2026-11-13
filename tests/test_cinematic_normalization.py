"""Typography equivalence and transactional recovery after cinematic review."""
from copy import deepcopy
import json

import pytest

pytestmark = pytest.mark.usefixtures("mock_scene_planning")

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca
from minimax_h3_novel_pipeline.core import narrative_choices as choices
from minimax_h3_novel_pipeline.core import narrative_validation as nv


def scene(source, state="Après."):
    return {"source": source, "initial_state": "Avant.", "events": ["Il attend."], "final_state": state}


def run(source, **kwargs):
    return ca.adapt_chapter(None, "mock", source, chunk_chars=4000, temperature=0.1,
                            max_tokens=1000, correction_attempts=kwargs.pop("correction_attempts", 1), **kwargs)


def test_normalization_preserves_content_and_paragraphs():
    original = "« L’e\u0301te\u0301\u00a0! »\r\n\r\n‘Oui’\u202f: A  B\rFin."
    expected = "« L'été ! »\n\n'Oui' : A  B\nFin."
    assert ca._normalize_text(original) == expected
    assert ca._normalize_text(expected) == expected


def test_unicode_quotes_and_line_separators_share_source_output_normalization():
    original = '“Oui”\u2028Il parle.\u2029« Non… » — Fin.'
    expected = '"Oui"\nIl parle.\n\n« Non… » — Fin.'
    assert ca._normalize_text(original) == expected
    assert ca._normalize_text(expected) == expected
    result = ca._validated_sequences({"sequences": [scene(original)]}, expected)
    assert result[0]["source"] == expected
    assert ca._normalized_output({"states": [original]}) == {"states": [expected]}


def test_unicode_paragraph_suffix_advances_normalized_cursor():
    source = ca._normalize_text('“Oui”.\u2029Il part.')
    sequences, consumed = ca._validated_window({
        "sequences": [scene('“Oui”.')], "unprocessed_source": '\u2029Il part.',
    }, source)
    assert consumed == len('"Oui".')
    assert "".join(s["source"] for s in sequences) + source[consumed:] == source


def test_source_requests_and_generated_descriptions_share_normalization(monkeypatch):
    original = "L’e\u0301te\u0301\u00a0!\r\n\r\nIl s’arrête."
    expected = "L'été !\n\nIl s'arrête."
    calls = []

    def chat(client, model, system, prompt, schema, *args):
        calls.append(schema["name"])
        data = json.loads(prompt)
        if schema == nv.SCHEMA:
            assert "".join(s["source"] for s in data["sequences"]) == expected
            assert data["sequences"][0]["initial_state"] == "L'été."
            return {"issues": []}
        assert schema == ca.SCHEMA and schema["strict"]
        assert data["current_passage"] == expected
        item = scene(original, "Il s’arrête.\r\nFin.")
        item.update(initial_state="L’e\u0301te\u0301.", events=["Il s’arrête\u202f!"])
        return {"sequences": [item]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run(original)
    assert calls == [ca.SCHEMA["name"], nv.SCHEMA["name"]]
    assert result["sequences"][0] == {
        "sequence": 1, "source": expected,
        "adaptation": {"initialState": "L'été.", "event": "1. Il s'arrête !", "endingState": "Il s'arrête.\nFin."},
    }


def test_normalized_deferred_suffix_uses_normalized_cursor():
    source = "L'été.\n\nIl s'arrête."
    sequences, consumed = ca._validated_window({
        "sequences": [scene("L’e\u0301te\u0301.")],
        "unprocessed_source": "\r\n\r\nIl s’arrête.",
    }, source)
    assert consumed == len("L'été.")
    assert "".join(s["source"] for s in sequences) + source[consumed:] == source


@pytest.mark.parametrize("excerpt", [
    "Il part.", "Il attend. Il attend.", "Il attend", "il attend.", "Il  attend.",
])
def test_normalization_still_rejects_content_changes(excerpt):
    with pytest.raises(ca._SourceCoverageError):
        ca._validated_sequences({"sequences": [scene(excerpt)]}, "Il attend.")


def test_retry_receives_first_difference_and_bounded_context(monkeypatch):
    prompts = []

    def chat(client, model, system, prompt, schema, *args):
        if schema == nv.SCHEMA:
            return {"issues": []}
        prompts.append(prompt)
        return {"sequences": [scene("Il part." if len(prompts) == 1 else "Il parle.")]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run("Il parle.")
    assert result["sequences"][0]["source"] == "Il parle."
    assert "passage character 7 (sequence starts at 1)" in prompts[1]
    assert 'Expected context: "Il parle."' in prompts[1]
    assert 'Received context: "Il part."' in prompts[1]
    with pytest.raises(ca._SourceCoverageError) as error:
        ca._validated_sequences({"sequences": [scene("A" * 500 + "X")]}, "A" * 500 + "Y")
    assert len(error.value.feedback) < 500


@pytest.mark.parametrize("targeted_valid", [True, False])
@pytest.mark.parametrize("failure", ["coverage", "fields", "length"])
def test_failed_local_repairs_retain_valid_work_and_resolve_issues(monkeypatch, targeted_valid, failure):
    fields = [scene("Il attend.", "Il attend encore."), scene(" Il part.")]
    repairs, full_rebuilds, proposals = [], [], []

    def chat(client, model, system, prompt, schema, *args):
        data, _ = json.JSONDecoder().raw_decode(prompt)
        assert schema["strict"]
        if schema == nv.SCHEMA:
            return {"issues": [{"target": i + 1, "message": "Vérifier la posture."}
                               for i, item in enumerate(data["sequences"]) if item["source"].strip() == "Il part."]}
        if schema == choices.PROPOSAL_SCHEMA:
            proposals.append(data)
            return {"proposals": [{"reason": reason, "adaptation": {
                "initialState": "Il s’apprête à partir.", "event": "Il part.", "endingState": "Il est sorti.",
            }} for reason in ("Départ immédiat.", "Départ après une pause.")]}
        if "rejected_adaptation" not in data:
            return {"sequences": deepcopy(fields)}
        full = isinstance(data["rejected_adaptation"], list)
        (full_rebuilds if full else repairs).append(data)
        assert schema == ca.SCHEMA
        if not full and targeted_valid:
            return {"sequences": [scene(data["current_passage"], "Réparé.")]}
        if failure == "length":
            raise ca.lmstudio_json.StructuredOutputLengthError("Output limit")
        if failure == "fields":
            return {"sequences": [{"source": data["current_passage"]}]}
        return {"sequences": [scene("Texte inventé.")]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run("Il attend. Il part.")
    assert len(repairs) == (2 if targeted_valid or failure == "length" else 4)
    assert not full_rebuilds
    assert len(proposals) == 1
    assert proposals[0]["current_passage"] == " Il part."
    assert proposals[0]["previous_final_state"] == "Il attend encore."
    assert all(p["current_passage"] == " Il part." for p in repairs)
    assert "".join(s["source"] for s in result["sequences"]) == "Il attend. Il part."
    assert result["sequences"][0]["adaptation"]["endingState"] == "Il attend encore."
    assert result["sequences"][1]["adaptation"]["initialState"] == "Il s'apprête à partir."
    assert result["narrative_review"]["issues"][0]["sequence"] == 2


@pytest.mark.parametrize("error", [ValueError("invalid client configuration"), RuntimeError("network failure"), KeyboardInterrupt()])
def test_repair_request_errors_are_not_hidden(monkeypatch, error):
    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema == nv.SCHEMA:
            return {"issues": [{"target": 1, "message": "Vérifier la posture."}]}
        if "rejected_adaptation" in data:
            raise error
        return {"sequences": [scene("Il attend.")]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    if isinstance(error, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            run("Il attend.")
    else:
        result = run("Il attend.")
        assert result["sequences"][0]["review"]["status"] == "incomplete"
        assert result["narrative_review"]["warnings"][0]["check"] == "scene_repair"

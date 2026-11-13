"""Lossless source-only recovery through strict, ordered segment references."""
import json
from copy import deepcopy

import pytest

pytestmark = pytest.mark.usefixtures("mock_scene_planning")
from jsonschema import validate

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca


def scene(source):
    return {"source": source, "initial_state": "Avant.", "events": ["Il parle."], "final_state": "Après."}


@pytest.mark.parametrize("source", ["  Il parle.\n\nElle répond.  ", "漢字。次。", "A\tB\nC", " "])
def test_segments_reconstruct_every_character(source):
    assert "".join(ca._source_segments(source)) == source


@pytest.mark.parametrize("ends", [[0, 4], [2, 2], [3, 2], [2, 5], [True, 4], [2], [2, 3]])
def test_invalid_boundaries_cannot_drop_repeat_or_skip_source(ends):
    with pytest.raises(ValueError):
        ca._restore_source_spans({"source_segment_ends": ends}, [scene("X"), scene("Y")],
                                 ca._source_segments("Il parle. Elle répond."), partial=False)


def test_partial_repair_defers_exact_suffix_and_preserves_fields():
    draft = [scene("Wrong.")]
    original = deepcopy(draft)
    source = "Il parle.\n\nElle répond."
    restored, consumed = ca._restore_source_spans({"source_segment_ends": [2]}, draft,
                                                 ca._source_segments(source), partial=True)
    assert restored[0]["source"] + source[consumed:] == source
    assert source[consumed:] == "Elle répond."
    assert restored[0]["events"] == draft[0]["events"]
    assert draft == original


def test_repair_retry_sends_strict_schema_and_all_scenes_then_reviews(monkeypatch):
    source = "Il parle.\n\nElle répond."
    draft = [scene("Il parlait."), scene("Elle répondait.")]
    repairs, reviews = [], []

    def chat(client, model, system, prompt, schema, temperature, max_tokens):
        payload, _ = json.JSONDecoder().raw_decode(prompt)
        if schema["name"] == "cinematic_source_spans":
            repairs.append(payload)
            assert schema["strict"] and temperature == 0.0
            assert payload["sequences"] == draft
            assert "".join(s["text"] for s in payload["source_segments"]) == source
            result = {"source_segment_ends": [2, 2] if len(repairs) == 1 else [2, 4]}
            validate(result, schema["schema"])
            return result
        return {"sequences": deepcopy(draft)}

    def review(request, client, model, payload, count, max_tokens):
        reviews.extend(payload["sequences"])
        return {}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    monkeypatch.setattr(ca.narrative_validation, "review", review)
    result = ca.adapt_chapter(None, "mock", source, chunk_chars=4000, temperature=0.15,
                              max_tokens=1000, correction_attempts=1)
    assert len(repairs) == 2
    assert "".join(s["source"] for s in result["sequences"]) == source
    assert len(reviews) == 2
    assert [s["events"] for s in reviews] == [s["events"] for s in draft]


def test_final_coverage_error_keeps_expected_and_received_context(monkeypatch):
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: {"sequences": [scene("Il part.")]})
    with pytest.raises(ValueError) as error:
        ca.adapt_chapter(None, "mock", "Il parle.", chunk_chars=4000, temperature=0.0,
                         max_tokens=1000, correction_attempts=0)
    assert 'Expected context: "Il parle."' in str(error.value)
    assert 'Received context: "Il part."' in str(error.value)


@pytest.mark.parametrize("error", [ValueError("invalid client configuration"), RuntimeError("network failure"), KeyboardInterrupt()])
def test_span_repair_request_errors_propagate_without_rebuilding(monkeypatch, error):
    calls = []

    def chat(*args):
        calls.append(args[4]["name"])
        if calls[-1] == "cinematic_source_spans":
            raise error
        return {"sequences": [scene("Wrong.")]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(type(error), match=str(error) or None):
        ca.adapt_chapter(None, "mock", "Il parle.", chunk_chars=4000, temperature=0.0,
                         max_tokens=1000, correction_attempts=0)
    assert calls == [ca.SCHEMA["name"], "cinematic_source_spans"]

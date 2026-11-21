"""Source boundaries are selected in reading and correction requests."""
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline.core import cinematic_adaptation as adaptation

pytestmark = pytest.mark.usefixtures("mock_scene_planning")


def scene(end_id, event="Elle attend."):
    return {"source_end_id": end_id, "initial_state": "Avant.", "events": [event], "final_state": "Après."}


@pytest.mark.parametrize("ends", [[0], [4], [True], [1], [2, 2], [2, 1]])
def test_invalid_boundaries_cannot_drop_repeat_or_skip_source(ends):
    with pytest.raises(adaptation._SourceCoverageError):
        adaptation._restore_source_spans({"sequences": [scene(end) for end in ends]},
                                         adaptation._source_segments("A. B. C."), partial=False)


def test_window_defers_untouched_exact_suffix():
    text = "  Elle attend.\n\n Elle entre.  "
    sequences, consumed = adaptation._restore_source_spans(
        {"sequences": [scene(2)]}, adaptation._source_segments(text), partial=True)
    assert sequences[0]["source"] == "  Elle attend."
    assert text[consumed:] == "\n\n Elle entre.  "


def test_boundary_retry_keeps_structured_schema_and_full_source(monkeypatch):
    text = "Élise attend.  Élise parle."
    requests = []

    def chat(client, model, system, prompt, schema, *args):
        data, _ = json.JSONDecoder().raw_decode(prompt)
        if schema["name"] == adaptation.narrative_validation.SCHEMA["name"]:
            return {"issues": []}
        assert schema == adaptation.SCHEMA
        requests.append((deepcopy(data), prompt))
        return {"sequences": [scene(2), scene(99 if len(requests) == 1 else 4, "Élise parle.")]}

    monkeypatch.setattr(adaptation.lmstudio_json, "chat_json", chat)
    accepted = adaptation.adapt_chapter(None, "mock", text, chunk_chars=4000, temperature=0.15,
                                       max_tokens=8192, correction_attempts=1)
    assert len(requests) == 2
    assert requests[0][0] == requests[1][0]
    assert "Previous response rejected" in requests[1][1]
    assert "".join(row["source"] for row in accepted["sequences"]) == text

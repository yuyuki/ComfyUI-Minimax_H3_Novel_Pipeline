"""Incremental source consumption and bounded review recovery, without a server."""
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca
from minimax_h3_novel_pipeline.core import narrative_validation as nv


def scene(source, state="Après."):
    return {"source": source, "initial_state": "Avant.", "events": ["Il attend."], "final_state": state}


def run(source, **kwargs):
    return ca.adapt_chapter(None, "mock", source, temperature=0.1, max_tokens=1000,
                            correction_attempts=kwargs.pop("correction_attempts", 0), **kwargs)


def test_partial_scene_is_read_again_with_new_source_and_only_committed_state(monkeypatch, capsys):
    source = "Il attend. Il saisit la poignée. Il ouvre. Il part."
    windows = []

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema == nv.SCHEMA:
            return {"issues": []}
        windows.append(data)
        passage = data["current_passage"]
        if len(windows) == 1:
            assert schema == ca.WINDOW_SCHEMA and schema["strict"]
            assert "ENTIRE source span" in system
            cut = passage.index("Il saisit")
            return {"sequences": [scene(passage[:cut], "Il attend toujours.")],
                    "unprocessed_source": passage[cut:]}
        result = {"sequences": [scene(passage)]}
        if schema == ca.WINDOW_SCHEMA:
            result["unprocessed_source"] = ""
        return result

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run(source, chunk_chars=35)
    assert windows[1]["current_passage"].startswith("Il saisit la poignée. Il ouvre.")
    assert windows[1]["previous_final_state"] == "Il attend toujours."
    assert all(len(window["current_passage"]) <= 35 for window in windows)
    assert "".join(item["source"] for item in result["sequences"]) == source
    assert [s["sequence"] for s in result["sequences"]] == list(range(1, len(result["sequences"]) + 1))
    output = capsys.readouterr().out
    assert f"committed={len(source)}/{len(source)}" in output
    assert "prompt_chars=" in output and "schema_chars=" in output


@pytest.mark.parametrize("tail,sequences", [
    ("invented", [scene("A.")]),
    (" B.", [scene("A. B.")]),  # Overlap with the deferred suffix.
    ("", [scene("A.")]),        # Missing source is never accepted as progress.
    (" B.", [scene("B.")]),     # A later excerpt cannot skip earlier source.
])
def test_window_rejects_invented_missing_or_overlapping_source(tail, sequences):
    with pytest.raises(ca._SourceCoverageError):
        ca._validated_window({"sequences": sequences, "unprocessed_source": tail}, "A. B.")


def test_window_preserves_whitespace_and_exact_cursor():
    source = "  A.\n\n B. C."
    sequences, consumed = ca._validated_window(
        {"sequences": [scene("A.")], "unprocessed_source": "\n\n B. C."}, source)
    assert consumed == 4
    assert sequences[0]["source"] == "  A."
    assert "".join(s["source"] for s in sequences) + source[consumed:] == source


def test_no_complete_scene_stops_without_looping_or_shortening_context(monkeypatch):
    calls = []

    def chat(*args):
        data, _ = json.JSONDecoder().raw_decode(args[3])
        calls.append(data)
        return {"sequences": [], "unprocessed_source": data["current_passage"]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(ValueError, match="No complete scene.*increase chunk_chars"):
        run("Il attend. " * 500, chunk_chars=4000, correction_attempts=1)
    assert len(calls) == 2 and calls[0] == calls[1]


def test_output_limited_adaptation_retries_unread_source_with_smaller_windows(monkeypatch):
    source = "Il attend. " * 300
    passages = []

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema == nv.SCHEMA:
            return {"issues": []}
        passage = data["current_passage"]
        passages.append(data)
        if len(passage) > 2000:
            raise ca.lmstudio_json.StructuredOutputLengthError("Thinking exhausted the output budget")
        result = {"sequences": [scene(passage)]}
        if schema == ca.WINDOW_SCHEMA:
            result["unprocessed_source"] = ""
        return result

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run(source, chunk_chars=4000)
    assert source.startswith(passages[1]["current_passage"])
    assert passages[1]["previous_final_state"] == ""
    assert len(passages[1]["current_passage"]) < len(passages[0]["current_passage"])
    assert "".join(s["source"] for s in result["sequences"]) == source


@pytest.mark.parametrize("oversized", [False, True])
def test_review_batches_preserve_coverage_target_indices_and_boundary_context(monkeypatch, oversized):
    sequences = [scene(f"Scène {index}. ", f"État {index}.") for index in range(4)]
    if oversized:
        for sequence in sequences:
            sequence["events"] = ["Action. " * 200]
    reviews, repairs = [], []

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema == nv.SCHEMA:
            reviews.append(data)
            batch = data["sequences"]
            if len(batch) > 1:
                raise ca.lmstudio_json.StructuredOutputLengthError("Output limit")
            index = int(batch[0]["source"].strip().split()[1].rstrip("."))
            assert data["previous_final_state"] in (
                sequences[index - 1]["final_state"] if index else "", "Corrigé." if index == 3 else "")
            if index:
                assert data["previous_sequence"]["source"].strip() == sequences[index - 1]["source"].strip()
            if index < 3:
                assert data["next_sequence"]["source"].strip() == sequences[index + 1]["source"].strip()
            return {"issues": [{"target": 1, "message": "Corriger la scène."}]
                    if batch[0]["final_state"] == "État 2." else []}
        if "rejected_adaptation" in data:
            repairs.append(data)
            assert data["target_sequence"] == 3
            return {"sequences": [scene(data["current_passage"], "Corrigé.")]}
        return {"sequences": deepcopy(sequences)}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run("".join(s["source"] for s in sequences), chunk_chars=4000, correction_attempts=1)
    assert len(repairs) == 1
    if oversized:
        assert len(reviews[0]["sequences"]) == 1  # Split before the first network call.
    assert result["sequences"][2]["adaptation"]["endingState"] == "Corrigé."
    assert result["narrative_review"] == {"issues": []}
    assert {r["sequences"][0]["source"].strip() for r in reviews if len(r["sequences"]) == 1} == {
        s["source"].strip() for s in sequences}


def test_rebuilt_passage_review_length_failure_is_split_and_rechecked(monkeypatch):
    source = "Il attend. Il part."
    fields = [scene("Il attend."), scene(" Il part.")]
    full_reviews = 0
    singleton_reviews = []

    def chat(client, model, system, prompt, schema, *args):
        nonlocal full_reviews
        data = json.loads(prompt)
        if schema != nv.SCHEMA:
            return {"sequences": deepcopy(fields if isinstance(data.get("rejected_adaptation"), list)
                                           or "rejected_adaptation" not in data else [scene(data["current_passage"])])}
        if len(data["sequences"]) == 2:
            full_reviews += 1
            if full_reviews <= 2:
                return {"issues": [{"target": 1, "message": "Vérifier la posture."}]}
            raise ca.lmstudio_json.StructuredOutputLengthError("Thinking exhausted the output budget")
        singleton_reviews.append(data)
        return {"issues": []}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run(source, chunk_chars=4000, correction_attempts=1)
    assert full_reviews == 3 and len(singleton_reviews) == 2
    assert "".join(s["source"] for s in result["sequences"]) == source
    assert result["narrative_review"] == {"issues": []}


def test_single_scene_review_exhaustion_remains_explicit_and_bounded(monkeypatch):
    calls = []

    def chat(client, model, system, prompt, schema, *args):
        calls.append(schema["name"])
        if schema == nv.SCHEMA:
            raise ca.lmstudio_json.StructuredOutputLengthError("Output limit")
        return {"sequences": [scene("Il attend.")]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(ca.lmstudio_json.StructuredOutputLengthError, match="sequence 1.*Disable thinking"):
        run("Il attend.", chunk_chars=4000)
    assert calls == [ca.SCHEMA["name"], nv.SCHEMA["name"]]

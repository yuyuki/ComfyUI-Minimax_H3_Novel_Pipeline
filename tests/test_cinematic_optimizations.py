"""Reduced request counts without skipping changed-context review or source."""
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca, narrative_validation as nv


def scene(source, state="Before."):
    return {"source": source, "initial_state": "Before.", "events": ["Action."], "final_state": state}


def run(source, **kwargs):
    return ca.adapt_chapter(None, "mock", source, chunk_chars=4000, temperature=0.1,
                            max_tokens=2000, correction_attempts=kwargs.pop("correction_attempts", 1), **kwargs)


@pytest.mark.parametrize("budget,expected", [(32768, 32768), (2000, 2000)])
def test_review_budget_is_scoped_and_previous_scene_crosses_windows(monkeypatch, budget, expected):
    source = "Il attend. Il part."
    reviews = []

    def chat(client, model, system, prompt, schema, temperature, max_tokens):
        data = json.loads(prompt)
        if schema == nv.SCHEMA:
            assert max_tokens == expected
            assert ca.lmstudio_json.REQUEST_OUTPUT_LIMIT.get() == expected
            assert "at most 8 seconds" in system
            reviews.append(data)
            return {"issues": []}
        assert max_tokens == budget
        assert ca.lmstudio_json.REQUEST_OUTPUT_LIMIT.get() == budget
        result = {"sequences": [scene(data["current_passage"])]}
        if schema == ca.WINDOW_SCHEMA:
            result["unprocessed_source"] = ""
        return result

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = ca.adapt_chapter(None, "mock", source, chunk_chars=11, temperature=0.1,
                              max_tokens=budget, correction_attempts=0)
    assert len(reviews) == 2
    assert "previous_sequence" not in reviews[0]
    assert reviews[1]["previous_sequence"]["source"] == result["sequences"][0]["source"]
    assert all(len(r["sequences"]) == 1 and "next_sequence" not in r for r in reviews)
    assert ca.lmstudio_json.REQUEST_OUTPUT_LIMIT.get() is None


def test_local_correction_rechecks_only_changed_scene_and_successor(monkeypatch):
    fields = [scene(f"Scene {i}. ") for i in range(7)]
    fields = ca._validated_sequences({"sequences": fields}, "".join(s["source"] for s in fields))
    fields = ca._validated_sequences({"sequences": fields}, "".join(s["source"] for s in fields))
    fields[3]["final_state"] = "Wrong."
    reviews, repairs = [], []

    def chat(client, model, system, prompt, schema, *args):
        assert schema["strict"] is True
        data = json.loads(prompt)
        if schema == nv.SCHEMA:
            reviews.append(deepcopy(data))
            return {"issues": [{"target": i + 1, "message": "Wrong ending."}
                               for i, s in enumerate(data["sequences"]) if s["final_state"] == "Wrong."]}
        if "rejected_adaptation" in data:
            repairs.append(data)
            assert data["current_passage"] == fields[3]["source"]
            assert data["previous_sequence"]["source"] == fields[2]["source"]
            assert data["previous_sequence"]["final_state"] == fields[2]["final_state"]
            assert "neighboring_sequences" not in data
            assert "next_sequence" not in data
            return {"sequences": [scene(data["current_passage"], "Correct.")]}
        return {"sequences": deepcopy(fields)}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    result = run("".join(s["source"] for s in fields))
    assert len(repairs) == 1
    assert [len(r["sequences"]) for r in reviews] == [1] * 8
    assert [r["sequences"][0]["source"] for r in reviews] == [fields[i]["source"] for i in (0, 1, 2, 3, 3, 4, 5, 6)]
    assert all("next_sequence" not in r for r in reviews)
    assert result["sequences"][3]["adaptation"]["endingState"] == "Correct."
    assert "".join(s["source"] for s in result["sequences"]) == "".join(s["source"] for s in fields)


@pytest.mark.parametrize("change_context", [False, True])
def test_resume_reuses_only_reviews_with_identical_context(monkeypatch, change_context):
    fields = [scene(f"Scene {i}. ") for i in range(4)]
    fields = ca._validated_sequences({"sequences": fields}, "".join(s["source"] for s in fields))
    fields = ca._validated_sequences({"sequences": fields}, "".join(s["source"] for s in fields))
    snapshots = []

    def chat(client, model, system, prompt, schema, *args):
        if schema == nv.SCHEMA:
            return {"issues": [{"target": 1, "message": "Check scene four."}]
                    if json.loads(prompt)["sequences"][0]["source"] == fields[3]["source"] else []}
        return {"sequences": deepcopy(fields)}

    def stop_after_review(state):
        if state["stage"] == "reviewed" and len(state["sequences"]) == 2:
            snapshots.append(deepcopy(state))
            raise RuntimeError("interrupted")

    source = "".join(s["source"] for s in fields)
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(source, on_checkpoint=stop_after_review)
    saved = snapshots[-1]
    assert len(saved["approved_reviews"]) == 3
    if change_context:
        saved["sequences"][-1]["adaptation"]["endingState"] = "Changed."
    reviews = []

    def review(chat, client, model, payload, count, tokens):
        reviews.append(deepcopy(payload))
        return {}

    monkeypatch.setattr(ca.narrative_validation, "review", review)
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: pytest.fail("No generation expected"))
    result = run(source, resume_state=saved)
    assert [s["source"] for r in reviews for s in r["sequences"]] == [s["source"] for s in fields[2 if change_context else 3:]]
    assert len(result["sequences"]) == 4


@pytest.mark.parametrize("failure", ["coverage", "length"])
def test_learned_window_avoids_repeated_large_failures_and_survives_resume(monkeypatch, failure):
    source = "Il attend. " * 1000
    sizes, checkpoints = [], []
    accepted = 0

    def chat(client, model, system, prompt, schema, *args):
        nonlocal accepted
        assert schema["strict"]
        if schema == nv.SCHEMA:
            return {"issues": []}
        payload, _ = json.JSONDecoder().raw_decode(prompt)
        passage = payload["current_passage"]
        sizes.append(len(passage))
        if len(passage) > 2000:
            if failure == "length":
                raise ca.lmstudio_json.StructuredOutputLengthError("Output limit")
            result = {"sequences": [scene("Wrong source.")]}
        else:
            accepted += 1
            result = {"sequences": [scene(passage)]}
        if schema == ca.WINDOW_SCHEMA:
            result["unprocessed_source"] = ""
        return result

    def checkpoint(state):
        if state["stage"] == "passage_committed":
            checkpoints.append(deepcopy(state))
            if accepted == 1:
                raise RuntimeError("restart here")

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    with pytest.raises(RuntimeError, match="restart here"):
        run(source, correction_attempts=5, on_checkpoint=checkpoint)
    assert sum(n > 2000 for n in sizes) == (2 if failure == "coverage" else 1)
    saved = checkpoints[-1]
    assert saved["window_chars"] <= 2000
    sizes.clear()
    result = run(source, correction_attempts=5, resume_state=saved)
    assert sizes and max(sizes) <= 2000
    assert "".join(s["source"] for s in result["sequences"]) == source

"""Persist real adapter drafts and accepted scenes across cancellation/failure."""
from contextlib import nullcontext
import json

import pytest

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca, path_access
from minimax_h3_novel_pipeline.nodes import cinematic_chapter_adapter as adapter


def scene(source, state="After."):
    return {"source": source, "initial_state": "Before.", "events": ["Action."], "final_state": state}


@pytest.fixture
def setup_node(tmp_path, monkeypatch):
    source = "Il attend. " * 20
    chapter = tmp_path / "chapter.md"
    chapter.write_text(source, encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(adapter, "stage_output", lambda *args: output)
    monkeypatch.setattr(adapter.lmstudio_pipeline, "make_client_and_model", lambda *args: (nullcontext(), "mock"))

    def run(**kwargs):
        return adapter.CinematicChapterAdapterNode().run(
            {"api_url": "unused", "run_folder": "test"}, {"chapter_paths": [str(chapter)]}, **kwargs)

    return run, output, source.strip()


def read_state(output):
    return json.loads((output / ".cache/001_chapter/progress.json").read_text(encoding="utf-8"))


def test_draft_persisted_before_review_cancellation(setup_node, monkeypatch):
    run, output, source = setup_node
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: {"sequences": [scene(source)]})

    def review(*args):
        state = read_state(output)
        assert state["stage"] == "source_validated_pending_review"
        assert state["pending_sequences"][0]["source"] == source
        assert state["sequences"] == []
        assert not list(output.glob("*.cinematic.json"))
        raise KeyboardInterrupt()

    monkeypatch.setattr(ca.narrative_validation, "review", review)
    with pytest.raises(KeyboardInterrupt):
        run()
    assert read_state(output)["pending_sequences"]
    assert not list(output.rglob("*.tmp"))


def test_correction_persisted_before_next_review_fails(setup_node, monkeypatch):
    run, output, source = setup_node
    calls = 0

    def chat(*args):
        nonlocal calls
        calls += 1
        return {"sequences": [scene(source, "Corrected." if calls > 1 else "After.")]}

    def review(*args):
        if calls == 1:
            return {1: ["Correct the ending."]}
        state = read_state(output)
        assert state["stage"] == "corrected_pending_review"
        assert state["pending_sequences"][0]["final_state"] == "Corrected."
        raise RuntimeError("review unavailable")

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    monkeypatch.setattr(ca.narrative_validation, "review", review)
    with pytest.raises(RuntimeError, match="review unavailable"):
        run(correction_attempts=1)


@pytest.mark.parametrize("fail_next_window", [False, True])
def test_accepted_scene_files_survive_later_failure(setup_node, monkeypatch, fail_next_window):
    run, output, source = setup_node
    calls = 0

    def chat(client, model, system, prompt, schema, *args):
        nonlocal calls
        calls += 1
        if fail_next_window and calls > 1:
            assert (output / ".cache/001_chapter/scene_001.json").is_file()
            raise RuntimeError("next window failed")
        passage = json.loads(prompt)["current_passage"]
        result = {"sequences": [scene(passage)]}
        if schema == ca.WINDOW_SCHEMA:
            result["unprocessed_source"] = ""
        return result

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    monkeypatch.setattr(ca.narrative_validation, "review", lambda *args: {})
    if fail_next_window:
        with pytest.raises(RuntimeError, match="next window failed"):
            run(chunk_chars=110)
    else:
        run()
        final = json.loads((output / "001_chapter.cinematic.json").read_text(encoding="utf-8"))
        assert final["sequences"] == read_state(output)["sequences"]
        assert "".join(s["source"] for s in final["sequences"]) == source
    state = read_state(output)
    saved = json.loads((output / ".cache/001_chapter/scene_001.json").read_text(encoding="utf-8"))
    assert saved == state["sequences"][0]
    assert saved["adaptation"]["endingState"] == "After."
    assert not state["pending_sequences"]
    assert not list(output.rglob("*.tmp"))

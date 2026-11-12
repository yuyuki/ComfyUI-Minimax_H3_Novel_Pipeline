"""Resume from a previous timestamped run without losing review or source coverage."""
from contextlib import nullcontext
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca, path_access
from minimax_h3_novel_pipeline.nodes import cinematic_chapter_adapter as adapter


OLD = "20261112124902"
NEW = "20261112140000"


def scene(source):
    return {"source": source, "initial_state": "Before.", "events": ["Action."], "final_state": "After."}


@pytest.fixture
def node(tmp_path, monkeypatch):
    source = "Il attend. " * 20
    chapter = tmp_path / "chapter.md"
    chapter.write_text(source, encoding="utf-8")
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(adapter.lmstudio_pipeline, "make_client_and_model", lambda *args: (nullcontext(), "mock"))
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: {"sequences": [scene(source.strip())]})
    monkeypatch.setattr(ca.narrative_validation, "review", lambda *args: {})

    def run(stamp=OLD, **kwargs):
        return adapter.CinematicChapterAdapterNode().run(
            {"api_url": "unused", "run_folder": stamp}, {"chapter_paths": [str(chapter)]}, **kwargs)

    return run, tmp_path, chapter


def cache(root, stamp=OLD):
    return root / stamp / "cinematic_chapters/.cache/001_chapter/progress.json"


def test_resume_pending_draft_then_completed_chapter_without_generation(node, monkeypatch):
    run, root, chapter = node

    def interrupted(*args):
        raise RuntimeError("review interrupted")

    monkeypatch.setattr(ca.narrative_validation, "review", interrupted)
    with pytest.raises(RuntimeError, match="review interrupted"):
        run()
    previous = cache(root).read_bytes()
    reviews = []

    def review(*args):
        reviews.append(args[3])
        return {}

    def no_generation(*args):
        pytest.fail("Saved source-validated draft must not be regenerated")

    monkeypatch.setattr(ca.narrative_validation, "review", review)
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", no_generation)
    result = run(NEW, resume_run=OLD)["result"][0][0]
    assert len(reviews) == 1
    assert "".join(s["source"] for s in result["sequences"]) == chapter.read_text().strip()
    assert cache(root).read_bytes() == previous
    assert cache(root, NEW).with_name("scene_001.json").is_file()
    reviews.clear()
    again = run("20261112150000", resume_run=NEW)["result"][0][0]
    assert again["sequences"] == result["sequences"]
    assert not reviews
    snapshot = json.loads((root / NEW / "cinematic_chapters/cinematic_adapter_configuration.json").read_text())
    assert snapshot["node_settings"]["resume_run"] == OLD


@pytest.mark.parametrize("change", ["settings", "source", "model", "corrupt_source"])
def test_resume_rejects_changed_inputs_and_invalid_coverage(node, monkeypatch, change):
    run, root, chapter = node
    run()
    kwargs = {}
    if change == "settings":
        kwargs["duration"] = 10
    elif change == "source":
        chapter.write_text("Autre texte. " * 20, encoding="utf-8")
    elif change == "model":
        monkeypatch.setattr(adapter.lmstudio_pipeline, "make_client_and_model", lambda *args: (nullcontext(), "different"))
    else:
        state = json.loads(cache(root).read_text())
        state["sequences"][0]["source"] = "Wrong source."
        cache(root).write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match|Invalid cinematic resume"):
        run(NEW, resume_run=OLD, **kwargs)


@pytest.mark.parametrize("resume", [OLD, "20261111111111", "../escape", 42])
def test_invalid_resume_folder_is_rejected(node, resume):
    run, _, _ = node
    with pytest.raises(ValueError, match="previous run|does not exist|Invalid run folder|timestamped"):
        run(resume_run=resume)


def test_core_resume_after_one_scene_committed_keeps_remaining_scene_and_choices(monkeypatch):
    source = "Il attend. Il part."
    snapshots = []

    def checkpoint(state):
        snapshots.append(deepcopy(state))
        if state["stage"] == "scene_committed":
            raise RuntimeError("interrupted after scene one")

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: {
        "sequences": [scene("Il attend."), scene(" Il part.")]})
    monkeypatch.setattr(ca.narrative_validation, "review", lambda *args: {})
    kwargs = dict(chunk_chars=4000, temperature=0.15, max_tokens=1000, correction_attempts=0)
    with pytest.raises(RuntimeError, match="interrupted after scene"):
        ca.adapt_chapter(None, "mock", source, on_checkpoint=checkpoint, **kwargs)
    saved = snapshots[-1]
    assert len(saved["sequences"]) == len(saved["pending_sequences"]) == 1
    # A committed scene can precede a scene resolved through an explicit proposal.
    saved["narrative_review"]["issues"] = [{"sequence": 2, "description": "Chosen proposal."}]
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: pytest.fail("No generation expected"))
    monkeypatch.setattr(ca.narrative_validation, "review", lambda *args: pytest.fail("Already accepted window"))
    result = ca.adapt_chapter(None, "mock", source, resume_state=saved, **kwargs)
    assert [s["sequence"] for s in result["sequences"]] == [1, 2]
    assert "".join(s["source"] for s in result["sequences"]) == source
    assert result["narrative_review"] == saved["narrative_review"]

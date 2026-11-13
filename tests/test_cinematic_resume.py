"""Resume from a previous timestamped run without losing review or source coverage."""
from contextlib import nullcontext
from copy import deepcopy
import json

import pytest

pytestmark = pytest.mark.usefixtures("mock_scene_planning")

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
        raise KeyboardInterrupt()

    monkeypatch.setattr(ca.narrative_validation, "review", interrupted)
    with pytest.raises(KeyboardInterrupt):
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
    assert saved["source_cursor"] == len("Il attend.")
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: pytest.fail("No generation expected"))
    reviews = []
    monkeypatch.setattr(ca.narrative_validation, "review", lambda *args: reviews.append(args[3]) or {})
    result = ca.adapt_chapter(None, "mock", source, resume_state=saved, **kwargs)
    assert [s["sequence"] for s in result["sequences"]] == [1, 2]
    assert "".join(s["source"] for s in result["sequences"]) == source
    assert result["narrative_review"] == saved["narrative_review"]
    assert [s["source"] for review in reviews for s in review["sequences"]] == [" Il part."]


@pytest.mark.parametrize("length_error", [False, True])
def test_third_scene_failure_preserves_files_and_resumes_only_pending_scenes(node, monkeypatch, capsys, length_error):
    run, root, chapter = node
    source = chapter.read_text().strip()
    fields = [scene(source[start:end]) for start, end in ((0, 55), (55, 110), (110, 165), (165, len(source)))]
    fields = ca._validated_sequences({"sequences": fields}, source)
    for index, item in enumerate(fields):
        item["final_state"] = f"After scene {index + 1}."
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: {"sequences": deepcopy(fields)})
    reviewed = []

    def review(chat, client, model, payload, count, tokens, **kwargs):
        item = payload["sequences"][0]
        reviewed.append(item["source"])
        if item["final_state"] == fields[2]["final_state"]:
            state = json.loads(cache(root).read_text())
            assert len(state["sequences"]) == len(state["pending_sequences"]) == 2
            assert state["source_cursor"] == len("".join(s["source"] for s in fields[:2]))
            assert cache(root).with_name("scene_001.json").is_file()
            assert cache(root).with_name("scene_002.json").is_file()
            assert not cache(root).with_name("scene_003.json").exists()
            if length_error:
                raise ca.lmstudio_json.StructuredOutputLengthError("output limit")
            raise RuntimeError("third scene unavailable")
        return {}

    monkeypatch.setattr(ca.narrative_validation, "review", review)
    result = run()["result"][0][0]
    assert len(result["sequences"]) == 4
    assert result["sequences"][2]["review"]["status"] == "incomplete"
    warnings = result["narrative_review"]["warnings"]
    assert all(w["sequence"] == 3 for w in warnings)
    assert warnings[0]["reason"] == ("output_limit" if length_error else "request_failed")
    previous = cache(root).read_bytes()
    assert "saved_scenes=2, pending_scenes=2" in capsys.readouterr().out
    reviewed.clear()

    def resumed_review(chat, client, model, payload, count, tokens):
        reviewed.append(deepcopy(payload))
        return {}

    monkeypatch.setattr(ca.narrative_validation, "review", resumed_review)
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: pytest.fail("No regeneration expected"))
    result = run(NEW, resume_run=OLD)["result"][0][0]
    assert not reviewed
    assert result["narrative_review"]["warnings"] == warnings
    assert [s["sequence"] for s in result["sequences"]] == [1, 2, 3, 4]
    assert "".join(s["source"] for s in result["sequences"]) == source
    assert cache(root).read_bytes() == previous
    assert all(cache(root, NEW).with_name(f"scene_{i:03d}.json").is_file() for i in range(1, 5))


def test_split_repair_commits_each_replacement_before_reviewing_the_next(monkeypatch):
    snapshots = []
    fields = [scene("A. B."), scene(" C.")]

    def chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if "rejected_adaptation" in payload:
            return {"sequences": [scene("A."), scene(" B.")]}
        return {"sequences": deepcopy(fields)}

    def review(chat, client, model, payload, count, tokens):
        source = payload["sequences"][0]["source"]
        if source == "A. B.":
            return {1: ["Split the scene."]}
        if source == " B.":
            saved = snapshots[-1]
            assert saved["stage"] == "scene_planned"
            assert saved["source_cursor"] == 2
            assert [s["source"] for s in saved["sequences"]] == ["A."]
            assert [s["source"] for s in saved["pending_sequences"]] == [" B.", " C."]
            raise KeyboardInterrupt()
        return {}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", chat)
    monkeypatch.setattr(ca.narrative_validation, "review", review)
    kwargs = dict(chunk_chars=4000, temperature=0.15, max_tokens=16384, correction_attempts=1)
    with pytest.raises(KeyboardInterrupt):
        ca.adapt_chapter(None, "mock", "A. B. C.", on_checkpoint=lambda s: snapshots.append(deepcopy(s)), **kwargs)
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: pytest.fail("No regeneration expected"))
    monkeypatch.setattr(ca.narrative_validation, "review", lambda *args: {})
    result = ca.adapt_chapter(None, "mock", "A. B. C.", resume_state=snapshots[-1], **kwargs)
    assert [s["source"] for s in result["sequences"]] == ["A.", " B.", " C."]


@pytest.mark.parametrize("stage", ["proposal_selected", "proposal_reviewed"])
def test_resume_selected_proposal_retains_choice_and_reviews_before_commit(monkeypatch, stage):
    snapshots = []
    proposal = {"initialState": "Before.", "event": "Wait.", "endingState": "After."}
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: {"sequences": [scene("Il attend.")]})
    monkeypatch.setattr(ca.narrative_validation, "review", lambda *args, **kwargs: {1: ["Uncertain posture."]})
    monkeypatch.setattr(ca.narrative_choices, "propose", lambda *args: [{"adaptation": deepcopy(proposal)}])

    def checkpoint(state):
        snapshots.append(deepcopy(state))
        if state["stage"] == stage:
            raise RuntimeError("interrupted proposal")

    kwargs = dict(chunk_chars=4000, temperature=0.15, max_tokens=16384, correction_attempts=0)
    with pytest.raises(RuntimeError, match="interrupted proposal"):
        ca.adapt_chapter(None, "mock", "Il attend.", on_checkpoint=checkpoint, **kwargs)
    monkeypatch.setattr(ca.lmstudio_json, "chat_json", lambda *args: pytest.fail("No regeneration expected"))
    monkeypatch.setattr(ca.narrative_choices, "propose", lambda *args: pytest.fail("Keep the selected proposal"))
    reviews = []
    monkeypatch.setattr(ca.narrative_validation, "review", lambda *args: reviews.append(args[3]) or {})
    result = ca.adapt_chapter(None, "mock", "Il attend.", resume_state=snapshots[-1], **kwargs)
    assert len(reviews) == 1
    assert result["sequences"][0]["adaptation"] == proposal
    record = result["narrative_review"]["issues"][0]
    assert record["sequence"] == 1 and record["proposals"][0]["review"] == {"status": "approved", "issues": []}

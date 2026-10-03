"""Timeline boundaries are enforced before mocked model extraction."""
from copy import deepcopy
from contextlib import nullcontext
import json

import pytest

from minimax_h3_novel_pipeline import cinematic_references as cr, lmstudio_pipeline, path_access, run_output, util
from minimax_h3_novel_pipeline.extract_chapter_references import ExtractChapterReferencesNode
from minimax_h3_novel_pipeline.load_cinematic_chapters import LoadCinematicChaptersNode


def chapter(name="chapter"):
    return {"chapter_name": name, "source_file": "/not/read.txt", "sequences": [
        {"sequence": 1, "source": "Original supporting text mentions a torch later.", "adaptation": {
            "initialState": "Indy hangs from a rope.", "event": "Indy waits.", "endingState": "Indy hangs."}},
        {"sequence": 2, "source": "Throwing.", "adaptation": {
            "initialState": "Indy hangs.", "event": "Doriane throws a torch toward Indy.",
            "endingState": "The torch is in flight."}},
        {"sequence": 3, "source": "Catching.", "adaptation": {
            "initialState": "The torch is in flight.", "event": "Indy catches the torch.",
            "endingState": "Indy holds the torch."}},
    ]}


@pytest.fixture
def execution(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock"))
    calls = []
    def extract(client, model, system, user, schema, temperature, max_tokens):
        data = json.loads(user)
        calls.append(data)
        result = {"chunk_summary": "", "characters": [], "locations": [], "objects": []}
        for name, kind in (("Indy", "characters"), ("Doriane", "characters"), ("torch", "objects")):
            if name in data["text"]:
                state = "chapter_appearance" if kind == "characters" else "chapter_state"
                result[kind].append({"canonical_name": name, "aliases": [], state: data["text"]})
        return result
    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", extract)
    params = {k: spec[1]["default"] for k, spec in ExtractChapterReferencesNode.INPUT_TYPES()["required"].items()
              if len(spec) > 1 and "default" in spec[1]}
    return calls, params, {"api_url": "http://localhost:1234/v1", "run_folder": "20261003120000"}


def test_multiple_chapters_ordered_phases_and_identity(execution):
    calls, params, config = execution
    chapters = [chapter(), chapter("second")]
    original = deepcopy(chapters)
    result, _ = ExtractChapterReferencesNode().run(config, chapters, **params)
    assert chapters == original
    assert len(result) == 2
    assert [c["chapter_id"] for c in result] == ["chapter", "second"]
    assert [(c["sequence"], c["phase"]) for c in calls[:9]] == [
        (i, p) for i in (1, 2, 3) for p in cr.PHASES]
    assert all("torch" not in c["text"] for c in calls if c["sequence"] == 1)
    for catalog in result:
        assert catalog["sequences"] == chapters[0]["sequences"]
        torch = catalog["objects"][0]
        assert torch["first_sequence"] == 2
        assert torch["first_phase"] == "event"
        states = torch["state_by_sequence"]
        assert "1" not in states and "initialState" not in states["2"]
        assert "throws" in states["2"]["event"][0]["chapter_state"]
        assert "in flight" in states["3"]["initialState"][0]["chapter_state"]
        assert "catches" in states["3"]["event"][0]["chapter_state"]
        assert "holds" in states["3"]["endingState"][0]["chapter_state"]
        assert torch["chapter_state"] == ""
        indy = [e for e in catalog["characters"] if e["canonical_name"] == "Indy"]
        assert len(indy) == 1
        assert list(indy[0]["state_by_sequence"]) == ["1", "2", "3"]
    count = len(calls)
    assert ExtractChapterReferencesNode().run(config, chapters, **params)[0] == result
    assert len(calls) == count
    chapters[0]["sequences"][0]["adaptation"]["event"] = "Indy shouts."
    ExtractChapterReferencesNode().run(config, chapters, **params)
    assert len(calls) > count


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(sequences=[]),
    lambda c: c["sequences"][1].update(sequence=1),
    lambda c: c["sequences"].reverse(),
    lambda c: c["sequences"][0].update(sequence=True),
    lambda c: c["sequences"][0]["adaptation"].pop("event"),
    lambda c: c["sequences"][0]["adaptation"].update(event=[]),
])
def test_invalid_timeline(mutate):
    value = chapter()
    mutate(value)
    with pytest.raises(ValueError):
        cr.parse_chapters([value])


def test_serialized_input_and_socket():
    assert cr.parse_chapters(json.dumps([chapter()])) == [chapter()]
    inputs = ExtractChapterReferencesNode.INPUT_TYPES()["required"]
    assert "chapter_selection" not in inputs
    assert inputs["cinematic_chapters"][0] == "MINIMAX_CINEMATIC_CHAPTERS"


def test_duplicate_chapter_names_have_separate_files(execution):
    _, params, config = execution
    results, _ = ExtractChapterReferencesNode().run(config, [chapter(), chapter()], **params)
    assert [c["chapter_id"] for c in results] == ["chapter_001", "chapter_002"]
    assert all(c["schema_version"] == util.CHAPTER_SCHEMA for c in results)


def test_extraction_resumes_from_saved_cinematic_chapter(execution, tmp_path):
    calls, params, config = execution
    value = chapter()
    value.pop("source_file")
    util.save_json(tmp_path / "previous/001_chapter.cinematic.json", value)
    loaded, _ = LoadCinematicChaptersNode().run("previous")
    results, _ = ExtractChapterReferencesNode().run(config, loaded, **params)
    assert results[0]["sequences"] == value["sequences"]
    assert [(c["sequence"], c["phase"]) for c in calls] == [
        (i, phase) for i in (1, 2, 3) for phase in cr.PHASES]

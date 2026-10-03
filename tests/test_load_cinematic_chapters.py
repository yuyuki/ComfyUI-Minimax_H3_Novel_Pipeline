"""Saved cinematic chapters resume without model calls or external reads."""
import json

import pytest

from minimax_h3_novel_pipeline import cinematic_references, lmstudio_pipeline, path_access, util
from minimax_h3_novel_pipeline.cinematic_chapter_adapter import CinematicChapterAdapterNode
from minimax_h3_novel_pipeline.extract_chapter_references import ExtractChapterReferencesNode
from minimax_h3_novel_pipeline.load_cinematic_chapters import LoadCinematicChaptersNode


@pytest.fixture
def output_root(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(path_access, "storage_root", lambda kind: root)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model",
                        lambda *args: pytest.fail("Loading adaptations must not call LM Studio"))
    return root


def chapter(name="Chapitre 1"):
    return {"chapter_name": name, "sequences": [
        {"sequence": 1, "source": "Il attend.\n", "adaptation": {
            "initialState": "Il est ici.", "event": "1. Il attend.", "endingState": "Il reste ici."}},
        {"sequence": 2, "source": "Il part.", "adaptation": {
            "initialState": "Il reste ici.", "event": "1. Il part.", "endingState": "Il est parti."}},
    ]}


def test_file_load_preserves_adapter_output_and_extraction_socket(output_root):
    saved = output_root / "001_chapter.cinematic.json"
    value = chapter()
    util.save_json(saved, value)
    loaded, files = LoadCinematicChaptersNode().run(str(saved))
    assert loaded == [{**value, "saved_file": str(saved)}]
    assert cinematic_references.parse_chapters(loaded) == loaded
    assert files == str(saved)
    assert json.loads(saved.read_text(encoding="utf-8")) == value
    socket = ExtractChapterReferencesNode.INPUT_TYPES()["required"]["cinematic_chapters"][0]
    assert LoadCinematicChaptersNode.RETURN_TYPES == CinematicChapterAdapterNode.RETURN_TYPES
    assert LoadCinematicChaptersNode.RETURN_TYPES[0] == socket


def test_folder_load_uses_natural_order_and_ignores_other_outputs(output_root):
    for number in (10, 2, 1):
        util.save_json(output_root / f"run/cinematic_chapters/{number}_chapter.cinematic.json", chapter(str(number)))
    util.save_json(output_root / "run/cinematic_chapters/cinematic_adapter_configuration.json", {})
    util.save_json(output_root / "run/cinematic_chapters/one_references.json", {})
    loaded, files = LoadCinematicChaptersNode().run("run/cinematic_chapters")
    assert [item["chapter_name"] for item in loaded] == ["1", "2", "10"]
    assert files.splitlines() == [item["saved_file"] for item in loaded]


@pytest.mark.parametrize("value", [None, [], {}, {"chapter_name": "one", "sequences": []},
                                    {"chapter_name": "one", "sequences": [{"sequence": 1, "source": "Text"}]}])
def test_invalid_saved_chapters_rejected_with_filename(output_root, value):
    util.save_json(output_root / "bad.cinematic.json", value)
    with pytest.raises(ValueError, match=r"bad\.cinematic\.json"):
        LoadCinematicChaptersNode().run("bad.cinematic.json")


def test_invalid_sequence_order_is_rejected(output_root):
    value = chapter()
    value["sequences"].reverse()
    util.save_json(output_root / "bad.cinematic.json", value)
    with pytest.raises(ValueError, match="increasing"):
        LoadCinematicChaptersNode().run("bad.cinematic.json")


@pytest.mark.parametrize("value", ["", "   ", None, "missing", "../outside"])
def test_missing_or_invalid_paths_rejected(output_root, value):
    with pytest.raises(ValueError):
        LoadCinematicChaptersNode().run(value)


def test_empty_folder_and_malformed_json_rejected(output_root):
    with pytest.raises(ValueError, match=r"No \*\.cinematic\.json"):
        LoadCinematicChaptersNode().run(str(output_root))
    (output_root / "bad.cinematic.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.cinematic\.json"):
        LoadCinematicChaptersNode().run("bad.cinematic.json")


def test_external_absolute_path_rejected_before_read(output_root, tmp_path, monkeypatch):
    outside = tmp_path / "private.cinematic.json"
    monkeypatch.setattr(util, "load_json", lambda *args: pytest.fail("External files must not be read"))
    with pytest.raises(ValueError, match="inside"):
        LoadCinematicChaptersNode().run(str(outside))

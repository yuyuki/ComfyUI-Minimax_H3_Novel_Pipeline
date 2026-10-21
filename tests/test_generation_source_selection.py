"""Source provenance and source-less cinematic chapter selection."""
from copy import deepcopy

import pytest

from minimax_h3_novel_pipeline import generate_h3_prompts as node, pipeline_step3_generate as step, util


def test_source_less_chapter_matches_extraction_id(tmp_path, cinematic_registry):
    path = tmp_path / "chapter.md"
    refs = cinematic_registry(path)
    refs["chapters"][0]["source_file"] = ""
    assert step.selected_timeline(refs, path)[0] == "chapter"
    with pytest.raises(ValueError, match="Expected one"):
        step.selected_timeline(refs, tmp_path / "other.md")
    refs["chapters"].append(deepcopy(refs["chapters"][0]))
    with pytest.raises(ValueError, match="Expected one"):
        step.selected_timeline(refs, path)


def test_explicit_provenance_is_not_overridden_by_filename(tmp_path, cinematic_registry):
    refs = cinematic_registry(tmp_path / "original/chapter.md")
    with pytest.raises(ValueError, match="Expected one"):
        step.selected_timeline(refs, tmp_path / "other/chapter.md")
    refs["chapters"].append({"chapter_id": "chapter", "source_file": ""})
    assert step.selected_timeline(refs, tmp_path / "original/chapter.md")[0] == "chapter"


def test_duplicate_source_less_selection_rejected_before_model_call(tmp_path, monkeypatch, cinematic_registry):
    paths = [tmp_path / folder / "chapter.md" for folder in ("one", "two")]
    refs = cinematic_registry(paths[0], schema_version=util.REGISTRY_SCHEMA)
    refs["chapters"][0]["source_file"] = ""
    monkeypatch.setattr(node, "stage_output", lambda *args: tmp_path)
    monkeypatch.setattr(node.util, "discover_inputs", lambda *args: paths)
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model",
                        lambda *args: pytest.fail("Ambiguous selection must not call LM Studio"))
    with pytest.raises(ValueError, match="Multiple selected files"):
        node.GenerateH3PromptsNode().run(refs, {}, {"chapter_paths": list(map(str, paths))}, "prompts")

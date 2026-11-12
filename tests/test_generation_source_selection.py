from tests.cinematic_generation_helpers import chapter_input
"""Cinematic chapter identity, provenance and stale-input rejection."""
from copy import deepcopy
import pytest
from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step, util
from minimax_h3_novel_pipeline.nodes import generate_h3_prompts as node


def test_source_less_chapter_matches_extraction_id(tmp_path, cinematic_registry):
    refs = cinematic_registry(tmp_path / "chapter.md")
    refs["chapters"][0]["source_file"] = ""
    chapter = chapter_input(refs)
    assert step.selected_timeline(refs, chapter)[0] == "chapter"
    with pytest.raises(ValueError, match="Expected one"):
        step.selected_timeline(refs, {**chapter, "chapter_name": "other"})
    refs["chapters"].append(deepcopy(refs["chapters"][0]))
    with pytest.raises(ValueError, match="Expected one"):
        step.selected_timeline(refs, chapter)


def test_explicit_provenance_is_not_overridden_by_filename(tmp_path, cinematic_registry):
    refs = cinematic_registry(tmp_path / "original/chapter.md")
    chapter = chapter_input(refs)
    with pytest.raises(ValueError, match="Expected one"):
        step.selected_timeline(refs, {**chapter, "source_file": str(tmp_path / "other/chapter.md")})
    assert step.selected_timeline(refs, chapter)[0] == "chapter"


def test_duplicate_selection_rejected_before_model_call(tmp_path, monkeypatch, cinematic_registry):
    refs = cinematic_registry(tmp_path / "chapter.md", schema_version=util.REGISTRY_SCHEMA)
    chapter = chapter_input(refs)
    monkeypatch.setattr(node, "stage_output", lambda *a: tmp_path)
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model",
                        lambda *a: pytest.fail("Duplicate chapter must not call LM Studio"))
    with pytest.raises(ValueError, match="Multiple"):
        node.GenerateH3PromptsNode().run(refs, {}, [chapter, deepcopy(chapter)], "prompts")

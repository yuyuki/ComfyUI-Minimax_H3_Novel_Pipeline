"""Shared current-contract generation fixtures."""
import pytest


@pytest.fixture
def narrative_review():
    return {"issues": [{"id": "scene_001_issue_01", "sequence": 1,
                        "description": "Uncertain framing.", "source_excerpt": "He waits.",
                        "proposals": [
                            {"id": name, "reason": "Filmable framing.", "selected": name == "B",
                             "adaptation": {phase: f"{name} chosen {phase}" for phase in
                                            ("initialState", "event", "endingState")}}
                            for name in ("A", "B")]}]}


@pytest.fixture
def cinematic_registry():
    from minimax_h3_novel_pipeline.core.util import REFERENCE_TIMELINE_VERSION

    def make(path, **extra):
        return {"timeline_version": REFERENCE_TIMELINE_VERSION,
                "chapters": [{"chapter_id": "chapter", "source_file": str(path)}],
                "chapter_timelines": {"chapter": {"sequences": [
                    {"sequence": 1, "source": "A character waits.", "adaptation": {
                        "initialState": "A character waits.", "event": "The character moves.",
                        "endingState": "The character stops."}}]}},
                "chapter_entity_map": {}, **extra}
    return make


@pytest.fixture(autouse=True)
def character_source_root(request, tmp_path, monkeypatch):
    """Keep synthetic chapter files inside each character test's confined root."""
    if request.module.__name__.split(".")[-1].startswith(("test_character_", "test_load_character_", "test_source_text")):
        from minimax_h3_novel_pipeline.core import path_access

        monkeypatch.setattr(path_access, "_FALLBACK_BASE", tmp_path)

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
    def make(path, **extra):
        return {"chapters": [{"chapter_id": "chapter", "source_file": str(path)}],
                "chapter_timelines": {"chapter": {"sequences": [
                    {"sequence": 1, "source": "A character waits.", "adaptation": {
                        "initialState": "A character waits.", "event": "The character moves.",
                        "endingState": "The character stops."}}]}},
                "chapter_entity_map": {}, **extra}
    return make

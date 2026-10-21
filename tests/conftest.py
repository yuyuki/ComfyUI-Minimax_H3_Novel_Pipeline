"""Shared current-contract generation fixtures."""
import pytest


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

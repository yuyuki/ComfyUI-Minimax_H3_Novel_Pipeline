"""Shared current-contract generation fixtures."""
import pytest


@pytest.fixture
def mock_scene_planning(monkeypatch):
    """Isolate existing generation/review tests from the separately tested planning stage."""
    from minimax_h3_novel_pipeline.core import cinematic_scene_planning as planning

    def assess(chat, client, model, scene, evidence, duration, max_tokens, correction_attempts, *, on_diagnostic=None):
        return {"timing": {"beats": [{"events": [{"event_index": i, "text": text}],
                                      "seconds": 0.1, "reason": "Mock activity."}
                                      for i, text in enumerate(scene["events"], 1)],
                           "pause_seconds": 0.1,
                           "estimated_seconds": round(0.1 * (len(scene["events"]) + 1), 3),
                           "reason": "Timing is isolated in this test."},
                "continuity": {"reset": False, "reset_evidence": "", "updates": []}}

    monkeypatch.setattr(planning, "assess", assess)


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

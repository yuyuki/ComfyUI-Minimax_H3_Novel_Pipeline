"""Offline H3 parameter and reference-budget regression tests."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline.nodes.generate_h3_prompts import GenerateH3PromptsNode as Node
from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step


@pytest.mark.parametrize("name,value", [
    ("duration", 3.9), ("duration", 15.1), ("duration", float("nan")),
    ("max_pictures", 10), ("max_pictures_per_subject", 10),
    ("max_audio", 4), ("max_audio", -1), ("max_audio", 1.5),
    ("max_shots", 7), ("max_scenes", -1),
])
def test_invalid_parameters_fail_before_execution(name, value):
    with pytest.raises(ValueError, match=name):
        Node().run(None, None, None, "", **{name: value})


@pytest.mark.parametrize("duration", [4, 8, 15])
def test_valid_boundaries(duration):
    Node.validate_parameters(dict(duration=duration, max_pictures=9,
                                  max_pictures_per_subject=9, max_audio=3, max_shots=6))


def test_example_uses_valid_current_defaults():
    data = json.loads((Path(__file__).parents[1] / "examples/ExtractChapter.json").read_text(encoding="utf-8"))
    node = next(n for n in data["nodes"] if n["type"] == "GenerateH3PromptsNode")
    values = node["widgets_values_named"]
    Node.validate_parameters(values)
    assert values["max_scenes"] == Node.INPUT_TYPES()["required"]["max_scenes"][1]["default"] == 0
    assert node["widgets_values"] == list(values.values())


def test_binding_rejects_excessive_budget():
    with pytest.raises(ValueError, match="max_audio"):
        step.build_bindings({}, None, "chapter", SimpleNamespace(
            max_pictures=9, max_pictures_per_subject=4, max_audio=4))


def test_no_audio_only_bindings(monkeypatch):
    monkeypatch.setattr(step, "entity_index", lambda refs: {})
    monkeypatch.setattr(step, "request_map", lambda scene: {})
    monkeypatch.setattr(step, "audio_asset_for", lambda *args: pytest.fail("Audio selected without image"))
    scene = SimpleNamespace(location_global_id="", visible_entity_ids=[], speaking_entity_ids=["speaker"])
    bindings = step.build_bindings({}, scene, "chapter", SimpleNamespace(
        max_pictures=9, max_pictures_per_subject=4, max_audio=3))
    assert bindings["audio"] == []

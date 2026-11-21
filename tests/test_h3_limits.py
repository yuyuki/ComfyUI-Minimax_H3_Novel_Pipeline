"""Offline H3 parameter and reference-budget regression tests."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline import NODE_CLASS_MAPPINGS
from minimax_h3_novel_pipeline.nodes.generate_h3_prompts import GenerateH3PromptsNode as Node
from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step


@pytest.mark.parametrize("name,value", [
    ("max_pictures", 10), ("max_pictures_per_subject", 10),
    ("max_audio", 4), ("max_audio", -1), ("max_audio", 1.5),
    ("max_shots", 7), ("max_scenes", -1),
])
def test_invalid_parameters_fail_before_execution(name, value):
    with pytest.raises(ValueError, match=name):
        Node().run(None, None, None, "", **{name: value})


def test_valid_boundaries():
    Node.validate_parameters(dict(max_pictures=9,
                                  max_pictures_per_subject=9, max_audio=3, max_shots=6))


def test_duration_override_is_not_supported():
    with pytest.raises(ValueError, match="Unknown Generate H3 parameters: duration"):
        Node.validate_parameters({"duration": 8})


def test_example_uses_valid_current_defaults():
    data = json.loads((Path(__file__).parents[1] / "examples/ExtractChapter.json").read_text(encoding="utf-8"))
    node = next(n for n in data["nodes"] if n["type"] == "GenerateH3PromptsNode")
    values = node["widgets_values_named"]
    Node.validate_parameters(values)
    inputs = Node.INPUT_TYPES()
    widget_names = [
        name for section in ("required", "optional")
        for name, spec in inputs[section].items()
        if spec[0] in {"BOOLEAN", "INT", "FLOAT", "STRING"}
    ]
    assert list(values) == widget_names
    assert values["max_scenes"] == inputs["required"]["max_scenes"][1]["default"] == 0
    assert node["widgets_values"] == list(values.values())


def test_max_tokens_default_matches_bundled_workflow():
    data = json.loads((Path(__file__).parents[1] / "examples/ExtractChapter.json").read_text(encoding="utf-8"))
    for node in NODE_CLASS_MAPPINGS.values():
        for inputs in node.INPUT_TYPES().values():
            if "max_tokens" in inputs:
                assert inputs["max_tokens"][1]["default"] == 32768, node.__name__
    for node in data["nodes"]:
        named = node.get("widgets_values_named", {})
        if "max_tokens" in named:
            position = list(named).index("max_tokens")
            assert node["widgets_values"][position] == named["max_tokens"] == 32768, node["type"]


def test_binding_rejects_excessive_budget():
    with pytest.raises(ValueError, match="max_audio"):
        step.build_bindings({}, None, "chapter", SimpleNamespace(
            max_pictures=9, max_pictures_per_subject=4, max_audio=4))


def test_no_audio_only_bindings(monkeypatch):
    monkeypatch.setattr(step, "entity_index", lambda refs: {})
    monkeypatch.setattr(step, "request_map", lambda scene: {})
    monkeypatch.setattr(step, "audio_asset_for", lambda *args: pytest.fail("Audio selected without image"))
    scene = step.Scene("Scene", "Text", "Event", "", [], ["speaker"], [], True, "")
    bindings = step.build_bindings({}, scene, "chapter", SimpleNamespace(
        max_pictures=9, max_pictures_per_subject=4, max_audio=3))
    assert bindings["audio"] == []

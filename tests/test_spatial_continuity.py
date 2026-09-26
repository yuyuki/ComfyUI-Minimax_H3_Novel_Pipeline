"""Offline validation of the spatial continuity interface and resolver."""
import pytest

from minimax_h3_novel_pipeline.spatial_continuity import SpatialContinuityNode
from minimax_h3_novel_pipeline import pipeline_step3_generate as generate


def test_anchors_validate_and_preserve_operator_text():
    node = SpatialContinuityNode()
    payload, summary = node.run('{"tablet.wall": "right wall"}')
    assert payload["anchors"] == {"tablet.wall": "right wall"}
    assert "tablet.wall: right wall" in summary
    with pytest.raises(ValueError, match="JSON object"):
        node.run("[]")
    with pytest.raises(ValueError, match="Invalid spatial anchors JSON"):
        node.run("{")


def test_resolver_receives_future_and_previous_without_mixing_facts(monkeypatch):
    scene = generate.Scene("Now", "Indy hangs", "Indy sways", "", [], [], [], False, "")
    future = generate.Scene("Later", "Tablet appears", "Indy faces tablet", "", [], [], [], False, "")
    captured = {}

    def fake_chat(client, model, system, user, schema, temperature, max_tokens):
        import json
        captured.update(json.loads(user))
        return {key: [] for key in generate.CONTINUITY_FIELDS}

    monkeypatch.setattr(generate, "chat_json", fake_chat)
    generate.resolve_continuity(None, "qwen", scene, {"final_state": ["Indy is suspended"]},
                                [future], {"tablet.wall": "right wall"})
    assert captured["previous_final_state"] == ["Indy is suspended"]
    assert captured["future_requirements"][0]["visual_event"] == "Indy faces tablet"
    assert captured["operator_anchors"]["tablet.wall"] == "right wall"

"""Camera post-processing preserves H3 content and rejects malformed notes."""
from types import SimpleNamespace

from minimax_h3_novel_pipeline import pipeline_step3_generate as step
from .test_h3_prompt_validation import BINDINGS, prompt_with_shots


def _scene():
    return step.Scene("Rope", "Indy hangs", "Follow the rope", "", [], [], [], False, "")


def _args():
    return SimpleNamespace(duration=5.0, max_shots=2, camera_direction="Rope in right third")


def test_camera_refinement_changes_only_detailed_description(monkeypatch):
    prompt = prompt_with_shots("[Shot 1] Rope sways.\n[Shot 2] At 00:02.500, Indy waits.")
    calls = []

    def fake_chat(client, model, system, user, schema, temperature, max_tokens):
        calls.append(user)
        return {"shot_cameras": [
            "Frame the taut rope in the right third, descending beside the wall.",
            "Continue the same axis, ending on Indy near that wall.",
        ]}

    monkeypatch.setattr(step, "chat_json", fake_chat)
    refined, warnings = step.refine_camera_prompt(None, "qwen", prompt, _scene(),
                                                   {"operator_anchors": {"tablet.wall": "right wall"}},
                                                   _args(), BINDINGS)
    assert not warnings
    assert step.validate_prompt(refined, BINDINGS, 5, 2).ok
    assert step.section_body(refined, "subject_definitions") == step.section_body(prompt, "subject_definitions")
    assert step.section_body(refined, "overall_soundscape") == step.section_body(prompt, "overall_soundscape")
    assert "Rope sways." in refined and "Indy waits." in refined
    assert "right wall" in calls[0] and "right third" in calls[0]


def test_camera_refinement_rejects_new_shot_without_touching_prompt(monkeypatch):
    prompt = prompt_with_shots("[Shot 1] Rope sways.")
    monkeypatch.setattr(step, "chat_json", lambda *a, **kw: {"shot_cameras": ["[Shot 2] At 00:02.500, cut away."]})
    refined, warnings = step.refine_camera_prompt(None, "qwen", prompt, _scene(), None, _args(), BINDINGS)
    assert refined == prompt
    assert "rejected" in warnings[0]

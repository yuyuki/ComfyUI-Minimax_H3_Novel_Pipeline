"""Offline coverage for H3 text formatting and shot validation."""
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline import pipeline_step3_generate as step


BINDINGS = {"subjects": [], "audio": []}


def prompt_with_shots(shots):
    return "\n".join([
        "subject_definitions:", "N/A",
        "summary:", "[reference generation] A quiet scene.",
        "retention_analysis:", "N/A",
        "detailed_description:", "Natural cinematic lighting.",
        shots, "Visible scenery. " * 175,
        "overall_soundscape:", "Wind through the trees.",
        "non_diegetic_music:", "N/A",
    ])


def test_first_shot_can_start_with_at_in_prose():
    prompt = prompt_with_shots("[Shot 1] At the doorway, a figure pauses.")
    assert step.validate_prompt(prompt, BINDINGS, 8).ok


@pytest.mark.parametrize("first", ["At 00:00.000, ", "00:00.000, "])
def test_first_shot_timestamp_does_not_blame_valid_later_shots(first):
    prompt = prompt_with_shots(f"[Shot 1] {first}A figure pauses.\n[Shot 2] At 00:03.000, The camera moves closer.")
    assert step.validate_prompt(prompt, BINDINGS, 8).errors == ["[Shot 1] must not have a timestamp."]


def test_first_shot_timestamp_cannot_mask_missing_later_timestamp():
    prompt = prompt_with_shots("[Shot 1] At 00:00.000, A figure pauses.\n[Shot 2] The camera moves closer.")
    errors = step.validate_prompt(prompt, BINDINGS, 8).errors
    assert "[Shot 1] must not have a timestamp." in errors
    assert "Every shot after Shot 1 must begin '[Shot N] At MM:SS.mmm, '." in errors


@pytest.mark.parametrize("shots,error", [
    ("[Shot 1] A scene.\n[Shot 2] At 00:09.000, A cut.", "exceeds target duration"),
    ("[Shot 1] A scene.\n[Shot 2] At 00:03.000, A cut.\n[Shot 3] At 00:02.000, Another cut.", "strictly increase"),
    ("[Shot 1] A scene.\n[Shot 3] At 00:03.000, A cut.", "not sequential"),
    ("[Shot 1] A scene.\n[Shot 2] At 00:60.000, A cut.", "seconds must be less than 60"),
])
def test_invalid_shots_still_require_repair(shots, error):
    result = step.validate_prompt(prompt_with_shots(shots), BINDINGS, 8)
    assert not result.ok
    assert any(error in message for message in result.errors)


def test_inline_sections_are_normalized_without_changing_content():
    canonical = prompt_with_shots("[Shot 1] At the doorway, a figure pauses.")
    inline = canonical
    for name in step.SECTIONS:
        inline = inline.replace(f"{name}:\n", f"{name}: ")
    assert step.normalize_prompt(inline) == canonical
    assert step.validate_prompt(step.normalize_prompt(inline), BINDINGS, 8).ok
    assert step.normalize_prompt(canonical) == canonical


@pytest.mark.parametrize("body", ["", "overall_soundscape: Wind.\noverall_soundscape: Rain."])
def test_normalization_preserves_missing_and_duplicate_section_errors(body):
    prompt = prompt_with_shots("[Shot 1] A figure pauses.")
    prompt = prompt.replace("overall_soundscape:\nWind through the trees.", body)
    result = step.validate_prompt(step.normalize_prompt(prompt), BINDINGS, 8)
    assert not result.ok
    assert any("Section 'overall_soundscape'" in error for error in result.errors)


@pytest.mark.parametrize("repair", [False, True])
def test_generation_and_repair_normalize_model_text(monkeypatch, repair):
    canonical = prompt_with_shots("[Shot 1] At the doorway, a figure pauses.")
    inline = canonical.replace("overall_soundscape:\n", "overall_soundscape: ")
    monkeypatch.setattr(step, "chat_json", lambda *args: {"prompt_text": inline})
    scene = SimpleNamespace(title="Scene", visual_event="Pause", dialogue_present=False,
                            adaptation_notes="", source_excerpt="A figure pauses.")
    args = SimpleNamespace(temperature=0.38, max_tokens=8000)
    if repair:
        result = step.repair_prompt(None, "model", inline, scene, BINDINGS,
                                    step.validate_prompt(inline, BINDINGS, 8), 8, args)
    else:
        result = step.generate_prompt(None, "model", scene, BINDINGS, 8, args)
    assert result == canonical
    assert step.validate_prompt(result, BINDINGS, 8).ok

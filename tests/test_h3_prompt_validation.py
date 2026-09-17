"""Offline coverage for H3 text formatting and shot validation."""
from types import SimpleNamespace
import copy
import json

import pytest

from minimax_h3_novel_pipeline import pipeline_step3_generate as step


BINDINGS = {"subjects": [], "audio": []}


@pytest.mark.parametrize("valid", [True, False])
def test_scene_exports_named_reference_sheet_and_copyable_prompt(tmp_path, valid):
    subjects, pictures = [], []
    for i, (name, kind) in enumerate([( "Indy", "character"), ("Temple", "location"),
                                     ("Lantern", "object")], 1):
        views = []
        for view in ("front", "rear"):
            picture = {"h3_picture_label": f"<Picture {len(pictures) + 1}>",
                       "asset_id": f"ASSET_{i}_{view}", "suggested_filename": f"{name}_{view}.png",
                       "view_type": view, "variant": "base"}
            pictures.append(picture)
            views.append(picture)
        subjects.append({"h3_subject_label": f"<Subject {i}>", "canonical_name": name,
                         "entity_type": kind, "global_id": f"ENTITY_{i}", "pictures": views})
    audio = {"h3_audio_label": "<Audio 1>", "canonical_name": "Indy",
             "asset_id": "VOICE_1", "suggested_filename": "Indy.wav"}
    bindings = {"subjects": subjects, "picture_input_order": pictures,
                "audio": [audio], "audio_input_order": [audio]}
    original = copy.deepcopy(bindings)
    scene = SimpleNamespace(title="Arrival", visual_event="Arrival", adaptation_notes="", source_excerpt="Text")
    prompt = prompt_with_shots("[Shot 1] A figure pauses.")
    validation = SimpleNamespace(ok=valid, errors=[] if valid else ["Needs repair"], word_count=350)
    entry = step.save_scene(tmp_path, 1, scene, bindings, prompt, validation)
    sheet = (tmp_path / entry["prompt_file"]).read_text(encoding="utf-8")
    assert sheet.split("COPY-PASTE PROMPT:\n", 1)[1] == prompt + "\n"
    assert "<Subject 1> = Indy (character; ENTITY_1)" in sheet
    assert "<Subject 2> = Temple (location; ENTITY_2)" in sheet
    assert "<Subject 3> = Lantern (object; ENTITY_3)" in sheet
    assert "<Picture 2> = Indy | rear | base | Indy_rear.png" in sheet
    assert "<Audio 1> = Indy | Indy.wav" in sheet
    assert ("VALIDATION WARNINGS:" in sheet) is not valid
    record = json.loads((tmp_path / entry["assets_file"]).read_text(encoding="utf-8"))
    assert record["copy_paste_prompt"] == prompt
    assert record["valid"] is valid
    assert record["picture_input_order"] == pictures
    assert bindings == original
    assert entry["prompt_file"].endswith("_prompt.txt")
    assert not list(tmp_path.glob("*_assets.txt"))


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


@pytest.mark.parametrize("separator", [": ", " ", " = "])
def test_subject_definitions_accept_colon_and_whitespace(separator):
    bindings = {"subjects": [
        {"h3_subject_label": f"<Subject {i}>",
         "pictures": [{"h3_picture_label": f"<Picture {i}>"}]}
        for i in (1, 2)
    ], "audio": []}
    definitions = "\n".join(
        f"<Subject {i}>{separator}{name} ({kind}), defined by <Picture {i}>."
        for i, name, kind in [(1, "The Crevasse", "location"), (2, "Indy", "character")]
    )
    prompt = prompt_with_shots("[Shot 1] A figure pauses.").replace(
        "subject_definitions:\nN/A", f"subject_definitions:\n{definitions}"
    )
    assert step.validate_prompt(prompt, bindings, 8).ok


@pytest.mark.parametrize("definition,error", [
    ("N/A", "<Subject 1> is missing from subject_definitions."),
    ("<Subject 1>extra <Picture 1>", "<Subject 1> is missing from subject_definitions."),
    ("<Subject 1>: Indy (character).", "<Subject 1> definition must cite assigned <Picture 1>."),
    ("<Subject 1>\n<Subject 2>: Temple from <Picture 1>.",
     "<Subject 1> definition must cite assigned <Picture 1>."),
])
def test_subject_definition_checks_still_reject_missing_bindings(definition, error):
    bindings = {"subjects": [{"h3_subject_label": "<Subject 1>",
                              "pictures": [{"h3_picture_label": "<Picture 1>"}]}], "audio": []}
    prompt = prompt_with_shots("[Shot 1] <Subject 1> pauses.").replace(
        "subject_definitions:\nN/A", f"subject_definitions:\n{definition}"
    )
    assert error in step.validate_prompt(prompt, bindings, 8).errors


@pytest.mark.parametrize("cuts,duration,short_shot", [
    ([2.5, 5, 7.5], 8, 4),
    ([1.5, 3.5, 5.5], 8, 1),
    ([3, 4], 8, 2),
    ([7.5], 8, 2),
    ([8], 8, 2),
    ([0], 8, 1),
    ([2.5], 5, 1),
])
def test_rushed_shots_require_repair(cuts, duration, short_shot):
    shots = "[Shot 1] A scene."
    for index, cut in enumerate(cuts, 2):
        shots += f"\n[Shot {index}] At 00:{cut:06.3f}, A reaction."
    result = step.validate_prompt(prompt_with_shots(shots), BINDINGS, duration)
    assert not result.ok
    assert any(f"Shot {short_shot} lasts only" in error for error in result.errors)


@pytest.mark.parametrize("duration,shots", [
    (0.1, "[Shot 1] A scene."),
    (2, "[Shot 1] A scene."),
    (5, "[Shot 1] A scene."),
    (6, "[Shot 1] A scene."),
    (8, "[Shot 1] A scene."),
    (12, "[Shot 1] A scene."),
])
def test_sustained_shots_and_short_continuous_clips_pass(duration, shots):
    assert step.validate_prompt(prompt_with_shots(shots), BINDINGS, duration).ok


@pytest.mark.parametrize("first", ["At 00:00.000, ", "00:00.000, "])
def test_first_shot_timestamp_does_not_blame_valid_later_shots(first):
    prompt = prompt_with_shots(f"[Shot 1] {first}A figure pauses.\n[Shot 2] At 00:03.000, The camera moves closer.")
    assert "[Shot 1] must not have a timestamp." in step.validate_prompt(prompt, BINDINGS, 8).errors


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
    def chat_json(*args):
        assert "Exactly one continuous [Shot 1] lasting the full 8s" in args[3]
        assert "Do not merge successive shots" in args[3]
        assert "Visual event: Pause" in args[3]
        return {"prompt_text": inline}
    monkeypatch.setattr(step, "chat_json", chat_json)
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


def test_even_well_spaced_cuts_require_single_shot_repair():
    prompt = prompt_with_shots("[Shot 1] Ignition.\n[Shot 2] At 00:04.000, A spoken reaction.")
    result = step.validate_prompt(prompt, BINDINGS, 8)
    assert not result.ok
    assert any("exactly one continuous" in error for error in result.errors)


def test_planning_keeps_successive_beats_with_shared_context(monkeypatch):
    excerpt = 'She lights the wood in the hearth. She says, "Stay here."'
    events = ["She lights the wood in the hearth.", "She speaks with a worried expression."]
    raw = [dict(title=f"Moment {i}", source_excerpt=excerpt, visual_event=event,
                location_global_id="", visible_entity_ids=[], speaking_entity_ids=[],
                reference_view_requests=[], dialogue_present=bool(i),
                adaptation_notes="Flames remain attached to the wood in the hearth.")
           for i, event in enumerate(events)]

    def chat_json(client, model, system, user, *args):
        assert "Split successive actions" in system
        assert "never pack leftover beats" in system
        assert "Full duration for EACH single-shot scene: 8 seconds" in user
        return {"scenes": raw}

    monkeypatch.setattr(step, "chat_json", chat_json)
    scenes = step.plan_scenes(None, "model", "chapter", excerpt, 1, 1, [],
                             SimpleNamespace(duration=8, scenes_per_chunk=4))
    assert [scene.visual_event for scene in step.dedupe_scenes(scenes + scenes)] == events
    assert all(scene.source_excerpt == excerpt for scene in scenes)


def test_offscreen_reply_survives_planning_and_reaches_generation(monkeypatch):
    excerpt = "— Doriane ! hurla-t-il. Envoyez une autre torche !\n— Jones ! cria Doriane."
    catalog = [{"global_id": gid} for gid in ("CHAR_001", "CHAR_002")]
    turns = [("CHAR_001", "Indy calls from below.", "Indy: Doriane ! Envoyez une autre torche !"),
             ("CHAR_002", "Indy hears Doriane's reply from above.", "Off-screen Doriane: Jones !")]
    raw = [dict(title=event, source_excerpt=excerpt, visual_event=event,
                location_global_id="", visible_entity_ids=["CHAR_001"],
                speaking_entity_ids=[gid], reference_view_requests=[],
                dialogue_present=True, adaptation_notes=notes)
           for gid, event, notes in turns]
    requests = []

    def chat(client, model, system, user, schema, *unused):
        requests.append(user)
        if schema is step.SCENE_SCHEMA:
            assert excerpt in user
            assert "retain its short replies and calls" in system
            assert "reference metadata, not a timeline" in system
            return {"scenes": raw}
        assert "Include every dialogue turn selected" in user
        assert "Off-screen Doriane: Jones !" in user
        return {"prompt_text": prompt_with_shots('[Shot 1] Indy listens. Off-screen Doriane calls <d>[French] Jones !</d>.')}

    monkeypatch.setattr(step, "chat_json", chat)
    args = SimpleNamespace(duration=8, scenes_per_chunk=4, temperature=0.12, max_tokens=8000)
    scenes = step.dedupe_scenes(step.plan_scenes(None, "model", "chapter", excerpt, 1, 1, catalog, args))
    assert [scene.speaking_entity_ids for scene in scenes] == [["CHAR_001"], ["CHAR_002"]]
    assert scenes[1].visible_entity_ids == ["CHAR_001"]
    prompt = step.generate_prompt(None, "model", scenes[1], BINDINGS, 8, args)
    assert "<d>[French] Jones !</d>" in prompt
    assert len(requests) == 2


def test_extraction_requests_attributed_speech_from_original_prose(monkeypatch):
    from minimax_h3_novel_pipeline import pipeline_step1_extract as extract

    excerpt = "— Doriane ! hurla-t-il. Envoyez une autre torche !\n— Jones ! cria Doriane."

    def chat(client, model, system, user, schema, *unused):
        assert excerpt in user
        assert "Off-screen" in system
        assert "actual words and attribution" in system
        assert "Summaries follow source event order" in system
        return {"chunk_summary": "Indy calls for a torch; Doriane calls back.",
                "characters": [], "locations": [], "objects": []}

    monkeypatch.setattr(extract, "chat_json", chat)
    result = extract.extract_chunk(None, "model", "chapter", excerpt, 1, 1,
                                   SimpleNamespace(temperature=0.12, max_tokens=8000))
    assert "Doriane calls back" in result["chunk_summary"]

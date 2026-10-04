"""Offline validation of the spatial continuity interface and resolver."""
import pytest

from minimax_h3_novel_pipeline.spatial_continuity import SpatialContinuityNode
from minimax_h3_novel_pipeline import pipeline_step3_generate as generate


def test_removed_anchor_argument_is_rejected():
    with pytest.raises(TypeError):
        SpatialContinuityNode().run(anchors_json='{"tablet.wall": "right wall"}')


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


def test_automatic_continuity_needs_no_example_constraints():
    assert SpatialContinuityNode.INPUT_TYPES() == {"required": {}}
    payload, summary = SpatialContinuityNode().run()
    assert payload["anchors"] == {}
    assert "Automatic" in summary


def test_review_receives_final_wording_bindings_and_distant_scenes(monkeypatch):
    import json
    scene = generate.Scene("Now", "He sees the tablet", "He looks", "room", [], [], [], False, "")
    later = generate.Scene("Later", "x" * 1500 + "tablet is on the right", "Tablet", "room", [], [], [], False, "")
    history = [{"prompt": "Tablet on the right", "bindings": {"global_id": "tablet"}}]
    captured = []

    def fake_chat(client, model, system, user, schema, temperature, max_tokens):
        captured.append(json.loads(user))
        if schema == generate.CONTINUITY_REVIEW_SCHEMA:
            return {"errors": ["Turn the character toward the tablet on the right."]}
        return {key: [] for key in generate.CONTINUITY_FIELDS}

    monkeypatch.setattr(generate, "chat_json", fake_chat)
    future = [later] * 7
    designs = {"entities": [{"global_id": "room", "added_details": {"layout": "Tablet on the right wall"}}]}
    generate.resolve_continuity(None, "qwen", scene, {}, future, {}, visual_designs=designs)
    errors = generate.check_prompt_continuity(None, "qwen", scene, "He looks left",
                                              {"subjects": ["tablet"]}, {}, history, future)
    assert len(captured[0]["future_requirements"]) == 7
    assert captured[0]["visual_designs"] == designs
    assert captured[0]["future_requirements"][-1]["source_excerpt"].endswith("tablet is on the right")
    assert captured[1]["previous_final_prompts"] == history
    assert captured[1]["current_prompt"] == "He looks left"
    assert captured[1]["bindings"] == {"subjects": ["tablet"]}
    assert errors == ["Turn the character toward the tablet on the right."]


@pytest.mark.parametrize("result", [{}, {"errors": "wrong"}, {"errors": [42]}, {"errors": [""]}])
def test_malformed_review_is_not_treated_as_success(monkeypatch, result):
    scene = generate.Scene("Now", "Source", "Event", "", [], [], [], False, "")
    monkeypatch.setattr(generate, "chat_json", lambda *args: result)
    with pytest.raises(ValueError, match="invalid scene continuity review"):
        generate.check_prompt_continuity(None, "qwen", scene, "prompt", {}, {}, [], [])


@pytest.mark.parametrize("resolved", [True, False])
def test_corrections_are_rechecked_and_unresolved_conflicts_invalidate_prompt(monkeypatch, resolved):
    from argparse import Namespace
    scene = generate.Scene("Now", "Source", "Event", "", [], [], [], False, "")
    checked = []
    history = [{"prompt": "Tablet on right"}]

    def review(client, model, scene, prompt, bindings, continuity, previous, future, camera):
        checked.append(prompt)
        return [] if resolved and prompt == "corrected" else ["Wrong eyeline"]

    def repair(client, model, prompt, scene, bindings, validation, duration, args, continuity):
        assert continuity["previous_final_prompts"] == history
        assert "Scene continuity: Wrong eyeline" in validation.errors
        return "corrected"

    monkeypatch.setattr(generate, "check_prompt_continuity", review)
    monkeypatch.setattr(generate, "repair_prompt", repair)
    monkeypatch.setattr(generate, "validate_prompt", lambda *args: generate.Validation(True, [], 100))
    prompt, validation, errors, attempts = generate.review_and_repair_continuity(
        None, "qwen", scene, "wrong", {}, {}, history, [], Namespace(repair_attempts=1, duration=8),
    )
    assert checked == ["wrong", "corrected"]
    assert prompt == "corrected" and attempts == 1
    assert validation.ok is resolved
    assert errors == ([] if resolved else ["Wrong eyeline"])


def test_chapter_reviews_after_camera_and_on_cache_hits_and_saves_corrected_text(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from minimax_h3_novel_pipeline import path_access

    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    chapter = tmp_path / "chapter.txt"
    chapter.write_text("He enters. He sees the tablet. " * 20, encoding="utf-8")
    scenes = [generate.Scene(title, title, title, "room", [], [], [], False, "")
              for title in ("Enter", "Tablet")]
    bindings = {"subjects": [{"h3_subject_label": "<Subject 1>", "canonical_name": "Hero",
                              "entity_type": "character", "global_id": "hero", "pictures": []}],
                "picture_input_order": [], "audio": []}
    args = SimpleNamespace(out_dir=tmp_path / "output", force=False, chunk_chars=3000,
                           overlap_paragraphs=0, max_scenes=0, duration=8, max_shots=1,
                           spatial_anchors={}, repair_attempts=1, refine_camera=True)
    monkeypatch.setattr(generate, "plan_scenes", lambda *a: scenes)
    designs = {"entities": [{"global_id": "hero", "added_details": {"hair": "Black hair"}}]}

    def resolve(*a):
        assert a[-1] == designs
        return {k: [] for k in generate.CONTINUITY_FIELDS}

    monkeypatch.setattr(generate, "resolve_continuity", resolve)
    monkeypatch.setattr(generate, "build_bindings", lambda *a: bindings)
    monkeypatch.setattr(generate, "generate_prompt", lambda *a: "draft")
    monkeypatch.setattr(generate, "validate_prompt", lambda *a: generate.Validation(True, [], 100))
    camera_calls = []

    def camera(*a):
        camera_calls.append(True)
        return "camera wording with wrong eyeline", []

    reviews = []

    def review(client, model, scene, prompt, bindings, continuity, previous, future, camera):
        assert continuity["visual_designs"] == designs
        reviews.append((prompt, list(previous)))
        return ["Wrong eyeline"] if prompt != "corrected" else []

    monkeypatch.setattr(generate, "refine_camera_prompt", camera)
    monkeypatch.setattr(generate, "check_prompt_continuity", review)
    monkeypatch.setattr(generate, "repair_prompt", lambda *a: "corrected")
    for _ in range(2):
        manifest = generate.process_chapter(chapter, {"chapter_entity_map": {}, "visual_designs": designs}, None, "qwen", args)
        for entry in manifest["outputs"]:
            asset_file = args.out_dir / "chapter" / entry["assets_file"]
            assert json.loads(asset_file.read_text())["copy_paste_prompt"] == "corrected"
            assert entry["valid"] and not entry["continuity_review"]["errors"]
    assert len(camera_calls) == 2  # Reused cached prompts retain their camera pass.
    assert len(reviews) == 6  # Four initial reviews/repairs, two cache-hit reviews.
    assert reviews[0][0] == "camera wording with wrong eyeline"
    assert reviews[2][1][0]["prompt"] == "corrected"
    report = json.loads((args.out_dir / "chapter/manifest.json").read_text())["spatial_continuity"]
    assert all(s["prompt_review"]["valid"] for s in report["scenes"])
    assert not (args.out_dir / "chapter/spatial_continuity.json").exists()
    assert "corrected" in (args.out_dir / "chapter/all_prompts.txt").read_text()

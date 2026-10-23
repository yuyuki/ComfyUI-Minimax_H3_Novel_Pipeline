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


def test_spatial_node_requires_consolidation_and_configuration():
    inputs = SpatialContinuityNode.INPUT_TYPES()["required"]
    assert inputs["consolidated_references"] == ("MINIMAX_REGISTRY",)
    assert inputs["lmstudio_config"] == ("MINIMAX_LMSTUDIO_CONFIG",)


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


@pytest.mark.parametrize("external_plan", [False, True])
def test_chapter_reviews_after_camera_and_on_cache_hits_and_saves_corrected_text(tmp_path, monkeypatch, cinematic_registry, external_plan):
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

    if external_plan:
        args.spatial_plan = {"chapters": {"chapter": {"sequences": [{"sequence": 1, "camera": "right"}]}}}

    def resolve(*a):
        assert not external_plan, "The supplied plan must not be replaced by another model call"
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
        if external_plan:
            assert continuity["sequence_plan"]["camera"] == "right"
            assert continuity["scenes_in_sequence"] == 2
        reviews.append((prompt, list(previous)))
        return ["Wrong eyeline"] if prompt != "corrected" else []

    monkeypatch.setattr(generate, "refine_camera_prompt", camera)
    monkeypatch.setattr(generate, "check_prompt_continuity", review)
    monkeypatch.setattr(generate, "repair_prompt", lambda *a: "corrected")
    for _ in range(2):
        manifest = generate.process_chapter(chapter, cinematic_registry(chapter, visual_designs=designs), None, "qwen", args)
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


def spatial_answer():
    from minimax_h3_novel_pipeline import spatial_continuity as spatial
    answer = {k: [] if v['type'] == 'array' else 'unknown' for k, v in spatial._FIELDS.items()}
    entity = {k: 'unknown' for k in spatial._ENTITY}
    entity.update(entity_id='hero', kind='character', evidence='inferred', source_sequence=3,
                  world_position='Beside the fixed tablet')
    answer['initial_entities'] = [entity]
    answer['final_entities'] = [dict(entity)]
    return answer


def test_complete_chapter_is_visible_from_first_sequence(tmp_path, monkeypatch):
    import json
    from contextlib import nullcontext
    from minimax_h3_novel_pipeline import spatial_continuity as spatial, util
    registry = {'schema_version': util.REGISTRY_SCHEMA, 'source_digest': 'source',
                'chapter_timelines': {'chapter': {'sequences': [
                    {'sequence': i, 'source': 'Near tablet' if i == 3 else 'Waits'} for i in (3, 1, 2)]}}}
    from minimax_h3_novel_pipeline import path_access
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    monkeypatch.setattr(spatial, 'stage_output', lambda *a: tmp_path)
    monkeypatch.setattr(spatial.lmstudio_pipeline, 'make_client_and_model', lambda *a: (nullcontext(), 'qwen'))
    requests = []

    def chat(client, model, system, user, *args):
        requests.append(json.loads(user))
        return spatial_answer()

    monkeypatch.setattr(spatial, 'chat_json', chat)
    payload, rendered, _ = spatial.SpatialContinuityNode().run(registry, {'api_url': 'unused'})['result']
    assert [s['sequence'] for s in requests[0]['whole_chapter_timeline']] == [1, 2, 3]
    assert requests[0]['whole_chapter_timeline'][2]['source'] == 'Near tablet'
    assert requests[1]['previous_final_state'] == spatial_answer()['final_entities']
    assert json.loads(rendered) == json.loads((tmp_path / 'spatial_continuity.json').read_text()) == payload
    spatial.validate_plan(payload, registry)
    registry['source_digest'] = 'changed'
    with pytest.raises(ValueError, match='stale'):
        spatial.validate_plan(payload, registry)
    registry['source_digest'] = 'source'
    payload['chapters']['chapter']['sequences'].pop()
    with pytest.raises(ValueError, match='every sequence'):
        spatial.validate_plan(payload, registry)


@pytest.mark.parametrize('field,value', [('kind', 'alien'), ('source_sequence', '3'), ('world_position', 42)])
def test_spatial_output_validates_entity_types(field, value):
    from minimax_h3_novel_pipeline import spatial_continuity as spatial
    answer = spatial_answer()
    answer['initial_entities'][0][field] = value
    with pytest.raises(ValueError, match='invalid spatial output'):
        spatial._verify(answer, 1)

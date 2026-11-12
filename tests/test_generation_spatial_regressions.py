from tests.cinematic_generation_helpers import chapter_input, metadata_scene
"""Regressions for future spatial states leaking into H3 generation."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
from tests.h3_review_helpers import issue

from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step
from minimax_h3_novel_pipeline.nodes import spatial_continuity as spatial


def test_spatial_future_facts_rejected_but_inferred_geometry_allowed():
    answer = {k: [] if v['type'] == 'array' else 'unknown' for k, v in spatial._FIELDS.items()}
    entity = {k: 'unknown' for k in spatial._ENTITY}
    entity.update(entity_id='tablet', kind='object', evidence='explicit', source_sequence=4)
    answer['final_entities'] = [entity]
    with pytest.raises(ValueError, match='future evidence'):
        spatial._verify(answer, 1)
    entity['evidence'] = 'inferred'
    spatial._verify(answer, 1)
    entity['evidence'] = 'explicit'
    spatial._verify(answer, 4)




def test_continuity_repair_does_not_replace_complete_prompt_with_missing_description(monkeypatch):
    scene = step.Scene('Wait', 'Wait', 'Wait', '', [], [], [], False, '')
    monkeypatch.setattr(step, 'check_prompt_continuity', lambda *a: [issue()])
    monkeypatch.setattr(step, 'repair_prompt', lambda *a: 'missing description')
    monkeypatch.setattr(step, 'validate_prompt', lambda prompt, *a: step.Validation(
        prompt == 'complete', [] if prompt == 'complete' else ['Missing detailed_description'], 400))
    prompt, validation, errors, attempts = step.review_and_repair_continuity(
        None, 'model', scene, 'complete', {}, {}, [], [], SimpleNamespace(repair_attempts=1, duration=8, max_tokens=8000))
    assert prompt == 'complete' and attempts == 1
    assert not validation.ok and errors == [issue()]
    assert validation.errors == ['Scene continuity: Wrong eyeline']


@pytest.mark.parametrize('failure', ['format', 'continuity'])
def test_invalid_prompts_do_not_pollute_history_or_get_reused(tmp_path, monkeypatch, cinematic_registry, failure):
    path = tmp_path / 'chapter.txt'
    designs = {'entities': [{'global_id': 'hero'}, {'global_id': 'future_tablet'}]}
    refs = cinematic_registry(path, visual_designs=designs)
    spatial_plan = {'sequence': 1, 'camera': 'right', 'future_detail': 'Tablet revealed'}
    args = SimpleNamespace(out_dir=tmp_path / 'generated', force=False, max_scenes=0,
                           duration=8, repair_attempts=0, max_tokens=8000,
                           spatial_plan={'chapters': {'chapter': {'sequences': [spatial_plan]}}})
    refs["chapter_timelines"]["chapter"]["sequences"].append({
        **deepcopy(refs["chapter_timelines"]["chapter"]["sequences"][0]), "sequence": 2})
    args.spatial_plan["chapters"]["chapter"]["sequences"].append({**spatial_plan, "sequence": 2})
    histories, generations = [], []

    bindings = {'subjects': [{'global_id': 'hero', 'h3_subject_label': '<Subject 1>',
                             'canonical_name': 'Hero', 'entity_type': 'character', 'pictures': []}],
                'audio': [], 'picture_input_order': []}

    def generate(client, model, scene, bindings, duration, args, continuity):
        generations.append(scene.title)
        assert continuity['visual_designs'] == {'entities': [{'global_id': 'hero'}]}
        return 'invalid prompt'

    def review(client, model, scene, prompt, bindings, continuity, previous, *unused):
        histories.append(deepcopy(previous))
        return [issue()]

    monkeypatch.setattr(step, "prepare_cinematic_scene", metadata_scene)
    monkeypatch.setattr(step, 'build_bindings', lambda *a: deepcopy(bindings))
    monkeypatch.setattr(step, 'generate_prompt', generate)
    monkeypatch.setattr(step, 'check_prompt_continuity', review)
    if failure == 'continuity':
        monkeypatch.setattr(step, 'validate_prompt', lambda *a: step.Validation(True, [], 400))
    for _ in range(2):
        manifest = step.process_chapter(chapter_input(refs), refs, None, 'model', args)
        assert all(not entry['valid'] for entry in manifest['outputs'])
    # Drafts can be reused, but invalid final results are never accepted or
    # introduced into continuity history; every resumed candidate is checked.
    assert len(generations) == 2
    assert histories == ([[], [], [], []] if failure == 'continuity' else [])
    assert refs['visual_designs'] == designs
    cached = json.loads((args.out_dir / '.cache/chapter/prompt_001.json').read_text())
    assert cached['valid'] is False


def test_review_deduplicates_model_errors(monkeypatch):
    monkeypatch.setattr(step, 'chat_json', lambda *a: {'errors': [issue(), issue()]})
    scene = step.Scene('Wait', 'Wait', 'Wait', '', [], [], [], False, '')
    errors = step.check_prompt_continuity(None, 'model', scene, 'prompt', {}, {}, [], [], 8000)
    assert [error['message'] for error in errors] == ['Wrong eyeline']


@pytest.mark.parametrize('recovers', [True, False])
def test_spatial_generation_repairs_future_evidence_before_carrying_state(tmp_path, monkeypatch, recovers):
    from contextlib import nullcontext
    from minimax_h3_novel_pipeline.core import path_access, util

    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)

    registry = {'schema_version': util.REGISTRY_SCHEMA, 'source_digest': 'source',
                'entities': [{'global_id': 'tablet', 'canonical_name': 'Tablet'}],
                'chapter_timelines': {'chapter': {'sequences': [
                    {'sequence': i, 'source': 'Wait' if i == 1 else 'Tablet revealed'} for i in (1, 4)]}}}
    monkeypatch.setattr(spatial, 'stage_output', lambda *a: tmp_path)
    monkeypatch.setattr(spatial.lmstudio_pipeline, 'make_client_and_model',
                        lambda *a: (nullcontext(), 'model'))
    requests = []

    def chat(client, model, system, user, *args):
        request = json.loads(user)
        requests.append(request)
        answer = {k: [] if v['type'] == 'array' else 'unknown' for k, v in spatial._FIELDS.items()}
        entity = {k: 'unknown' for k in spatial._ENTITY}
        entity.update(entity_id='tablet', canonical_name='Tablet', kind='object',
                      evidence='explicit', source_sequence=4)
        if recovers and len(requests) > 1 and request['target_sequence'] == 1:
            entity.update(evidence='inferred', screen_position='offscreen')
        answer['final_entities'] = [entity]
        return answer

    monkeypatch.setattr(spatial, 'chat_json', chat)
    if recovers:
        payload = spatial.SpatialContinuityNode().run(registry, {'api_url': 'unused'})['result'][0]
        spatial.validate_plan(payload, registry)
        assert requests[-1]['previous_final_state'][0]['evidence'] == 'inferred'
        assert (tmp_path / 'spatial_continuity.json').exists()
    else:
        with pytest.raises(ValueError, match='future evidence'):
            spatial.SpatialContinuityNode().run(registry, {'api_url': 'unused'})
        assert not (tmp_path / 'spatial_continuity.json').exists()
    assert len(requests) == 3
    assert 'correction_required' in requests[1]

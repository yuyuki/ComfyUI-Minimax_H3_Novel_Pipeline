from tests.cinematic_generation_helpers import chapter_input, metadata_scene
"""Stable physical imports and source-grounded scene references."""
from contextlib import nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline.core import (
    character_visual_profiles as vp, character_visual_integration as integration,
    lmstudio_json, path_access, pipeline_step2_consolidate, util, visual_designs,
)
from minimax_h3_novel_pipeline.nodes import consolidate_references as node
from tests.test_character_evidence import observation
from tests.test_character_profiles import settings
from tests.test_character_visual_profiles import consolidated, cinematic
from tests.test_reference_timeline import catalog


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    data = consolidated(observation('She wears a red dress.', attribute='clothing', persistence='temporary'),
                        observation('Her eyes are green.', value='green eyes'))
    p = data['profiles'][0]
    p['temporary_states'] = [{'trait': 'clothing', 'value': 'red dress',
                             'evidence_ids': [p['supporting_evidence'][0]['evidence_id']]}]
    result, saved = vp.generate(data, ['chapter1.txt'], tmp_path / 'visual')
    chapter = catalog('chapter1', name='Élise')
    chapter['source'] = {'file': 'chapter1.txt'}
    chapter['sequences'] = cinematic(data)[0]['sequences']
    entity = {'global_id': 'CHAR_042', 'entity_type': 'character', 'canonical_name': 'Élise',
              'aliases': [], 'chapters_seen': ['chapter1'], 'stable_visual_description': 'Wrong eyes',
              'distinguishing_features': [], 'timeline': {'chapter1': chapter['characters'][0]['state_by_sequence']}, 'source_entities': [{'chapter_id': 'chapter1', 'local_id': 'LOCAL_1'}]}
    entity["timeline"]["chapter1"] = {
        "1": deepcopy(entity["timeline"]["chapter1"]["1"]),
    }
    def unexpected_request(*args):
        pytest.fail("Physical profile integration must not call the model.")
    monkeypatch.setattr(lmstudio_json, "chat_json", unexpected_request)
    return tmp_path, data, result, chapter, entity


@pytest.mark.parametrize('disk', [False, True])
@pytest.mark.parametrize('mapped', [False, True])
def test_consolidation_consumes_facts_via_socket_and_file(setup, monkeypatch, disk, mapped):
    root, data, result, chapter, entity = setup
    if not mapped:
        result['visual_designs']['character_profiles']['profiles'][0]['identity_status'] = 'ambiguous'
        util.save_json(root / 'visual/visual_designs.json', result['visual_designs'])
    pipeline = SimpleNamespace(reconcile_chapter=lambda *a: [deepcopy(entity)],
        audit_registry=lambda c, m, r, a: r, natural_key=lambda value: value,
        chat_json=lambda *a: {'added_details': []},
        build_picture_specs=lambda *a: [], generate_picture_assets=lambda *a: [],
        build_audio_specs=lambda *a: [], generate_audio_assets=lambda *a: [],
        build_chapter_map=lambda *a: {}, build_entity_asset_index=lambda *a: {},
        write_asset_prompts=lambda *a: None)
    monkeypatch.setattr(node.lmstudio_pipeline, 'load', lambda *a: pipeline)
    monkeypatch.setattr(node.lmstudio_pipeline, 'make_client_and_model', lambda *a: (nullcontext(), 'mock'))
    monkeypatch.setattr(node, 'stage_output', lambda *a: root / 'references')
    monkeypatch.setattr(node, 'validate_catalogs', lambda *a: None)
    monkeypatch.setattr(node, 'parse_chapters', lambda chapters: chapters)
    options = {k: v[1]['default'] for k, v in node.ConsolidateReferencesNode.INPUT_TYPES()['required'].items()
               if len(v) > 1 and 'default' in v[1]}
    options['max_attempts'] = 2
    extra = {'visual_designs_path': str(root / 'visual/visual_designs.json')} if disk else {'character_visual_profiles': result}
    with nullcontext() if mapped else pytest.warns(UserWarning):
        payload, _ = node.ConsolidateReferencesNode().run([chapter], {'api_url': 'http://127.0.0.1:1234/v1', 'run_folder': '20261028120000'}, **options, **extra)['result']
    if not mapped:
        assert 'character_id' not in payload['entities'][0]
        assert 'character_enrichment' not in payload
        report = util.load_json(root / 'references/consolidated_references.diagnostics.json')
        assert report['status'] == 'incomplete'
        return
    assert payload['entities'][0]['stable_visual_description'] == 'green eyes'
    assert payload['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] == 'green eyes'
    assert payload['entities'][0]['global_id'] == 'CHAR_042'
    assert payload['entities'][0]['character_id'] == data['profiles'][0]['local_id']
    assert 'character_enrichment' not in payload
    assert payload['entities'][0]['visual_reference']['scenes']['chapter1']['1']['presence'] == 'visible'
    assert not (root / 'references/.cache/chapter1').exists()
    assert util.load_json(root / 'references/visual_designs.json') == payload['visual_designs']
    assert util.load_json(root / 'references/consolidate_configuration.json')['node_settings']['max_attempts'] == 2


@pytest.mark.parametrize('max_attempts', [1, 2, 4])
def test_reference_validation_respects_attempt_limit(max_attempts):
    from minimax_h3_novel_pipeline.core.reference_requests import InvalidResponse, validated_request

    attempts = []

    def chat(*args):
        attempts.append(args)
        return {}

    def reject(result):
        raise ValueError('Missing requested entities.')

    with pytest.raises(InvalidResponse, match='Missing requested entities'):
        validated_request(chat, None, 'mock', 'system', {'entities': ['A', 'B']}, pipeline_step2_consolidate.RECONCILE_SCHEMA,
                          settings(max_attempts=max_attempts), reject)
    assert len(attempts) == max_attempts
    assert all(call[4] == pipeline_step2_consolidate.RECONCILE_SCHEMA and '"A", "B"' in call[3] for call in attempts)


@pytest.mark.parametrize('kind', ['ambiguous', 'two_registry_entities'])
def test_no_arbitrary_identity_mapping(setup, monkeypatch, kind):
    root, data, result, chapter, entity = setup
    designs = result['visual_designs']
    entities = [entity]
    if kind == 'ambiguous':
        designs['character_profiles']['profiles'][0]['identity_status'] = 'ambiguous'
    else:
        entities.append({**entity, 'global_id': 'CHAR_043'})
    diagnostics = []
    with pytest.warns(UserWarning):
        imported = integration.apply(designs, [chapter], entities, diagnostics)
    assert imported == {}
    assert len(diagnostics) == len(entities)
    assert all(row['status'] == 'incomplete' and row['chapter_ids'] == ['chapter1'] for row in diagnostics)
    assert all('character_id' not in e for e in entities)


def test_embedded_later_chapter_profile_is_used_without_exported_design(setup):
    from minimax_h3_novel_pipeline.core import character_profiles as cp

    root, data, result, chapter, entity = setup
    designs = result['visual_designs']
    # The prologue mentions the person, but all physical evidence comes later.
    document = designs['character_profiles']
    first = document['source_document']['chapters'][0]
    first['evidence'] = []
    document['source_digest'] = cp._digest(document['source_document'])
    profile = document['profiles'][0]
    removed = profile['supporting_evidence'][0]['evidence_id']
    profile['supporting_evidence'] = [r for r in profile['supporting_evidence'] if r['evidence_id'] != removed]
    profile['temporary_states'] = []
    profile['identity_support'] = [i for i in profile['identity_support'] if i != removed]
    for key in profile['attributes']:
        profile['attributes'][key] = [r for r in profile['attributes'][key] if r['evidence_id'] != removed]
    designs['entities'] = []
    imported = integration.apply(designs, [chapter], [entity], [])
    assert entity['stable_visual_description'] == 'green eyes'
    assert imported['CHAR_042']['source_facts']['stable_visual_description'] == 'green eyes'


@pytest.mark.parametrize('name', ['Indy', ' indy ', 'Henry Jones Jr.'])
def test_validated_alias_imports_unclassified_permanent_facts_without_identity_llm(setup, name):
    root, data, result, chapter, entity = setup
    designs = result['visual_designs']
    profile = designs['character_profiles']['profiles'][0]
    profile['canonical_name'], profile['aliases'] = 'Henry Jones Jr.', ['Indy']
    profile['global_traits'] = []  # The validated permanent observation still exists.
    entity['canonical_name'] = name
    original = deepcopy(designs)
    imported = integration.apply(designs, [chapter], [entity], [])
    assert entity['character_id'] == profile['local_id']
    assert imported['CHAR_042']['source_facts']['stable_visual_description'] == 'green eyes'
    assert designs == original


def test_shared_alias_between_profiles_is_not_chosen_arbitrarily(setup):
    root, data, result, chapter, entity = setup
    designs = result['visual_designs']
    p = designs['character_profiles']['profiles'][0]
    # Separate valid profiles own distinct observations but share the same alias.
    other = deepcopy(p)
    other['local_id'] = 'independent_identity'
    other['canonical_name'] = 'Another person'
    other['aliases'] = [entity['canonical_name']]
    other['supporting_evidence'] = p['supporting_evidence'][:1]
    ids = {r['evidence_id'] for r in other['supporting_evidence']}
    p['supporting_evidence'] = p['supporting_evidence'][1:]
    for target, keep in ((other, ids), (p, {r['evidence_id'] for r in p['supporting_evidence']})):
        target['identity_support'] = [k for k in target['identity_support'] if k in keep]
        target['attributes'] = {k: [r for r in rows if r['evidence_id'] in keep] for k, rows in target['attributes'].items()}
        for field in ('global_traits', 'temporary_states'):
            target[field] = [r for r in target[field] if set(r['evidence_ids']) <= keep]
    designs['character_profiles']['profiles'].append(other)
    with pytest.warns(UserWarning):
        imported = integration.apply(designs, [chapter], [entity], [])
    assert imported == {}
    assert 'character_id' not in entity


def test_manual_defaults_are_separate_and_ignored_fields_cannot_return(setup):
    root, data, result, chapter, entity = setup
    previous = result['visual_designs']
    previous['entities'][0]['added_details'] = {'eyes': 'black eyes', 'hair': 'cropped hair', 'default_outfit': 'red dress'}
    regenerated, _ = vp.generate(data, ['chapter1.txt'], root / 'next', previous)
    item = regenerated['visual_designs']['entities'][0]
    assert item['added_details'] == {'hair': 'cropped hair', 'default_outfit': 'red dress'}
    assert item['source_facts']['stable_visual_description'] == 'green eyes'


def test_imported_profile_gets_period_outfit_and_retains_it_on_reimport(setup):
    root, data, result, chapter, entity = setup
    chapter['default_temporal_context'] = '1922'
    chapter['sequences'][0]['adaptation']['initialState'] = 'En 1890, Élise paraît.'
    imported = integration.apply(result['visual_designs'], [chapter], [entity], [])
    entity['visual_reference'] = {'scenes': {'chapter1': {
        '1': {'presence': 'visible', 'traits': [], 'excluded_attributes': []},
    }}}
    args = settings(image_style='cinematic photographic', character_designs=imported,
                    temporal_context_by_chapter={'chapter1': {'default_temporal_context': '1922',
                                                              'sequences': chapter['sequences']}},
                    reference_diagnostics=[])
    requests = []
    def chat(client, model, system, user, schema, *rest):
        payload, _ = json.JSONDecoder().raw_decode(user)
        requests.append((schema['name'], system, payload))
        if schema is visual_designs.DESIGN_SCHEMA:
            assert schema['strict'] is True
            assert payload['required_traits'] == ['default_outfit']
            assert payload['scenes_without_wardrobe'] == {'chapter1': [1]}
            if len(requests) == 1:
                return {'added_details': []}
            assert 'Previous response was invalid' in user and 'Return default_outfit' in user
            return {'added_details': [{'trait': 'default_outfit', 'description': 'Robe bleue en laine et veste grise.'}]}
        return {'checks': [{'trait': key, 'verdict': 'compatible_addition', 'reason': 'Compatible.'}
                           for key in payload['added_details']]}
    designs = visual_designs.prepare_designs(chat, None, 'mock', [entity], args)
    assert designs['entities'][0]['added_details']['default_outfit'] == 'Robe bleue en laine et veste grise.'
    _, system, context = requests[0]
    assert context['temporal_context_by_chapter']['chapter1']['default_temporal_context'] == '1922'
    assert '1890' in json.dumps(context)
    assert 'flashbacks' in system and 'override' in system
    designs['character_profiles'] = data
    util.save_json(root / 'designs.json', designs)
    imported_again = integration.apply(util.load_json(root / 'designs.json'), [chapter], [entity], [])
    args.character_designs = imported_again
    requests.clear()
    again = visual_designs.prepare_designs(chat, None, 'mock', [entity], args, root / 'designs.json')
    assert again['entities'][0]['added_details'] == designs['entities'][0]['added_details']
    assert all(name != 'visual_design_additions' for name, _, _ in requests)






def test_wrong_source_identity_fails_instead_of_ignoring_facts(setup):
    root, data, result, chapter, entity = setup
    chapter['source']['file'] = 'another/chapter1.txt'
    with pytest.raises(ValueError, match='no matching character source'):
        integration.apply(result['visual_designs'], [chapter], [entity], [])


def test_source_matching_preserves_verbatim_passages_and_checks_order(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    first = 'Élise’s eyes are green.\r\n'
    second = 'She waits on the stairs.'
    data = consolidated(observation(first + second, value='green eyes'))
    (tmp_path / 'chapter1.txt').write_bytes((first + second).encode('utf-8'))
    result, _ = vp.generate(data, ['chapter1.txt'], tmp_path / 'visual')
    designs = result['visual_designs']
    chapter = {'source_file': 'chapter1.txt', 'sequences': [
        {'sequence': 1, 'source': first}, {'sequence': 2, 'source': second},
    ]}
    entity = {'global_id': 'CHAR_042', 'entity_type': 'character', 'canonical_name': 'Élise'}
    original = deepcopy(chapter)

    imported = integration.apply(designs, [chapter], [entity], [])
    assert imported['CHAR_042']['source_facts']['stable_visual_description'] == 'green eyes'
    assert chapter == original

    for excerpts in [(first.replace('green', 'blue'), second), (second, first)]:
        for sequence, excerpt in zip(chapter['sequences'], excerpts, strict=True):
            sequence['source'] = excerpt
        with pytest.raises(ValueError, match='unique ordered passage in chapter1.txt'):
            integration.apply(designs, [chapter], [entity], [])


def test_consolidated_designs_can_be_imported_with_independent_registry_ids(setup):
    from minimax_h3_novel_pipeline.core.editable_schemas import DESIGN_SCHEMA, validate_document
    root, data, result, chapter, entity = setup
    result['visual_designs']['entities'][0]['exclude_image'] = True
    imported = integration.apply(result['visual_designs'], [chapter], [entity], [])
    args = settings(image_style='cinematic photographic', character_designs=imported)
    designs = visual_designs.prepare_designs(lambda *a: pytest.fail('No invented character facts'), None, 'mock', [entity], args)
    designs['character_profiles'] = data
    validate_document(designs, DESIGN_SCHEMA, 'visual_designs.json')
    assert designs['entities'][0]['global_id'] == 'CHAR_042'
    assert designs['entities'][0]['character_id'] == data['profiles'][0]['local_id']
    assert designs['entities'][0]['exclude_image'] is True
    prop = {'global_id': 'OBJ_012', 'entity_type': 'object', 'canonical_name': 'Cloche'}
    designs['entities'].append({**prop, 'global_id': 'OBJ_003', 'source_facts': visual_designs.source_facts(prop),
                                'added_details': {}, 'exclude_image': True})
    path = root / 'designs.json'
    util.save_json(path, designs)
    reimported = visual_designs.prepare_designs(lambda *a: pytest.fail('No new design requests'), None, 'mock',
                                               [entity, prop], args, path)
    assert reimported['entities'][1]['global_id'] == 'OBJ_012'
    assert reimported['entities'][1]['exclude_image'] is True
    regenerated, _ = vp.generate(data, ['chapter1.txt'], root / 'next', designs)
    assert regenerated['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] == 'green eyes'
    assert regenerated['visual_designs']['entities'][0]['exclude_image'] is True


def test_changed_source_snapshot_is_not_silently_ignored_or_trusted(setup):
    root, data, result, chapter, entity = setup
    result['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] = 'Contradictory black eyes'
    integration.apply(result['visual_designs'], [chapter], [entity], [])
    assert entity['stable_visual_description'] == 'green eyes'


@pytest.mark.parametrize('field', ['character_id', 'identity_status'])
def test_profile_backed_import_requires_identity_metadata(setup, field):
    root, data, result, chapter, entity = setup
    del result['visual_designs']['entities'][0][field]
    with pytest.raises(ValueError, match=field):
        integration.apply(result['visual_designs'], [chapter], [entity], [])


def test_h3_prompts_receive_current_adaptation_and_stable_facts(tmp_path, monkeypatch, cinematic_registry):
    from minimax_h3_novel_pipeline.core import pipeline_step3_generate as generate
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    chapter = tmp_path / 'chapter.txt'
    sequences = [{'sequence': i, 'source': text, 'adaptation': {p: text for p in
                  ('initialState', 'event', 'endingState')}}
                 for i, text in enumerate(['Sweat runs down the hero\'s forehead.', 'Hero waits.'], 1)]
    refs = cinematic_registry(chapter)
    refs['chapter_timelines']['chapter']['sequences'] = sequences
    refs['visual_designs'] = {'character_profiles': {'private': 'Other chapter clothes and conflicting traits'},
        'entities': [{'global_id': 'hero', 'source_facts': {'stable_visual_description': 'green eyes'},
                      'added_details': {}}]}
    refs['entities'] = [{'global_id': 'hero', 'character_id': 'profile', 'ignored_character_traits': ['hair']}]
    planning, prompts = [], []
    def prepare(client, model, row, period, args):
        planning.append(deepcopy(row['sequence']['adaptation']))
        scene = metadata_scene(client, model, row, period, args)
        scene.visible_entity_ids = ['hero']
        return scene
    def prompt(client, model, scene, bindings, duration, args, continuity):
        prompts.append((scene.adaptation_notes, deepcopy(continuity)))
        return 'draft'
    monkeypatch.setattr(generate, "prepare_cinematic_scene", prepare)
    monkeypatch.setattr(generate, 'build_bindings', lambda *a: {'subjects': [{
        'global_id': 'hero', 'entity_type': 'character', 'canonical_name': 'Hero',
        'h3_subject_label': '<Subject 1>', 'pictures': []}], 'picture_input_order': [], 'audio': []})
    monkeypatch.setattr(generate, 'generate_prompt', prompt)
    monkeypatch.setattr(generate, 'validate_prompt', lambda *a: generate.Validation(True, [], 100))
    monkeypatch.setattr(generate, 'check_prompt_continuity', lambda *a: [])
    args = SimpleNamespace(out_dir=tmp_path / 'prompts', force=False, max_scenes=0,
                           duration=8, max_shots=1, max_tokens=8000, repair_attempts=0, refine_camera=False)
    # Run twice: cached planning must preserve the same scene-local constraints.
    for _ in range(2):
        manifest = generate.process_chapter(chapter_input(refs), refs, None, 'mock', args)
        assert manifest['saved_prompt_count'] == 2
    assert len(planning) == 4 and len(prompts) == 2
    assert 'Sweat' in planning[0]['event'] and 'Sweat' not in planning[1]['event']
    assert 'Sweat' in prompts[0][0] and 'Sweat' not in prompts[1][0]
    for _, context in prompts:
        assert context['ignored_character_traits'] == {'hero': ['hair']}
        assert context['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] == 'green eyes'
        assert 'character_profiles' not in context['visual_designs']


def test_spatial_planning_keeps_local_facts_and_schema_on_retries(setup, monkeypatch):
    from unittest.mock import MagicMock
    from minimax_h3_novel_pipeline.nodes import spatial_continuity as spatial
    from tests.test_spatial_continuity import spatial_answer
    from tests.test_structured_json import Stream, client_for
    root, data, result, chapter, entity = setup
    integration.apply(result['visual_designs'], [chapter], [entity], [])
    answer = spatial_answer()
    for row in answer['initial_entities'] + answer['final_entities']:
        row.update(entity_id='CHAR_042', source_sequence=1)
    invalid = deepcopy(answer)
    invalid['initial_entities'][0].update(evidence='explicit', source_sequence=2)
    raw, create = client_for(Stream(['{"initial_entities":']), Stream([json.dumps(invalid)]),
                             Stream([json.dumps(answer)]))
    client = MagicMock(chat=raw.chat)
    monkeypatch.setattr(lmstudio_json, 'QWEN35_LENGTH_RETRIES', 1)
    monkeypatch.setattr(spatial.lmstudio_pipeline, 'make_client_and_model', lambda *a: (client, 'qwen3.5'))
    monkeypatch.setattr(spatial, 'stage_output', lambda *a: root / 'spatial')
    refs = {'schema_version': util.REGISTRY_SCHEMA, 'timeline_version': util.REFERENCE_TIMELINE_VERSION, 'source_digest': 'source', 'entities': [entity],
            'chapter_timelines': {'chapter1': chapter}, 'visual_designs': result['visual_designs']}
    spatial.SpatialContinuityNode().run(refs, {'api_url': 'unused'})
    assert create.call_count == 3
    for call in create.call_args_list:
        response_format = call.kwargs['response_format']
        assert response_format['type'] == 'json_schema'
        assert response_format['json_schema']['name'] == spatial._SCHEMA['name']
        assert response_format['json_schema']['strict'] is True
        message = next(m['content'] for m in call.kwargs['messages'] if m['role'] == 'user')
        assert 'red dress' in message and 'green eyes' in message
        assert 'character_profiles' not in message
        assert data['profiles'][0]['local_id'] in message

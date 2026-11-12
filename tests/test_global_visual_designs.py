from tests.cinematic_generation_helpers import chapter_input, metadata_scene
"""Stable physical imports, validated alias joins and scene-local enrichment."""
from contextlib import nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline.core import (
    character_visual_profiles as vp, character_visual_integration as integration,
    character_chapter_enrichment as local, lmstudio_json, path_access, util, visual_designs,
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
    calls = []
    def chat(client, model, system, prompt, schema, *args):
        payload = json.loads(prompt.split('\nRejected:')[0])
        calls.append((schema, payload))
        if schema == local.PRESENCE:
            return {'characters': [{'local_id': c['local_id'], 'presence': 'visible',
                'source_quote': payload['sequence']['source'], 'label': c['canonical_name']} for c in payload['candidates']],
                'unresolved': []}
        if schema == local.STATES:
            return {'states': [{'state_id': s['state_id'], 'applies': True,
                'source_quote': payload['sequence']['source'], 'reason': 'Attested here.'} for s in payload['states']]}
        if schema == local.LOCAL_STATES:
            return {'observations': []}
        if schema == local.LOCAL_REVIEW:
            return {'issues': []}
        raise AssertionError(schema)
    monkeypatch.setattr(lmstudio_json, 'chat_json', chat)
    return tmp_path, data, result, chapter, entity, calls


def test_enrichment_resumes_chapter_cache_and_force_bypasses_it(setup, caplog):
    caplog.set_level('INFO')
    root, data, result, chapter, entity, calls = setup
    args = settings(context_tokens=131072)
    first = root / '20261101000000/references'
    integration.apply(result['visual_designs'], [chapter], [deepcopy(entity)], first, None, 'mock', args)
    count = len(calls)
    assert count > 0
    previous = {p.relative_to(first): p.read_bytes() for p in first.rglob('*') if p.is_file()}
    args.resume_cache_dir = first / '.cache'
    current = root / '20261101000001/references'
    integration.apply(result['visual_designs'], [chapter], [deepcopy(entity)], current, None, 'mock', args)
    assert len(calls) == count
    assert 'cache hit (previous run)' in caplog.text
    assert list((current / '.cache/chapter1').glob('*.json'))
    assert previous == {p.relative_to(first): p.read_bytes() for p in first.rglob('*') if p.is_file()}
    args.force = True
    integration.apply(result['visual_designs'], [chapter], [deepcopy(entity)], current, None, 'mock', args)
    assert len(calls) > count
    assert 'resume cache bypassed (force=True)' in caplog.text


@pytest.mark.parametrize('disk', [False, True])
@pytest.mark.parametrize('mapped', [False, True])
def test_consolidation_consumes_facts_via_socket_and_file(setup, monkeypatch, disk, mapped):
    root, data, result, chapter, entity, calls = setup
    if not mapped:
        result['visual_designs']['character_profiles']['profiles'][0]['identity_status'] = 'ambiguous'
        util.save_json(root / 'visual/visual_designs.json', result['visual_designs'])
    pipeline = SimpleNamespace(reconcile_chapter=lambda *a: [deepcopy(entity)],
        audit_registry=lambda c, m, r, a: r, natural_key=lambda value: value,
        chat_json=lambda *a: pytest.fail('No new design needed for imported character') if mapped else {'added_details': []},
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
    options['context_tokens'] = 131072
    original_requests = integration.Requests
    def configured_requests(output, client, model, args):
        assert args.context_tokens == 131072
        return original_requests(output, client, model, args)
    monkeypatch.setattr(integration, 'Requests', configured_requests)
    extra = {'visual_designs_path': str(root / 'visual/visual_designs.json')} if disk else {'character_visual_profiles': result}
    with nullcontext() if mapped else pytest.warns(UserWarning), \
            nullcontext() if mapped else pytest.raises(ValueError, match='no unique validated registry mapping'):
        payload, _ = node.ConsolidateReferencesNode().run([chapter], {'api_url': 'http://127.0.0.1:1234/v1', 'run_folder': '20261028120000'}, **options, **extra)['result']
    if not mapped:
        assert not (root / 'references/visual_designs.json').exists()
        assert not (root / 'references/consolidated_references.json').exists()
        return
    assert payload['entities'][0]['stable_visual_description'] == 'green eyes'
    assert payload['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] == 'green eyes'
    assert payload['entities'][0]['global_id'] == 'CHAR_042'
    context = integration.sequence_context(payload, 'chapter1', 1)
    assert context['registry_ids'] == {data['profiles'][0]['local_id']: 'CHAR_042'}
    assert context['sequence']['states'][0]['value'] == 'red dress'
    assert 'offscreen' in context['instruction'].lower()
    assert (root / 'references/.cache/chapter1').is_dir()
    assert not (root / 'visual/.cache').exists()
    assert util.load_json(root / 'references/visual_designs.json') == payload['visual_designs']
    assert util.load_json(root / 'references/consolidate_configuration.json')['node_settings']['max_attempts'] == 2
    assert util.load_json(root / 'references/consolidate_configuration.json')['node_settings']['context_tokens'] == 131072
    assert integration.sequence_context(payload, 'chapter1', 2)['sequence'] is None






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
        validated_request(chat, None, 'mock', 'system', {'entities': ['A', 'B']}, local.PRESENCE,
                          settings(max_attempts=max_attempts), reject)
    assert len(attempts) == max_attempts
    assert all(call[4] == local.PRESENCE and '"A", "B"' in call[3] for call in attempts)


@pytest.mark.parametrize('kind', ['ambiguous', 'two_registry_entities'])
def test_no_arbitrary_identity_mapping(setup, monkeypatch, kind):
    root, data, result, chapter, entity, calls = setup
    designs = result['visual_designs']
    entities = [entity]
    if kind == 'ambiguous':
        designs['character_profiles']['profiles'][0]['identity_status'] = 'ambiguous'
    else:
        entities.append({**entity, 'global_id': 'CHAR_043'})
    with pytest.warns(UserWarning), pytest.raises(ValueError, match='no unique validated registry mapping'):
        integration.apply(designs, [chapter], entities, root / 'refs', None, 'mock', settings())
    assert util.load_json(root / 'refs/chapter1/character_chapter_enrichment.json')['validation']['confidence'] == 'low'
    assert all('character_id' not in e for e in entities)


def test_embedded_later_chapter_profile_is_used_without_exported_design(setup):
    from minimax_h3_novel_pipeline.core import character_profiles as cp

    root, data, result, chapter, entity, calls = setup
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
    _, imported = integration.apply(designs, [chapter], [entity], root / 'refs', None, 'mock', settings())
    assert entity['stable_visual_description'] == 'green eyes'
    assert imported['CHAR_042']['source_facts']['stable_visual_description'] == 'green eyes'
    assert all(schema['name'] != 'character_registry_identity' for schema, _ in calls)


@pytest.mark.parametrize('name', ['Indy', ' indy ', 'Henry Jones Jr.'])
def test_validated_alias_imports_unclassified_permanent_facts_without_identity_llm(setup, name):
    root, data, result, chapter, entity, calls = setup
    designs = result['visual_designs']
    profile = designs['character_profiles']['profiles'][0]
    profile['canonical_name'], profile['aliases'] = 'Henry Jones Jr.', ['Indy']
    profile['global_traits'] = []  # The validated permanent observation still exists.
    entity['canonical_name'] = name
    original = deepcopy(designs)
    _, imported = integration.apply(designs, [chapter], [entity], root / 'refs', None, 'mock', settings())
    assert entity['character_id'] == profile['local_id']
    assert imported['CHAR_042']['source_facts']['stable_visual_description'] == 'green eyes'
    assert all(schema['name'] != 'character_registry_identity' for schema, _ in calls)
    assert designs == original


def test_shared_alias_between_profiles_is_not_chosen_arbitrarily(setup):
    root, data, result, chapter, entity, calls = setup
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
    with pytest.warns(UserWarning), pytest.raises(ValueError, match='no unique validated registry mapping'):
        integration.apply(designs, [chapter], [entity], root / 'refs', None, 'mock', settings())
    assert 'character_id' not in entity


def test_manual_defaults_are_separate_and_ignored_fields_cannot_return(setup):
    root, data, result, chapter, entity, calls = setup
    previous = result['visual_designs']
    previous['entities'][0]['added_details'] = {'eyes': 'black eyes', 'hair': 'cropped hair', 'default_outfit': 'red dress'}
    regenerated, _ = vp.generate(data, ['chapter1.txt'], root / 'next', previous)
    item = regenerated['visual_designs']['entities'][0]
    assert item['added_details'] == {'hair': 'cropped hair'}
    assert item['source_facts']['stable_visual_description'] == 'green eyes'






def test_wrong_source_identity_fails_instead_of_ignoring_facts(setup):
    root, data, result, chapter, entity, calls = setup
    chapter['source']['file'] = 'another/chapter1.txt'
    with pytest.raises(ValueError, match='source must resolve uniquely'):
        integration.apply(result['visual_designs'], [chapter], [entity], root / 'refs', None, 'mock', settings())
    assert calls == []


def test_consolidated_designs_can_be_imported_with_independent_registry_ids(setup):
    from minimax_h3_novel_pipeline.core.editable_schemas import DESIGN_SCHEMA, validate_document
    root, data, result, chapter, entity, calls = setup
    enrichment, imported = integration.apply(result['visual_designs'], [chapter], [entity], root / 'refs', None, 'mock', settings())
    args = settings(image_style='cinematic photographic', character_designs=imported)
    designs = visual_designs.prepare_designs(lambda *a: pytest.fail('No invented character facts'), None, 'mock', [entity], args)
    designs['character_profiles'] = data
    validate_document(designs, DESIGN_SCHEMA, 'visual_designs.json')
    assert designs['entities'][0]['global_id'] == 'CHAR_042'
    assert designs['entities'][0]['character_id'] == data['profiles'][0]['local_id']
    regenerated, _ = vp.generate(data, ['chapter1.txt'], root / 'next', designs)
    assert regenerated['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] == 'green eyes'


def test_changed_source_snapshot_is_not_silently_ignored_or_trusted(setup):
    root, data, result, chapter, entity, calls = setup
    result['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] = 'Contradictory black eyes'
    integration.apply(result['visual_designs'], [chapter], [entity], root / 'refs', None, 'mock', settings())
    assert entity['stable_visual_description'] == 'green eyes'


@pytest.mark.parametrize('field', ['character_id', 'identity_status'])
def test_profile_backed_import_requires_identity_metadata(setup, field):
    root, data, result, chapter, entity, calls = setup
    del result['visual_designs']['entities'][0][field]
    with pytest.raises(ValueError, match=field):
        integration.apply(result['visual_designs'], [chapter], [entity], root / 'refs', None, 'mock', settings())
    assert calls == []


@pytest.mark.parametrize('presence', ['visible', 'offscreen', 'mentioned', 'uncertain', 'absent'])
def test_h3_bindings_respect_local_presence_without_changing_other_entities(presence):
    from minimax_h3_novel_pipeline.core import pipeline_step3_generate as generate
    row = {'local_id': 'profile', 'presence': presence, 'label': 'Hero', 'source_quote': 'Hero waits.'}
    refs = {'character_enrichment': {'chapter': {'registry_ids': {'profile': 'CHAR_042'}, 'ignored_traits': {},
        'sequences': [{'sequence': 1, 'characters': [] if presence == 'absent' else [row], 'states': []}]}}}
    scene = generate.Scene('Wait', 'Hero waits.', 'Wait', 'room', ['CHAR_042', 'other'],
        ['CHAR_042', 'other'], [generate.ViewRequest('CHAR_042', ['face'], 'high', '')], True, '', sequence=1)
    generate.constrain_character_presence(refs, 'chapter', scene)
    assert ('CHAR_042' in scene.visible_entity_ids) == (presence == 'visible')
    assert ('CHAR_042' in scene.speaking_entity_ids) == (presence in ('visible', 'offscreen'))
    assert bool(scene.reference_view_requests) == (presence == 'visible')
    assert 'other' in scene.visible_entity_ids and 'other' in scene.speaking_entity_ids


def test_h3_planning_and_prompts_receive_only_current_states_and_stable_facts(tmp_path, monkeypatch, cinematic_registry):
    from minimax_h3_novel_pipeline.core import pipeline_step3_generate as generate
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    chapter = tmp_path / 'chapter.txt'
    sequences = [{'sequence': i, 'source': text, 'adaptation': {p: text for p in
                  ('initialState', 'event', 'endingState')}}
                 for i, text in enumerate(['Hero wears a red dress.', 'Hero waits.'], 1)]
    refs = cinematic_registry(chapter)
    refs['chapter_timelines']['chapter']['sequences'] = sequences
    refs['visual_designs'] = {'character_profiles': {'private': 'Other chapter clothes and conflicting traits'},
        'entities': [{'global_id': 'hero', 'source_facts': {'stable_visual_description': 'green eyes'},
                      'added_details': {}}]}
    refs['character_enrichment'] = {'chapter': {'registry_ids': {'profile': 'hero'}, 'ignored_traits': {},
        'sequences': [{'sequence': seq['sequence'], 'characters': [{'local_id': 'profile', 'presence': 'visible',
            'label': 'Hero', 'source_quote': seq['source']}], 'states':
            [{'local_id': 'profile', 'trait': 'clothing', 'value': 'red dress'}] if seq['sequence'] == 1 else []}
                      for seq in sequences]}}
    planning, prompts = [], []
    def prepare(client, model, row, period, args):
        planning.append(deepcopy(row['character_context']))
        return metadata_scene(client, model, row, period, args)
    def prompt(client, model, scene, bindings, duration, args, continuity):
        prompts.append(deepcopy(continuity))
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
    assert planning[0]['sequence']['states'][0]['value'] == 'red dress'
    assert planning[1]['sequence']['states'] == []
    assert prompts[0]['character_enrichment']['sequence']['states'][0]['value'] == 'red dress'
    assert prompts[1]['character_enrichment']['sequence']['states'] == []
    assert all(c['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] == 'green eyes'
               for c in prompts)
    assert all('character_profiles' not in c['visual_designs'] for c in prompts)


def test_spatial_planning_keeps_local_facts_and_schema_on_retries(setup, monkeypatch):
    from unittest.mock import MagicMock
    from minimax_h3_novel_pipeline.nodes import spatial_continuity as spatial
    from tests.test_spatial_continuity import spatial_answer
    from tests.test_structured_json import Stream, client_for
    root, data, result, chapter, entity, calls = setup
    enrichment, _ = integration.apply(result['visual_designs'], [chapter], [entity],
                                      root / 'refs', None, 'mock', settings())
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
    refs = {'schema_version': util.REGISTRY_SCHEMA, 'source_digest': 'source', 'entities': [entity],
            'chapter_timelines': {'chapter1': chapter}, 'visual_designs': result['visual_designs'],
            'character_enrichment': enrichment}
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

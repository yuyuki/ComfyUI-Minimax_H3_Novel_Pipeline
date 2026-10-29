"""Stable physical imports, contextual identity joins and scene-local enrichment."""
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
        if schema == integration.IDENTITY:
            return {'same': True, 'source_quote': 'She wears a red dress.', 'reason': 'Context identifies the person.'}
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


@pytest.mark.parametrize('disk', [False, True])
def test_consolidation_consumes_facts_via_socket_and_file(setup, monkeypatch, disk):
    root, data, result, chapter, entity, calls = setup
    pipeline = SimpleNamespace(reconcile_chapter=lambda *a: [deepcopy(entity)],
        audit_registry=lambda c, m, r, a: r, natural_key=lambda value: value,
        chat_json=lambda *a: pytest.fail('No new design needed for imported character'),
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
    extra = {'visual_designs_path': str(root / 'visual/visual_designs.json')} if disk else {'character_visual_profiles': result}
    payload, _ = node.ConsolidateReferencesNode().run([chapter], {'api_url': 'http://127.0.0.1:1234/v1', 'run_folder': '20261028120000'}, **options, **extra)['result']
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
    assert integration.sequence_context(payload, 'chapter1', 2)['sequence'] is None


@pytest.mark.parametrize('kind', ['ambiguous', 'homonym', 'two_registry_entities', 'exhausted'])
def test_no_arbitrary_identity_mapping(setup, monkeypatch, kind):
    root, data, result, chapter, entity, calls = setup
    designs = result['visual_designs']
    entities = [entity]
    if kind == 'ambiguous':
        designs['character_profiles']['profiles'][0]['identity_status'] = 'ambiguous'
    elif kind in ('homonym', 'exhausted'):
        original = lmstudio_json.chat_json
        monkeypatch.setattr(lmstudio_json, 'chat_json', lambda *a: {
            'same': kind == 'exhausted', 'source_quote': '', 'reason': 'Name alone is insufficient.'}
            if a[4] == integration.IDENTITY else original(*a))
    else:
        entities.append({**entity, 'global_id': 'CHAR_043'})
    with pytest.warns(UserWarning, match='Unresolved'):
        enrichment, imported = integration.apply(designs, [chapter], entities, root / 'refs', None, 'mock', settings())
    assert imported == {}
    assert all('character_id' not in e for e in entities)
    assert enrichment['chapter1']['registry_ids'] == {}
    assert enrichment['chapter1']['validation']['confidence'] == 'low'


def test_manual_defaults_are_separate_and_ignored_fields_cannot_return(setup):
    root, data, result, chapter, entity, calls = setup
    previous = result['visual_designs']
    previous['entities'][0]['added_details'] = {'eyes': 'black eyes', 'hair': 'cropped hair', 'default_outfit': 'red dress'}
    regenerated, _ = vp.generate(data, ['chapter1.txt'], root / 'next', previous)
    item = regenerated['visual_designs']['entities'][0]
    assert item['added_details'] == {'hair': 'cropped hair'}
    assert item['source_facts']['stable_visual_description'] == 'green eyes'


def test_identity_retry_uses_shared_schema_and_keeps_all_observations(tmp_path, monkeypatch):
    from tests.test_structured_json import Stream, client_for
    from minimax_h3_novel_pipeline.core.character_profile_batches import Requests
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    data = consolidated(observation(), observation('Other chapter.'))
    client, create = client_for(Stream(['{"same":']), Stream([json.dumps({
        'same': False, 'source_quote': '', 'reason': 'Insufficient context'})]))
    requests = Requests(tmp_path / 'requests', client, 'mock', settings())
    evidence = data['profiles'][0]['supporting_evidence']
    local.request(requests, integration.IDENTITY_SYSTEM, integration.IDENTITY,
                  {'supporting_evidence': evidence}, lambda value: None)
    assert create.call_count == 2
    for call in create.call_args_list:
        assert call.kwargs['response_format'] == {'type': 'json_schema', 'json_schema': integration.IDENTITY}
        assert all(row['evidence_id'] in str(call.kwargs['messages']) for row in evidence)


def test_identity_budget_deduplicates_timeline_without_losing_evidence(setup, monkeypatch):
    from minimax_h3_novel_pipeline.core.character_profile_batches import Requests, encoded

    root, data, result, chapter, entity, calls = setup
    # Real catalogs repeat their sequence states in each reconciled registry entity.
    states = {'1': {'event': [{'chapter_appearance': 'Detailed observation. ' * 1400,
                              'evidence': ['Independent catalog observation.']}]}}
    chapter['characters'][0]['state_by_sequence'] = states
    entity['timeline'] = {'chapter1': deepcopy(states)}
    other = deepcopy(chapter['characters'][0])
    other.update(local_id='OTHER', canonical_name='Another person', state_by_sequence={})
    chapter['characters'].append(other)
    original_entity = deepcopy(entity)
    original_chapter = deepcopy(chapter)
    options = settings(max_tokens=8500)
    requests = Requests(root / 'probe', None, 'mock', options)
    capacity = requests.capacity(integration.IDENTITY_SYSTEM, integration.IDENTITY)
    identity_calls = []
    original_chat = lmstudio_json.chat_json

    def chat(client, model, system, prompt, schema, *args):
        if schema == integration.IDENTITY:
            payload = json.JSONDecoder().raw_decode(prompt)[0]
            identity_calls.append(payload)
            assert len(prompt.encode()) <= requests.context - options.max_tokens
            if len(identity_calls) == 1:
                return {'same': True, 'source_quote': 'Not in the source.', 'reason': 'Invalid quotation.'}
        return original_chat(client, model, system, prompt, schema, *args)

    monkeypatch.setattr(lmstudio_json, 'chat_json', chat)
    enriched, imported = integration.apply(result['visual_designs'], [chapter], [entity],
                                          root / 'refs', None, 'mock', options)
    assert len(identity_calls) == 2  # The correction retains the complete payload.
    assert identity_calls[0] == identity_calls[1]
    payload = identity_calls[0]
    assert len(encoded(payload).encode()) <= capacity
    assert len(encoded({**payload, 'registry_character': original_entity}).encode()) > capacity
    assert 'timeline' not in payload['registry_character']
    assert payload['registry_character'] == {k: v for k, v in original_entity.items() if k != 'timeline'}
    assert payload['chapter_id'] == 'chapter1'
    assert payload['catalog_characters'] == original_chapter['characters']
    assert payload['chapter_sources'] == [s['source'] for s in original_chapter['sequences']]
    profile = vp.prepare_profiles(data)[0]
    assert payload['profile']['supporting_evidence'] == profile['supporting_evidence']
    assert payload['profile']['identity_support'] == profile['identity_support']
    assert chapter == original_chapter
    assert entity['timeline'] == original_entity['timeline']
    assert enriched['chapter1']['registry_ids'][profile['local_id']] == entity['global_id']
    assert entity['global_id'] in imported


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
    with pytest.raises(ValueError, match='source_facts differs from validated physical traits'):
        integration.apply(result['visual_designs'], [chapter], [entity], root / 'refs', None, 'mock', settings())
    assert calls == []


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
    def plan(client, model, cid, chunk, index, count, catalog, args):
        planning.append(json.loads(chunk))
        return [generate.Scene('Wait', 'Hero waits.', 'Wait', '', ['hero'], [], [], False, '')]
    def prompt(client, model, scene, bindings, duration, args, continuity):
        prompts.append(deepcopy(continuity))
        return 'draft'
    monkeypatch.setattr(generate, 'plan_scenes', plan)
    monkeypatch.setattr(generate, 'build_bindings', lambda *a: {'subjects': [{
        'global_id': 'hero', 'entity_type': 'character', 'canonical_name': 'Hero',
        'h3_subject_label': '<Subject 1>', 'pictures': []}], 'picture_input_order': [], 'audio': []})
    monkeypatch.setattr(generate, 'generate_prompt', prompt)
    monkeypatch.setattr(generate, 'validate_prompt', lambda *a: generate.Validation(True, [], 100))
    monkeypatch.setattr(generate, 'check_prompt_continuity', lambda *a: [])
    args = SimpleNamespace(out_dir=tmp_path / 'prompts', force=False, max_scenes=0,
                           duration=8, max_shots=1, repair_attempts=0, refine_camera=False)
    # Run twice: cached planning must preserve the same scene-local constraints.
    for _ in range(2):
        manifest = generate.process_chapter(chapter, refs, None, 'mock', args)
        assert manifest['saved_prompt_count'] == 2
    assert len(planning) == len(prompts) == 2
    for context in (planning, prompts):
        assert context[0]['character_enrichment']['sequence']['states'][0]['value'] == 'red dress'
        assert context[1]['character_enrichment']['sequence']['states'] == []
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

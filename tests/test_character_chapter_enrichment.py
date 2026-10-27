"""Chapter projection, source boundaries, identity edits and downstream transport."""
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline.core import character_chapter_enrichment as ce
from minimax_h3_novel_pipeline.core import character_visual_profiles as vp
from minimax_h3_novel_pipeline.core import character_visual_integration as integration
from minimax_h3_novel_pipeline.core import character_profile_batches as batches, path_access, util
from tests.test_character_visual_profiles import consolidated, cinematic, design
from tests.test_character_profiles import settings
from tests.test_character_evidence import observation


@pytest.fixture
def output(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    return tmp_path / 'visual'


def mock_chat(monkeypatch, data, presence='visible'):
    calls = []
    def chat(client, model, system, prompt, schema, *args):
        payload = json.loads(prompt.split('\nRejected:')[0])
        calls.append((schema, payload))
        if schema == ce.PRESENCE:
            return {'characters': [{'local_id': c['local_id'], 'presence': presence,
                'source_quote': payload['sequence']['source'], 'label': c['canonical_name']}
                for c in payload['candidates']], 'unresolved': []}
        if schema == ce.STATES:
            return {'states': [{'state_id': s['state_id'], 'applies': s['kind'] == 'temporary',
                'source_quote': payload['sequence']['source'], 'reason': 'Directly attested.'} for s in payload['states']]}
        if schema == ce.LOCAL_STATES:
            return {'observations': []}
        if schema == ce.LOCAL_REVIEW:
            return {'issues': []}
        if schema == vp.REVIEW_SCHEMA:
            return {'issues': []}
        if schema == vp.LINK_SCHEMA:
            return {'compatible': False, 'reason': 'Conflicting values.'}
        return design(data)
    monkeypatch.setattr(vp.lmstudio_json, 'chat_json', chat)
    return calls


def test_late_trait_enriches_earlier_chapter_with_schema_and_downstream(output, monkeypatch):
    data = consolidated(observation(), observation('She has green eyes.'))
    # Only the second chapter supplies the selected global description.
    data['profiles'][0]['global_traits'] = data['profiles'][0]['global_traits'][1:]
    response_data = deepcopy(data)
    response_data['profiles'][0]['supporting_evidence'] = response_data['profiles'][0]['supporting_evidence'][1:]
    mock_chat(monkeypatch, response_data)
    chapter = ce.selected_chapter(data, cinematic(data))
    result, _ = vp.generate(data, vp.prepare_profiles(data), output, None, 'qwen', settings(mode='strict_canon'), chapter)
    assert result['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] == 'Green eyes.'
    assert result['chapter_enrichment']['sequences'][0]['states'] == []
    assert (output / 'character_chapter_enrichment.schema.json').is_file()
    assert list((output / '.cache' / 'chapter1').glob('*.json'))
    catalog = {**chapter, 'chapter_id': 'chapter1'}
    registry = [{'global_id': 'CHAR_001', 'entity_type': 'character', 'canonical_name': data['profiles'][0]['canonical_name'],
                 'aliases': [], 'chapters_seen': ['chapter1']}]
    local = integration.apply(result, [catalog], registry)
    assert registry[0]['stable_visual_description'] == 'Green eyes.'
    context = integration.sequence_context({'character_enrichment': local}, 'chapter1', 1)
    assert context['registry_ids'][data['profiles'][0]['local_id']] == 'CHAR_001'
    assert context['sequence']['characters'][0]['presence'] == 'visible'
    changed = deepcopy(catalog)
    changed['sequences'][0]['adaptation']['event'] += ' edited'
    with pytest.raises(ValueError, match='different source or cinematic'):
        integration.validate_input(result, [changed])


def test_temporary_clothing_stays_in_attested_source_and_not_design(output, monkeypatch):
    data = consolidated(observation('She wears a red evening dress.', attribute='clothing', persistence='temporary'))
    profile = data['profiles'][0]
    key = profile['supporting_evidence'][0]['evidence_id']
    profile['temporary_states'] = [{'trait': 'clothing', 'value': 'red evening dress', 'evidence_ids': [key]}]
    chapter = ce.selected_chapter(data, cinematic(data))
    mock_chat(monkeypatch, data)
    p = vp.prepare_profiles(data)[0]
    requests = batches.Requests(output, None, 'qwen', settings(mode='strict_canon'))
    result = ce.enrich(data, [p], chapter, requests)
    assert result['sequences'][0]['states'][0]['value'] == 'red evening dress'
    elsewhere = {**chapter['sequences'][0], 'source': 'The next morning.'}
    assert ce.state_candidates(p, chapter, elsewhere) == []
    wrong_source = {**chapter, 'source_file': 'another.txt'}
    assert ce.state_candidates(p, wrong_source, chapter['sequences'][0]) == []
    assert vp.effective_evidence(p) == {}
    assert not p['chronological_changes']  # No injury-to-scar inference.


def test_contextual_pronouns_offscreen_and_mentions(output, monkeypatch):
    data = consolidated()
    chapter = ce.selected_chapter(data, cinematic(data))
    chapter['sequences'].append({'sequence': 2, 'source': 'She waits.',
        'adaptation': {'initialState': 'She waits.', 'event': 'A voice is heard.', 'endingState': 'She waits.'}})
    calls = mock_chat(monkeypatch, data, 'offscreen')
    result = ce.enrich(data, vp.prepare_profiles(data), chapter,
                       batches.Requests(output, None, 'qwen', settings(mode='strict_canon')))
    assert [s['characters'][0]['presence'] for s in result['sequences']] == ['offscreen', 'offscreen']
    presence_calls = [p for schema, p in calls if schema == ce.PRESENCE]
    assert presence_calls[1]['previous_context'][0]['characters'][0]['local_id'] == data['profiles'][0]['local_id']
    mock_chat(monkeypatch, data, 'mentioned')
    result = ce.enrich(data, vp.prepare_profiles(data), chapter,
                       batches.Requests(output, None, 'qwen', settings(mode='strict_canon', force=True)))
    assert all(not s['states'] for s in result['sequences'])


def test_manual_links_merge_sources_and_conflicting_linked_traits_are_ignored(output, monkeypatch):
    data = consolidated(observation(), observation('Another has blue eyes.', value='blue eyes'))
    p = data['profiles'][0]
    alias = deepcopy(p)
    alias.update(local_id='char_alias', canonical_name='Alias', supporting_evidence=p['supporting_evidence'][1:],
                 global_traits=p['global_traits'][1:], identity_support=[], link_to_local_id=p['local_id'])
    p['supporting_evidence'] = p['supporting_evidence'][:1]
    p['global_traits'] = p['global_traits'][:1]
    p['identity_support'] = []
    for profile in (p, alias):
        own = {r['evidence_id'] for r in profile['supporting_evidence']}
        profile['attributes'] = {k: [r for r in v if r['evidence_id'] in own] for k, v in profile['attributes'].items()}
    data['profiles'] = [p, alias]
    projected = vp.prepare_profiles(data)
    assert len(projected) == 1
    assert len(projected[0]['supporting_evidence']) == 2
    assert 'Alias' in projected[0]['aliases']
    mock_chat(monkeypatch, data)
    vp.resolve_linked_traits(projected, batches.Requests(output, None, 'qwen', settings(mode='strict_canon')))
    assert projected[0]['effective_traits'] == []
    assert 'eye_color' in projected[0]['ignored_traits']
    assert len(data['profiles']) == 2


def test_chronological_scar_requires_sequence_evidence_not_chapter_order(output, monkeypatch):
    data = consolidated(observation('Her cheek is unmarked.', value='unmarked cheek'),
                        observation('After healing she has a scar.', value='scar', attribute='distinguishing_feature'))
    profile = data['profiles'][0]
    before, after = [r['evidence_id'] for r in profile['supporting_evidence']]
    profile['chronological_changes'] = [{'trait': 'distinguishing_mark', 'before_id': before,
        'after_id': after, 'support_id': after, 'temporal_quote': 'After healing'}]
    chapter = ce.selected_chapter(data, cinematic(data))
    p = vp.prepare_profiles(data)[0]
    assert not vp.effective_evidence(p)
    mock_chat(monkeypatch, data)
    local = ce.enrich(data, [p], chapter, batches.Requests(output, None, 'qwen', settings(mode='strict_canon')))
    assert not local['sequences'][0]['states']
    rows = ce.state_candidates(p, chapter, chapter['sequences'][0])
    assert {r['chronology']['side'] for r in rows} == {'before_id', 'after_id'}
    assert all(r['chronology']['temporal_quote'] == 'After healing' for r in rows)


@pytest.mark.parametrize('chapter_index,side', [(0, 'before_id'), (1, 'after_id')])
def test_attested_chronology_selects_only_the_matching_state(output, monkeypatch, chapter_index, side):
    data = consolidated(observation('Before the accident her cheek was unmarked.', value='unmarked cheek'),
                        observation('After healing she has a scar.', value='scar', attribute='distinguishing_feature'))
    profile = data['profiles'][0]
    before, after = [r['evidence_id'] for r in profile['supporting_evidence']]
    profile['chronological_changes'] = [{'trait': 'distinguishing_mark', 'before_id': before,
        'after_id': after, 'support_id': after, 'temporal_quote': 'After healing'}]
    source = data['source_document']['chapters'][chapter_index]
    chapter = cinematic(data)
    chapter[0].update(source_file=source['source_file'], chapter_name=source['chapter_name'])
    chapter[0]['sequences'][0]['source'] = source['source_text']
    chapter = ce.selected_chapter(data, chapter)
    mock_chat(monkeypatch, data)
    default_chat = vp.lmstudio_json.chat_json
    def chat(client, model, system, prompt, schema, *args):
        if schema != ce.STATES:
            return default_chat(client, model, system, prompt, schema, *args)
        payload = json.loads(prompt)
        return {'states': [{'state_id': r['state_id'], 'applies': r['chronology']['side'] == side,
                            'source_quote': source['source_text'], 'reason': 'Explicit before/after source wording.'}
                           for r in payload['states']]}
    monkeypatch.setattr(vp.lmstudio_json, 'chat_json', chat)
    local = ce.enrich(data, vp.prepare_profiles(data), chapter,
                      batches.Requests(output, None, 'qwen', settings(mode='strict_canon')))
    states = local['sequences'][0]['states']
    assert len(states) == 1
    assert states[0]['chronology']['side'] == side
    assert states[0]['value'] == ('unmarked cheek' if side == 'before_id' else 'scar')


def test_bounded_visual_batches_preserve_all_evidence(output, monkeypatch):
    data = consolidated(*(observation(f'Green eyes {i}.') for i in range(18)))
    seen = []
    def chat(client, model, system, prompt, schema, *args):
        payload = json.loads(prompt)
        if schema == vp.REVIEW_SCHEMA:
            return {'issues': []}
        seen.extend(r['evidence_id'] for r in payload['evidence'])
        response = {f: [] for f in vp.FIELDS}
        response['eyes'] = [{'description': 'Green eyes.', 'classification': 'canonical',
                             'evidence': [{'evidence_id': r['evidence_id']}]} for r in payload['evidence']]
        return response
    monkeypatch.setattr(vp.lmstudio_json, 'chat_json', chat)
    args = settings(mode='strict_canon', context_tokens=20000, max_tokens=1024)
    result, _ = vp.generate(data, vp.prepare_profiles(data), output, None, 'qwen', args)
    assert set(seen) == {r['evidence_id'] for r in data['profiles'][0]['supporting_evidence']}
    assert len(seen) == 18
    assert len(result['profiles'][0]['eyes']) == 18


def test_design_reuse_keeps_absent_characters_and_manual_defaults(output, monkeypatch):
    data = consolidated(observation(), observation('She waits for dawn.', value='green eyes'))
    data['profiles'][0]['global_traits'] = data['profiles'][0]['global_traits'][:1]
    mock_chat(monkeypatch, data)
    first, _ = vp.generate(data, vp.prepare_profiles(data), output, None, 'qwen', settings(mode='balanced'),
                           ce.selected_chapter(data, cinematic(data)))
    previous = first['visual_designs']
    previous['entities'][0]['added_details']['hair'] = 'Short brown hair.'
    absent = deepcopy(previous['entities'][0])
    absent.update(global_id='char_absent', canonical_name='Absent')
    previous['entities'].append(absent)
    second_source = data['source_document']['chapters'][1]
    second = cinematic(data)
    second[0].update(source_file=second_source['source_file'], chapter_name=second_source['chapter_name'])
    second[0]['sequences'][0]['source'] = second_source['source_text']
    result, _ = vp.generate(data, vp.prepare_profiles(data), output, None, 'qwen', settings(mode='balanced'),
                            ce.selected_chapter(data, second), previous=previous)
    assert first['chapter_enrichment']['source_file'] != result['chapter_enrichment']['source_file']
    assert result['visual_designs']['entities'][0]['added_details']['hair'] == 'Short brown hair.'
    assert result['visual_designs']['entities'][1] == absent
    assert util.load_json(output / 'visual_designs.json') == result['visual_designs']


def test_wrong_source_and_multiple_chapters_fail_before_network():
    data = consolidated()
    chapter = cinematic(data)
    with pytest.raises(ValueError, match='exactly one cinematic'):
        ce.selected_chapter(data, chapter * 2)
    chapter[0]['source_file'] = 'unrelated.txt'
    with pytest.raises(ValueError, match='source_file'):
        ce.selected_chapter(data, chapter)


def test_whitespace_mapping_preserves_original_offsets():
    original = 'Start.\n\nShe wears\t a red dress.\r\nEnd.'
    start, end = ce.source_span(original, 'She wears a red dress.')
    assert original[start:end] == 'She wears\t a red dress.'
    assert ce.source_span(original, 'She wears a blue dress.') is None
    assert ce.source_span('She waits. She waits.', 'She waits.') is None


@pytest.mark.parametrize('quote', ['She wears a red dress.', 'She\nwears a red dress.'])
def test_presence_restores_source_whitespace(quote):
    row = {'source_quote': quote}
    source = 'Start. She wears\t a red dress.\r\nEnd.'
    ce.presence_quote(row, source, 'hero')
    assert row['source_quote'] == 'She wears\t a red dress.'


@pytest.mark.parametrize('quote', ['', '  ', 'She wears a blue dress.', 'She wears ... dress.',
                                  'She wears a red dress!'])
def test_presence_rejects_changed_or_empty_evidence(quote):
    with pytest.raises(ValueError, match='Presence hero:'):
        ce.presence_quote({'source_quote': quote}, 'She wears a red dress.', 'hero')


def test_presence_correction_identifies_row_and_preserves_candidates(output, monkeypatch):
    data = consolidated()
    chapter = ce.selected_chapter(data, cinematic(data))
    pid = data['profiles'][0]['local_id']
    prompts = []
    def chat(client, model, system, prompt, schema, *args):
        assert schema == ce.PRESENCE
        prompts.append(prompt)
        return {'characters': [{'local_id': pid, 'presence': 'mentioned', 'label': 'Hero',
                'source_quote': 'Invented quote.' if len(prompts) == 1 else chapter['sequences'][0]['source']}],
                'unresolved': []}
    monkeypatch.setattr(vp.lmstudio_json, 'chat_json', chat)
    result = ce.enrich(data, vp.prepare_profiles(data), chapter,
                       batches.Requests(output, None, 'qwen', settings(mode='strict_canon', correction_attempts=1)))
    assert result['sequences'][0]['characters'][0]['presence'] == 'mentioned'
    assert len(prompts) == 2
    assert json.loads(prompts[0]) == json.loads(prompts[1].split('\nRejected:')[0])
    correction = prompts[1].split('\nRejected:')[1]
    assert pid in correction and 'Invented quote.' in correction and 'sequence.source' in correction


def test_invalid_presence_saves_diagnostics_before_generation_fails(output, monkeypatch):
    data = consolidated()
    pid = data['profiles'][0]['local_id']
    response = {'characters': [{'local_id': pid, 'presence': 'uncertain', 'source_quote': '',
                                'label': 'Hero'}], 'unresolved': []}
    monkeypatch.setattr(vp.lmstudio_json, 'chat_json', lambda *args: deepcopy(response))
    with pytest.raises(ValueError, match='no valid decision'):
        vp.generate(data, vp.prepare_profiles(data), output, None, 'qwen',
                    settings(mode='strict_canon', correction_attempts=1), ce.selected_chapter(data, cinematic(data)))
    warning = util.load_json(output / 'character_profiles_requests.json')['warnings'][0]
    assert warning['attempts'] == 2
    assert warning['response'] == response
    assert pid in warning['reason']
    assert warning['input']['sequence']['source'] == cinematic(data)[0]['sequences'][0]['source']


def test_presence_bounds_48_candidates_without_losing_any(output, monkeypatch):
    data = consolidated()
    chapter = ce.selected_chapter(data, cinematic(data))
    profile = vp.prepare_profiles(data)[0]
    profiles = [{**deepcopy(profile), 'local_id': f'char_{i}', 'canonical_name': f'Person {i}'} for i in range(48)]
    seen = []
    def chat(client, model, system, prompt, schema, *args):
        assert schema == ce.PRESENCE
        payload = json.loads(prompt)
        assert payload['sequence'] == chapter['sequences'][0]
        assert 1 <= len(payload['candidates']) <= 8
        seen.extend(c['local_id'] for c in payload['candidates'])
        return {'characters': [{'local_id': c['local_id'], 'presence': 'mentioned',
                'label': c['canonical_name'], 'source_quote': payload['sequence']['source']}
                for c in payload['candidates']], 'unresolved': []}
    monkeypatch.setattr(vp.lmstudio_json, 'chat_json', chat)
    result = ce.enrich(data, profiles, chapter, batches.Requests(output, None, 'qwen', settings(mode='strict_canon')))
    assert seen == [p['local_id'] for p in profiles]
    assert [r['local_id'] for r in result['sequences'][0]['characters']] == seen


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'unknown'])
def test_presence_splits_failed_batches_with_schema_and_full_context(output, defect):
    from tests.test_structured_json import Stream, client_for
    candidates = [{'local_id': 'char_a', 'canonical_name': 'Same name'},
                  {'local_id': 'char_b', 'canonical_name': 'Same name'}]
    context = {'sequence': {'sequence': 2, 'source': 'He waits.', 'adaptation': {'event': 'Waiting.'}},
               'previous_context': [{'sequence': 1, 'source': 'Earlier identity evidence.'}]}
    rows = [{'local_id': c['local_id'], 'presence': 'mentioned', 'label': 'Same name',
             'source_quote': 'He waits.'} for c in candidates]
    invalid = {'missing': rows[:1], 'duplicate': [rows[0], rows[0], rows[1]],
               'unknown': [*rows, {**rows[0], 'local_id': 'char_unknown'}]}[defect]
    client, create = client_for(*(Stream([json.dumps(response)]) for response in [
        {'characters': invalid, 'unresolved': []}, {'characters': invalid, 'unresolved': []},
        {'characters': rows[:1], 'unresolved': []}, {'characters': rows[1:], 'unresolved': []}]))
    requests = batches.Requests(output, client, 'qwen', settings(mode='strict_canon', correction_attempts=1))
    results = list(ce._presence_decisions(requests, context, candidates))
    assert [r for result in results for r in result['characters']] == rows
    assert create.call_count == 4
    for call in create.call_args_list:
        assert call.kwargs['response_format'] == {'type': 'json_schema', 'json_schema': ce.PRESENCE}
        messages = str(call.kwargs['messages'])
        assert 'Earlier identity evidence.' in messages and 'He waits.' in messages
    correction = str(create.call_args_list[1].kwargs['messages'])
    assert {'missing': 'Missing:', 'duplicate': 'duplicate:', 'unknown': 'unknown:'}[defect] in correction
    assert ('char_unknown' if defect == 'unknown' else 'char_a' if defect == 'duplicate' else 'char_b') in correction
    warning = util.load_json(output / 'character_profiles_requests.json')['warnings'][0]
    assert warning['input']['candidates'] == candidates
    assert warning['response']['characters'] == invalid


def test_local_injury_is_sourced_without_creating_a_global_scar(output, monkeypatch):
    data = consolidated(observation('Her arm is burned.', persistence='temporary', attribute='distinguishing_feature'))
    chapter = ce.selected_chapter(data, cinematic(data))
    pid = data['profiles'][0]['local_id']
    def chat(client, model, system, prompt, schema, *args):
        if schema == ce.PRESENCE:
            return {'characters': [{'local_id': pid, 'presence': 'visible', 'source_quote': 'Her arm is burned.',
                                    'label': 'Hero'}], 'unresolved': []}
        if schema == ce.LOCAL_STATES:
            return {'observations': [{'local_id': pid, 'trait': 'injury', 'value': 'Burned arm',
                                      'source_quote': 'Her arm is burned.'}]}
        if schema in (ce.LOCAL_REVIEW, vp.REVIEW_SCHEMA):
            return {'issues': []}
        return {f: [] for f in vp.FIELDS}
    monkeypatch.setattr(vp.lmstudio_json, 'chat_json', chat)
    result, _ = vp.generate(data, vp.prepare_profiles(data), output, None, 'qwen', settings(mode='strict_canon'), chapter)
    state = result['chapter_enrichment']['sequences'][0]['states'][0]
    assert state['kind'] == 'temporary'
    assert state['evidence'][0]['observation']['provenance']['locations'] == [{'start_char': 0, 'end_char': 18}]
    assert not result['profiles'][0]['effective_traits']
    assert not result['visual_designs']['entities'][0]['source_facts']['stable_visual_description']
    assert result['source_document'] == data


def test_presence_compact_retry_uses_schema_and_complete_candidates(output):
    from tests.test_structured_json import Stream, client_for
    data = consolidated()
    chapter = ce.selected_chapter(data, cinematic(data))
    pid = data['profiles'][0]['local_id']
    presence = {'characters': [{'local_id': pid, 'presence': 'visible', 'source_quote': chapter['sequences'][0]['source'],
                                'label': 'Hero'}], 'unresolved': []}
    client, create = client_for(Stream(['{"characters":']), Stream([json.dumps(presence)]),
                                Stream(['{"observations": []}']), Stream(['{"issues": []}']),
                                Stream([json.dumps(design(data))]), Stream(['{"issues": []}']))
    vp.generate(data, vp.prepare_profiles(data), output, client, 'qwen', settings(mode='strict_canon'), chapter)
    assert create.call_count == 6
    for call in create.call_args_list:
        assert call.kwargs['response_format']['type'] == 'json_schema'
        assert call.kwargs['response_format']['json_schema']['strict'] is True
    for call in create.call_args_list[:2]:
        assert pid in str(call.kwargs['messages'])
        assert call.kwargs['response_format']['json_schema']['name'] == ce.PRESENCE['name']


def test_absent_profiles_are_filtered_before_design_requests(output, monkeypatch):
    data = consolidated()
    def chat(client, model, system, prompt, schema, *args):
        assert schema == ce.PRESENCE
        return {'characters': [{'local_id': data['profiles'][0]['local_id'], 'presence': 'absent',
                                'label': '', 'source_quote': ''}], 'unresolved': [
                                    {'label': 'Unknown visitor', 'source_quote': cinematic(data)[0]['sequences'][0]['source']}]}
    monkeypatch.setattr(vp.lmstudio_json, 'chat_json', chat)
    result, _ = vp.generate(data, vp.prepare_profiles(data), output, None, 'qwen', settings(mode='strict_canon'),
                            ce.selected_chapter(data, cinematic(data)))
    assert result['profiles'] == []
    assert result['visual_designs']['entities'] == []
    assert result['chapter_enrichment']['sequences'][0]['unresolved'][0]['label'] == 'Unknown visitor'


def test_changed_selected_values_invalidate_design_cache(output, monkeypatch):
    data = consolidated(observation(), observation('Blue eyes.', value='blue eyes'))
    profile = data['profiles'][0]
    ids = [r['evidence_id'] for r in profile['supporting_evidence']]
    profile['unresolved_conflicts'] = [{'conflict_id': 'eyes', 'trait': 'eye_color', 'scope': 'same period',
        'evidence_ids': ids, 'choices': [{'id': str(i), 'value': v, 'selected': i == 0, 'evidence_ids': [ids[i]]}
                                        for i, v in enumerate(['green', 'blue'])]}]
    seen = []
    def chat(client, model, system, prompt, schema, *args):
        if schema == vp.REVIEW_SCHEMA:
            return {'issues': []}
        payload = json.loads(prompt)
        seen.append(payload['evidence'][0]['evidence_id'])
        response = {f: [] for f in vp.FIELDS}
        response['eyes'] = [{'description': payload['traits'][0]['value'], 'classification': 'canonical',
                             'evidence': [{'evidence_id': seen[-1]}]}]
        return response
    monkeypatch.setattr(vp.lmstudio_json, 'chat_json', chat)
    for selection in (0, 1):
        for i, c in enumerate(profile['unresolved_conflicts'][0]['choices']):
            c['selected'] = i == selection
        vp.generate(data, vp.prepare_profiles(data), output, None, 'qwen', settings(mode='strict_canon'))
    assert seen == ids

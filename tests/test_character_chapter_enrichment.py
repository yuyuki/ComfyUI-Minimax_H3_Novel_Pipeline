"""Chapter projection, source boundaries, identity edits and downstream transport."""
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline.core import character_chapter_enrichment as ce
from minimax_h3_novel_pipeline.core import character_visual_profiles as vp
from minimax_h3_novel_pipeline.core import character_profile_batches as batches, path_access, util
from tests.test_character_visual_profiles import consolidated, cinematic
from tests.test_character_profiles import settings
from tests.test_character_evidence import observation


def selected_chapter(data, chapters):
    chapter = deepcopy(chapters[0])
    chapter['_source_text'] = next(c['source_text'] for c in data['source_document']['chapters']
                                   if c['source_file'] == chapter['source_file'])
    return chapter


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
        raise AssertionError(schema)
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', chat)
    return calls




def test_temporary_clothing_stays_in_attested_source_and_not_design(output, monkeypatch):
    data = consolidated(observation('She wears a red evening dress.', attribute='clothing', persistence='temporary'))
    profile = data['profiles'][0]
    key = profile['supporting_evidence'][0]['evidence_id']
    profile['temporary_states'] = [{'trait': 'clothing', 'value': 'red evening dress', 'evidence_ids': [key]}]
    chapter = selected_chapter(data, cinematic(data))
    mock_chat(monkeypatch, data)
    p = vp.prepare_profiles(data)[0]
    requests = batches.Requests(output, None, 'qwen', settings())
    result = ce.enrich(data, [p], chapter, requests)
    assert result['sequences'][0]['states'][0]['value'] == 'red evening dress'
    elsewhere = {**chapter['sequences'][0], 'source': 'The next morning.'}
    assert ce.state_candidates(p, chapter, elsewhere) == []
    wrong_source = {**chapter, 'source_file': 'another.txt'}
    assert ce.state_candidates(p, wrong_source, chapter['sequences'][0]) == []
    assert p['effective_traits'] == []
    assert not p['chronological_changes']  # No injury-to-scar inference.


def test_contextual_pronouns_offscreen_and_mentions(output, monkeypatch):
    data = consolidated()
    chapter = selected_chapter(data, cinematic(data))
    chapter['sequences'].append({'sequence': 2, 'source': 'She waits.',
        'adaptation': {'initialState': 'She waits.', 'event': 'A voice is heard.', 'endingState': 'She waits.'}})
    calls = mock_chat(monkeypatch, data, 'offscreen')
    result = ce.enrich(data, vp.prepare_profiles(data), chapter,
                       batches.Requests(output, None, 'qwen', settings()))
    assert [s['characters'][0]['presence'] for s in result['sequences']] == ['offscreen', 'offscreen']
    presence_calls = [p for schema, p in calls if schema == ce.PRESENCE]
    assert presence_calls[1]['previous_context'][0]['characters'][0]['local_id'] == data['profiles'][0]['local_id']
    mock_chat(monkeypatch, data, 'mentioned')
    result = ce.enrich(data, vp.prepare_profiles(data), chapter,
                       batches.Requests(output, None, 'qwen', settings(force=True)))
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
    chapter = selected_chapter(data, cinematic(data))
    p = vp.prepare_profiles(data)[0]
    assert not p['effective_traits']
    mock_chat(monkeypatch, data)
    local = ce.enrich(data, [p], chapter, batches.Requests(output, None, 'qwen', settings()))
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
    chapter = selected_chapter(data, chapter)
    mock_chat(monkeypatch, data)
    default_chat = batches.lmstudio_json.chat_json
    def chat(client, model, system, prompt, schema, *args):
        if schema != ce.STATES:
            return default_chat(client, model, system, prompt, schema, *args)
        payload = json.loads(prompt)
        return {'states': [{'state_id': r['state_id'], 'applies': r['chronology']['side'] == side,
                            'source_quote': source['source_text'], 'reason': 'Explicit before/after source wording.'}
                           for r in payload['states']]}
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', chat)
    local = ce.enrich(data, vp.prepare_profiles(data), chapter,
                      batches.Requests(output, None, 'qwen', settings()))
    states = local['sequences'][0]['states']
    assert len(states) == 1
    assert states[0]['chronology']['side'] == side
    assert states[0]['value'] == ('unmarked cheek' if side == 'before_id' else 'scar')








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
    chapter = selected_chapter(data, cinematic(data))
    pid = data['profiles'][0]['local_id']
    prompts = []
    def chat(client, model, system, prompt, schema, *args):
        assert schema == ce.PRESENCE
        prompts.append(prompt)
        return {'characters': [{'local_id': pid, 'presence': 'mentioned', 'label': 'Hero',
                'source_quote': 'Invented quote.' if len(prompts) == 1 else chapter['sequences'][0]['source']}],
                'unresolved': []}
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', chat)
    result = ce.enrich(data, vp.prepare_profiles(data), chapter,
                       batches.Requests(output, None, 'qwen', settings(correction_attempts=1)))
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
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', lambda *args: deepcopy(response))
    with pytest.raises(ValueError, match='no valid decision'):
        ce.enrich(data, vp.prepare_profiles(data), selected_chapter(data, cinematic(data)),
                  batches.Requests(output, None, 'qwen', settings(correction_attempts=1)))
    warning = util.load_json(output / 'character_profiles_requests.json')['warnings'][0]
    assert warning['attempts'] == 2
    assert warning['response'] == response
    assert pid in warning['reason']
    assert warning['input']['sequence']['source'] == cinematic(data)[0]['sequences'][0]['source']


def test_presence_bounds_48_candidates_without_losing_any(output, monkeypatch):
    data = consolidated()
    chapter = selected_chapter(data, cinematic(data))
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
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', chat)
    result = ce.enrich(data, profiles, chapter, batches.Requests(output, None, 'qwen', settings()))
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
    requests = batches.Requests(output, client, 'qwen', settings(correction_attempts=1))
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




def test_presence_compact_retry_uses_schema_and_complete_candidates(output):
    from tests.test_structured_json import Stream, client_for
    data = consolidated()
    chapter = selected_chapter(data, cinematic(data))
    pid = data['profiles'][0]['local_id']
    presence = {'characters': [{'local_id': pid, 'presence': 'visible', 'source_quote': chapter['sequences'][0]['source'],
                                'label': 'Hero'}], 'unresolved': []}
    client, create = client_for(Stream(['{"characters":']), Stream([json.dumps(presence)]),
                                Stream(['{"observations": []}']), Stream(['{"issues": []}']),
                                )
    ce.enrich(data, vp.prepare_profiles(data), chapter, batches.Requests(output, client, 'qwen', settings()))
    assert create.call_count == 4
    for call in create.call_args_list:
        assert call.kwargs['response_format']['type'] == 'json_schema'
        assert call.kwargs['response_format']['json_schema']['strict'] is True
    for call in create.call_args_list[:2]:
        assert pid in str(call.kwargs['messages'])
        assert call.kwargs['response_format']['json_schema']['name'] == ce.PRESENCE['name']






def test_chapter_candidates_exclude_other_chapters_but_keep_named_mentions():
    data = consolidated()
    chapter = selected_chapter(data, cinematic(data))
    hero = vp.prepare_profiles(data)[0]
    outsider = deepcopy(hero)
    outsider.update(local_id='other', canonical_name='Ted Conrad', aliases=[])
    outsider['supporting_evidence'][0]['observation']['provenance']['file'] = 'elsewhere.txt'
    mentioned = deepcopy(outsider)
    mentioned.update(local_id='mentioned', canonical_name='Elise', aliases=['Élise'])
    candidates = ce.chapter_candidates([hero, outsider, mentioned], chapter)
    assert [r['local_id'] for r in candidates] == [hero['local_id'], 'mentioned']
    assert candidates[0]['chapter_evidence'] == [chapter['sequences'][0]['source']]
    assert candidates[1]['chapter_evidence'] == []
    mentioned['aliases'] = ['Éli']
    assert ce.chapter_candidates([mentioned], chapter) == []


@pytest.mark.parametrize('defect', ['whitespace', 'identity', 'wording'])
def test_local_quote_repair_and_precise_corrections(output, monkeypatch, defect):
    data = consolidated(observation('Her arm\nis burned.', persistence='temporary', attribute='distinguishing_feature'))
    chapter = selected_chapter(data, cinematic(data))
    pid = data['profiles'][0]['local_id']
    mock_chat(monkeypatch, data)
    default_chat = batches.lmstudio_json.chat_json
    prompts = []
    def chat(client, model, system, prompt, schema, *args):
        if schema != ce.LOCAL_STATES:
            return default_chat(client, model, system, prompt, schema, *args)
        prompts.append(prompt)
        first = len(prompts) == 1
        return {'observations': [{'local_id': 'Hero' if first and defect == 'identity' else pid,
            'trait': 'injury', 'value': 'Burned arm',
            'source_quote': 'Her arm is injured.' if first and defect == 'wording' else 'Her arm is burned.'}]}
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', chat)
    result = ce.enrich(data, vp.prepare_profiles(data), chapter,
                       batches.Requests(output, None, 'qwen', settings()))
    state = result['sequences'][0]['states'][0]
    assert state['source_quote'] == 'Her arm\nis burned.'
    assert state['evidence'][0]['observation']['provenance']['locations'] == [{'start_char': 0, 'end_char': 18}]
    assert len(prompts) == (1 if defect == 'whitespace' else 2)
    if defect != 'whitespace':
        assert pid in prompts[1].split('\nRejected:')[1]
        assert ('character.local_id' if defect == 'identity' else 'sequence.source') in prompts[1].split('\nRejected:')[1]


def test_ambiguous_speaker_does_not_become_multiple_cast_identities(output, monkeypatch):
    data = consolidated()
    chapter = selected_chapter(data, cinematic(data))
    profile = vp.prepare_profiles(data)[0]
    profiles = [{**deepcopy(profile), 'local_id': f'char_{i}', 'canonical_name': name}
                for i, name in enumerate(['Shannon (Jack)', 'Ted Conrad', 'Un homme'])]
    mock_chat(monkeypatch, data, 'uncertain')
    result = ce.enrich(data, profiles, chapter, batches.Requests(output, None, 'qwen', settings()))
    seq = result['sequences'][0]
    assert seq['characters'] == []
    assert seq['states'] == []
    assert seq['unresolved'] == [{'label': 'Unidentified character', 'source_quote': chapter['sequences'][0]['source']}]


def test_no_candidates_still_detects_unresolved_mentions(output, monkeypatch):
    data = consolidated()
    chapter = selected_chapter(data, cinematic(data))
    quote = chapter['sequences'][0]['source']
    def chat(client, model, system, prompt, schema, *args):
        assert schema == ce.PRESENCE
        assert json.loads(prompt)['candidates'] == []
        return {'characters': [], 'unresolved': [{'label': 'Élise', 'source_quote': quote}]}
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', chat)
    result = ce.enrich(data, [], chapter, batches.Requests(output, None, 'qwen', settings()))
    assert result['sequences'][0]['unresolved'] == [{'label': 'Élise', 'source_quote': quote}]


@pytest.mark.parametrize('source,quote', [
    ('Indy se balançait dans l’obscurité.', "Indy se balançait dans l'obscurité."),
    ("Indy se balançait dans l'obscurité.", 'Indy se balançait dans l’obscurité.'),
    ('Indy\nse balançait dans l’obscurité.', "Indy se balançait dans l'obscurité."),
])
def test_presence_restores_apostrophes_and_whitespace(source, quote):
    row = {'source_quote': quote}
    ce.presence_quote(row, 'Start. ' + source + ' End.', 'Indy')
    assert row['source_quote'] == source


@pytest.mark.parametrize('quote', [
    "Indy se balançait dans l'obscurite.",  # Accent removed.
    "Indy se balançait dans l'obscurité!",  # Punctuation changed.
    "Indy se balançait dans lobscurité.",  # Apostrophe removed.
    "Indy attendait dans l'obscurité.",  # Word replaced.
])
def test_apostrophe_repair_does_not_accept_changed_evidence(quote):
    with pytest.raises(ValueError, match='not a contiguous quote'):
        ce.presence_quote({'source_quote': quote}, 'Indy se balançait dans l’obscurité.', 'Indy')


def test_real_presence_transport_caches_visible_with_original_apostrophe(output):
    from tests.test_structured_json import Stream, client_for
    source = ('Indy se balançait dans l’obscurité, suspendu tel un croissant de lune '
              'à une corde qui lui meurtrissait le torse et les aisselles.')
    candidates = [{'local_id': 'char_indy', 'canonical_name': 'Indy'},
                  {'local_id': 'char_doriane', 'canonical_name': 'Doriane'}]
    response = {'characters': [
        {'local_id': 'char_indy', 'presence': 'visible', 'label': 'Indy',
         'source_quote': source.replace('’', "'")},
        {'local_id': 'char_doriane', 'presence': 'absent', 'label': '', 'source_quote': ''}], 'unresolved': []}
    client, create = client_for(Stream([json.dumps(response)]))
    requests = batches.Requests(output, client, 'qwen', settings(correction_attempts=0))
    context = {'sequence': {'sequence': 1, 'source': source}, 'previous_context': []}
    for _ in range(2):
        result = list(ce._presence_decisions(requests, context, candidates))
        assert result[0]['characters'][0]['source_quote'] == source
        assert result[0]['characters'][0]['presence'] == 'visible'
    assert create.call_count == 1
    assert requests.stats['cache_hits'] == 1
    assert requests.stats['warnings'] == []
    assert create.call_args.kwargs['response_format'] == {'type': 'json_schema', 'json_schema': ce.PRESENCE}
    cached = util.load_json(next(requests.cache_dir.glob('*.json')))
    assert cached['characters'][0]['presence'] == 'visible'
    assert cached['characters'][0]['source_quote'] == source


@pytest.mark.parametrize('existing', [False, True])
def test_temporary_states_restore_source_apostrophes(output, monkeypatch, existing):
    quote = 'L’homme porte une veste.'
    data = consolidated(observation(quote, attribute='clothing', persistence='temporary', value='une veste'))
    profile = data['profiles'][0]
    if existing:
        key = profile['supporting_evidence'][0]['evidence_id']
        profile['temporary_states'] = [{'trait': 'clothing', 'value': 'une veste', 'evidence_ids': [key]}]
    chapter = selected_chapter(data, cinematic(data))
    mock_chat(monkeypatch, data)
    default_chat = batches.lmstudio_json.chat_json
    def chat(client, model, system, prompt, schema, *args):
        if schema == ce.LOCAL_STATES and not existing:
            return {'observations': [{'local_id': profile['local_id'], 'trait': 'clothing',
                                     'value': 'une veste', 'source_quote': quote.replace('’', "'")}]}
        response = default_chat(client, model, system, prompt, schema, *args)
        if schema == ce.STATES:
            for state in response['states']:
                state['source_quote'] = quote.replace('’', "'")
        return response
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', chat)
    result = ce.enrich(data, vp.prepare_profiles(data), chapter,
                       batches.Requests(output, None, 'qwen', settings(correction_attempts=0)))
    assert result['sequences'][0]['states'][0]['source_quote'] == quote

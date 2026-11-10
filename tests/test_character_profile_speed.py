"""Request reduction must preserve source text, semantic checks and retry budgets."""
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline.core import character_profile_batches as batches, character_profiles as cp, path_access
from tests.test_character_evidence import observation
from tests.test_character_profiles import document, group, settings
from tests.test_character_profile_batches import fake_model, request_payload
from tests.test_character_profile_recovery import narrative_document


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)


def test_overlapping_context_is_losslessly_shared_and_source_scoped():
    text = "0123456789" * 100
    rows = []
    for i, (start, end, chapter) in enumerate([(0, 700, 1), (200, 900, 1), (0, 700, 2)]):
        rows.append({'evidence_id': str(i), 'observation': {
            'source_file': 'novel.txt', 'chapter': chapter, 'quotation': text[start:start + 10],
            'source_context': [{'start_char': start, 'end_char': end, 'text': text[start:end]}]}})
    original = deepcopy(rows)
    payload = batches.shared_context_payload({'observations': rows})
    assert rows == original
    assert len(payload['source_contexts']) == 2
    assert len(batches.encoded(payload)) < len(batches.encoded({'observations': rows}))
    assert request_payload(batches.encoded(payload))['observations'] == rows
    assert batches.shared_context_payload(payload) == payload


def test_identity_comparison_reused_with_new_job_numbers_but_revalidated(tmp_path, monkeypatch):
    records = cp.prepare_evidence(document(observation(), observation('Other eyes.')))
    rows = [batches.compact_record(k, v) for k, v in records.items()]
    calls = []
    def model(*args):
        payload = request_payload(args[3])
        calls.append(payload)
        return {'decisions': [{'job_id': j['job_id'], 'relation': 'same',
                              'left_support': [0], 'right_support': [0]} for j in payload['jobs']]}
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', model)
    requests = batches.Requests(tmp_path, None, 'qwen', settings(correction_attempts=0))
    def job(number):
        return {'job_id': number, 'left_group': int(number), 'right_group': 99,
                'left': rows[:1], 'right': rows[1:]}
    assert list(requests.decisions([job('0')], batches.IDENTITY_SYSTEM, batches.IDENTITY_PAIR))[0][0]['relation'] == 'same'
    assert list(requests.decisions([job('1')], batches.IDENTITY_SYSTEM, batches.IDENTITY_PAIR))[0][0]['job_id'] == '1'
    assert len(calls) == 1
    assert requests.stats['identity_decisions_reused'] == 1
    def reject(decision, job):
        if decision['relation'] == 'same':
            raise ValueError('New dossier has contradictory evidence')
    assert list(requests.decisions([job('2')], batches.IDENTITY_SYSTEM, batches.IDENTITY_PAIR, reject))[0][0]['relation'] == 'uncertain'
    assert len(calls) == 2


def test_partial_partition_retries_only_unresolved_group(tmp_path, monkeypatch):
    data = narrative_document('Élise avait les yeux verts. Paul était grand. Paul avait les yeux bleus.',
        observation(), observation('Paul était grand.', character='Paul'),
        observation('Paul avait les yeux bleus.', character='Paul'))
    records = cp.prepare_evidence(data)
    calls = []
    def model(*args):
        assert args[4] == cp.SCHEMA
        rows = request_payload(args[3])['observations']
        calls.append([row['evidence_id'] for row in rows])
        if len(calls) == 1:
            return {'profiles': [group(calls[0][:1]), group(calls[0][1:], canonical_name='Paul', identity_status='ambiguous')]}
        return {'profiles': [group(calls[-1], canonical_name='Paul')]}
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', model)
    requests = batches.Requests(tmp_path, None, 'qwen', settings(correction_attempts=1))
    groups = batches.partition_identities(records, requests)
    assert calls == [list(records), list(records)[1:]]
    assert sorted(len(g['evidence']) for g in groups) == [1, 2]
    assert {r['evidence_id'] for g in groups for r in g['evidence']} == set(records)


@pytest.mark.parametrize('fails', [False, True])
def test_request_timings_include_failures_and_exclude_cache_hits(tmp_path, monkeypatch, fails):
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(batches, 'perf_counter', lambda: next(ticks))
    def model(*args):
        if fails:
            raise ValueError('Invalid response')
        return {'observations': []}
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', model)
    requests = batches.Requests(tmp_path, None, 'qwen', settings(correction_attempts=0))
    requests.call('Classify', batches.TRAITS, {'observations': []}, lambda result: None)
    if not fails:
        requests.call('Classify', batches.TRAITS, {'observations': []}, lambda result: None)
    assert requests.stats['request_seconds_by_schema'] == {'character_precise_traits': 2.5}
    assert requests.stats['request_seconds'] == 2.5
    assert len(requests.stats['request_timings']) == 1
    json.dumps(requests.stats)


def test_cross_chapter_batching_reduces_requests_with_identical_profiles(tmp_path, monkeypatch):
    records = cp.prepare_evidence(document(*[
        observation(f'Élise avait les yeux verts. Chapitre {i}.') for i in range(24)]))
    original = deepcopy(records)
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', fake_model)
    previous = batches.Requests(tmp_path / 'chapter_batches', None, 'qwen', settings(correction_attempts=0))
    chapter_groups = []
    for key, item in records.items():
        chapter_groups.extend(batches.partition_identities({key: item}, previous))
    old_profiles = batches.reconcile_identities(chapter_groups, records, previous)
    current = batches.Requests(tmp_path / 'combined_batches', None, 'qwen', settings(correction_attempts=0))
    groups = batches.partition_identities(records, current)
    new_profiles = batches.reconcile_identities(groups, records, current)
    assert new_profiles == old_profiles
    assert records == original
    assert previous.stats['requests'] >= 29
    assert current.stats['requests'] == 1
    assert current.stats['requests_by_schema'] == {'character_profile_plan': 1}


def test_cross_chapter_group_without_support_is_not_accepted(tmp_path, monkeypatch):
    records = cp.prepare_evidence(document(observation(), observation('Élise avait les yeux bleus.')))
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', lambda *args: {'profiles': [group(list(records))]})
    requests = batches.Requests(tmp_path, None, 'qwen', settings(correction_attempts=0))
    groups = batches.partition_identities(records, requests)
    assert len(groups) == len(records)
    assert all(row['identity_status'] == 'ambiguous' for row in groups)
    assert 'every source chapter' in requests.stats['warnings'][0]['reason']


@pytest.mark.parametrize('corrections', [0, 1, 2])
def test_invalid_partition_never_restarts_correction_budget(tmp_path, monkeypatch, corrections):
    records = cp.prepare_evidence(document(observation(), observation('Other eyes.')))
    calls = []
    def model(*args):
        calls.append(request_payload(args[3]))
        return {'profiles': []}
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', model)
    requests = batches.Requests(tmp_path, None, 'qwen', settings(correction_attempts=corrections))
    groups = batches.partition_identities(records, requests)
    assert len(calls) == corrections + 1
    assert len(groups) == len(records)


@pytest.mark.parametrize('change', ['quotation', 'alias', 'force'])
def test_changed_identity_input_or_force_prevents_reuse(tmp_path, monkeypatch, change):
    records = cp.prepare_evidence(document(observation(), observation('Other eyes.')))
    rows = [batches.compact_record(k, v) for k, v in records.items()]
    requests = batches.Requests(tmp_path, None, 'qwen', settings(correction_attempts=0, force=change == 'force'))
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', fake_model)
    job = {'job_id': '0', 'left': rows[:1], 'right': rows[1:],
           'left_identity': {'canonical_name': 'Élise', 'aliases': [], 'identity_status': 'confirmed'}}
    list(requests.decisions([job], batches.IDENTITY_SYSTEM, batches.IDENTITY_PAIR))
    job = deepcopy(job)
    job['job_id'] = '1'
    if change == 'quotation':
        job['left'][0]['observation']['quotation'] += ' Additional evidence.'
    elif change == 'alias':
        job['left_identity']['aliases'] = ['Lise']
    list(requests.decisions([job], batches.IDENTITY_SYSTEM, batches.IDENTITY_PAIR))
    assert requests.stats['requests'] == 2
    assert requests.stats['identity_decisions_reused'] == 0


def test_shared_contexts_survive_transport_retry(tmp_path):
    from tests.test_structured_json import Stream, client_for
    text = 'Élise avait les yeux verts. Elle était grande.'
    records = cp.prepare_evidence(narrative_document(text, observation(), observation('Elle était grande.')))
    payload = {'observations': [batches.compact_record(k, v) for k, v in records.items()]}
    response = {'profiles': [group(list(records))]}
    client, create = client_for(Stream(['{']), Stream([json.dumps(response)]))
    requests = batches.Requests(tmp_path, client, 'qwen3.5', settings(correction_attempts=0))
    assert requests.call(cp.SYSTEM, cp.SCHEMA, payload, lambda result: cp.build_profiles(result, records)) == response
    assert create.call_count == 2
    for call in create.call_args_list:
        assert call.kwargs['response_format'] == {'type': 'json_schema', 'json_schema': cp.SCHEMA}
        prompts = [m['content'] for m in call.kwargs['messages'] if m['role'] == 'user']
        packed = next(request_payload(p) for p in prompts if p.startswith('{'))
        assert packed['observations'] == payload['observations']


def test_ambiguous_multi_observation_group_is_forbidden_by_transport_schema():
    from jsonschema import Draft202012Validator
    validator = Draft202012Validator(cp.SCHEMA['schema'])
    assert list(validator.iter_errors({'profiles': [group(['a', 'b'], identity_status='ambiguous')]}))
    assert not list(validator.iter_errors({'profiles': [group(['a'], identity_status='ambiguous')]}))
    assert not list(validator.iter_errors({'profiles': [group(['a', 'b'])]}))
    compact = batches.lmstudio_json._qwen35_compact_schema(cp.SCHEMA)
    assert list(Draft202012Validator(compact['schema']).iter_errors(
        {'profiles': [group(['a', 'b'], identity_status='ambiguous')]}))
    with pytest.raises(ValueError, match='non-empty'):
        batches.lmstudio_json._validate_canonical_names(
            {'profiles': [group(['a'], canonical_name=' ')]}, cp.SCHEMA['schema'])


def test_failed_grouping_does_not_compare_unrelated_cooccurring_names(tmp_path, monkeypatch):
    names = [f'Person{i}' for i in range(12)]
    scene = ' '.join(f'{name} avait les yeux verts.' for name in names)
    records = cp.prepare_evidence(narrative_document(scene, *[
        observation(f'{name} avait les yeux verts.', character=name) for name in names]))
    calls = []
    def model(*args):
        calls.append(args[4]['name'])
        assert args[4] == cp.SCHEMA  # No possible alias merge; no pair requests.
        return {'profiles': [group(list(records), identity_status='ambiguous')]}
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', model)
    requests = batches.Requests(tmp_path, None, 'qwen', settings(correction_attempts=0))
    groups = batches.partition_identities(records, requests)
    profiles = batches.reconcile_identities(groups, records, requests)
    assert len(profiles) == len(records)
    assert calls == ['character_profile_plan']
    assert requests.stats['identity_pairs_pruned'] == 66
    assert {r['evidence_id'] for p in profiles for r in p['supporting_evidence']} == set(records)


def test_alias_candidate_components_keep_indirect_bridges_and_homonyms(tmp_path, monkeypatch):
    data = document(
        observation('Henry Jones was called Indy.', character='Henry'),
        observation('Jones avait une cicatrice.', character='Jones'),
        observation('Indy avait les yeux verts.', character='Indy'),
        observation('Henry, un autre homme, entra.', character='Henry'),
        observation('Paul regardait Henry et Indy.', character='Paul'))
    records = cp.prepare_evidence(data)
    components = batches._identity_components(records)
    assert components['Henry'] == components['Jones'] == components['Indy']
    assert components['Paul'] != components['Henry']
    seen = []
    def model(*args):
        payload = request_payload(args[3])
        seen.extend(payload['jobs'])
        return {'decisions': [{'job_id': j['job_id'], 'relation': 'different',
                              'left_support': [0], 'right_support': [0]} for j in payload['jobs']]}
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', model)
    groups = [group([key], canonical_name=item['character']) for key, item in records.items()]
    profiles = batches.reconcile_identities(groups, records,
        batches.Requests(tmp_path, None, 'qwen', settings(correction_attempts=0)))
    assert len(profiles) == len(groups)
    assert any(j['left_identity']['canonical_name'] == j['right_identity']['canonical_name'] == 'Henry' for j in seen)
    assert all('Paul' not in (j['left_identity']['canonical_name'], j['right_identity']['canonical_name']) for j in seen)

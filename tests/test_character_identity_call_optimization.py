"""A failed identity job must not repeatedly regenerate validated neighbors."""
from copy import deepcopy

import pytest

from minimax_h3_novel_pipeline.core import character_profile_batches as batches, character_profiles as cp, path_access
from tests.test_character_evidence import observation
from tests.test_character_profiles import document, group, settings
from tests.test_character_profile_batches import request_payload


@pytest.mark.parametrize('failure', ['missing', 'invalid_support', 'semantic', 'duplicate'])
def test_partial_identity_recovery_reduces_calls_without_changing_decisions(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    records = cp.prepare_evidence(document(observation(), observation('Other eyes.'), observation('Third eyes.')))
    rows = [batches.compact_record(k, v) for k, v in records.items()]
    jobs = [{'job_id': str(i), 'left': rows[:1], 'right': rows[i + 1:i + 2]} for i in range(2)]
    original = deepcopy(jobs)
    def check(decision, job):
        if failure == 'semantic' and job['job_id'] == '1' and decision['relation'] == 'same':
            raise ValueError('Unsupported identity merge.')
    def exercise(optimized):
        calls = []
        def model(*args):
            assert args[4] == batches.IDENTITY_PAIR
            payload = request_payload(args[3])
            calls.append(payload)
            decisions = [{'job_id': j['job_id'], 'relation': 'different',
                          'left_support': [0], 'right_support': [0]} for j in payload['jobs']]
            if len(decisions) == 2:
                if failure == 'missing':
                    decisions.pop()
                elif failure == 'invalid_support':
                    decisions[-1]['right_support'] = [999]
                elif failure == 'semantic':
                    decisions[-1]['relation'] = 'same'
                else:
                    decisions.append(deepcopy(decisions[-1]))
            return {'decisions': decisions}
        monkeypatch.setattr(batches.lmstudio_json, 'chat_json', model)
        requests = batches.Requests(tmp_path / str(optimized), None, 'qwen', settings(correction_attempts=2))
        if not optimized:
            call = requests.call
            def previous_call(*args, **kwargs):
                kwargs['recover_partial'] = False
                return call(*args, **kwargs)
            requests.call = previous_call
        result = list(requests.decisions(jobs, batches.IDENTITY_SYSTEM, batches.IDENTITY_PAIR, check))
        return sorted(result, key=lambda pair: pair[0]['job_id']), calls, requests
    before, old_calls, _ = exercise(False)
    after, new_calls, requests = exercise(True)
    assert before == after
    assert jobs == original
    assert len(old_calls) == 4
    assert len(new_calls) == 2
    assert [j['job_id'] for j in new_calls[1]['jobs']] == ['1']
    assert new_calls[1]['jobs'][0] == new_calls[0]['jobs'][1]
    assert requests.stats['early_identity_recoveries'] == 1
    assert requests.stats['identity_jobs_preserved'] == 1
    diagnostic = requests.stats['identity_attempt_diagnostics'][0]
    assert diagnostic['accepted_job_ids'] == ['0']
    assert diagnostic['early_recovery_enabled'] is True
    assert diagnostic['corrections_remaining'] == 2
    # Only the fully validated single-job response is cached, never the partial lot.
    assert len(list(requests.cache_dir.glob('*.json'))) == 1


def test_no_valid_identity_decision_keeps_retry_budget_and_uncertainty(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    records = cp.prepare_evidence(document(observation(), observation('Other eyes.')))
    rows = [batches.compact_record(k, v) for k, v in records.items()]
    jobs = [{'job_id': '0', 'left': rows[:1], 'right': rows[1:]}]
    calls = []
    def model(*args):
        calls.append(args)
        assert args[4] == batches.IDENTITY_PAIR
        return {'decisions': [{'job_id': '0', 'relation': 'same', 'left_support': [999], 'right_support': [0]}]}
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', model)
    requests = batches.Requests(tmp_path, None, 'qwen', settings(correction_attempts=2))
    result = list(requests.decisions(jobs, batches.IDENTITY_SYSTEM, batches.IDENTITY_PAIR))
    assert len(calls) == 3
    assert result[0][0]['relation'] == 'uncertain'
    assert not requests.stats.get('early_identity_recoveries')
    diagnostics = requests.stats['identity_attempt_diagnostics']
    assert [row['attempt'] for row in diagnostics] == [1, 2, 3]
    assert [row['corrections_remaining'] for row in diagnostics] == [2, 1, 0]
    assert all(row['accepted_job_ids'] == [] for row in diagnostics)
    assert all(row['response_received'] for row in diagnostics)
    assert not list(requests.cache_dir.glob('*.json'))


def test_unsupported_alias_retry_explains_uncertainty_without_losing_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    records = cp.prepare_evidence(document(
        observation('Élise regardait Lise.'),
        observation('Lise regardait Élise.', character='Lise')))
    groups = [group([key], canonical_name=item['character']) for key, item in records.items()]
    prompts = []

    def model(*args):
        assert args[4] == batches.IDENTITY_PAIR
        prompts.append(args[3])
        return {'decisions': [{'job_id': '0', 'relation': 'same' if len(prompts) == 1 else 'uncertain',
                              'left_support': [0], 'right_support': [0]}]}

    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', model)
    requests = batches.Requests(tmp_path, None, 'qwen', settings(correction_attempts=2))
    profiles = batches.reconcile_identities(groups, records, requests)
    assert len(prompts) == 2
    assert 'job_id=0' in prompts[1]
    assert 'return uncertain' in prompts[1]
    assert 'Missing identity proof does not justify different' in prompts[1]
    assert request_payload(prompts[0]) == request_payload(prompts[1])
    assert len(profiles) == 2
    assert {row['evidence_id'] for p in profiles for row in p['supporting_evidence']} == set(records)


def test_invalid_positions_retry_names_job_side_and_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    records = cp.prepare_evidence(document(observation(), observation('Other eyes.')))
    rows = [batches.compact_record(k, v) for k, v in records.items()]
    jobs = [{'job_id': '7', 'left': rows[:1], 'right': rows[1:]}]
    prompts = []

    def model(*args):
        prompts.append(args[3])
        return {'decisions': [{'job_id': '7', 'relation': 'uncertain', 'left_support': [0],
                              'right_support': [4] if len(prompts) == 1 else [0]}]}

    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', model)
    requests = batches.Requests(tmp_path, None, 'qwen', settings(correction_attempts=2))
    decisions = list(requests.decisions(jobs, batches.IDENTITY_SYSTEM, batches.IDENTITY_PAIR))
    assert len(prompts) == 2
    assert 'job_id=7: right_support must contain unique integers from 0 to 0' in prompts[1]
    assert 'A side with one observation requires [0]' in prompts[1]
    assert request_payload(prompts[0]) == request_payload(prompts[1])
    assert decisions[0][0]['relation'] == 'uncertain'

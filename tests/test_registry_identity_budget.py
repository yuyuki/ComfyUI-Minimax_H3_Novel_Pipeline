"""Oversized registry comparisons preserve evidence, disambiguation and strict requests."""
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline.core import character_profile_batches as batches
from minimax_h3_novel_pipeline.core import character_visual_integration as integration
from minimax_h3_novel_pipeline.core import path_access
from tests.test_character_profiles import settings


def context():
    return {'registry_character': {'global_id': 'C1', 'canonical_name': 'Élise'},
            'chapter_id': 'chapter1', 'chapter_sources': ['Élise attends.'],
            'catalog_characters': [{'local_id': 'c1', 'canonical_name': 'Élise'},
                                   {'local_id': 'c2', 'canonical_name': 'Another Élise'}],
            'profile': {'local_id': 'p1', 'canonical_name': 'Élise', 'aliases': [],
                        'linked_local_ids': [], 'identity_support': [],
                        'supporting_evidence': [
                            {'evidence_id': str(i), 'observation': {'quotation': 'é' * 1800}}
                            for i in range(10)]}}


@pytest.mark.parametrize('negative_batch', [None, 0, 1])
def test_oversized_identity_preserves_every_observation_and_retries(tmp_path, monkeypatch, negative_batch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    requests = batches.Requests(tmp_path, None, 'qwen', settings(context_tokens=32768))
    payload = context()
    original = deepcopy(payload)
    with pytest.raises(ValueError, match='budget'):
        requests.call(integration.IDENTITY_SYSTEM, integration.IDENTITY, payload, lambda value: None)
    calls, accepted = [], []

    def chat(client, model, system, prompt, contract, temperature, max_tokens):
        supplied = json.JSONDecoder().raw_decode(prompt)[0]
        assert contract == integration.IDENTITY
        assert contract['strict'] is True
        assert len(batches.encoded(supplied).encode()) <= requests.capacity(system, contract)
        calls.append(supplied)
        # Each batch needs one semantic correction, preserving the exact payload.
        if len(calls) % 2:
            return {'same': True, 'source_quote': 'invented', 'reason': 'Bad quote.'}
        accepted.append(supplied)
        return {'same': len(accepted) - 1 != negative_batch,
                'source_quote': 'Élise attends.', 'reason': 'Reviewed.'}

    def check(value):
        if value['same'] and value['source_quote'] != 'Élise attends.':
            raise ValueError('Identity mapping requires an exact current source quote.')

    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', chat)
    assert integration._identity_matches(requests, payload, check) is (negative_batch is None)
    assert len(accepted) > 1
    assert payload == original
    assert [r for p in accepted for r in p['profile']['supporting_evidence']] == original['profile']['supporting_evidence']
    for p in accepted:
        for field in ('chapter_sources', 'catalog_characters', 'registry_character'):
            assert p[field] == original[field]
    assert all(calls[i] == calls[i + 1] for i in range(0, len(calls), 2))
    # A repeat uses the same validated per-batch caches, including negative decisions.
    count = len(calls)
    assert integration._identity_matches(requests, payload, check) is (negative_batch is None)
    assert len(calls) == count


def test_small_identity_stays_one_request(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    requests = batches.Requests(tmp_path, None, 'qwen', settings(context_tokens=131072))
    payload = context()
    calls = []

    def chat(client, model, system, prompt, contract, *args):
        calls.append(json.loads(prompt))
        assert system == integration.IDENTITY_SYSTEM
        return {'same': True, 'source_quote': 'Élise attends.', 'reason': 'Attested.'}

    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', chat)
    assert integration._identity_matches(requests, payload, lambda value: None)
    assert calls == [payload]


def test_indivisible_context_reports_actionable_budget_before_any_call(tmp_path, monkeypatch):
    requests = batches.Requests(tmp_path, None, 'qwen', settings(context_tokens=32768))
    payload = context()
    payload['chapter_sources'] = ['é' * 30000]
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', lambda *args: pytest.fail('No request should be sent'))
    with pytest.raises(ValueError, match='context_tokens=32768, max_tokens=16384'):
        integration._identity_matches(requests, payload, lambda value: None)

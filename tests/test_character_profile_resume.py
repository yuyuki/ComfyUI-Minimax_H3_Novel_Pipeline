"""Reuse previous-run decisions only after current request and content validation."""
from contextlib import nullcontext
import json
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import character_profile_batches as batches, path_access
from minimax_h3_novel_pipeline.nodes import consolidate_character_profiles as node
from tests.test_character_evidence import observation
from tests.test_character_profiles import document
from tests.test_character_profile_batches import fake_model


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    factory = Mock(side_effect=lambda *a: (nullcontext(None), 'qwen'))
    monkeypatch.setattr(node.lmstudio_pipeline, 'make_client_and_model', factory)
    chat = Mock(side_effect=fake_model)
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', chat)
    def run(stamp, **kwargs):
        return node.ConsolidateCharacterProfilesNode().run(
            document(observation()), {'api_url': 'http://127.0.0.1:1234/v1', 'run_folder': stamp},
            out_dir='profiles', **kwargs)['result'][0]
    return run, chat, factory, tmp_path


def test_resume_same_result_zero_calls_and_promotes_only_hits(setup):
    run, chat, _, root = setup
    first = run('20261031000001')
    assert chat.call_count == 2
    previous = root / '20261031000001/profiles/.cache'
    before = {p.name: p.read_bytes() for p in previous.iterdir()}
    chat.reset_mock()
    assert run('20261031000002', resume_run='20261031000001') == first
    chat.assert_not_called()
    stats = json.loads((root / '20261031000002/profiles/character_profiles_requests.json').read_text())
    assert stats['requests'] == 0
    assert stats['cache_hits'] == stats['resume_cache_hits'] == 2
    assert {p.name: p.read_bytes() for p in previous.iterdir()} == before
    assert {p.name: p.read_bytes() for p in (root / '20261031000002/profiles/.cache').iterdir()} == before


def test_prompt_change_recomputes_classification_only(setup, monkeypatch):
    run, chat, _, root = setup
    first = run('20261031000001')
    monkeypatch.setattr(batches, 'TRAIT_SYSTEM', batches.TRAIT_SYSTEM + '\nNew classification instruction.')
    chat.reset_mock()
    assert run('20261031000002', resume_run='20261031000001') == first
    assert chat.call_count == 1
    assert chat.call_args.args[4] == batches.TRAITS
    stats = json.loads((root / '20261031000002/profiles/character_profiles_requests.json').read_text())
    assert stats['resume_cache_hits'] == 1


@pytest.mark.parametrize('change', ['force', 'temperature', 'invalid_cache'])
def test_resume_does_not_bypass_settings_or_validation(setup, change):
    run, chat, _, root = setup
    run('20261031000001')
    options = {'force': True} if change == 'force' else {'temperature': 0.4} if change == 'temperature' else {}
    if change == 'invalid_cache':
        for p in (root / '20261031000001/profiles/.cache').iterdir():
            data = json.loads(p.read_text())
            if 'observations' in data:
                # Structurally valid but anatomically incorrect: current check must reject it.
                data['observations'][0]['traits'][0]['trait'] = 'facial_hair'
                p.write_text(json.dumps(data))
    chat.reset_mock()
    run('20261031000002', resume_run='20261031000001', **options)
    assert chat.call_count == (1 if change == 'invalid_cache' else 2)


@pytest.mark.parametrize('resume', ['../escape', '20261031000002', '20261031000003', None])
def test_invalid_resume_fails_before_model_access(setup, resume):
    run, chat, factory, _ = setup
    with pytest.raises(ValueError):
        run('20261031000002', resume_run=resume)
    factory.assert_not_called()
    chat.assert_not_called()


def test_configuration_records_effective_controls_and_resume_without_changing_cache(setup):
    from minimax_h3_novel_pipeline.core import configuration_snapshot as snapshots
    run, chat, _, root = setup
    first = run('20261031000001', thinking=True, correction_attempts=2, persistence_filter='everything')
    chat.reset_mock()
    assert run('20261031000002', thinking=True, correction_attempts=2, persistence_filter='everything',
               resume_run='20261031000001') == first
    chat.assert_not_called()
    folder = root / '20261031000002/profiles'
    record = json.loads((folder / 'character_profiles_configuration.json').read_text())
    assert record['node_settings'] == {
        'out_dir': 'profiles', 'thinking': True, 'temperature': 0.12, 'max_tokens': 16384,
        'context_tokens': 65536, 'correction_attempts': 2, 'force': False,
        'persistence_filter': 'everything', 'resume_run': '20261031000001',
    }
    assert record['run_folder'] == '20261031000002'
    assert record['resolved_model'] == 'qwen'
    assert record['model_controls']['settings']['thinking'] is True
    assert record['status'] == 'completed'
    assert record['inputs']['character_evidence_sha256'] == snapshots.content_digest(first['source_document'])
    assert len(record['outputs']) == 4
    for artifact in record['outputs']:
        assert '.cache' not in artifact['file']
        assert snapshots.file_digest(folder / artifact['file']) == artifact['sha256']


def test_configuration_survives_failure_without_recording_secrets(setup, monkeypatch):
    _, _, _, root = setup
    def fail(*args):
        raise RuntimeError('secret-must-not-be-saved')
    monkeypatch.setattr(node.character_profiles, 'consolidate', fail)
    with pytest.raises(RuntimeError):
        node.ConsolidateCharacterProfilesNode().run(document(observation()), {
            'api_url': 'http://127.0.0.1:1234/v1', 'run_folder': '20261031000001',
            'api_key': 'secret-must-not-be-saved', 'unexpected': 'secret-must-not-be-saved',
        })
    text = (root / '20261031000001/character_profiles/character_profiles_configuration.json').read_text()
    assert 'secret-must-not-be-saved' not in text
    record = json.loads(text)
    assert record['status'] == 'started'
    assert record['outputs'] == []

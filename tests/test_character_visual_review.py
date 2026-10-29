"""The visual node repairs its own generated files without editing upstream evidence."""
from copy import deepcopy
import json
from unittest.mock import MagicMock, Mock

import pytest

from minimax_h3_novel_pipeline.core import character_profile_batches as batches
from minimax_h3_novel_pipeline.core import character_profiles as cp, character_visual_profiles as visual
from minimax_h3_novel_pipeline.core import character_visual_review as review, path_access
from minimax_h3_novel_pipeline.core.character_profile_document import make_document, validate_document
from minimax_h3_novel_pipeline.nodes import character_visual_profile_generator as node
from minimax_h3_novel_pipeline.nodes.load_character_visual_profiles import LoadCharacterVisualProfilesNode
from tests.test_character_evidence import observation
from tests.test_character_profiles import document, group, settings
from tests.test_character_profile_batches import fake_model
from tests.test_structured_json import Stream, client_for


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path / kind)
    evidence = document(observation('Doriane avait les yeux verts.', character='Doriane'),
                        observation('Doriane Belecamus avait les cheveux noirs.',
                                    character='Doriane Belecamus', value='cheveux noirs'))
    paths = []
    for chapter in evidence['chapters']:
        path = tmp_path / 'input' / chapter['source_file']
        path.parent.mkdir(exist_ok=True)
        path.write_text(chapter['source_text'], encoding='utf-8')
        chapter['source_file'] = str(path)
        chapter['evidence'][0]['provenance']['file'] = str(path)
        paths.append(path)
    records = cp.prepare_evidence(evidence)
    profiles = cp.build_profiles({'profiles': [group([k], canonical_name=r['character'])
                                               for k, r in records.items()]}, records)
    for profile in profiles:
        row = profile['supporting_evidence'][0]
        profile['global_traits'] = [{'trait': 'eye_color' if row['observation']['character'] == 'Doriane' else 'hair_color',
                                    'value': row['observation']['value'], 'evidence_ids': [row['evidence_id']]}]
    return make_document(evidence, records, profiles), paths, tmp_path / 'output'


def test_node_second_pass_updates_same_files_and_shared_transport_retries(inputs, monkeypatch):
    data, paths, output = inputs
    original = deepcopy(data)
    answer = {'decisions': [{'job_id': '0', 'relation': 'same', 'left_support': [0], 'right_support': [0]}]}
    transport, create = client_for(Stream(['{']), Stream([json.dumps(answer)]))
    client = MagicMock()
    client.chat = transport.chat
    client.__enter__.return_value = client
    factory = Mock(return_value=(client, 'qwen3.5'))
    monkeypatch.setattr(node.lmstudio_pipeline, 'make_client_and_model', factory)
    generated = []
    generate = visual.generate

    def track(*args):
        result = generate(*args)
        generated.append(deepcopy(result[0]))
        return result

    monkeypatch.setattr(visual, 'generate', track)
    result, saved = node.CharacterVisualProfileGeneratorNode().run(data,
        {'chapter_paths': [str(paths[0])]}, {'api_url': 'http://127.0.0.1:1234/v1', 'run_folder': '20261029120000'})['result']
    assert len(generated) == 2
    assert all(p['link_to_local_id'] is None for p in generated[0]['source_document']['profiles'])
    assert sum(p['link_to_local_id'] is not None for p in result['source_document']['profiles']) == 1
    assert len(result['profiles']) == 1 and len(result['profiles'][0]['supporting_evidence']) == 2
    assert result['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] == 'yeux verts; cheveux noirs'
    assert data == original
    assert result['source_document']['source_document'] == original['source_document']
    assert validate_document(result['source_document'])
    folder = output / '20261029120000' / 'character_visual_profiles'
    assert json.loads((folder / 'visual_designs.json').read_text('utf-8')) == result['visual_designs']
    assert json.loads((folder / 'character_visual_profiles.json').read_text('utf-8')) == result
    assert LoadCharacterVisualProfilesNode().run(saved)[0] == result
    assert list(folder.glob('.cache/*.json'))
    assert not (output / '.cache').exists()
    assert create.call_count == 2
    for call in create.call_args_list:
        assert call.kwargs['response_format'] == {'type': 'json_schema', 'json_schema': batches.IDENTITY_PAIR}
        for profile in original['profiles']:
            for row in profile['supporting_evidence']:
                assert row['evidence_id'] in str(call.kwargs['messages'])
    factory.assert_called_once()
    client.__exit__.assert_called_once()


@pytest.mark.parametrize('relation', ['different', 'uncertain'])
def test_review_never_merges_on_names_alone(inputs, monkeypatch, relation):
    data, paths, output = inputs
    initial, _ = visual.generate(data, paths[:1], output)
    def model(*args):
        result = fake_model(*args)
        for decision in result['decisions']:
            decision['relation'] = relation
        return result
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', model)
    corrected = review.repair(initial['visual_designs'], review.candidates(initial['visual_designs']),
                              output, None, 'qwen', settings())
    assert corrected == initial['visual_designs']
    assert json.loads((output / 'character_visual_identity_review.json').read_text())['unresolved']


def test_conflicting_manual_defaults_prevent_silent_loss(inputs, monkeypatch):
    data, paths, output = inputs
    initial, _ = visual.generate(data, paths, output)
    for entity, value in zip(initial['visual_designs']['entities'], ['narrow nose', 'broad nose']):
        entity['added_details']['facial_features'] = value
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', fake_model)
    corrected = review.repair(initial['visual_designs'], review.candidates(initial['visual_designs']),
                              output, None, 'qwen', settings())
    assert corrected == initial['visual_designs']


def test_existing_manual_links_are_not_reconsidered(inputs):
    data, paths, output = inputs
    data['profiles'][1]['link_to_local_id'] = data['profiles'][0]['local_id']
    initial, _ = visual.generate(data, paths, output)
    assert review.candidates(initial['visual_designs']) == []


def test_linked_trait_conflicts_remain_blocked(inputs, monkeypatch):
    data, paths, output = inputs
    # Editable trait decisions are preserved, never regenerated by the review.
    data['profiles'][1]['global_traits'][0]['trait'] = 'eye_color'
    initial, _ = visual.generate(data, paths, output)
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', fake_model)
    corrected = review.repair(initial['visual_designs'], review.candidates(initial['visual_designs']),
                              output, None, 'qwen', settings())
    final, _ = visual.generate(corrected['character_profiles'], paths, output, corrected)
    assert 'eye_color' in final['profiles'][0]['ignored_traits']
    assert final['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] == ''
    assert [p['global_traits'] for p in final['source_document']['profiles']] == [p['global_traits'] for p in data['profiles']]

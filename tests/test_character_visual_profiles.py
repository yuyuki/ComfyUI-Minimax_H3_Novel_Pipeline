"""Deterministic projection and current node wiring."""
from copy import deepcopy
import inspect
import json

import pytest

from minimax_h3_novel_pipeline.core import character_profiles as cp, character_visual_profiles as vp, path_access
from minimax_h3_novel_pipeline.nodes.character_visual_profile_generator import CharacterVisualProfileGeneratorNode as Generator
from tests.test_character_evidence import observation
from tests.test_character_profiles import document, plan


def consolidated(*items):
    data = document(*(items or (observation(),)))
    records = cp.prepare_evidence(data)
    from minimax_h3_novel_pipeline.core.character_profile_document import make_document
    profiles = cp.build_profiles(plan(records), records)
    for profile in profiles:
        profile['global_traits'] = [{'trait': 'eye_color', 'value': r['observation']['value'],
                                     'evidence_ids': [r['evidence_id']]}
                                    for r in profile['supporting_evidence']
                                    if r['observation']['persistence'] == 'persistent']
    return make_document(data, records, profiles)


def cinematic(data):
    source = data['source_document']['chapters'][0]
    return [{'source_file': source['source_file'], 'chapter_name': source['chapter_name'],
             'sequences': [{'sequence': 1, 'source': source['source_text'],
                            'adaptation': {p: source['source_text'] for p in ('initialState', 'event', 'endingState')}}]}]



def test_node_uses_selection_without_model_or_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path / kind)
    source = tmp_path / 'input' / 'chapter1.txt'
    source.parent.mkdir()
    source.write_text('Élise avait les yeux verts.', encoding='utf-8')
    data = consolidated()
    chapter = data['source_document']['chapters'][0]
    chapter['source_file'] = str(source)
    chapter['evidence'][0]['provenance']['file'] = str(source)
    records = cp.prepare_evidence(data['source_document'])
    from minimax_h3_novel_pipeline.core.character_profile_document import make_document
    p = cp.build_profiles(plan(records), records)
    p[0]['global_traits'] = [{'trait': 'eye_color', 'value': 'yeux verts', 'evidence_ids': list(records)}]
    data = make_document(data['source_document'], records, p)
    original = deepcopy(data)
    result, saved = Generator().run(data, {'chapter_paths': [str(source)]},
        {'run_folder': '20261028120000'})['result']
    assert data == original
    assert result['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] == 'yeux verts'
    assert json.loads(open(saved, encoding='utf-8').read()) == result
    assert not list((tmp_path / 'output').rglob('.cache'))
    assert 'cinematic_chapters' not in inspect.signature(Generator.run).parameters
    assert set(Generator.INPUT_TYPES()['required']) == {'character_profiles', 'chapter_selection', 'lmstudio_config', 'out_dir'}


def test_cross_chapter_traits_and_temporary_exclusion(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    data = consolidated(observation('She enters.', persistence='temporary', attribute='clothing'),
                        observation('Her eyes are green.'))
    result, _ = vp.generate(data, ['chapter1.txt'], tmp_path / 'design')
    assert result['visual_designs']['entities'][0]['source_facts']['stable_visual_description'] == 'yeux verts'
    assert len(result['profiles'][0]['supporting_evidence']) == 2
    assert 'chapter_enrichment' not in result


@pytest.mark.parametrize('choice', [None, 0, 1])
def test_manual_choice_and_ignored_traits(tmp_path, monkeypatch, choice):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    data = consolidated(observation(), observation('Blue eyes.', value='blue'))
    p = data['profiles'][0]
    ids = [r['evidence_id'] for r in p['supporting_evidence']]
    p['unresolved_conflicts'] = [{'conflict_id': 'eye', 'trait': 'eye_color', 'scope': 'same period',
        'evidence_ids': ids, 'choices': [{'id': str(i), 'value': v, 'selected': choice == i, 'evidence_ids': [ids[i]]}
                                      for i, v in enumerate(['edited green', 'edited blue'])]}]
    result, _ = vp.generate(data, ['chapter1.txt'], tmp_path / 'design')
    fact = result['visual_designs']['entities'][0]['source_facts']['stable_visual_description']
    assert fact == ('' if choice is None else ['edited green', 'edited blue'][choice])


def test_homonymous_source_paths_do_not_select_by_name():
    data = consolidated(observation(), observation('Other chapter.'))
    # Change source identities before building the validated profile document.
    evidence = data['source_document']
    for c, folder in zip(evidence['chapters'], ['one', 'two']):
        c['source_file'] = folder + '/same.txt'
        c['chapter_name'] = 'same'
        c['evidence'][0]['provenance'].update(file=c['source_file'], chapter_name='same')
    records = cp.prepare_evidence(evidence)
    from minimax_h3_novel_pipeline.core.character_profile_document import make_document
    groups = {'profiles': [{'canonical_name': 'Élise', 'identity_status': 'confirmed',
        'identity_support': [], 'evidence': [{'evidence_id': k}]} for k in records]}
    data = make_document(evidence, records, cp.build_profiles(groups, records))
    first = vp.select_profiles(data, ['one/same.txt'])
    second = vp.select_profiles(data, ['two/same.txt'])
    assert len(first) == len(second) == 1
    assert first[0]['local_id'] != second[0]['local_id']
    with pytest.raises(ValueError, match='exactly one evidence chapter'):
        vp.select_profiles(data, ['same.txt'])


def test_chronology_cannot_return_through_edited_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    data = consolidated(observation('Her hair is black.', value='black hair'),
                        observation('Years later her hair is white.', value='white hair'))
    profile = data['profiles'][0]
    for row in profile['global_traits']:
        row['trait'] = 'hair_color'
    previous, _ = vp.generate(data, ['chapter1.txt'], tmp_path / 'first')
    previous['visual_designs']['entities'][0]['added_details'] = {
        'hair': 'white hair', 'facial_features': 'A narrow nose'}
    before, after = [r['evidence_id'] for r in profile['supporting_evidence']]
    profile['chronological_changes'] = [{'trait': 'hair_color', 'before_id': before,
        'after_id': after, 'support_id': after, 'temporal_quote': 'Years later'}]
    result, _ = vp.generate(data, ['chapter1.txt'], tmp_path / 'next', previous['visual_designs'])
    item = result['visual_designs']['entities'][0]
    assert item['source_facts']['stable_visual_description'] == ''
    assert item['added_details'] == {'facial_features': 'A narrow nose'}
    # The evolution remains available to scene-local decisions.
    assert 'hair_color' not in result['profiles'][0]['ignored_traits']
    assert result['profiles'][0]['chronological_changes'] == profile['chronological_changes']


def test_ignored_facial_traits_block_defaults_in_the_correct_field(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    data = consolidated(observation('Her nose is narrow.', value='narrow nose'),
                        observation('Her nose is broad.', value='broad nose'))
    profile = data['profiles'][0]
    for row in profile['global_traits']:
        row['trait'] = 'nose_shape'
    previous, _ = vp.generate(data, ['chapter1.txt'], tmp_path / 'first')
    previous['visual_designs']['entities'][0]['added_details'] = {
        'facial_features': 'A narrow nose', 'build': 'slender'}
    ids = [r['evidence_id'] for r in profile['supporting_evidence']]
    profile['unresolved_conflicts'] = [{'conflict_id': 'nose', 'trait': 'nose_shape', 'scope': 'same period',
        'evidence_ids': ids, 'choices': [{'id': str(i), 'value': value, 'selected': False, 'evidence_ids': [key]}
                                      for i, (key, value) in enumerate(zip(ids, ['narrow nose', 'broad nose']))]}]
    result, _ = vp.generate(data, ['chapter1.txt'], tmp_path / 'next', previous['visual_designs'])
    item = result['visual_designs']['entities'][0]
    assert item['source_facts']['stable_visual_description'] == ''
    assert item['added_details'] == {'build': 'slender'}

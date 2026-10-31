"""Linked profiles preserve proven equivalence without guessing new synonyms."""
from copy import deepcopy

import pytest

from minimax_h3_novel_pipeline.core import character_profiles as cp, character_visual_profiles as visual
from minimax_h3_novel_pipeline.core.character_profile_document import make_document
from tests.test_character_evidence import observation
from tests.test_character_profiles import document, group


def linked_document(relation='equivalent', trait='age', last_value='jeune'):
    values = ['jeune', 'jeune femme', 'young woman', last_value]
    evidence = document(*(observation(f'Observation {i}: {value}.', value=value, attribute='age')
                          for i, value in enumerate(values)))
    records = cp.prepare_evidence(evidence)
    ids = list(records)
    profiles = cp.build_profiles({'profiles': [group(ids[:3]), group(ids[3:])]}, records)
    for profile in profiles:
        profile['global_traits'] = [{'trait': 'age', 'value': row['observation']['value'],
                                     'evidence_ids': [row['evidence_id']]}
                                    for row in profile['supporting_evidence']]
    profiles[1]['link_to_local_id'] = profiles[0]['local_id']
    profiles[0]['trait_relations'] = [
        {'trait': trait, 'relation': relation, 'scope': 'same period',
         'before_id': '', 'after_id': '', 'support_id': '', 'temporal_quote': '',
         'evidence': [{'evidence_id': key} for key in pair]}
        for pair in (ids[:2], ids[1:3])]
    return make_document(evidence, records, profiles)


def test_transitive_equivalence_survives_linking_and_preserves_sources():
    data = linked_document()
    original = deepcopy(data)
    profile, = visual.prepare_profiles(data)
    assert profile['linked_trait_conflicts'] == []
    assert profile['ignored_traits'] == []
    assert profile['blocked_fields'] == []
    assert visual.source_facts(profile)['stable_visual_description'] == 'jeune; jeune femme; young woman'
    assert len(profile['effective_traits']) == 4
    assert data == original


@pytest.mark.parametrize('options', [
    {'relation': 'uncertain'}, {'relation': 'independent'}, {'relation': 'conflict'},
    {'trait': 'gender'}, {'last_value': 'vieille femme'},
])
def test_unproven_or_different_traits_remain_blocked(options):
    profile, = visual.prepare_profiles(linked_document(**options))
    assert profile['ignored_traits'] == ['age']
    assert profile['blocked_fields'] == ['age_appearance']
    assert visual.source_facts(profile)['stable_visual_description'] == ''


def test_unresolved_conflict_is_not_overridden_by_equivalence():
    data = linked_document()
    profile = data['profiles'][0]
    rows = profile['global_traits']
    profile['unresolved_conflicts'] = [{
        'conflict_id': 'age', 'trait': 'age', 'scope': 'same period',
        'evidence_ids': [key for row in rows for key in row['evidence_ids']],
        'choices': [{'id': str(i), 'value': row['value'], 'evidence_ids': row['evidence_ids'],
                     'selected': False} for i, row in enumerate(rows)],
    }]
    projected, = visual.prepare_profiles(data)
    assert projected['ignored_traits'] == ['age']
    assert projected['effective_traits'] == []

import pytest

from minimax_h3_novel_pipeline.core import character_profiles as cp
from tests.test_character_evidence import observation
from tests.test_character_profiles import document, group, refs


@pytest.mark.parametrize('left,right', [('Doriane', 'Belecamus'), ('Belecamus', 'Doriane')])
def test_source_full_name_connects_separate_given_and_family_labels(left, right):
    assert cp._alias_link(left, right, ['Puis du plus profond de Doriane Belecamus s’éleva une plainte.'])


@pytest.mark.parametrize('text', [
    'Doriane regardait Belecamus.', 'Doriane et Belecamus attendaient.',
    'Doriane, Belecamus et Indy.', 'Doriane. Belecamus entra.',
    'Doriane\nBelecamus', 'Doriane\r\n\r\nBelecamus',
    'Doriane-Belecamus', 'Doriane Belecamusson',
])
def test_cooccurrence_is_not_a_full_name_bridge(text):
    assert not cp._alias_link('Doriane', 'Belecamus', [text])


def test_generic_label_is_not_a_name_component():
    assert not cp._alias_link('la jeune femme', 'Doriane', ['la jeune femme Doriane sourit.'])


def test_full_name_bridge_needs_cited_explicit_support_and_preserves_original_labels():
    records = cp.prepare_evidence(document(
        observation('Doriane était jeune.', character='Doriane', value='jeune'),
        observation('Le corps mince de Doriane Belecamus.', character='Belecamus', value='mince')))
    ids = list(records)
    with pytest.raises(ValueError, match='connecting quotations'):
        cp.build_profiles({'profiles': [group(ids, canonical_name='Doriane')]}, records)
    result = cp.build_profiles({'profiles': [group(ids, canonical_name='Belecamus',
                                                 identity_support=refs(ids))]}, records)
    assert result[0]['aliases'] == ['Doriane']
    assert {r['evidence_id'] for r in result[0]['supporting_evidence']} == set(ids)
    # The full name is evidence for a link, not permission to invent a new label.
    with pytest.raises(ValueError, match='canonical_name'):
        cp.build_profiles({'profiles': [group(ids, canonical_name='Doriane Belecamus',
                                             identity_support=refs(ids))]}, records)

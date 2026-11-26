"""Chapter-independent identities retain aliases, facts and temporal scope."""
from copy import deepcopy

import pytest

from minimax_h3_novel_pipeline.core import character_profile_batches as batches, character_profiles as cp, path_access
from minimax_h3_novel_pipeline.core.character_visual_profiles import prepare_profiles
from tests.test_character_evidence import observation
from tests.test_character_profile_batches import fake_model, request_payload
from tests.test_character_profile_recovery import narrative_document
from tests.test_character_profiles import document, group, settings


@pytest.mark.parametrize('name,alias,text', [
    ('Jones', 'Indy', 'Henry Jones Jr. Mais ceux qui le connaissaient l’appelaient Indy.'),
    ('Jones', 'Indy', 'Henry Jones Jr. était surnommé Indy.'),
    ('Lovelace', 'Ada', 'Augusta Ada Lovelace was known as Ada.'),
])
def test_short_name_and_alias_can_use_explicit_full_name_bridge(name, alias, text):
    assert cp._alias_link(name, alias, [text])
    assert cp._alias_link(alias, name, [text])


@pytest.mark.parametrize('text', [
    'Jones regardait Henry. Henry était surnommé Indy.',
    'Jones, Henry était surnommé Indy.',
    'Jones et Henry étaient amis. Henry était surnommé Indy.',
    'Henry Jones Jr. parlait avec Indy.',
    'Henry Jones Jr. rencontra Ted Conrad, surnommé Indy.',
])
def test_full_name_bridge_does_not_connect_other_people(text):
    assert not cp._alias_link('Jones', 'Indy', [text])


def test_shared_surname_cannot_connect_two_full_names():
    first = "Robin Park had black hair."
    second = "Morgan Park had grey eyes."
    surname = "Park was tall."
    data = narrative_document(" ".join((first, second, surname)),
        observation(first, character="Robin Park", value="black hair"),
        observation(second, character="Morgan Park", value="grey eyes"),
        observation(surname, character="Park", value="tall"))
    records = cp.prepare_evidence(data)
    refs = [{"evidence_id": key} for key in records]
    with pytest.raises(ValueError, match="connecting quotations"):
        cp.build_profiles({"profiles": [group(records, canonical_name="Robin Park", identity_support=refs)]}, records)


def test_one_attested_occurrence_connects_short_and_reordered_qualified_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    baker = "Camille la boulangère avait les yeux bleus."
    sailor = "Le marin Camille avait les yeux verts."
    data = narrative_document(baker + " " + sailor,
        observation(baker, character="Camille la boulangère", value="yeux bleus"),
        observation(sailor, character="Camille", value="yeux verts"),
        observation(sailor, character="Camille le marin", value="yeux verts"))

    def chat(*args):
        reply = fake_model(*args)
        if args[4] == batches.IDENTITY_PAIR:
            for job, decision in zip(request_payload(args[3])["jobs"], reply["decisions"]):
                names = {job[side]["canonical_name"] for side in ("left_identity", "right_identity")}
                if names != {"Camille", "Camille le marin"}:
                    decision["relation"] = "different"
        return reply

    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    result, _ = cp.consolidate(data, cp.prepare_evidence(data), tmp_path, None, "qwen", settings())
    assert {p["canonical_name"]: p["aliases"] for p in result["profiles"]} == {
        "Camille la boulangère": [], "Camille le marin": ["Camille"],
    }


def test_pair_consolidation_accepts_bridge_without_inventing_canonical_label(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    text = ('Henry Jones Jr. Mais ceux qui le connaissaient l’appelaient Indy. '
            'Jones avait les yeux noisette. Indy avait une cicatrice sur le menton.')
    data = narrative_document(text,
        observation('les yeux noisette', character='Jones', value='yeux noisette'),
        observation('une cicatrice sur le menton', character='Indy', value='cicatrice sur le menton'))
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', fake_model)
    result, _ = cp.consolidate(data, cp.prepare_evidence(data), tmp_path, None, 'qwen', settings(correction_attempts=0))
    assert len(result['profiles']) == 1
    profile = result['profiles'][0]
    assert {profile['canonical_name'], *profile['aliases']} == {'Jones', 'Indy'}
    assert {r['value'] for r in profile['global_traits']} == {'yeux noisette', 'cicatrice sur le menton'}


def test_identity_requests_include_established_names_and_aliases(tmp_path, monkeypatch):
    text = 'Henry Jones Jr. était surnommé Indy. Indy avait une cicatrice.'
    data = narrative_document(text,
        observation('Henry Jones Jr.', character='Henry Jones Jr.'),
        observation('une cicatrice', character='Indy', value='cicatrice'))
    records = cp.prepare_evidence(data)
    ids = list(records)
    extra = cp.prepare_evidence(document(observation('Jones avait les yeux noisette.', character='Jones')))
    records.update(extra)
    groups = [group(ids, canonical_name='Henry Jones Jr.', identity_support=[{'evidence_id': k} for k in ids]),
              group(list(extra), canonical_name='Jones')]
    original = deepcopy(groups)
    calls = []

    def model(*args):
        payload = request_payload(args[3])
        calls.append(payload)
        job = payload['jobs'][0]
        assert job['left_identity'] == {'canonical_name': 'Henry Jones Jr.', 'aliases': ['Indy'],
                                        'identity_status': 'confirmed'}
        assert job['right_identity'] == {'canonical_name': 'Jones', 'aliases': [], 'identity_status': 'confirmed'}
        return fake_model(*args)

    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', model)
    result = batches.reconcile_identities(groups, records, batches.Requests(tmp_path, None, 'qwen', settings()))
    assert calls
    assert groups == original
    assert len(result) == 1
    assert result[0]['aliases'] == ['Indy', 'Jones']


def test_three_labels_keep_all_chapter_facts_and_temporary_outfits(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    data = document(
        observation('Henry Jones Jr. était surnommé Indy. Il avait une cicatrice sur le menton.',
                    character='Henry Jones Jr.', value='cicatrice sur le menton'),
        observation('Indy avait les yeux noisette.', character='Indy', value='yeux noisette'),
        observation('Jones portait une veste rouge.', character='Jones', value='veste rouge',
                    attribute='clothing', persistence='temporary'),
        observation('Indy portait une veste bleue.', character='Indy', value='veste bleue',
                    attribute='clothing', persistence='temporary'))
    original = deepcopy(data)
    monkeypatch.setattr(batches.lmstudio_json, 'chat_json', fake_model)
    records = cp.prepare_evidence(data)
    result, _ = cp.consolidate(data, records, tmp_path, None, 'qwen', settings(correction_attempts=0))
    assert data == original
    assert len(result['profiles']) == 1
    profile = result['profiles'][0]
    assert profile['canonical_name'] == 'Henry Jones Jr.'
    assert profile['aliases'] == ['Indy', 'Jones']
    assert {r['evidence_id'] for r in profile['supporting_evidence']} == set(records)
    assert len(profile['temporary_states']) == 2
    assert {r['value'] for r in prepare_profiles(result)[0]['effective_traits']} == {
        'cicatrice sur le menton', 'yeux noisette'}

"""Scene planning must distinguish entities from their picture assets."""
from copy import deepcopy
from types import SimpleNamespace

from jsonschema import Draft202012Validator
import pytest

from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step


def scene():
    return dict(title='Fall', source_excerpt='Il tomba.', visual_event='Indy tombe.',
                location_global_id='', visible_entity_ids=['CHAR_001'],
                speaking_entity_ids=[], reference_view_requests=[dict(
                    global_id='CHAR_001', view_types=['full_body_front'],
                    prominence='primary', reason='Full-body action')],
                dialogue_present=False, adaptation_notes='Indy est en chute libre.')


@pytest.mark.parametrize('field', ['visible_entity_ids', 'speaking_entity_ids',
                                  'location_global_id', 'reference_view_requests'])
def test_planning_schema_rejects_asset_ids_and_repairs_with_field_feedback(monkeypatch, field):
    original = deepcopy(step.SCENE_SCHEMA)
    asset = 'PIC_CHAR_001_FULL_BODY_FRONT'
    catalog = [dict(global_id='CHAR_001', available_picture_views=[dict(
        asset_id=asset, view_type='full_body_front', variant='base')])]
    calls = []

    def chat(client, model, system, user, schema, *args):
        calls.append(user)
        assert schema['strict'] is True
        assert 'Never put an' in system
        validator = Draft202012Validator(schema['schema'])
        Draft202012Validator.check_schema(schema['schema'])
        valid = {'scenes': [scene()]}
        assert validator.is_valid(valid)
        invalid = deepcopy(valid)
        if field == 'reference_view_requests':
            invalid['scenes'][0][field][0]['global_id'] = asset
        elif field == 'location_global_id':
            invalid['scenes'][0][field] = asset
        else:
            invalid['scenes'][0][field] = [asset]
        assert not validator.is_valid(invalid)
        if len(calls) == 1:
            # Simulate a backend ignoring constraints; semantic validation still applies.
            return invalid
        assert f'scenes[0].{field}' in user
        assert 'never picture asset_id' in user
        assert "Allowed entity global_id values: ['CHAR_001']" in user
        return valid

    monkeypatch.setattr(step, 'chat_json', chat)
    result = step.plan_scenes(None, 'mock', 'chapter', 'Il tomba.', 15, 15, catalog,
                             SimpleNamespace(duration=8, scenes_per_chunk=4, repair_attempts=1))
    assert len(calls) == 2
    assert result[0].reference_view_requests[0].global_id == 'CHAR_001'
    assert step.SCENE_SCHEMA == original


def test_empty_and_successive_catalogs_do_not_share_schema_constraints(monkeypatch):
    schemas = []

    def chat(client, model, system, user, schema, *args):
        schemas.append(deepcopy(schema))
        raw = scene()
        raw.update(visible_entity_ids=[], reference_view_requests=[])
        validator = Draft202012Validator(schema['schema'])
        Draft202012Validator.check_schema(schema['schema'])
        assert validator.is_valid({'scenes': [raw]})
        return {'scenes': [raw]}

    monkeypatch.setattr(step, 'chat_json', chat)
    for ids in (['CHAR_001'], [], ['CHAR_002']):
        step.plan_scenes(None, 'mock', 'chapter', 'Wait', 1, 1,
                         [{'global_id': gid} for gid in ids],
                         SimpleNamespace(duration=8, scenes_per_chunk=4))
    for schema in schemas[1:]:
        assert not Draft202012Validator(schema['schema']).is_valid({'scenes': [scene()]})
    empty = schemas[1]['schema']['properties']['scenes']['items']['properties']
    for field in ('visible_entity_ids', 'speaking_entity_ids', 'reference_view_requests'):
        assert empty[field]['maxItems'] == 0
    assert empty['location_global_id']['enum'] == ['']

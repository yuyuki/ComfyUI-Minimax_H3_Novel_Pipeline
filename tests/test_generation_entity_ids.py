"""Reference metadata uses only registered entity IDs and available views."""
from copy import deepcopy
import pytest
from jsonschema import Draft202012Validator
from minimax_h3_novel_pipeline.core import continuity_issues
from tests.test_cinematic_generation import catalog, metadata


@pytest.mark.parametrize("field", ["visible_entity_ids", "speaking_entity_ids", "location_global_id", "reference_view_requests"])
def test_metadata_schema_rejects_asset_ids(field):
    schema = continuity_issues.metadata_schema(catalog())
    result = metadata()
    result[field] = {"visible_entity_ids": ["PIC_001"], "speaking_entity_ids": ["PIC_001"],
                    "location_global_id": "PIC_001", "reference_view_requests": [
                        {"global_id": "PIC_001", "view_types": [], "prominence": "primary", "reason": ""}]}[field]
    assert not Draft202012Validator(schema["schema"]).is_valid(result)


def test_request_schemas_do_not_share_entity_constraints():
    first = continuity_issues.metadata_schema(catalog())
    saved = deepcopy(first)
    empty = continuity_issues.metadata_schema([])
    assert first == saved
    assert empty["schema"]["properties"]["visible_entity_ids"]["maxItems"] == 0

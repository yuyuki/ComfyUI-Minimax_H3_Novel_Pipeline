from tests.neutral_reference_helpers import entity as neutral_entity
"""Generation exposes only exact source observations at the requested sequence."""
import json
import pytest
from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step


def test_source_catalog_preserves_conflicts_without_future_or_carried_states():
    entity = {"global_id": "CHAR_001", "entity_type": "character", "canonical_name": "Aster",
              "chapters_seen": ["chapter"], "timeline": {"chapter": {
                  "1": {"source": [{"state": "holding rope"}, {"state": "empty hands"}]},
                  "2": {"source": [{"state": "injured"}]}}}}
    entity = neutral_entity(entity)
    refs = {"entities": [entity]}
    catalog = step.sequence_catalog(refs, "chapter", 1)
    assert catalog[0]["source_observations"] == [{"state": "holding rope"}, {"state": "empty hands"}]
    assert "state_by_phase" not in catalog[0]
    assert "injured" not in json.dumps(catalog)
    assert step.sequence_catalog(refs, "chapter", 3) == []
    catalog[0]["source_observations"].clear()
    assert len(entity["timeline"]["chapter"]["1"]["source"]) == 2


def test_generation_rejects_obsolete_phase_observations():
    entity = {"global_id": "CHAR_001", "entity_type": "character", "canonical_name": "Aster",
              "chapters_seen": ["chapter"], "timeline": {"chapter": {"1": {"event": [{}]}}}}
    with pytest.raises(ValueError, match="source observations only"):
        step.sequence_catalog({"entities": [entity]}, "chapter", 1)

"""Generation consumes consolidated cinematic provenance without reading prose."""
import json


from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step




def test_catalog_preserves_conflicts_without_future_or_carried_states():
    entity = {"global_id": "CHAR_001", "entity_type": "character", "canonical_name": "Aster",
              "chapters_seen": ["chapter"], "timeline": {"chapter": {
                  "1": {"event": [{"state": "holding rope"}, {"state": "empty hands"}]},
                  "2": {"endingState": [{"state": "injured"}]}}}}
    refs = {"entities": [entity]}
    catalog = step.sequence_catalog(refs, "chapter", 1)
    assert catalog[0]["state_by_phase"] == {
        "initialState": [], "event": [{"state": "holding rope"}, {"state": "empty hands"}], "endingState": []}
    assert "injured" not in json.dumps(catalog)
    assert step.sequence_catalog(refs, "chapter", 3) == []
    catalog[0]["state_by_phase"]["event"].clear()
    assert len(entity["timeline"]["chapter"]["1"]["event"]) == 2

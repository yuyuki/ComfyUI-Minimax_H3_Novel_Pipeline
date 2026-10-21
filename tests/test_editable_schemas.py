import pytest
from jsonschema import Draft202012Validator
from minimax_h3_novel_pipeline import editable_schemas, path_access, util, visual_designs


def test_schema_exports_and_visual_traits(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    editable_schemas.export_schemas(tmp_path)
    for filename in ("visual_designs.schema.json",):
        assert (tmp_path / filename).read_bytes() == (editable_schemas.SCHEMA_DIR / filename).read_bytes()
        Draft202012Validator.check_schema(util.load_json(tmp_path / filename))
    data = {"$schema": "https://invalid.example/no-network", "schema_version": visual_designs.DESIGN_SCHEMA_VERSION,
            "image_style": "realistic photographic", "entities": [{"global_id": "CHAR_001", "entity_type": "character",
            "canonical_name": "Indy", "source_facts": {"stable_visual_description": "", "distinguishing_features": []},
            "added_details": {"hair": "bruns"}}]}
    path = tmp_path / "visual_designs.json"
    util.save_json(path, data)
    assert visual_designs.resolve_designs_path(str(path)) == path
    data["entities"][0]["added_details"] = {"architecture": "pierre"}
    util.save_json(path, data)
    with pytest.raises(ValueError, match=r"entities\[0\].added_details"):
        visual_designs.resolve_designs_path(str(path))


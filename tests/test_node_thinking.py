"""Per-node reasoning controls reach the client without changing shared settings."""
import pytest

from minimax_h3_novel_pipeline.core import lmstudio_pipeline, path_access, run_output, util
from minimax_h3_novel_pipeline.nodes.cinematic_chapter_adapter import CinematicChapterAdapterNode
from minimax_h3_novel_pipeline.nodes.consolidate_references import ConsolidateReferencesNode
from minimax_h3_novel_pipeline.nodes.extract_chapter_references import ExtractChapterReferencesNode
from minimax_h3_novel_pipeline.nodes.generate_h3_prompts import GenerateH3PromptsNode
from minimax_h3_novel_pipeline.nodes.spatial_continuity import SpatialContinuityNode


@pytest.mark.parametrize("cls,default", [
    (CinematicChapterAdapterNode, True),
    (ExtractChapterReferencesNode, True),
    (ConsolidateReferencesNode, True),
    (SpatialContinuityNode, True),
    (GenerateH3PromptsNode, False),
])
@pytest.mark.parametrize("thinking", [None, True, False])
def test_node_thinking_reaches_client(tmp_path, monkeypatch, cinematic_registry, cls, default, thinking):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: tmp_path)
    chapter = tmp_path / "chapter.txt"
    chapter.write_text("Aster waits.", encoding="utf-8")
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261024120000"}
    registry = cinematic_registry(chapter, schema_version=util.REGISTRY_SCHEMA,
                                 entities=[], picture_assets=[], audio_assets=[])
    catalog = {"schema_version": util.CHAPTER_SCHEMA, "chapter_id": "chapter",
               "chapter_name": "chapter", "source": {},
               "timeline_version": "cinematic-reference-timeline.v1",
               "characters": [], "locations": [], "objects": [],
               "sequences": registry["chapter_timelines"]["chapter"]["sequences"]}
    inputs = {
        "lmstudio_config": config,
        "chapter_selection": {"chapter_paths": [str(chapter)]},
        "cinematic_chapters": [catalog],
        "chapter_catalogs": [catalog],
        "consolidated_references": registry,
    }
    params = {}
    for fields in cls.INPUT_TYPES().values():
        for key, spec in fields.items():
            if key == "thinking":
                assert spec[1]["default"] is default
                if thinking is not None:
                    params[key] = thinking
            elif key in inputs:
                params[key] = inputs[key]
            elif len(spec) > 1 and "default" in spec[1]:
                params[key] = spec[1]["default"]

    class ClientReached(Exception):
        pass

    def capture(module, url, settings):
        assert settings["thinking"] is (default if thinking is None else thinking)
        assert settings is not config
        assert "thinking" not in config
        raise ClientReached

    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", capture)
    with pytest.raises(ClientReached):
        cls().run(**params)
    assert "thinking" not in config

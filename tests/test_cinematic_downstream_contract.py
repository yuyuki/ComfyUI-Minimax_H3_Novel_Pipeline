"""Exercise current adapter files through downstream nodes with offline LLM replies."""
from contextlib import nullcontext
from copy import deepcopy
import json

import pytest

from tests.passages_helpers import with_passage_ids

from minimax_h3_novel_pipeline.core import (
    cinematic_adaptation as adaptation,
    lmstudio_pipeline,
    path_access,
    run_output,
    util,
)
from minimax_h3_novel_pipeline.nodes.cinematic_chapter_adapter import CinematicChapterAdapterNode
from minimax_h3_novel_pipeline.nodes.consolidate_references import ConsolidateReferencesNode
from minimax_h3_novel_pipeline.nodes.extract_chapter_references import ExtractChapterReferencesNode
from minimax_h3_novel_pipeline.nodes.generate_h3_prompts import GenerateH3PromptsNode
from minimax_h3_novel_pipeline.nodes.load_cinematic_chapters import LoadCinematicChaptersNode
from minimax_h3_novel_pipeline.nodes import spatial_continuity as spatial


def defaults(cls):
    return {name: spec[1]["default"]
            for fields in cls.INPUT_TYPES().values() for name, spec in fields.items()
            if len(spec) > 1 and "default" in spec[1]}


@pytest.mark.parametrize("reload_file", [False, True], ids=["socket", "saved-file"])
@pytest.mark.parametrize("review_unavailable", [False, True], ids=["reviewed", "incomplete-review"])
def test_adapter_output_reaches_downstream_nodes(tmp_path, monkeypatch, reload_file, review_unavailable):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock"))
    source = tmp_path / "chapter.md"
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261114120000"}
    passage = ("Indy attend près de la porte et regarde la sortie en silence. "
               "Puis Indy part sans se retourner et quitte la pièce.")
    source.write_text(passage, encoding="utf-8")

    def adapt_chat(client, model, system, user, schema, *args):
        payload = json.loads(user)
        if schema == adaptation.SCHEMA:
            segments = payload["source_segments"]
            boundary = next(row["id"] for row in segments if row["text"].endswith("silence."))
            return {
                "opening_state": [{"id": "indy", "description": "Indy est près de la porte."}],
                "beats": [
                    {"source_end_id": boundary, "estimated_seconds": 3,
                     "description": "Indy regarde la sortie en silence.", "changes": []},
                    {"source_end_id": len(segments), "estimated_seconds": 4,
                     "description": "Indy quitte la pièce sans se retourner.",
                     "changes": [{"id": "indy", "description": "Indy est sorti de la pièce."}]},
                ],
            }
        assert schema == adaptation.REVIEW_SCHEMA
        if review_unavailable:
            raise ConnectionError("Offline review")
        merged, = payload["sequences"]
        assert merged["estimated_seconds"] == 7 and len(merged["events"]) == 2
        return {"source_language": "fr",
                "language_checks": [{"sequence": 1, "initialState": "fr", "event": "fr", "endingState": "fr"}],
                "descriptive_checks": [{"sequence": 1, "unsupported_transfers": [],
                                        "lost_visual_or_spatial_meaning": []}],
                "findings": [], "choices": []}

    monkeypatch.setattr(adaptation.lmstudio_json, "chat_json", adapt_chat)
    direct, saved_files = CinematicChapterAdapterNode().run(
        config, {"chapter_paths": [str(source)]})["result"]
    before = deepcopy(direct)
    saved = util.output_path(saved_files)
    diagnostics = saved.with_name(saved.name.replace(".cinematic.json", ".cinematic.diagnostics.json"))
    report = util.load_json(diagnostics)
    assert report["status"] == ("incomplete" if review_unavailable else "completed")
    assert report["summary"]["incomplete_scenes"] == (1 if review_unavailable else 0)
    assert all("review" not in seq for seq in direct[0]["sequences"])
    assert "narrative_review" not in direct[0]

    # Neither loading nor any downstream stage should read the diagnostic sidecar.
    original_load = util.load_json

    def load_without_diagnostics(path):
        assert "diagnostics" not in str(path)
        return original_load(path)

    monkeypatch.setattr(util, "load_json", load_without_diagnostics)
    loaded, _ = LoadCinematicChaptersNode().run(str(saved.parent))
    assert loaded == direct
    chapters = loaded if reload_file else direct
    source_inputs = []

    def extract_chat(client, model, system, user, schema, *args):
        source_inputs.append(json.loads(user))
        return {"chunk_summary": "", "characters": [], "locations": [], "objects": []}

    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", with_passage_ids(extract_chat))
    catalogs, _ = ExtractChapterReferencesNode().run(config, chapters, **defaults(ExtractChapterReferencesNode))["result"]
    sequences = chapters[0]["sequences"]
    assert [(p["sequence"], " ".join(row["text"] for row in p["source_passages"])) for p in source_inputs] == [
        (seq["sequence"], " ".join(seq["adaptation"][p].replace("\n", " ") for p in ("initialState", "event")
                                    if seq["adaptation"][p].strip())) for seq in sequences]
    assert all("phase" not in p for p in source_inputs)
    assert catalogs[0]["sequences"] == sequences
    registry, _ = ConsolidateReferencesNode().run(catalogs, config, **defaults(ConsolidateReferencesNode))["result"]
    assert registry["chapter_timelines"]["chapter"]["sequences"] == sequences

    def spatial_chat(client, model, system, user, schema, *args):
        request = json.loads(user)
        assert request["whole_chapter_timeline"] == sequences
        return {"location_id": "", "location_layout": "Unknown.",
                "camera_position": "Unknown.", "camera_direction": "Unknown.",
                "camera_movement": "Static.", "camera_axis": "Unknown.",
                "initial_entities": [], "final_entities": [], "justified_changes": [], "warnings": []}

    monkeypatch.setattr(spatial, "chat_json", with_passage_ids(spatial_chat))
    spatial_plan = spatial.SpatialContinuityNode().run(registry, config)["result"][0]
    generator = lmstudio_pipeline.load("generate")
    monkeypatch.setattr(generator, "chat_json", with_passage_ids(lambda *a: {
        "dialogue_present": False, "visible_entity_ids": [], "speaking_entity_ids": [],
        "location_global_id": "", "reference_view_requests": []}))
    prepared = []

    def generate_chapter(chapter, refs, client, model, args):
        # Execute the actual H3 input matching and metadata preparation; stop before prompt drafting.
        assert args.duration == 8 and args.spatial_plan == spatial_plan
        cid, rows = generator.selected_timeline(refs, chapter)
        assert rows == sequences
        for seq in rows:
            scene = generator.prepare_cinematic_scene(client, model, {
                "sequence": seq, "catalog": []}, "", args)
            assert scene.source_excerpt == seq["source"]
            assert scene.visual_event == seq["adaptation"]["event"]
            assert json.loads(scene.adaptation_notes.split(": ", 1)[1]) == seq["adaptation"]
            prepared.append(scene.sequence)
        (args.out_dir / cid).mkdir(parents=True, exist_ok=True)
        return {"chapter_id": cid, "outputs": []}

    monkeypatch.setattr(generator, "process_chapter", generate_chapter)
    GenerateH3PromptsNode().run(registry, config, chapters, spatial_continuity=spatial_plan,
                              **defaults(GenerateH3PromptsNode))
    assert prepared == [1]
    assert direct == before
    assert original_load(diagnostics) == report

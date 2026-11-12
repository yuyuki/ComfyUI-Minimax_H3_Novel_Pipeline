from tests.cinematic_generation_helpers import chapter_input, metadata_scene
"""Upstream evidence compilation is lossless, deterministic and resumable."""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import generation_evidence as evidence, path_access, lmstudio_json


@pytest.fixture
def inputs(tmp_path, monkeypatch, cinematic_registry):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(lmstudio_json, "chat_json", Mock(side_effect=AssertionError("No LLM for evidence compilation")))
    refs = cinematic_registry(tmp_path / "chapter.txt", source_digest="source-v1",
                              entities=[{"global_id": "hero", "canonical_name": "Indy"}],
                              picture_assets=[{"asset_id": "pic", "linked_global_id": "hero"}],
                              audio_assets=[], visual_designs={"entities": [{"global_id": "hero"}]})
    sequences = refs["chapter_timelines"]["chapter"]["sequences"]
    catalog = Mock(return_value=[{"global_id": "hero", "state_by_phase": {
        "initialState": ["Hanging"], "event": ["Calls for help"], "endingState": []}}])
    context = Mock(return_value={"registry_ids": {"local": "hero"}, "sequence": {
        "sequence": 1, "characters": [{"local_id": "local", "presence": "offscreen"}]}})
    spatial = {"source_digest": "source-v1", "chapters": {"chapter": {"sequences": [
        {"sequence": 1, "initial_entities": [{"entity_id": "hero", "evidence": "inferred", "world_position": "unknown"}]}]}}}
    return dict(refs=refs, chapter_id="chapter", sequences=sequences, spatial_plan=spatial,
                period="1922 is adaptation context", cache_dirs=[tmp_path / "first/h3/.cache/chapter"],
                catalog_for=catalog, context_for=context, dependencies={"builder": "v1"})


def test_compile_preserves_provenance_and_never_mutates_inputs(inputs):
    before = deepcopy({key: inputs[key] for key in ("refs", "sequences", "spatial_plan")})
    bundle, status = evidence.prepare(**inputs)
    assert status["status"] == "compiled"
    row = bundle["sequences"][0]
    assert row["sequence"] == inputs["sequences"][0]
    assert {p["phase"]: p["text"] for p in row["phases"]} == inputs["sequences"][0]["adaptation"]
    assert len({p["id"] for p in row["phases"]}) == 3
    assert all(p["chapter_id"] == "chapter" and p["sequence"] == 1 for p in row["phases"])
    assert row["spatial_plan"]["initial_entities"][0]["evidence"] == "inferred"
    assert row["spatial_plan"]["initial_entities"][0]["world_position"] == "unknown"
    assert row["character_context"]["sequence"]["characters"][0]["presence"] == "offscreen"
    bundle["references"]["entities"][0]["canonical_name"] = "mutated"
    row["catalog"].clear()
    assert before == {key: inputs[key] for key in before}
    cached, status = evidence.prepare(**inputs)
    assert status["status"] == "cached"
    assert cached["references"]["entities"][0]["canonical_name"] == "Indy"
    assert cached["sequences"][0]["catalog"]
    assert inputs["catalog_for"].call_count == inputs["context_for"].call_count == 1


def test_resume_carries_bundle_forward_without_recomputing_or_writing_old_run(inputs, tmp_path):
    expected, _ = evidence.prepare(**inputs)
    old = inputs["cache_dirs"][0]
    before = (old / "evidence_bundle.json").read_bytes()
    current = tmp_path / "second/h3/.cache/chapter"
    inputs["cache_dirs"] = [current, old]
    actual, status = evidence.prepare(**inputs)
    assert status["status"] == "cached"
    assert actual == expected
    assert (old / "evidence_bundle.json").read_bytes() == before
    assert (current / "evidence_bundle.json").read_bytes() == before
    inputs["cache_dirs"] = [tmp_path / "third/h3/.cache/chapter", current]
    assert evidence.prepare(**inputs)[0] == expected
    assert inputs["catalog_for"].call_count == inputs["context_for"].call_count == 1


@pytest.mark.parametrize("change", ["phase", "source", "name", "asset", "design", "spatial", "implementation", "force"])
def test_actual_upstream_changes_invalidate_even_with_unchanged_source_digest(inputs, change):
    first, _ = evidence.prepare(**inputs)
    if change == "phase":
        inputs["sequences"][0]["adaptation"]["event"] = "Selected different event"
    elif change == "source":
        inputs["sequences"][0]["source"] = "Exact changed source"
    elif change == "name":
        inputs["refs"]["entities"][0]["canonical_name"] = "Corrected name"
    elif change == "asset":
        inputs["refs"]["picture_assets"][0]["asset_id"] = "replacement"
    elif change == "design":
        inputs["refs"]["visual_designs"]["entities"][0]["appearance"] = "approved"
    elif change == "spatial":
        inputs["spatial_plan"]["chapters"]["chapter"]["sequences"][0]["camera"] = "left"
    elif change == "implementation":
        inputs["dependencies"]["builder"] = "v2"
    else:
        inputs["force"] = True
    _, status = evidence.prepare(**inputs)
    assert status["status"] == "compiled"
    assert inputs["catalog_for"].call_count == 2
    assert first["source_digest"] == inputs["refs"]["source_digest"]


@pytest.mark.parametrize("corruption", ["truncated", "digest", "shape", "version"])
def test_corrupt_bundle_is_rebuilt(inputs, corruption):
    original, _ = evidence.prepare(**inputs)
    path = inputs["cache_dirs"][0] / "evidence_bundle.json"
    record = json.loads(path.read_text())
    if corruption == "truncated":
        path.write_text("{")
    else:
        if corruption == "digest":
            record["bundle"]["references"]["entities"] = []
        elif corruption == "shape":
            record["bundle"]["references"] = []
            record["bundle_digest"] = evidence.digest(record["bundle"])
        else:
            record["version"] = -1
        path.write_text(json.dumps(record))
    rebuilt, status = evidence.prepare(**inputs)
    assert rebuilt == original
    assert status["status"] == "compiled"
    assert inputs["catalog_for"].call_count == 2


def test_cancelled_compilation_never_writes_a_complete_bundle(inputs, monkeypatch):
    class Interrupted(Exception):
        pass
    monkeypatch.setattr(evidence, "comfy_interrupt_check", Mock(side_effect=[None, Interrupted()]))
    with pytest.raises(Interrupted):
        evidence.prepare(**inputs)
    assert not (inputs["cache_dirs"][0] / "evidence_bundle.json").exists()


@pytest.mark.parametrize("change", ["stale", "duplicate", "missing"])
def test_invalid_spatial_provenance_is_rejected_before_cache_reuse(inputs, change):
    evidence.prepare(**inputs)
    if change == "stale":
        inputs["spatial_plan"]["source_digest"] = "old"
    elif change == "duplicate":
        rows = inputs["spatial_plan"]["chapters"]["chapter"]["sequences"]
        rows.append(deepcopy(rows[0]))
    else:
        inputs["spatial_plan"]["chapters"]["chapter"]["sequences"] = []
    with pytest.raises(ValueError, match="spatial evidence"):
        evidence.prepare(**inputs)
    assert inputs["catalog_for"].call_count == 1


def test_chapter_reuses_compiled_context_for_all_scenes_and_resume(tmp_path, monkeypatch, cinematic_registry):
    from types import SimpleNamespace
    from minimax_h3_novel_pipeline.core import pipeline_step3_generate as step
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    chapter = tmp_path / "chapter.txt"
    refs = cinematic_registry(chapter)
    args = SimpleNamespace(out_dir=tmp_path / "first/h3", force=False, max_scenes=0,
                           duration=8, max_tokens=8192, repair_attempts=0)
    catalog, context = Mock(return_value=[]), Mock(return_value=None)
    monkeypatch.setattr(step, "sequence_catalog", catalog)
    monkeypatch.setattr(step, "sequence_context", context)
    monkeypatch.setattr(step, "prepare_cinematic_scene", metadata_scene)
    monkeypatch.setattr(step, "build_bindings", lambda *a: {"subjects": [{"global_id": "hero"}], "audio": []})
    monkeypatch.setattr(step, "generate_prompt", lambda *a: "draft")
    monkeypatch.setattr(step, "validate_prompt", lambda *a: step.Validation(True, [], 400))
    monkeypatch.setattr(step, "save_scene", lambda *a: {"subject_count": 1, "picture_count": 1, "audio_count": 0})
    first = step.process_chapter(chapter_input(refs), refs, None, "gemma", args)
    assert first["evidence_bundle"]["status"] == "compiled"
    args.resume_cache_dir = args.out_dir / ".cache"
    args.out_dir = tmp_path / "second/h3"
    second = step.process_chapter(chapter_input(refs), refs, None, "gemma", args)
    assert len(second["outputs"]) == len(refs["chapter_timelines"]["chapter"]["sequences"])
    assert second["evidence_bundle"]["status"] == "cached"
    assert first["evidence_bundle"]["digest"] == second["evidence_bundle"]["digest"]
    assert catalog.call_count == context.call_count == 1

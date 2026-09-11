"""Offline coverage for queued runs, editable designs and copy-paste exports."""
import copy
from contextlib import nullcontext
from datetime import datetime
import json
import math
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline import (
    image_prompt_export, lmstudio_json, lmstudio_pipeline, path_access, prompt_cache,
    reference_requests, run_output, util, visual_designs,
)
from minimax_h3_novel_pipeline.consolidate_references import ConsolidateReferencesNode
from minimax_h3_novel_pipeline.generate_h3_prompts import GenerateH3PromptsNode
from minimax_h3_novel_pipeline.lmstudio_config import LMStudioConfigurationNode
from minimax_h3_novel_pipeline.load_consolidated_references import LoadConsolidatedReferencesNode


@pytest.fixture
def output_root(tmp_path, monkeypatch):
    root = tmp_path / "output"
    root.mkdir()
    monkeypatch.setattr(path_access, "storage_root", lambda kind: root)
    monkeypatch.setattr(run_output, "storage_root", lambda kind: root)
    monkeypatch.setattr(run_output, "execution_id", lambda: "queue-one")
    return root


def entity(kind="character", gid="CHAR_001"):
    return {"global_id": gid, "entity_type": kind, "canonical_name": "Aster",
            "stable_visual_description": "A silver silhouette.", "distinguishing_features": ["A star marking."],
            "importance": "major", "reference_priority": "optional", "chapters_seen": ["chapter"],
            "chapter_variations": [], "source_entities": [], "speaks": False}


def options():
    return SimpleNamespace(max_character_base_views=4, max_location_base_views=3, max_object_base_views=2,
                           picture_threshold="recommended", no_variants=False, asset_batch_size=4,
                           temperature=0.12, max_tokens=2500, image_style="realistic photographic")


def node_defaults(cls):
    result = {}
    for section in ("required", "optional"):
        for name, spec in cls.INPUT_TYPES().get(section, {}).items():
            if len(spec) > 1 and "default" in spec[1]:
                result[name] = spec[1]["default"]
    return result


def design_checks(user, verdict="compatible_addition", reason="Compatible with source."):
    return {"checks": [{"trait": trait, "verdict": verdict, "reason": reason}
                       for trait in json.loads(user)["added_details"]]}


def test_runs_share_queue_and_advance_on_collision(output_root, monkeypatch):
    class Clock:
        @staticmethod
        def now():
            return datetime(2026, 9, 11, 15, 30, 42)

    monkeypatch.setattr(run_output, "datetime", Clock)
    first = run_output.reserve_run()
    assert first == "20260911153042"
    assert run_output.reserve_run() == first
    monkeypatch.setattr(run_output, "execution_id", lambda: "queue-two")
    assert run_output.reserve_run() == "20260911153043"
    assert (output_root / first).is_dir()
    assert math.isnan(LMStudioConfigurationNode.IS_CHANGED())


def test_configuration_nodes_share_run_and_repeat_queues(output_root, monkeypatch):
    from minimax_h3_novel_pipeline import lmstudio_settings
    monkeypatch.setattr(lmstudio_settings, "get_api_key", lambda: "test-secret")
    monkeypatch.setattr(lmstudio_settings, "validate_api_url", lambda url: url)
    first, _ = LMStudioConfigurationNode().run("http://127.0.0.1:1234/v1")
    other, _ = LMStudioConfigurationNode().run("http://127.0.0.1:1234/v1")
    assert first["run_folder"] == other["run_folder"]
    monkeypatch.setattr(run_output, "execution_id", lambda: "next-queue")
    second, _ = LMStudioConfigurationNode().run("http://127.0.0.1:1234/v1")
    assert first["run_folder"] != second["run_folder"]
    assert "test-secret" not in json.dumps(first)


def test_stage_paths_normalize_legacy_absolute_and_confine(output_root):
    config = {"run_folder": run_output.reserve_run()}
    target = output_root / config["run_folder"] / "references"
    assert run_output.stage_output(config, "references") == target
    assert run_output.stage_output(config, str(output_root / "references")) == target
    assert run_output.stage_output(config, str(output_root / "20250101010101" / "references")) == target
    for value in ("../escape", str(output_root.parent / "outside"), "C:escape", "references/file:stream"):
        with pytest.raises(ValueError):
            run_output.stage_output(config, value)
    with pytest.raises(ValueError):
        run_output.stage_output({"run_folder": "../bad"}, "references")


@pytest.mark.parametrize("kind,gid,expected", [
    ("character", "CHAR_001", ["face_front", "full_body_front", "three_quarter", "back_view"]),
    ("location", "LOC_001", ["wide_establishing", "secondary_angle", "key_detail"]),
    ("object", "OBJ_001", ["hero_three_quarter", "detail_closeup"]),
])
def test_all_entity_coverage_prioritizes_core_views(kind, gid, expected):
    step = lmstudio_pipeline.load("consolidate")
    item = entity(kind, gid)
    item["reference_view_hints"] = step.ALLOWED_VIEWS[kind]
    args = options()
    specs = step.build_picture_specs([item], args)
    assert [spec["view_type"] for spec in specs] == expected
    args.image_asset_scope = "existing priority threshold"
    assert step.build_picture_specs([item], args) == []


def test_prompts_repeat_identity_design_style_and_variant(monkeypatch):
    step = lmstudio_pipeline.load("consolidate")
    item = entity()
    item["chapter_variations"] = [{"chapter_id": "chapter", "variant_reference_recommended": True,
                                    "visual_state": "Wearing a red coat."}]
    args = options()
    args.image_style = "watercolor"
    args.visual_designs = {"CHAR_001": {"added_details": {"hair": "Short copper hair."}}}
    seen_temperatures = []

    def chat(client, model, system, user, schema, temperature, max_tokens):
        seen_temperatures.append(temperature)
        return {"assets": [{"asset_id": s["asset_id"], "description": "A clear reference.",
                            "generation_prompt": "Soft even lighting, plain background."} for s in json.loads(user)]}

    monkeypatch.setattr(step, "chat_json", chat)
    assets = step.generate_picture_assets(None, "qwen", step.build_picture_specs([item], args), args)
    assert len(assets) == 6
    assert seen_temperatures == [args.temperature, args.temperature]
    for asset in assets:
        prompt = asset["generation_prompt"]
        assert all(text in prompt for text in ("Aster", "A silver silhouette.", "Short copper hair.", "watercolor"))
        assert step.VIEW_FRAMING[asset["view_type"]] in prompt
        assert ("Wearing a red coat." in prompt) == (asset["variant"] == "chapter")
        assert "<Picture" not in prompt


@pytest.mark.parametrize("bad", [
    [],
    [{"asset_id": "a", "description": "d", "generation_prompt": "p"}] * 2,
    [{"asset_id": "unknown", "description": "d", "generation_prompt": "p"}],
    [{"asset_id": "a", "description": "d", "generation_prompt": "  "}],
])
def test_invalid_asset_batches_retry_then_succeed(monkeypatch, bad):
    step = lmstudio_pipeline.load("consolidate")
    args = options()
    args.max_character_base_views = 1
    specs = step.build_picture_specs([entity()], args)
    specs[0]["asset_id"] = "a"
    responses = iter([{"assets": bad}, {"assets": [{"asset_id": "a", "description": "d", "generation_prompt": "Even light."}]}])
    calls = []

    def chat(*args):
        calls.append(args)
        return next(responses)

    monkeypatch.setattr(step, "chat_json", chat)
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    assert len(step.generate_picture_assets(None, "qwen", specs, args)) == 1
    assert len(calls) == 2


def test_asset_retries_are_bounded_and_cancelable(monkeypatch):
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    calls = []
    def chat(*args):
        calls.append(1)
        return {"assets": []}
    with pytest.raises(ValueError, match="bounded retries"):
        reference_requests.validated_request(chat, None, "model", "system", [], {"name": "assets"}, options(),
                                             lambda result: reference_requests.validate_assets(result, [{"asset_id": "a"}]))
    assert len(calls) == 2
    def stop():
        raise RuntimeError("interrupted")
    monkeypatch.setattr(reference_requests, "comfy_interrupt_check", stop)
    with pytest.raises(RuntimeError, match="interrupted"):
        reference_requests.validated_request(chat, None, "model", "system", [], {"name": "assets"}, options(), lambda x: None)
    assert len(calls) == 2


def test_edit_design_then_import_reuses_additions(output_root):
    calls = []
    def chat(client, model, system, user, schema, *args):
        calls.append(schema["name"])
        if schema is visual_designs.CONFLICT_SCHEMA:
            return design_checks(user)
        return {"added_details": [{"trait": "hair", "description": "Copper hair."}]}
    item = entity()
    original = copy.deepcopy(item)
    payload = visual_designs.prepare_designs(chat, None, "model", [item], options())
    assert item == original
    payload["entities"][0]["added_details"] = {"hair": "Black hair."}
    util.save_json(output_root / "visual_designs.json", payload)
    calls.clear()
    # Reworded source text on a later consolidation run must not block editing.
    item["stable_visual_description"] = "Silver silhouette."
    edited = visual_designs.prepare_designs(chat, None, "model", [item], options(), "visual_designs.json")
    assert edited["entities"][0]["added_details"] == {"hair": "Black hair."}
    assert calls == ["visual_design_conflicts"]


@pytest.mark.parametrize("change", ["name", "type", "id", "duplicate", "conflict"])
def test_bad_design_imports_fail_actionably(output_root, change):
    item = entity()
    design = {"global_id": item["global_id"], "canonical_name": item["canonical_name"],
              "entity_type": item["entity_type"], "source_facts": visual_designs.source_facts(item),
              "added_details": {"color": "Gold instead of silver."}}
    if change == "name":
        design["canonical_name"] = "Another character"
    if change == "type":
        design["entity_type"] = "location"
    if change == "id":
        design["global_id"] = "CHAR_999"
    util.save_json(output_root / "designs.json", {"schema_version": visual_designs.DESIGN_SCHEMA_VERSION,
                                                "entities": [design, design] if change == "duplicate" else [design]})
    def chat(client, model, system, user, *args):
        return design_checks(user, "conflict", "color contradicts the silver source description")
    with pytest.raises(ValueError, match="mismatch|Unknown|conflicts"):
        visual_designs.prepare_designs(chat, None, "model", [item], options(), "designs.json")


def test_export_preserves_assets_and_supports_old_v3_registry(output_root):
    registry = {"schema_version": util.REGISTRY_SCHEMA, "entities": [entity()], "picture_assets": [{
        "asset_id": "PIC_CHAR_001_FACE_FRONT", "linked_global_id": "CHAR_001", "view_type": "face_front",
        "variant": "base", "description": "portrait", "generation_prompt": "A silver figure facing forward.",
    }]}
    before = copy.deepcopy(registry)
    records, text = image_prompt_export.export_image_prompts(registry, output_root / "image_prompts")
    assert registry == before
    assert "None recorded." in text
    assert records[0]["prompts"][0]["generation_prompt"] in text
    assert (output_root / "image_prompts" / records[0]["text_file"]).is_file()
    saved = util.load_json(output_root / "image_prompts/image_prompts.json")
    assert saved["entities"] == records


def test_consolidation_loader_and_generation_export_in_new_run(output_root, monkeypatch):
    step = lmstudio_pipeline.load("consolidate")
    monkeypatch.setattr(lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock-qwen"))
    monkeypatch.setattr(step, "reconcile_chapter", lambda *a: [entity()])
    monkeypatch.setattr(step, "audit_registry", lambda client, model, registry, args: registry)
    def chat(client, model, system, user, schema, *args):
        if schema is visual_designs.DESIGN_SCHEMA:
            return {"added_details": [{"trait": "hair", "description": "Copper hair."}]}
        if schema is visual_designs.CONFLICT_SCHEMA:
            return design_checks(user)
        return {"assets": [{"asset_id": s["asset_id"], "description": "Reference.",
                            "generation_prompt": "Soft light."} for s in json.loads(user)]}
    monkeypatch.setattr(step, "chat_json", chat)
    config = dict(api_url="http://127.0.0.1:1234/v1", thinking=False, qwen35_length_retries=2,
                  qwen35_top_k=20, qwen35_min_p=0, qwen35_repeat_penalty=1.05,
                  run_folder=run_output.reserve_run())
    chapter = {"schema_version": util.CHAPTER_SCHEMA, "chapter_id": "chapter", "source": {}}
    result, = ConsolidateReferencesNode().run([chapter], config, **node_defaults(ConsolidateReferencesNode))
    saved = output_root / config["run_folder"] / "references/consolidated_references.json"
    loaded, = LoadConsolidatedReferencesNode().run(str(saved))
    assert loaded == result
    assert (saved.parent / "visual_designs.json").is_file()
    assert (saved.parent / "image_prompts/characters/CHAR_001_Aster.txt").is_file()
    assert saved.parent.joinpath("reference_asset_prompts.txt").is_file()
    monkeypatch.setattr(run_output, "execution_id", lambda: "next-run")
    next_config = {**config, "run_folder": run_output.reserve_run()}
    generate = lmstudio_pipeline.load("generate")
    monkeypatch.setattr(generate, "process_chapter", lambda *a: {"chapter_id": "chapter", "outputs": []})
    from minimax_h3_novel_pipeline import generate_h3_prompts as wrapper
    monkeypatch.setattr(wrapper, "selected_chapter_paths", lambda value: "chapter.txt")
    monkeypatch.setattr(util, "discover_inputs", lambda paths: paths)
    prompts, scene_text, image_text = GenerateH3PromptsNode().run(
        loaded, next_config, {}, **node_defaults(GenerateH3PromptsNode))
    assert scene_text == ""
    assert "Copper hair." in image_text
    assert prompts["image_prompts"][0]["global_id"] == "CHAR_001"
    assert (output_root / next_config["run_folder"] / "h3_prompts/image_prompts/image_prompts.json").is_file()
    assert saved.is_file()
    assert GenerateH3PromptsNode.RETURN_TYPES == ("MINIMAX_PROMPTS", "STRING", "STRING")


def test_cache_changes_with_prompt_sampling_and_scene_settings(monkeypatch):
    args = SimpleNamespace(temperature=0.1, max_tokens=2000, scenes_per_chunk=4, force=False, out_dir="old")
    first = prompt_cache.fingerprint("qwen", args, "prompt.v1")
    args.force, args.out_dir = True, "new"
    assert first == prompt_cache.fingerprint("qwen", args, "prompt.v1")
    assert first != prompt_cache.fingerprint("qwen", args, "prompt.v2")
    args.scenes_per_chunk = 5
    assert first != prompt_cache.fingerprint("qwen", args, "prompt.v1")
    args.scenes_per_chunk = 4
    monkeypatch.setattr(lmstudio_json, "QWEN35_TOP_K", 42)
    assert first != prompt_cache.fingerprint("qwen", args, "prompt.v1")


def test_design_verdicts_distinguish_source_facts_from_additions(output_root):
    def chat(client, model, system, user, schema, *args):
        if schema is visual_designs.DESIGN_SCHEMA:
            return {"added_details": [{"trait": "skin", "description": "A silver silhouette."},
                                      {"trait": "hair", "description": "Copper hair."}]}
        return {"checks": [{"trait": "skin", "verdict": "already_in_source", "reason": "Explicitly stated."},
                           {"trait": "hair", "verdict": "compatible_addition", "reason": "Source does not specify hair."}]}
    designs = visual_designs.prepare_designs(chat, None, "model", [entity()], options())
    assert designs["entities"][0]["added_details"] == {"hair": "Copper hair."}


def test_new_design_conflict_regenerates_within_retry_budget(output_root, monkeypatch):
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    generations = []
    def chat(client, model, system, user, schema, *args):
        if schema is visual_designs.DESIGN_SCHEMA:
            generations.append(1)
            return {"added_details": [{"trait": "skin", "description": "Gold." if len(generations) == 1 else "Silver."}]}
        if len(generations) == 1:
            return design_checks(user, "conflict", "Gold contradicts source silver.")
        return design_checks(user, "already_in_source", "Silver is source-supported.")
    designs = visual_designs.prepare_designs(chat, None, "model", [entity()], options())
    assert len(generations) == 2
    assert designs["entities"][0]["added_details"] == {}


def test_completed_extraction_invalidates_when_prompt_changes(output_root, monkeypatch):
    step = lmstudio_pipeline.load("extract")
    chapter = output_root / "chapter.txt"
    chapter.write_text("A long enough novel paragraph. " * 10, encoding="utf-8")
    args = SimpleNamespace(force=False, chunk_chars=5500, overlap_paragraphs=0, merge_batch_size=4,
                           temperature=0.12, max_tokens=2200, base_url="http://127.0.0.1:1234/v1")
    calls = []
    def extract(*unused):
        calls.append(1)
        return {"characters": [], "locations": [], "objects": [], "chunk_summary": "A passage."}
    monkeypatch.setattr(step, "extract_chunk", extract)
    monkeypatch.setattr(step, "hierarchical_merge_candidates", lambda *a: ({}, {}))
    monkeypatch.setattr(step, "assign_local_ids", lambda *a: {"characters": [], "locations": [], "objects": []})
    step.process_chapter(chapter, output_root, None, "qwen", args)
    step.process_chapter(chapter, output_root, None, "qwen", args)
    assert len(calls) == 1
    monkeypatch.setattr(step, "EXTRACT_SYSTEM", step.EXTRACT_SYSTEM + "\nNew extraction instruction.")
    step.process_chapter(chapter, output_root, None, "qwen", args)
    assert len(calls) == 2

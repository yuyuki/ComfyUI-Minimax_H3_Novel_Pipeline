"""Progress remains accurate on success, cache hits, failure and interruption."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from minimax_h3_novel_pipeline.core import progress
from minimax_h3_novel_pipeline.core import lmstudio_pipeline
from minimax_h3_novel_pipeline.nodes import NODE_CLASS_MAPPINGS


@pytest.fixture
def bars(monkeypatch):
    instances = []

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.current = 0
            self.values = []
            instances.append(self)

        def update_absolute(self, value):
            self.current = value
            self.values.append(value)

    comfy = ModuleType("comfy")
    utils = ModuleType("comfy.utils")
    utils.ProgressBar = ProgressBar
    comfy.utils = utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    return instances


def test_nested_work_and_skipped_items(bars):
    @progress.node_progress
    def run():
        for index in range(2):
            with progress.scope(index / 2, (index + 1) / 2):
                for cached in progress.steps([False, True]):
                    if cached:
                        continue
        # Saving output is still pending, so the bar cannot say 100 yet.
        assert bars[0].current == 99
        return "saved"

    assert run() == "saved"
    assert bars[0].values == [0, 25, 50, 75, 99, 100]
    assert progress._active.get() is None


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failure_does_not_complete_or_leak_context(bars, error, capsys):
    @progress.node_progress
    def run():
        with progress.scope(0, 0.5):
            progress.report(0.5)
            raise error("stopped")

    with pytest.raises(error, match="stopped"):
        run()
    assert bars[0].values == [0, 25]
    assert progress._active.get() is None
    assert "complete in" not in capsys.readouterr().out


@pytest.mark.parametrize("with_comfy", [True, False])
@pytest.mark.parametrize("elapsed, expected", [
    (0.0023, "00:00:00"),
    (280.4964, "00:04:40"),
    (3661.9, "01:01:01"),
    (90061, "25:01:01"),
])
def test_node_completion_time(bars, monkeypatch, capsys, with_comfy, elapsed, expected):
    if not with_comfy:
        monkeypatch.setitem(sys.modules, "comfy.utils", None)
    ticks = iter([100, 100 + elapsed])
    monkeypatch.setattr(progress.time, "perf_counter", lambda: next(ticks))
    node = NODE_CLASS_MAPPINGS["SelectChaptersNode"]()
    assert node.run("chapter.txt", "") == ({"chapter_paths": ["chapter.txt"]},)
    assert capsys.readouterr().out == (
        f"[minimax_h3_novel] SelectChaptersNode.run complete in {expected}\n"
    )


def test_empty_stages_and_monotonic_progress(bars):
    @progress.node_progress
    def run():
        assert list(progress.steps([], 0, 0.5)) == []
        progress.report(0.2)
        progress.report(1)

    run()
    assert bars[0].values == [0, 50, 99, 100]


def test_all_registered_nodes_have_progress_and_picker_reports(bars):
    for node in NODE_CLASS_MAPPINGS.values():
        assert hasattr(getattr(node, node.FUNCTION), "__wrapped__")
    result = NODE_CLASS_MAPPINGS["SelectChaptersNode"]().run("chapter.txt", "")
    assert result == ({"chapter_paths": ["chapter.txt"]},)
    assert bars[0].values == [0, 100]


@pytest.mark.parametrize("passages", [1, 2])
@pytest.mark.parametrize("review_fails", [False, True])
def test_cinematic_adaptation_advances_before_review(bars, monkeypatch, passages, review_fails):
    from minimax_h3_novel_pipeline.core import cinematic_adaptation

    reviewed = []

    def chat(client, model, system, prompt, *args):
        import json

        source = json.loads(prompt)["current_passage"]
        return {"sequences": [{"source": source, "initial_state": "Before.",
                               "events": ["Action."], "final_state": "After."}]}

    def review(*args):
        reviewed.append(bars[0].current)
        assert bars[0].current == int(100 * (len(reviewed) - 0.5) / passages)
        if review_fails:
            raise RuntimeError("review failed")
        return {}

    monkeypatch.setattr(cinematic_adaptation.lmstudio_json, "chat_json", chat)
    monkeypatch.setattr(cinematic_adaptation.narrative_validation, "review", review)

    @progress.node_progress
    def run():
        return cinematic_adaptation.adapt_chapter(
            None, "mock", "Il attend. " * passages, chunk_chars=11,
            temperature=0.15, max_tokens=8192, correction_attempts=0,
        )

    if review_fails:
        with pytest.raises(RuntimeError, match="review failed"):
            run()
        assert 100 not in bars[0].values
    else:
        assert len(run()["sequences"]) == passages
        assert bars[0].values[-2:] == [99, 100]
        assert len(reviewed) == passages
    assert bars[0].values == sorted(bars[0].values)
    assert progress._active.get() is None


def test_standalone_without_comfy(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.utils", None)

    @progress.node_progress
    def run():
        with progress.scope(0, 1):
            return list(progress.steps([1, 2]))

    assert run() == [1, 2]


def test_merge_batches_report_progress_on_fresh_and_cached_runs(bars, tmp_path, monkeypatch):
    step = lmstudio_pipeline.load("extract")
    args = SimpleNamespace(merge_batch_size=2, force=False, temperature=0.1, max_tokens=3000)
    calls = []

    def merge(*unused):
        calls.append(True)
        return {"chapter_summary": "summary", "characters": [], "locations": [], "objects": []}

    monkeypatch.setattr(step, "merge_candidates", merge)

    @progress.node_progress
    def run():
        return step.hierarchical_merge_candidates(None, "model", "chapter", [{}] * 5, args, tmp_path)

    first = run()
    assert len(calls) == 6  # Three batches, then two, then one final merge.
    assert bars[0].values == [0, 16, 33, 50, 66, 83, 99, 100]
    assert run() == first
    assert len(calls) == 6
    assert bars[1].values == bars[0].values


@pytest.fixture
def execution_context(monkeypatch):
    context = SimpleNamespace(prompt_id="workflow", node_id="extract")
    package = ModuleType("comfy_execution")
    utils = ModuleType("comfy_execution.utils")
    utils.get_executing_context = lambda: context if context.node_id else None
    package.utils = utils
    monkeypatch.setitem(sys.modules, "comfy_execution", package)
    monkeypatch.setitem(sys.modules, "comfy_execution.utils", utils)
    return context


def test_workflow_completion_metadata_preserves_socket_values(bars, execution_context):
    completed = []
    payload = ({"entities": []}, "summary")

    @progress.node_progress
    def run():
        progress.report(0.55)
        # Mid-node progress must not claim this node has completed.
        assert execution_context.node_id not in completed
        return payload

    for stage in ("extract", "consolidate", "generate"):
        execution_context.node_id = stage
        response = run()
        assert response["result"] is payload
        # ComfyUI emits `executed` only for nonempty UI output. The frontend
        # then marks this node completed in its workflow-total counter.
        assert response["ui"] == {"minimax_h3_completed": [True]}
        completed.append(stage)
    assert len(completed) == 3
    assert all(bar.values == [0, 55, 100] for bar in bars)


@pytest.mark.parametrize("name", [
    "ExtractChapterReferencesNode", "ConsolidateReferencesNode",
    "CinematicChapterAdapterNode", "GenerateH3PromptsNode",
])
def test_preview_nodes_are_execution_roots(name):
    assert NODE_CLASS_MAPPINGS[name].OUTPUT_NODE is True


def test_preview_survives_progress_wrapper(bars, execution_context):
    payload = {"ui": {"text": ["Résumé"]}, "result": ({"entities": []}, "Résumé")}

    @progress.node_progress
    def run():
        return payload

    assert run() is payload
    assert bars[0].values == [0, 100]


def test_direct_calls_with_comfy_installed_keep_tuple(bars, execution_context):
    execution_context.node_id = None
    payload = ({"chapter_paths": "chapter.txt"},)

    @progress.node_progress
    def run():
        return payload

    assert run() is payload


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_graph_failure_never_returns_completion_metadata(bars, execution_context, error):
    @progress.node_progress
    def run():
        progress.report(0.55)
        raise error("stopped")

    with pytest.raises(error, match="stopped"):
        run()
    assert bars[0].values == [0, 55]
    assert progress._active.get() is None


@pytest.mark.parametrize("fail_view", [False, True])
def test_picture_progress_counts_appearances_views_and_batches(bars, monkeypatch, capsys, fail_view):
    step = lmstudio_pipeline.load("consolidate")
    args = SimpleNamespace(asset_batch_size=4, temperature=0.1, max_tokens=3000, max_attempts=1)
    template = {"linked_global_id": "CHAR_001", "variant": "base", "entity_type": "character",
                "canonical_name": "Alice", "image_style": "cinematic photographic"}
    specs = [{**template, "asset_id": "front", "view_type": "face_front"},
             {**template, "asset_id": "profile", "view_type": "profile"}]
    observed = []

    def request(chat, client, model, system, user, schema, args, validate):
        observed.append(bars[0].current)
        if schema["name"] == "reference_appearance":
            if fail_view and len(observed) == 2:
                raise RuntimeError("view failed")
            return {"appearance": "A person with brown eyes."}
        return {"assets": [{"asset_id": s["asset_id"], "description": "Portrait",
                            "generation_prompt": "Neutral pose."} for s in user]}

    monkeypatch.setattr(step, "validated_request", request)
    monkeypatch.setattr(step, "complete_image_prompt", lambda *args: "Portrait")

    @progress.node_progress
    def run():
        return step.generate_picture_assets(None, "mock", specs, args)

    if fail_view:
        with pytest.raises(RuntimeError, match="view failed"):
            run()
        assert observed == [0, 33]
        assert bars[0].current == 33
    else:
        assert len(run()) == 2
        # Both facial views share one request; the batch is the third work unit.
        assert observed == [0, 33, 66]
        assert bars[0].values == [0, 33, 66, 99, 100]
    output = capsys.readouterr().out
    assert "Reference appearances: 1/1 starting" in output
    assert "View appearances: 1/1 starting" in output


def test_labeled_steps_report_start_and_completion(bars, capsys):
    @progress.node_progress
    def run():
        for item in progress.steps([1, 2], label="Visual designs"):
            output = capsys.readouterr().out
            assert f"Visual designs: {item}/2 starting" in output
            if item == 2:
                assert "Visual designs: 1/2 complete" in output

    run()
    assert "Visual designs: 2/2 complete" in capsys.readouterr().out
    assert bars[0].values == [0, 50, 99, 100]

"""Progress remains accurate on success, cache hits, failure and interruption."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from minimax_h3_novel_pipeline import progress
from minimax_h3_novel_pipeline import lmstudio_pipeline
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
    assert node.run("chapter.txt", "") == ({"chapter_paths": "chapter.txt"},)
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
    assert result == ({"chapter_paths": "chapter.txt"},)
    assert bars[0].values == [0, 100]


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

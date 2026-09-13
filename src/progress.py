"""Native ComfyUI progress, with scoped fractions for nested pipeline work."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps


_active = ContextVar("minimax_h3_progress", default=None)


def report(fraction):
    state = _active.get()
    if state is not None:
        bar, start, end = state
        # Only the node's successful return may report 100%.
        value = min(99, int(100 * (start + (end - start) * fraction)))
        if value > bar.current:
            bar.update_absolute(value)


@contextmanager
def scope(start, end):
    """Map child work onto a fraction of the current task."""
    state = _active.get()
    if state is None:
        yield
        return
    bar, parent_start, parent_end = state
    width = parent_end - parent_start
    token = _active.set((bar, parent_start + width * start, parent_start + width * end))
    try:
        report(0)
        yield
        report(1)
    finally:
        _active.reset(token)


def steps(items, start=0, end=1):
    """Advance after each finished item, including cached or skipped items."""
    total = len(items)
    report(start)
    for index, item in enumerate(items):
        yield item
        report(start + (end - start) * (index + 1) / total)
    report(end)


def node_progress(function):
    """Keep ComfyUI optional for standalone imports and offline tests."""
    @wraps(function)
    def run(*args, **kwargs):
        try:
            from comfy.utils import ProgressBar
        except ImportError:
            return function(*args, **kwargs)
        bar = ProgressBar(100)
        token = _active.set((bar, 0, 1))
        try:
            bar.update_absolute(0)
            result = function(*args, **kwargs)
            bar.update_absolute(100)
            return result
        finally:
            _active.reset(token)
    return run

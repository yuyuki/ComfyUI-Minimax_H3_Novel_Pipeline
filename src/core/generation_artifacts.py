"""Dependency-scoped H3 artifacts and explicit, ordered review coverage."""
from copy import deepcopy
import json
from pathlib import Path

from . import generation_requests as requests
from .lmstudio_pipeline import comfy_interrupt_check
from .path_access import confined_path


VERSION = 2


def _stats(state, stage, outcome):
    counts = state.setdefault("artifacts", {}).setdefault(stage, {})
    counts[outcome] = counts.get(outcome, 0) + 1


def cached(stage, dependencies, compute, valid):
    """Revalidate cached artifacts, never cache exceptions or partial results."""
    state = requests.cache_context()
    if state is None:
        return compute()
    comfy_interrupt_check()
    key = requests.digest([VERSION, stage, dependencies])
    requests.cache_dependencies(stage, {"contract": VERSION, **dependencies})
    relative = Path("artifacts") / stage / f"{key}.json"
    target = confined_path(relative, state["directory"])
    if state["force"]:
        requests.cache_event(stage, "force")
    for index, directory in enumerate(state["sources"] if not state["force"] else []):
        try:
            record = json.loads(confined_path(relative, directory).read_text(encoding="utf-8"))
            result = record["result"]
            reason = ("contract_mismatch" if record.get("version") != VERSION else
                      "key_mismatch" if record.get("key") != key else
                      "integrity_mismatch" if record.get("result_digest") != requests.digest(result) else
                      "validation_failed" if not valid(result) else None)
            if reason:
                requests.cache_event(stage, reason, source=index)
                _stats(state, stage, "invalid")
                continue
        except FileNotFoundError:
            requests.cache_event(stage, "missing", source=index)
            continue
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
            requests.cache_event(stage, "unreadable_record", source=index)
            _stats(state, stage, "invalid")
            continue
        comfy_interrupt_check()
        if directory != state["directory"]:
            requests.atomic_json(target, record)
        _stats(state, stage, "hit")
        requests.cache_event(stage, "hit", source=index)
        return deepcopy(result)
    _stats(state, stage, "forced" if state["force"] else "miss")
    result = compute()
    comfy_interrupt_check()
    if valid(result):
        requests.atomic_json(target, {"version": VERSION, "key": key, "result": result,
                                      "result_digest": requests.digest(result)})
    else:
        _stats(state, stage, "not_cacheable")
        requests.cache_event(stage, "not_cacheable")
    return deepcopy(result)


class ReviewCoverage:
    """Audit each required occurrence; a prior manifest never grants approval."""

    def __init__(self, dependencies, base, groups, batches):
        self.state = requests.cache_context()
        expected = {name: [requests.digest(row) for row in rows] for name, rows in groups.items()}
        covered = {name: [requests.digest(row) for batch in batches for row in batch[name]] for name in groups}
        if covered != expected:
            raise ValueError("Continuity batches do not cover every evidence occurrence in order.")
        self.key = requests.digest([VERSION, dependencies, base, groups, batches])
        self.record = {"version": VERSION, "key": self.key, "dependencies_digest": requests.digest(dependencies),
                       "base_digest": requests.digest(base), "expected": expected,
                       "batches": [{"input_digest": requests.digest(batch), "status": "pending",
                                    "coverage": {name: [requests.digest(row) for row in batch[name]] for name in groups}}
                                   for batch in batches], "status": "in_progress", "approved": False}
        self._write()

    def _write(self):
        if self.state is not None:
            path = confined_path(Path("reviews") / f"{self.key}.json", self.state["directory"])
            requests.atomic_json(path, {**self.record, "digest": requests.digest(self.record)})
            self.state["review_coverage"] = {"key": self.key, "status": self.record["status"],
                "completed_batches": sum(row["status"] == "completed" for row in self.record["batches"]),
                "required_batches": len(self.record["batches"]), "approved": self.record["approved"]}

    def completed(self, index, result, errors):
        self.record["batches"][index].update(status="completed", result_digest=requests.digest(result),
                                             issue_count=len(errors))
        self._write()

    def finish(self):
        if not all(row["status"] == "completed" for row in self.record["batches"]):
            raise ValueError("Incomplete continuity review cannot be approved.")
        self.record["status"] = "completed"
        self.record["approved"] = not any(row["issue_count"] for row in self.record["batches"])
        self._write()

    def failed(self):
        self.record.update(status="failed", approved=False)
        self._write()

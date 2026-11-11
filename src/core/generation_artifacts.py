"""Dependency-scoped H3 artifacts and explicit, ordered review coverage."""
import ast
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from . import generation_requests as requests
from .lmstudio_pipeline import comfy_interrupt_check
from .path_access import confined_path


VERSION = 1


@lru_cache(maxsize=32)
def _declarations(source):
    declarations = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        else:
            continue
        dependencies = {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}
        for name in names:
            declarations[name] = (ast.get_source_segment(source, node), dependencies)
    return declarations


def implementation(path, roots, extra_files=()):
    """Include transitive local helpers/constants, excluding unrelated stages."""
    declarations = _declarations(Path(path).read_text(encoding="utf-8"))
    if any(root not in declarations for root in roots):
        raise ValueError("H3 artifact implementation dependencies are incomplete.")
    pending, included = list(roots), {}
    while pending:
        name = pending.pop()
        if name in included or name not in declarations:
            continue
        source, dependencies = declarations[name]
        included[name] = sha256(source.encode()).hexdigest()
        pending.extend(dependencies - included.keys())
    return {"declarations": included, "modules": {
        Path(file).name: sha256(Path(file).read_bytes()).hexdigest() for file in (*extra_files, __file__)}}


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
    relative = Path("artifacts") / stage / f"{key}.json"
    target = confined_path(relative, state["directory"])
    for directory in state["sources"] if not state["force"] else []:
        try:
            record = json.loads(confined_path(relative, directory).read_text(encoding="utf-8"))
            result = record["result"]
            if (record.get("version") != VERSION or record.get("key") != key
                    or record.get("result_digest") != requests.digest(result) or not valid(result)):
                _stats(state, stage, "invalid")
                continue
        except FileNotFoundError:
            continue
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
            _stats(state, stage, "invalid")
            continue
        comfy_interrupt_check()
        if directory != state["directory"]:
            requests.atomic_json(target, record)
        _stats(state, stage, "hit")
        return deepcopy(result)
    _stats(state, stage, "forced" if state["force"] else "miss")
    result = compute()
    comfy_interrupt_check()
    if valid(result):
        requests.atomic_json(target, {"version": VERSION, "key": key, "result": result,
                                      "result_digest": requests.digest(result)})
    else:
        _stats(state, stage, "not_cacheable")
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

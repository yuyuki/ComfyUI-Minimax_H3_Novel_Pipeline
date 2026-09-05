"""Offline regression coverage for the adapted v3 scalability changes."""
import copy
import json
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline import lmstudio_pipeline


def test_hierarchical_merge_resumes_and_preserves_order(tmp_path, monkeypatch):
    step = lmstudio_pipeline.load("extract")
    args = SimpleNamespace(merge_batch_size=2, force=False, delay=0, temperature=0.1, max_tokens=3000)
    chunks = [{"chunk_summary": str(i), "characters": [{"canonical_name": str(i)}]} for i in range(5)]
    calls = []

    def merge(client, model, chapter, combined, options):
        calls.append(copy.deepcopy(combined))
        return {"chapter_summary": "summary", "characters": [
            {**e, "source_candidate_ids": [e["candidate_id"]]}
            for e in reversed(combined["characters"])
        ]}

    monkeypatch.setattr(step, "merge_candidates", merge)
    merged, combined = step.hierarchical_merge_candidates(None, "model", "chapter", chunks, args, tmp_path)
    assert all(len(c["chunk_summaries"]) <= 2 for c in calls)
    catalog = step.assign_local_ids(merged, combined)
    assert [e["canonical_name"] for e in catalog["characters"]] == list(map(str, range(5)))
    count = len(calls)
    step.hierarchical_merge_candidates(None, "model", "chapter", chunks, args, tmp_path)
    assert len(calls) == count
    args.force = True
    step.hierarchical_merge_candidates(None, "model", "chapter", chunks, args, tmp_path)
    assert len(calls) == count * 2


def test_cached_merge_checks_cancellation(tmp_path, monkeypatch):
    step = lmstudio_pipeline.load("extract")
    def stop():
        raise RuntimeError("interrupted")
    monkeypatch.setattr(lmstudio_pipeline, "comfy_interrupt_check", stop)
    with pytest.raises(RuntimeError, match="interrupted"):
        step.hierarchical_merge_candidates(None, "model", "chapter", [{}], SimpleNamespace(), tmp_path)


def entity(gid, name, kind="character"):
    return {"global_id": gid, "canonical_name": name, "entity_type": kind,
            "aliases": [], "importance": "minor", "reference_priority": "optional",
            "chapters_seen": [gid], "source_entities": [{"chapter_id": gid, "local_id": "CHAR_001"}],
            "chapter_variations": []}


def test_large_audit_is_bounded_and_rejects_unseen_ids(monkeypatch):
    step = lmstudio_pipeline.load("consolidate")
    registry = [entity(str(i), "Alice") for i in range(4)] + [entity("outside", "Castle", "location")]
    args = SimpleNamespace(no_audit=False, audit_max_entities=2, audit_cluster_size=2,
                           audit_similarity=0.68, temperature=0.1, max_tokens=6500, delay=0)
    calls = []

    def chat(client, model, system, user, *unused):
        group = json.loads(user)
        calls.append(group)
        return {"merge_groups": [{"keep_global_id": group[0]["global_id"],
                                  "merge_global_ids": [group[1]["global_id"], "outside"]}]}

    monkeypatch.setattr(step, "chat_json", chat)
    result = step.audit_registry(None, "model", registry, args)
    assert len(calls) == 2
    assert all(len(c) == 2 for c in calls)
    assert len(result) == 3
    assert result[-1]["global_id"] == "outside"
    assert len(result[0]["source_entities"]) == 2
    args.no_audit = True
    assert step.audit_registry(None, "model", registry, args) is registry
    assert len(calls) == 2


def test_cluster_matching_respects_aliases_and_types():
    step = lmstudio_pipeline.load("consolidate")
    alice = entity("a", "Lady Alice")
    alice["aliases"] = ["Alice"]
    registry = [alice, entity("b", "Alice"), entity("c", "Alice", "location")]
    assert step._audit_candidate_clusters(registry, 0.68, 24) == [["a", "b"]]

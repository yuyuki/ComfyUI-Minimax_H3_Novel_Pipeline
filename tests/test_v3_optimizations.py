"""Offline regression coverage for the adapted v3 scalability changes."""
import json
from types import SimpleNamespace

import pytest

from tests.passages_helpers import with_passage_ids

from minimax_h3_novel_pipeline.core import lmstudio_pipeline


def test_chapter_identity_batches_resume_and_preserve_order(tmp_path, monkeypatch):
    from tests.test_chapter_reference_identity import job, run
    jobs = [job(i, f"Person {i} waits.", f"Person {i}") for i in range(1, 10)]
    calls = []
    def reply(payload, count):
        calls.append(payload)
        return {"groups": []}
    catalog, _, _ = run(tmp_path, monkeypatch, jobs, reply)
    assert [e["canonical_name"] for e in catalog["characters"]] == [f"Person {i}" for i in range(1, 10)]
    assert len(calls) == 1
    cached, _, _ = run(tmp_path, monkeypatch, jobs, lambda *a: pytest.fail("Identity cache should be reused"))
    assert cached == catalog


def test_chapter_identity_checks_cancellation(tmp_path, monkeypatch):
    from tests.test_chapter_reference_identity import job, run
    def stop():
        raise RuntimeError("interrupted")
    monkeypatch.setattr(lmstudio_pipeline, "comfy_interrupt_check", stop)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(tmp_path, monkeypatch, [job(1, "A person waits.", "A person")], lambda *a: {"groups": []})


def entity(gid, name, kind="character"):
    return {"global_id": gid, "canonical_name": name, "entity_type": kind,
            "aliases": [], "importance": "minor", "reference_priority": "optional",
            "chapters_seen": [gid], "source_entities": [{"chapter_id": gid, "local_id": "CHAR_001"}],
            "timeline": {}}


def test_large_audit_is_bounded_and_rejects_unseen_ids(monkeypatch):
    step = lmstudio_pipeline.load("consolidate")
    registry = [entity(str(i), "Alice") for i in range(4)] + [entity("outside", "Castle", "location")]
    args = SimpleNamespace(no_audit=False, audit_max_entities=2, audit_cluster_size=2,
                           audit_similarity=0.68, temperature=0.1, max_tokens=6500, delay=0, max_attempts=1)
    calls = []

    def chat(client, model, system, user, *unused):
        data = json.JSONDecoder().raw_decode(user)[0]
        group = data["entities"]
        calls.append(group)
        return {"merge_groups": [{"keep_global_id": group[0]["global_id"],
                                  "merge_global_ids": [group[1]["global_id"], "outside"],
                                  "canonical_name": group[0]["canonical_name"], "aliases": [],
                                  "stable_visual_description": "", "distinguishing_features": [],
                                  "voice_description": "", "reference_view_hints": [], "reason": "",
                                  "passage_ids": ["outside"]}]}


    monkeypatch.setattr(step, "chat_json", with_passage_ids(chat))
    result = step.audit_registry(None, "model", registry, args)
    assert len(calls) == 2
    assert all(len(c) == 2 for c in calls)
    assert result == registry
    args.no_audit = True
    assert step.audit_registry(None, "model", registry, args) is registry
    assert len(calls) == 2


def test_cluster_matching_respects_aliases_and_types():
    step = lmstudio_pipeline.load("consolidate")
    alice = entity("a", "Lady Alice")
    alice["aliases"] = ["Alice"]
    registry = [alice, entity("b", "Alice"), entity("c", "Alice", "location")]
    assert step._audit_candidate_clusters(registry, 0.68, 24) == [["a", "b"]]


@pytest.mark.parametrize("model", ["qwen3.5-9b", "other-model"])
@pytest.mark.parametrize("chunk_chars,expected_count", [(1000, 6), (5500, 1)])
def test_extraction_uses_node_chunk_size(tmp_path, monkeypatch, model, chunk_chars, expected_count):
    from minimax_h3_novel_pipeline.core import cinematic_references as cr, path_access
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    extracted = []
    text = "\n\n".join(str(i) * 800 for i in range(6))
    chapter = {"chapter_name": "chapter", "sequences": [{"sequence": 1, "source": "Unused.",
        "adaptation": {"initialState": text, "event": "The scene continues.", "endingState": "Unused."}}]}
    def extract(client, model, system, user, schema, *unused):
        if schema["name"] == "narrative_semantic_review":
            return {"issues": []}
        extracted.append(" ".join(p["text"] for p in json.loads(user)["source_passages"]))
        return {"chunk_summary": "", "characters": [], "locations": [], "objects": []}
    monkeypatch.setattr(lmstudio_pipeline.load("extract"), "chat_json", with_passage_ids(extract))
    args = SimpleNamespace(force=False, chunk_chars=chunk_chars, temperature=0.1, max_tokens=3000)
    cr.process_chapter(chapter, tmp_path, None, model, args, "chapter")
    assert len(extracted) == expected_count
    assert all(len(chunk) <= chunk_chars for chunk in extracted)

"""Identity decisions use ordered source while occurrence data stays lossless."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline.core import chapter_reference_identity as identity, lmstudio_pipeline, path_access, util


def job(sequence, source, *names):
    return {"sequence": sequence, "chunk": 1, "text": source, "result": {
        "characters": [{"canonical_name": name, "aliases": [], "chapter_appearance": source,
                        "importance": "major", "reference_priority": "required", "evidence": [source]}
                       for name in names], "locations": [], "objects": [],
    }}


def group(ids, name, quote, aliases=None):
    return {"member_ids": ids, "canonical_name": name, "aliases": aliases or [], "source_quotes": [quote]}


def run(tmp_path, monkeypatch, jobs, reply):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    step = lmstudio_pipeline.load("extract")
    calls = []
    def chat(client, model, system, user, schema, *args):
        assert schema == identity.SCHEMA and schema["strict"] is True
        assert "adaptation" not in json.loads(user)
        payload = json.loads(user)
        calls.append(payload)
        return reply(payload, len(calls))
    monkeypatch.setattr(step, "chat_json", chat)
    args = SimpleNamespace(temperature=0.18, max_tokens=8192, force=False)
    diagnostics = {}
    original = deepcopy(jobs)
    catalog = identity.consolidate(jobs, step, None, "mock", args, tmp_path, diagnostics, tmp_path / "diagnostics.json")
    assert jobs == original
    return catalog, diagnostics, calls


def test_pronouns_and_named_aliases_merge_without_changing_observations(tmp_path, monkeypatch):
    jobs = [job(1, "Indy se suspend à une corde.", "Indy"),
            job(2, "Il lève les yeux.", "l'homme"),
            job(3, "— Jones ! appelle Doriane en regardant Indy.", "Jones", "Doriane")]
    def reply(payload, count):
        assert [p["text"] for p in payload["source_passages"]] == [j["text"] for j in jobs]
        assert [e["canonical_name"] for e in payload["entities"]] == ["Indy", "l'homme", "Jones", "Doriane"]
        return {"groups": [group(["REF_00003", "REF_00001", "REF_00002"], "Indy", jobs[2]["text"], ["Jones"])]}
    catalog, diagnostics, calls = run(tmp_path, monkeypatch, jobs, reply)
    indy, doriane = catalog["characters"]
    assert (indy["canonical_name"], indy["aliases"], doriane["canonical_name"]) == ("Indy", ["Jones"], "Doriane")
    assert indy["first_sequence"] == 1 and indy["chapter_appearance"] == ""
    assert [indy["state_by_sequence"][str(n)]["source"][0]["canonical_name"] for n in (1, 2, 3)] == ["Indy", "l'homme", "Jones"]
    assert len(diagnostics["identity_occurrences"]) == 4
    assert len(calls) == 1
    cached, _, cached_calls = run(tmp_path, monkeypatch, jobs, lambda *a: pytest.fail("Identity cache should be reused"))
    assert cached == catalog and cached_calls == []


def test_same_generic_name_and_identical_object_names_do_not_force_merge(tmp_path, monkeypatch):
    jobs = [job(1, "Un homme entre.", "l'homme"), job(2, "Ailleurs, un autre homme attend.", "l'homme")]
    for entry in jobs:
        entry["result"]["objects"] = [{"canonical_name": "la corde", "aliases": [], "chapter_state": entry["text"]}]
    catalog, diagnostics, _ = run(tmp_path, monkeypatch, jobs, lambda *a: {"groups": []})
    assert len(catalog["characters"]) == len(catalog["objects"]) == 2
    assert [e["first_sequence"] for e in catalog["characters"]] == [1, 2]
    assert diagnostics["identity_consolidation"][0]["status"] == "passed"


def test_later_batch_can_unify_previously_separate_identities(tmp_path, monkeypatch):
    jobs = [job(1, "Indy avance.", "Indy"), job(2, "Un homme attend.", "l'homme")]
    jobs += [job(n, "Le silence persiste.") for n in range(3, 9)]
    jobs.append(job(9, "L'homme est Indy, que Doriane appelle Jones.", "Jones"))
    def reply(payload, count):
        if count == 1:
            return {"groups": []}
        assert [p["sequence"] for p in payload["source_passages"]] == [8, 9]
        assert {e["canonical_name"] for e in payload["entities"]} == {"Indy", "l'homme", "Jones"}
        return {"groups": [group([e["id"] for e in payload["entities"]], "Indy", jobs[-1]["text"], ["Jones"])]}
    catalog, _, calls = run(tmp_path, monkeypatch, jobs, reply)
    assert len(calls) == 2 and len(catalog["characters"]) == 1
    assert list(catalog["characters"][0]["state_by_sequence"]) == ["1", "2", "9"]


@pytest.mark.parametrize("defect", ["unknown_id", "duplicate_id", "cross_type", "invented_name", "invented_alias", "invented_quote", "overlap", "schema"])
def test_invalid_merge_is_retried_and_preserves_every_occurrence(tmp_path, monkeypatch, defect):
    jobs = [job(1, "Indy et un homme parlent.", "Indy", "l'homme")]
    jobs[0]["result"]["objects"] = [{"canonical_name": "la corde", "aliases": []}]
    def reply(payload, count):
        invalid = group(["REF_00001", "REF_00002"], "Indy", jobs[0]["text"])
        if defect == "unknown_id":
            invalid["member_ids"][1] = "missing"
        elif defect == "duplicate_id":
            invalid["member_ids"][1] = "REF_00001"
        elif defect == "cross_type":
            invalid["member_ids"][1] = "REF_00003"
        elif defect == "invented_name":
            invalid["canonical_name"] = "Indiana Jones"
        elif defect == "invented_alias":
            invalid["aliases"] = ["Henry"]
        elif defect == "invented_quote":
            invalid["source_quotes"] = ["private fabricated quote"]
        elif defect == "schema":
            return {"groups": "private malformed data"}
        return {"groups": [invalid, invalid] if defect == "overlap" else [invalid]}
    catalog, diagnostics, calls = run(tmp_path, monkeypatch, jobs, reply)
    assert len(calls) == 2 and calls[0]["entities"] == calls[1]["entities"]
    assert len(catalog["characters"]) == 2 and len(catalog["objects"]) == 1
    assert diagnostics["identity_consolidation"][0]["status"] == "incomplete"
    assert diagnostics["identity_consolidation"][0]["validation_error"]
    assert calls[1]["correction"]["validation_error"]
    assert "rejected_response" in calls[1]["correction"]
    assert "private" not in json.dumps(diagnostics)
    assert not list(tmp_path.glob("identity_*.json"))


def test_retry_can_recover_without_discarding_evidence(tmp_path, monkeypatch):
    jobs = [job(1, "Indy attend. Il regarde.", "Indy", "l'homme")]
    def reply(payload, count):
        if count == 1:
            return {"groups": None}
        assert "correction" in payload
        return {"groups": [group([e["id"] for e in payload["entities"]], "Indy", jobs[0]["text"])]}
    catalog, diagnostics, calls = run(tmp_path, monkeypatch, jobs, reply)
    assert len(calls) == 2 and len(catalog["characters"]) == 1
    assert len(catalog["characters"][0]["state_by_sequence"]["1"]["source"]) == 2
    assert diagnostics["identity_consolidation"][0]["status"] == "passed"


def test_invalid_cached_identity_is_corrected_with_all_evidence(tmp_path, monkeypatch):
    jobs = [job(1, "Indy attend. Il regarde.", "Indy", "l'homme")]
    _, _, calls = run(tmp_path, monkeypatch, jobs, lambda *a: {"groups": []})
    cached, = tmp_path.glob("identity_*.json")
    util.save_json(cached, {"groups": [group(["REF_00001", "missing"], "Indy", jobs[0]["text"])]})

    def reply(payload, count):
        assert payload["entities"] == calls[0]["entities"]
        assert "distinct supplied IDs" in payload["correction"]["validation_error"]
        return {"groups": []}

    catalog, diagnostics, repaired = run(tmp_path, monkeypatch, jobs, reply)
    assert len(repaired) == 1 and len(catalog["characters"]) == 2
    assert diagnostics["identity_consolidation"][0]["status"] == "passed"


@pytest.mark.parametrize("error", [OSError("disk"), KeyboardInterrupt(),
                                 type("InterruptProcessingException", (RuntimeError,), {})("stop")])
def test_filesystem_and_cancellation_errors_propagate(tmp_path, monkeypatch, error):
    def reply(*args):
        raise error
    with pytest.raises(type(error)):
        run(tmp_path, monkeypatch, [job(1, "Indy regarde un homme.", "Indy", "l'homme")], reply)


def test_source_budget_failure_retains_data_and_marks_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(identity.continuity_context, "REQUEST_BYTES", 10)
    catalog, diagnostics, calls = run(tmp_path, monkeypatch, [job(1, "Indy attend.", "Indy", "l'homme")],
                                    lambda *a: pytest.fail("Oversized request must not be sent"))
    assert calls == [] and len(catalog["characters"]) == 2
    assert diagnostics["identity_consolidation"][0]["status"] == "incomplete"


def test_shared_aliases_are_removed_symmetrically(tmp_path, monkeypatch):
    jobs = [job(1, "Deux hommes distincts attendent.", "Indy", "Jones")]
    for entity in jobs[0]["result"]["characters"]:
        entity["aliases"] = ["l'homme"]
    catalog, _, _ = run(tmp_path, monkeypatch, jobs, lambda *a: {"groups": []})
    assert all(e["aliases"] == [] for e in catalog["characters"])


def test_identity_diagnostics_are_written_separately(tmp_path, monkeypatch):
    catalog, diagnostics, _ = run(tmp_path, monkeypatch, [job(1, "Indy attend.", "Indy")], lambda *a: {"groups": []})
    assert "identity_consolidation" not in catalog
    assert util.load_json(tmp_path / "diagnostics.json")["identity_consolidation"] == diagnostics["identity_consolidation"]

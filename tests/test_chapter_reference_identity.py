"""Identity decisions use ordered source while occurrence data stays lossless."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline.core import chapter_reference_identity as identity, lmstudio_pipeline, path_access, util


def job(sequence, source, *names):
    return {"sequence": sequence, "chunk": 1, "text": source, "result": {
        "characters": [{"canonical_name": name, "aliases": [], "presence": "visible", "visual_traits": [],
                        "reference_view_hints": [], "voice_description": "", "speaks": False,
                        "importance": "major", "reference_priority": "required", "evidence": [source], "evidence_passages": [{"chunk": 1, "start": 0, "end": len(source), "text": source}]}
                       for name in names], "locations": [], "objects": [],
    }}


def group(ids, name_id, passage_ids):
    return {"member_ids": ids, "canonical_member_id": name_id, "passage_ids": passage_ids}


def run(tmp_path, monkeypatch, jobs, reply, *, review=None, resume=None, ambiguous=True):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path if resume is None else tmp_path.parent)
    step = lmstudio_pipeline.load("extract")
    calls = []
    def chat(client, model, system, user, schema, *args):
        if schema == identity.IDENTITY_REVIEW_SCHEMA:
            return {"decisions": [{"decision_number": row["decision_number"],
                                   "verdict": review(system, row) if review else "supported"}
                                  for row in json.loads(user)["decisions"]]}
        assert schema["name"] == identity.SCHEMA["name"] and schema["strict"] is True
        assert "adaptation" not in json.loads(user)
        payload = json.loads(user)
        fields = schema["schema"]["properties"]["groups"]["items"]["properties"]
        offered = [entity["id"] for entity in payload["entities"]]
        assert fields["member_ids"]["items"]["enum"] == offered
        assert fields["canonical_member_id"]["enum"] == offered
        assert fields["passage_ids"]["items"]["enum"] == [p["id"] for p in payload["source_passages"]]
        passages_by_id = {p["id"]: p for p in payload["source_passages"]}
        for entity in payload["entities"]:
            for occurrence in entity["occurrences"]:
                for identifier in occurrence.get("evidence_passage_ids", []):
                    passage = passages_by_id[identifier]
                    assert (passage["sequence"], passage["chunk"]) == (occurrence["sequence"], occurrence["chunk"])
        calls.append(payload)
        return reply(payload, len(calls))
    monkeypatch.setattr(step, "chat_json", chat)
    args = SimpleNamespace(temperature=0.18, max_tokens=8192, force=False, resume_cache_dir=resume)
    diagnostics = {}
    original = deepcopy(jobs)
    candidate_jobs = deepcopy(jobs)
    for kind in ("characters", "locations", "objects"):
        rows = [row for job in candidate_jobs for row in job["result"][kind]]
        for row in rows:
            row.setdefault("reference_id", "new")
        if ambiguous and len(rows) > 1 and len({row["canonical_name"] for row in rows}) == len(rows):
            rows[0]["reference_id"] = "unresolved"
    entities, order = {}, 0
    for row in candidate_jobs:
        order = identity.append_occurrences(entities, row, step, order)
    catalog = identity.finish_dictionary(entities, candidate_jobs, step, None, "mock", args, tmp_path, diagnostics)
    util.save_json(tmp_path / "diagnostics.json", diagnostics)
    assert jobs == original
    return catalog, diagnostics, calls


def test_pronouns_and_named_aliases_merge_without_changing_observations(tmp_path, monkeypatch):
    jobs = [job(1, "Indy se suspend à une corde.", "Indy"),
            job(2, "Il lève les yeux.", "l'homme"),
            job(3, "— Jones ! appelle Doriane en regardant Indy.", "Jones", "Doriane")]
    def reply(payload, count):
        assert {p["sequence"] for p in payload["source_passages"]} == {1, 2, 3}
        assert [e["canonical_name"] for e in payload["entities"]] == ["Indy", "l'homme", "Jones", "Doriane"]
        passages = [p["id"] for p in payload["source_passages"]]
        assert payload["entities"][0]["occurrences"][0]["evidence_passage_ids"] == passages[:1]
        return {"groups": [group(["REF_00003", "REF_00001", "REF_00003", "REF_00002"], "REF_00001",
                                 [*passages, passages[-1]])]}
    catalog, diagnostics, calls = run(tmp_path, monkeypatch, jobs, reply)
    indy, doriane = catalog["characters"]
    assert (indy["canonical_name"], set(indy["aliases"]), doriane["canonical_name"]) == ("Indy", {"Jones", "l'homme"}, "Doriane")
    assert indy["first_sequence"] == 1
    assert [indy["state_by_sequence"][str(n)]["source"][0]["canonical_name"] for n in (1, 2, 3)] == ["Indy", "l'homme", "Jones"]
    assert len(diagnostics["identity_occurrences"]) == 4
    proof = diagnostics["identity_consolidation"][0]["groups"][0]
    assert len(proof["passage_ids"]) == len(set(proof["passage_ids"])) == len(proof["source_passages"])
    assert len(calls) == 1
    cached, _, cached_calls = run(tmp_path, monkeypatch, jobs, lambda *a: pytest.fail("Identity cache should be reused"))
    assert cached == catalog and cached_calls == []


def test_same_generic_name_and_identical_object_names_do_not_force_merge(tmp_path, monkeypatch):
    jobs = [job(1, "Un homme entre.", "l'homme"), job(2, "Ailleurs, un autre homme attend.", "l'homme")]
    for entry in jobs:
        entry["result"]["objects"] = [{"canonical_name": "la corde", "aliases": [], "presence": "visible",
                                       "visual_traits": []}]
    catalog, diagnostics, _ = run(tmp_path, monkeypatch, jobs, lambda *a: {"groups": []})
    assert len(catalog["characters"]) == len(catalog["objects"]) == 2
    assert [e["first_sequence"] for e in catalog["characters"]] == [1, 2]
    assert diagnostics["identity_consolidation"][0]["status"] == "passed"


@pytest.mark.parametrize("defect", ["unknown_id", "duplicate_id", "cross_type", "invalid_canonical", "invented_quote", "overlap", "schema"])
def test_invalid_merge_is_retried_and_preserves_every_occurrence(tmp_path, monkeypatch, defect):
    jobs = [job(1, "Indy et un homme parlent.", "Indy", "l'homme")]
    jobs[0]["result"]["objects"] = [{"canonical_name": "la corde", "aliases": [], "presence": "visible",
                                    "visual_traits": []}]
    def reply(payload, count):
        invalid = group(["REF_00001", "REF_00002"], "REF_00001", [payload["source_passages"][0]["id"]])
        if defect == "unknown_id":
            invalid["member_ids"][1] = "missing"
        elif defect == "duplicate_id":
            invalid["member_ids"][1] = "REF_00001"
        elif defect == "cross_type":
            invalid["member_ids"][1] = "REF_00003"
        elif defect == "invalid_canonical":
            invalid["canonical_member_id"] = "REF_00003"
        elif defect == "invented_quote":
            invalid["passage_ids"] = ["private fabricated passage ID"]
        elif defect == "schema":
            return {"groups": "private malformed data"}
        return {"groups": [invalid, invalid] if defect == "overlap" else [invalid]}
    catalog, diagnostics, calls = run(tmp_path, monkeypatch, jobs, reply)
    assert len(calls) == 2 and calls[0]["entities"] == calls[1]["entities"]
    assert len(catalog["characters"]) == 2 and len(catalog["objects"]) == 1
    assert diagnostics["identity_consolidation"][0]["status"] == "incomplete"
    failures = diagnostics["identity_consolidation"][0]["windows"][0]["attempts"][-1]["failures"]
    assert failures
    if defect == "duplicate_id":
        assert failures[0]["reason"] == "insufficient_distinct_members"
    assert calls[1]["correction"]["reason"]
    assert "rejected_response" not in calls[1]["correction"]
    assert "private" not in json.dumps(diagnostics)
    assert not list(tmp_path.glob("identity_*.json"))


def test_retry_can_recover_without_discarding_evidence(tmp_path, monkeypatch):
    jobs = [job(1, "Indy attend. Il regarde.", "Indy", "l'homme")]
    def reply(payload, count):
        if count == 1:
            return {"groups": None}
        assert "correction" in payload
        return {"groups": [group([e["id"] for e in payload["entities"]], "REF_00001",
                                 [payload["source_passages"][0]["id"]])]}
    catalog, diagnostics, calls = run(tmp_path, monkeypatch, jobs, reply)
    assert len(calls) == 2 and len(catalog["characters"]) == 1
    assert len(catalog["characters"][0]["state_by_sequence"]["1"]["source"]) == 2
    assert diagnostics["identity_consolidation"][0]["status"] == "passed"


def test_invalid_cached_identity_is_corrected_with_all_evidence(tmp_path, monkeypatch):
    jobs = [job(1, "Indy attend. Il regarde.", "Indy", "l'homme")]
    _, _, calls = run(tmp_path, monkeypatch, jobs, lambda *a: {"groups": []})
    cached, = tmp_path.glob("identity_*.json")
    util.save_json(cached, {"groups": [group(["REF_00001", "missing"], "REF_00001",
                                             [calls[0]["source_passages"][0]["id"]])]})

    def reply(payload, count):
        assert payload["entities"] == calls[0]["entities"]
        assert payload["correction"]["reason"] == "unknown_member_id"
        return {"groups": []}

    catalog, diagnostics, repaired = run(tmp_path, monkeypatch, jobs, reply)
    assert len(repaired) == 1 and len(catalog["characters"]) == 2
    assert diagnostics["identity_consolidation"][0]["status"] == "incomplete"


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


@pytest.mark.parametrize("reverse", [False, True])
def test_invalid_and_overlapping_groups_leave_disjoint_confirmed_group(tmp_path, monkeypatch, reverse):
    jobs = [job(1, "Mara is Vale. Two different visitors arrive.", "Mara", "Vale", "Visitor A",
                "Visitor B", "Visitor C", "Visitor D")]

    def reply(payload, count):
        ids = [e["id"] for e in payload["entities"]]
        proof = [payload["source_passages"][0]["id"]]
        if count == 2:
            assert ids == ["REF_00003", "REF_00004", "REF_00005", "REF_00006"]
            return {"groups": []}
        groups = [group(ids[:2], ids[0], proof), group(ids[2:4], ids[2], proof),
                  group(ids[3:5], ids[3], proof), group([ids[5], "invalid"], ids[5], proof)]
        return {"groups": groups[::-1] if reverse else groups}

    catalog, diagnostics, _ = run(tmp_path, monkeypatch, jobs, reply)
    assert [e["canonical_name"] for e in catalog["characters"]] == [
        "Mara", "Visitor A", "Visitor B", "Visitor C", "Visitor D"]
    assert len(catalog["characters"][0]["state_by_sequence"]["1"]["source"]) == 2
    report, = diagnostics["identity_consolidation"]
    window, = report["windows"]
    assert {f["reason"] for f in window["attempts"][0]["failures"]} == {"overlapping_groups", "unknown_member_id"}
    assert report["status"] == "incomplete"
    assert not list(tmp_path.glob("identity_*.json"))

    def resume(payload, count):
        assert [e["id"] for e in payload["entities"]] == ["REF_00003", "REF_00004", "REF_00005", "REF_00006"]
        return {"groups": []}

    resumed, diagnostics, _ = run(tmp_path, monkeypatch, jobs, resume)
    assert resumed == catalog
    assert diagnostics["identity_consolidation"][0]["status"] == "incomplete"
    assert len(diagnostics["identity_occurrences"]) == 6


def test_validated_occurrence_citation_repairs_missing_member_evidence(tmp_path, monkeypatch):
    jobs = [job(1, "Mara is nicknamed Vale.", "Mara", "Vale"),
            job(2, "Ivo, also called Ren, enters beside a guard.", "Ivo", "guard"),
            job(3, "Ren stays by the door.", "Ren")]

    def reply(payload, count):
        passages = {p["sequence"]: p["id"] for p in payload["source_passages"]}
        assert count == 1
        return {"groups": [
            group(["REF_00001", "REF_00002"], "REF_00001", [passages[1]]),
            group(["REF_00003", "REF_00003", "REF_00005"], "REF_00003", [passages[2]]),
        ]}

    catalog, diagnostics, calls = run(tmp_path, monkeypatch, jobs, reply)
    assert len(calls) == 1
    decision = diagnostics["identity_consolidation"][0]
    assert decision["status"] == "passed"
    assert [p["sequence"] for p in decision["groups"][1]["source_passages"]] == [2, 3]
    mara, ivo, guard = catalog["characters"]
    assert mara["state_by_sequence"]["1"]["source"] == jobs[0]["result"]["characters"]
    assert ivo["state_by_sequence"]["2"]["source"] == [jobs[1]["result"]["characters"][0]]
    assert ivo["state_by_sequence"]["3"]["source"] == jobs[2]["result"]["characters"]
    assert guard["canonical_name"] == "guard"


def test_oversized_rejected_response_is_not_echoed_in_retry(tmp_path, monkeypatch):
    jobs = [job(1, "A visitor waits.", "Visitor"), job(2, "The same visitor leaves.", "Visitor")]

    def reply(payload, count):
        return {"groups": "unexpected" * 30000} if count == 1 else {"groups": []}

    _, diagnostics, calls = run(tmp_path, monkeypatch, jobs, reply)
    first, second = calls
    assert first["entities"] == second["entities"]
    assert first["source_passages"] == second["source_passages"]
    assert second["correction"]["reason"] == "response_schema"
    assert len(json.dumps(second)) - len(json.dumps(first)) < 400
    assert diagnostics["identity_consolidation"][0]["status"] == "passed"


def test_split_windows_compare_occurrences_across_batches(tmp_path, monkeypatch):
    original_fits = identity.continuity_context.fits

    def fits(system, payload, schema, reserve=0):
        return len(payload.get("entities", [])) <= 4 and original_fits(system, payload, schema, reserve)

    monkeypatch.setattr(identity.continuity_context, "fits", fits)
    jobs = [job(1, "The northern gate's bell hangs above the arch."),
            *[job(n, f"A visitor enters at stop {n}.", f"Visitor {n}") for n in range(2, 9)],
            job(9, "The bell still hangs above the northern arch. A separate bell hangs at the harbour.")]
    for scene, names in ((jobs[0], ["Northern gate bell"]), (jobs[-1], ["Bell", "Bell"])):
        scene["result"]["objects"] = [{"canonical_name": name, "aliases": [], "presence": "visible",
                                       "visual_traits": []} for name in names]

    jobs[-1]["result"]["objects"][0]["reference_id"] = "unresolved"

    def reply(payload, count):
        ids = {e["id"] for e in payload["entities"]}
        if {"REF_00001", "REF_00009"} <= ids:
            passages = [p["id"] for p in payload["source_passages"]
                        if p["sequence"] == 1 or (p["sequence"] == 9 and "northern arch" in p["text"])]
            return {"groups": [group(["REF_00001", "REF_00009"], "REF_00001", passages)]}
        return {"groups": []}

    catalog, diagnostics, calls = run(tmp_path, monkeypatch, jobs, reply)
    assert all(len(call["entities"]) <= 4 for call in calls)
    first, second = catalog["objects"]
    assert list(first["state_by_sequence"]) == ["1", "9"]
    assert list(second["state_by_sequence"]) == ["9"]
    assert first["canonical_name"] == "Northern gate bell"
    assert len(diagnostics["identity_occurrences"]) == 10


def test_prologue_coexistence_blocks_merge_between_confirmed_groups(tmp_path, monkeypatch):
    names = ["maillet", "bâton enflammé"]
    sentences = [
        "Il porte un sac à dos et tient un maillet ainsi qu'un bâton enflammé.",
        "Indy est devant la tablette, tenant un bâton enflammé et un maillet, avec une torchère plantée dans le roc.",
    ]
    jobs = []
    for number in range(4, 13):
        row = job(number, sentences[number != 10] if number >= 10 else "Il attend.")
        if number >= 10:
            row["result"]["objects"] = [
                {"canonical_name": name, "aliases": [], "presence": "visible", "visual_traits": [],
                 "evidence": [row["text"]], "evidence_passages": [{"chunk": 1, "start": 0, "end": len(row["text"]),
                                                                "text": row["text"]}]} for name in names]
        jobs.append(row)

    for row in jobs[-2:]:
        for index, occurrence in enumerate(row["result"]["objects"], 1):
            occurrence["reference_id"] = f"REF_{index:05d}"
    jobs[-3]["result"]["objects"][0]["reference_id"] = "unresolved"

    def reply(payload, count):
        ids = [e["id"] for e in payload["entities"]]
        proof = [p["id"] for p in payload["source_passages"]]
        return {"groups": [group(ids, ids[0], proof)]}

    def review(system, decision):
        assert system != identity.COEXISTENCE_SYSTEM
        return "supported"

    catalog, diagnostics, calls = run(tmp_path, monkeypatch, jobs, reply, review=review, ambiguous=False)
    assert len(calls) == 2
    assert [row["canonical_name"] for row in catalog["objects"]] == names
    assert [list(row["state_by_sequence"]) for row in catalog["objects"]] == [["10", "11", "12"]] * 2
    assert any(f.get("stage") == "coexistence_review" and f["reason"] == "evidence_distinct_instances"
               and f["evidence"] in sentences
               for w in diagnostics["identity_consolidation"][0]["windows"]
               for a in w["attempts"] for f in a["failures"])


def test_same_sentence_object_aliases_can_merge(tmp_path, monkeypatch):
    source = "Mara holds a brooch and a clasp, two names for the same object."
    scene = job(1, source)
    scene["result"]["objects"] = [{"canonical_name": name, "aliases": [], "presence": "visible",
                                  "visual_traits": [], "evidence": [source]}
                                 for name in ("brooch", "clasp")]

    def reply(payload, count):
        ids = [row["id"] for row in payload["entities"]]
        return {"groups": [group(ids, ids[0], [payload["source_passages"][0]["id"]])]}

    catalog, _, _ = run(tmp_path, monkeypatch, [scene], reply)
    merged, = catalog["objects"]
    assert merged["aliases"] == ["clasp"]
    assert [o["canonical_name"] for o in merged["state_by_sequence"]["1"]["source"]] == ["brooch", "clasp"]


@pytest.mark.parametrize("citing_distinction", [0, 1, 2])
def test_coordinated_objects_require_citations_from_both_occurrences(tmp_path, monkeypatch, citing_distinction):
    alias = "Mara calls her heirloom a brooch or a clasp."
    distinction = "In the play, Mara holds a brooch and a clasp."
    source = f"{alias} {distinction}"
    offset = 190
    scene = {**job(1, source), "offset": offset}
    for index, name in enumerate(("brooch", "clasp")):
        quote = distinction if index < citing_distinction else alias
        start = offset + source.index(quote)
        scene["result"]["objects"].append(
            {"canonical_name": name, "aliases": [], "presence": "visible", "visual_traits": [],
             "evidence": [quote], "evidence_passages": [
                 {"chunk": 1, "start": start, "end": start + len(quote), "text": quote}]})
    coexistence = []

    def reply(payload, count):
        ids = [row["id"] for row in payload["entities"]]
        return {"groups": [group(ids, ids[0], [p["id"] for p in payload["source_passages"]])]}

    def review(system, decision):
        if system == identity.COEXISTENCE_SYSTEM:
            coexistence.append(decision)
        return "supported"

    catalog, diagnostics, _ = run(tmp_path, monkeypatch, [scene], reply, review=review)
    if citing_distinction == 2:
        assert len(catalog["objects"]) == 2 and not coexistence
        failures = diagnostics["identity_consolidation"][0]["windows"][0]["attempts"][0]["failures"]
        assert any(f["reason"] == "evidence_distinct_instances" and f["evidence"] == distinction for f in failures)
    else:
        assert len(catalog["objects"]) == 1 and len(coexistence) == 1
        assert coexistence[0]["source_passages"][0]["text"] == source


def test_resume_rechecks_identity_decisions_from_previous_validation_version(tmp_path, monkeypatch):
    jobs = [job(1, "Mara opens a gate.", "Mara"), job(2, "Mara walks through it.", "Mara")]
    older = tmp_path / "older"
    resumed = tmp_path / "resumed"
    older.mkdir()
    resumed.mkdir()
    with monkeypatch.context() as old_rules:
        old_rules.setattr(identity, "VERSION", "chapter-reference-identity.v13")
        _, _, calls = run(older, old_rules, jobs, lambda *args: {"groups": []})
    assert len(calls) == 1 and list(older.glob("identity_*.json"))

    def reply(payload, count):
        identifiers = [row["id"] for row in payload["entities"]]
        return {"groups": [group(identifiers, identifiers[0], [p["id"] for p in payload["source_passages"]])]}

    catalog, _, calls = run(resumed, monkeypatch, jobs, reply, resume=older)
    assert len(calls) == 1 and len(catalog["characters"]) == 1

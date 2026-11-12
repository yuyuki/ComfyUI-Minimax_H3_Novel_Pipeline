"""User selections reach extraction and H3 planning without alternative leakage."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline.core import cinematic_references as cr, narrative_choices as nc, path_access, util, pipeline_step3_generate as generate
from minimax_h3_novel_pipeline.nodes.load_cinematic_chapters import LoadCinematicChaptersNode


def chapter(review):
    return {"chapter_name": "chapter", "sequences": [{"sequence": 1, "source": "He waits.",
            "adaptation": {p: "Original " + p for p in nc.PHASES}}], "narrative_review": review}


def test_proposal_rejects_source_metadata_in_events():
    proposals = [{"reason": "Choix", "adaptation": {
        "initialState": "Il attend.", "event": "1. Il attend.\n2. source_excerpt: Il attend.",
        "endingState": "Il attend encore.",
    }} for _ in range(2)]
    with pytest.raises(ValueError, match="source_excerpt metadata"):
        nc.propose(lambda *args: {"proposals": proposals}, None, "mock", {}, 1000)


def test_dialogue_mentioning_source_excerpt_is_preserved():
    nc.validate_scene_text('1. Il dit : « source_excerpt: est une clé JSON. »')


@pytest.mark.parametrize("flags,chosen", [([False, False], 0), ([True, True], 0), ([False, True], 1)])
def test_normalize_apply_without_mutating_input(narrative_review, flags, chosen, capsys):
    for proposal, flag in zip(narrative_review["issues"][0]["proposals"], flags):
        proposal["selected"] = flag
    value = chapter(narrative_review)
    original = deepcopy(value)
    resolved = cr.parse_chapters([value])[0]
    proposals = resolved["narrative_review"]["issues"][0]["proposals"]
    assert [p["selected"] for p in proposals] == [i == chosen for i in range(2)]
    assert resolved["sequences"][0]["adaptation"] == proposals[chosen]["adaptation"]
    assert value == original
    assert nc.resolve_chapter(resolved) == resolved
    assert bool(capsys.readouterr().out) == (sum(flags) != 1)


@pytest.mark.parametrize("mutate", [
    lambda r: r["issues"][0].update(sequence=99),
    lambda r: r["issues"][0].update(proposals=[]),
    lambda r: r["issues"][0]["proposals"][0].update(selected="true"),
    lambda r: r["issues"][0]["proposals"][0]["adaptation"].update(event=[]),
])
def test_malformed_choices_remain_technical_errors(narrative_review, mutate):
    mutate(narrative_review)
    with pytest.raises(ValueError):
        cr.parse_chapters([chapter(narrative_review)])


def test_file_edit_reloads_changes_extraction_cache_and_reaches_h3(tmp_path, monkeypatch, narrative_review):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    path = tmp_path / "chapter.cinematic.json"
    value = chapter(narrative_review)
    util.save_json(path, value)
    loader = LoadCinematicChaptersNode()
    before = loader.IS_CHANGED(str(path))
    chapters, _ = loader.run(str(path))
    calls = []

    def extract(client, model, system, user, schema, *args):
        if schema["name"] == "narrative_semantic_review":
            return {"issues": []}
        calls.append(json.loads(user))
        return {"characters": [], "locations": [], "objects": [], "chunk_summary": ""}

    monkeypatch.setattr(cr.lmstudio_pipeline.load("extract"), "chat_json", extract)
    args = SimpleNamespace(force=False, chunk_chars=4000, max_tokens=2000, temperature=0.1)
    output = tmp_path / "catalogs"
    output.mkdir()
    saved = cr.process_chapter(chapters[0], output, None, "mock", args, "chapter")
    first = util.load_json(saved)
    assert len(calls) == 3
    assert all(c["text"].startswith("B chosen") for c in calls)
    assert "A chosen" not in json.dumps(calls)
    proposals = value["narrative_review"]["issues"][0]["proposals"]
    proposals[0]["selected"], proposals[1]["selected"] = True, False
    util.save_json(path, value)
    assert loader.IS_CHANGED(str(path)) != before
    chapters, _ = loader.run(str(path))
    cr.process_chapter(chapters[0], output, None, "mock", args, "chapter")
    second = util.load_json(saved)
    assert first["cache_key"] != second["cache_key"]
    assert len(calls) == 6
    assert all(c["text"].startswith("A chosen") for c in calls[3:])
    assert second["narrative_review"]["issues"] == value["narrative_review"]["issues"]
    refs = {"chapters": [{"chapter_id": "chapter", "source_file": str(path)}],
            "chapter_timelines": {"chapter": second}}
    cid, sequences = generate.selected_timeline(refs, {**chapters[0], "source_file": str(path)})
    assert cid == "chapter"
    assert "A chosen event" in json.dumps(sequences)
    assert "B chosen" not in json.dumps(sequences) and "proposals" not in json.dumps(sequences)


def test_proposal_failure_is_not_swallowed():
    with pytest.raises(ValueError, match="two or three"):
        nc.propose(lambda *a: {"proposals": []}, None, "mock", {}, 1000)
    def unavailable(*args):
        raise ConnectionError("Offline")
    with pytest.raises(ConnectionError):
        nc.propose(unavailable, None, "mock", {}, 1000)

"""Source fidelity, chapter isolation, caching and structured shared transport."""
import argparse
from contextlib import nullcontext
import json
from unittest.mock import Mock
from types import SimpleNamespace

import httpx
from openai import APIConnectionError
import pytest

from minimax_h3_novel_pipeline import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from minimax_h3_novel_pipeline.core import character_evidence as evidence, path_access
from minimax_h3_novel_pipeline.nodes import character_evidence_extractor as node
from tests.test_structured_json import Stream, client_for


def observation(quote="Élise avait les yeux verts.", **overrides):
    return {"character": "Élise", "attribute": "physical_appearance", "value": "yeux verts",
            "classification": "explicit", "confidence": 0.95, "persistence": "persistent",
            "quotation": quote, "rationale": "Description directe.", **overrides}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path / kind)
    source = tmp_path / "input"
    source.mkdir()
    return source, tmp_path / "output"


def settings(**overrides):
    return argparse.Namespace(**{**dict(chunk_chars=1000, temperature=0.15, max_tokens=8192,
                                      force=False, correction_attempts=1), **overrides})


def test_output_overflow_splits_with_overlap_and_preserves_cache(workspace, monkeypatch):
    monkeypatch.setattr(evidence.lmstudio_json, "THINKING_ENABLED", True)
    monkeypatch.setattr(evidence.lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    source, output = workspace
    path = source / "chapter.txt"
    quote = "Élise avait les yeux verts."
    text = quote + " " * 350 + quote + " " * 350 + quote
    path.write_text(text, encoding="utf-8")
    truncated = SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content='{"evidence":[' + ' ' * 1000, reasoning_content="brief"),
        finish_reason="length")])
    streams = [Stream([truncated]), Stream([truncated]),
               Stream([json.dumps({"evidence": [observation()]})]),
               Stream([json.dumps({"evidence": [observation()]})])]
    client, create = client_for(*streams)
    diagnostics = []
    args = settings(character_scope="all_characters")
    result = evidence.extract_chapter(path, output, client, "qwen3.8", args, 1, diagnostics=diagnostics)
    assert create.call_count == 4
    assert all(s.closed for s in streams)
    for call in create.call_args_list:
        schema = call.kwargs["response_format"]
        assert schema == {"type": "json_schema", "json_schema": evidence.SCHEMA}
        assert call.kwargs["max_tokens"] == args.max_tokens
    left, right = [json.loads(call.kwargs["messages"][-1]["content"])["passages"][0]
                   for call in create.call_args_list[-2:]]
    assert text.startswith(left) and text.endswith(right)
    assert len(left) + len(right) > len(text)
    locations = [loc for item in result["evidence"] for loc in item["provenance"]["locations"]]
    assert len(locations) == 3
    assert all(text[loc["start_char"]:loc["end_char"]] == quote for loc in locations)
    assert diagnostics[0]["overflow_splits"] == 1
    assert evidence.extract_chapter(path, output, client, "qwen3.8", args, 1) == result
    assert create.call_count == 4


def test_overflow_splitting_is_bounded_and_never_caches_partial_result(workspace, monkeypatch):
    source, output = workspace
    path = source / "chapter.txt"
    path.write_text("Élise " * 150, encoding="utf-8")
    chat = Mock(side_effect=evidence.lmstudio_json.StructuredOutputLengthError("output limit"))
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    with pytest.raises(RuntimeError, match="character offset 0"):
        evidence.extract_chapter(path, output, None, "qwen3.8",
                                 settings(character_scope="all_characters", correction_attempts=10), 1)
    assert chat.call_count == 3
    assert not list(output.rglob("*.json"))


def test_overflow_splits_disjoint_parts_without_joining_them(monkeypatch):
    chat = Mock(side_effect=[evidence.lmstudio_json.StructuredOutputLengthError("output limit"),
                            {"evidence": [observation("first")]},
                            {"evidence": [observation("second")]}])
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    result = evidence._extract_parts(None, "qwen3.8", evidence.SYSTEM, "chapter",
                                     ["first", "second"], settings(), {"llm_calls": 0})
    assert [item["quotation"] for item in result["evidence"]] == ["first", "second"]
    assert [json.loads(call.args[3])["passages"] for call in chat.call_args_list] == [
        ["first", "second"], ["first"], ["second"]]


def test_node_keeps_every_chapter_and_exact_source(workspace, monkeypatch):
    source, output = workspace
    text = "\ufeffÉlise avait les yeux verts.  \r\n\r\nÉlise avait les yeux verts."
    for name in ("chapter10.txt", "chapter2.txt"):
        (source / name).write_bytes(text.encode("utf-8"))
    chat = Mock(return_value={"evidence": [observation()]})
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model",
                        lambda *args: (nullcontext(), "qwen3.5"))
    result = node.CharacterEvidenceExtractorNode().run(
        {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261024120000"},
        {"chapter_paths": [str(source)]})["result"]
    data, filename = result
    assert filename == str(output / "20261024120000/character_evidence/character_evidence.json")
    assert json.loads(open(filename, encoding="utf-8").read()) == data
    assert [c["chapter_name"] for c in data["chapters"]] == ["chapter2", "chapter10"]
    assert chat.call_count == 2
    for i, chapter in enumerate(data["chapters"], 1):
        assert chapter["source_text"] == text.removeprefix("\ufeff")
        item = chapter["evidence"][0]
        assert item["provenance"]["chapter"] == i
        assert len(item["provenance"]["locations"]) == 2
        for loc in item["provenance"]["locations"]:
            assert chapter["source_text"][loc["start_char"]:loc["end_char"]] == item["quotation"]


def test_cache_force_source_settings_and_corruption(workspace, monkeypatch):
    source, output = workspace
    path = source / "chapter.txt"
    path.write_text("Élise avait les yeux verts.", encoding="utf-8")
    chat = Mock(return_value={"evidence": [observation()]})
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    def run(**kwargs):
        return evidence.extract_chapter(path, output, None, "qwen3.5", settings(**kwargs), 1)
    assert run() == run()
    assert chat.call_count == 1
    run(force=True)
    assert chat.call_count == 2
    run(temperature=0.2)
    assert chat.call_count == 3
    path.write_text("Élise avait les yeux verts. Elle attend.", encoding="utf-8")
    run()
    assert chat.call_count == 4
    for cache in (output / ".cache").glob("*.json"):
        cache.write_text('{"broken": true}', encoding="utf-8")
    run()
    assert chat.call_count == 5


@pytest.mark.parametrize("overrides,expected_hits", [({}, 1), ({"force": True}, 0),
                                                    ({"max_tokens": 4096}, 0)])
def test_resume_run_reuses_completed_work_and_copies_cache(workspace, monkeypatch, overrides, expected_hits):
    source, output = workspace
    first = source / "chapter1.txt"
    second = source / "chapter2.txt"
    for path in (first, second):
        path.write_text("Élise avait les yeux verts.", encoding="utf-8")
    chat = Mock(return_value={"evidence": [observation()]})
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model",
                        lambda *args: (nullcontext(), "qwen3.5"))
    extractor = node.CharacterEvidenceExtractorNode()
    def run(stamp, paths, **kwargs):
        return extractor.run({"api_url": "http://127.0.0.1:1234/v1", "run_folder": stamp},
                             {"chapter_paths": [str(p) for p in paths]}, **kwargs)["result"][0]
    old_stamp, new_stamp = "20261025154413", "20261025160000"
    run(old_stamp, [first])
    previous = output / old_stamp
    original = {p.relative_to(previous): p.read_bytes() for p in previous.rglob("*") if p.is_file()}
    result = run(new_stamp, [first, second], resume_run=old_stamp, **overrides)
    assert chat.call_count == 3 - expected_hits
    assert len(result["chapters"]) == 2
    diagnostics = json.loads((output / new_stamp / "character_evidence/character_evidence_diagnostics.json").read_text())
    assert diagnostics["resume_run"] == old_stamp
    assert sum(c["resumed_cache_hits"] for c in diagnostics["chapters"]) == expected_hits
    assert len(list((output / new_stamp / ".cache/character_evidence").glob("*.json"))) == 2
    assert original == {p.relative_to(previous): p.read_bytes() for p in previous.rglob("*") if p.is_file()}
    # The new run contains both recovered and newly extracted batches, so it can
    # itself be the only resume source on the next execution.
    calls = chat.call_count
    run("20261025170000", [first, second], resume_run=new_stamp,
        **{k: v for k, v in overrides.items() if k != "force"})
    assert chat.call_count == calls


@pytest.mark.parametrize("resume_run", ["../20261025154413", "20261025154413/.cache", "C:\\cache",
                                       "unknown", None, "20261025154413", "20261025160000"])
def test_resume_run_rejects_invalid_missing_or_current_run(workspace, monkeypatch, resume_run):
    source, output = workspace
    path = source / "chapter.txt"
    path.write_text("Élise attend.", encoding="utf-8")
    make_client = Mock(side_effect=AssertionError("must validate before client creation"))
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model", make_client)
    with pytest.raises(ValueError):
        node.CharacterEvidenceExtractorNode().run(
            {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261025160000"},
            {"chapter_paths": [str(path)]}, resume_run=resume_run)
    make_client.assert_not_called()
    assert not output.exists()


def test_resume_corrupt_cache_recomputed_without_changing_source(workspace, monkeypatch):
    source, output = workspace
    path = source / "chapter.txt"
    path.write_text("Élise avait les yeux verts.", encoding="utf-8")
    chat = Mock(return_value={"evidence": [observation()]})
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    previous = output / "20261025154413/.cache/character_evidence"
    evidence.extract_chapter(path, output, None, "qwen3.5", settings(), 1, cache_dir=previous)
    cached_file = next(previous.glob("*.json"))
    cached_file.write_text("broken", encoding="utf-8")
    current = output / "20261025160000/.cache/character_evidence"
    evidence.extract_chapter(path, output, None, "qwen3.5", settings(), 1,
                             cache_dir=current, resume_cache_dir=previous)
    assert chat.call_count == 2
    assert cached_file.read_text() == "broken"
    assert json.loads(next(current.glob("*.json")).read_text(encoding="utf-8")) == [{"evidence": [observation()]}]


@pytest.mark.parametrize("overrides", [
    {"quotation": "Elise had green eyes."}, {"classification": "fact"},
    {"confidence": True}, {"confidence": float("nan")}, {"confidence": 1.1},
    {"attribute": "clothing", "persistence": "persistent"},
    {"attribute": "accessory", "persistence": "unknown"}, {"rationale": ""},
])
def test_invalid_evidence_warns_and_keeps_valid_observations(overrides, caplog):
    valid = observation()
    result = evidence.validate_result({"evidence": [observation(**overrides), valid]}, valid["quotation"])
    assert len(result) == 2
    assert result[1] == valid
    assert result[0]["warnings"]
    assert {k: v for k, v in result[0].items() if k != "warnings"} == observation(**overrides)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "item 1" in caplog.text


@pytest.mark.parametrize("raw", [None, [], {}, {"evidence": None},
                                 {"evidence": [], "extra": True}])
def test_invalid_result_warns_without_raising(raw, caplog):
    result = evidence.validate_result(raw, "Élise attend.")
    if raw != {"evidence": [], "extra": True}:
        assert result[0]["warnings"]
    assert caplog.records
    assert all(record.levelname == "WARNING" for record in caplog.records)


@pytest.mark.parametrize("item", [None, {}, {**observation(), "extra": True}])
def test_malformed_item_warns_without_losing_valid_evidence(item, caplog):
    valid = observation()
    result = evidence.validate_result({"evidence": [item, valid]}, valid["quotation"])
    assert len(result) == 2
    assert result[1] == valid
    assert result[0]["warnings"]
    if not isinstance(item, dict):
        assert result[0]["raw_evidence"] == item
    else:
        assert {k: v for k, v in result[0].items() if k != "warnings"} == item
    assert "requested fields" in caplog.text


@pytest.mark.parametrize("valid_items", [[], [observation()]])
def test_failed_repairs_keep_all_observations_and_do_not_block(workspace, monkeypatch, caplog, valid_items):
    source, output = workspace
    path = source / "chapter.txt"
    path.write_text("Élise avait les yeux verts.", encoding="utf-8")
    chat = Mock(return_value={"evidence": [observation("translated"), *valid_items]})
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    result = evidence.extract_chapter(path, output, None, "qwen3.5", settings(), 1)
    assert len(result["evidence"]) == 1 + len(valid_items)
    assert result["evidence"][0]["quotation"] == "translated"
    assert result["evidence"][0]["warnings"]
    assert result["evidence"][0]["provenance"]["locations"] == []
    assert chat.call_count == 2
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert "observations need review" in caplog.text
    cache = next((output / ".cache").glob("*.json"))
    assert json.loads(cache.read_text(encoding="utf-8")) == [{"evidence": [observation("translated"), *valid_items]}]
    assert evidence.extract_chapter(path, output, None, "qwen3.5", settings(), 1) == result
    assert chat.call_count == 2
    assert len(caplog.records) == 2


@pytest.mark.parametrize("item", [None, {}, observation(quote=""), observation(quote=123),
                                observation(attribute=[], classification={}, persistence=[]),
                                observation(confidence="unknown")])
def test_malformed_observations_are_saved_without_blocking(workspace, monkeypatch, item):
    source, output = workspace
    path = source / "chapter.txt"
    path.write_text("Élise avait les yeux verts.", encoding="utf-8")
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", Mock(return_value={"evidence": [item]}))
    result = evidence.extract_chapter(path, output, None, "qwen3.5", settings(), 1)
    saved = result["evidence"][0]
    assert saved["warnings"]
    if isinstance(item, dict):
        assert {k: v for k, v in saved.items() if k not in {"warnings", "provenance"}} == item
    else:
        assert saved["raw_evidence"] == item
    assert json.loads(json.dumps(result)) == result


def test_all_validation_problems_are_recorded(caplog):
    item = observation(quote="translated", confidence=2, persistence="persistent", attribute="clothing")
    result = evidence.validate_result({"evidence": [item]}, "Élise attend.")
    assert len(result[0]["warnings"]) == 3
    assert [record.message.split(": ", 1)[1] for record in caplog.records] == result[0]["warnings"]


def test_chunks_cover_source_with_boundary_overlap():
    text = "x" * 998 + "Élise avait les yeux verts." + "y" * 1100
    chunks = list(evidence.passages(text, 1000))
    covered = set()
    for offset, part in chunks:
        assert text[offset:offset + len(part)] == part
        assert len(part) <= 1000
        covered.update(range(offset, offset + len(part)))
    assert len(covered) == len(text)
    assert any("Élise avait les yeux verts." in part for _, part in chunks)


def test_extraction_uses_json_schema_and_repairs_malformed_json(workspace, monkeypatch):
    monkeypatch.setattr(evidence.lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    first, second = Stream(['{"evidence":']), Stream(['```json\n{"evidence": []}\n```'])
    client, create = client_for(first, second)
    source, output = workspace
    path = source / "chapter.txt"
    path.write_text("Élise attend.", encoding="utf-8")
    result = evidence.extract_chapter(path, output, client, "qwen3.5", settings(), 1)
    assert result["evidence"] == []
    assert len(create.call_args_list) == 2
    assert all(call.kwargs["response_format"] == {"type": "json_schema", "json_schema": evidence.SCHEMA}
               for call in create.call_args_list)
    items = evidence.SCHEMA["schema"]["properties"]["evidence"]["items"]
    assert set(items["required"]) == evidence.FIELDS == set(items["properties"])
    assert first.closed and second.closed


def test_registration_and_inputs():
    cls = NODE_CLASS_MAPPINGS["CharacterEvidenceExtractorNode"]
    assert NODE_DISPLAY_NAME_MAPPINGS["CharacterEvidenceExtractorNode"] == "Character Evidence Extractor"
    assert cls.RETURN_TYPES == ("MINIMAX_CHARACTER_EVIDENCE", "STRING")
    assert "api_key" not in cls.INPUT_TYPES()["required"]
    assert cls.IS_CHANGED() != cls.IS_CHANGED()
    with pytest.raises(ValueError, match="requires Qwen"):
        cls().run({"model_family": "Mistral"}, {})


def test_interrupt_is_not_retried(workspace, monkeypatch):
    class InterruptProcessingException(RuntimeError):
        pass
    source, output = workspace
    path = source / "chapter.txt"
    path.write_text("Élise avait les yeux verts.", encoding="utf-8")
    chat = Mock(side_effect=InterruptProcessingException())
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    with pytest.raises(InterruptProcessingException):
        evidence.extract_chapter(path, output, None, "qwen3.5", settings(), 1)
    assert chat.call_count == 1
    assert not (output / ".cache").exists()


def test_pdf_locations_use_extracted_text(workspace, monkeypatch):
    source, output = workspace
    path = source / "chapter.pdf"
    path.write_bytes(b"mock PDF")
    quote = "Élise avait les yeux verts."
    monkeypatch.setattr("pypdf.PdfReader", lambda _: Mock(pages=[
        Mock(extract_text=lambda: "Page one."), Mock(extract_text=lambda: quote)]))
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", lambda *a: {"evidence": [
        observation(classification="inferred", confidence=0.4, persistence="unknown")]})
    result = evidence.extract_chapter(path, output, None, "qwen3.5", settings(), 1)
    assert result["location_basis"] == "pdf_extracted_text"
    assert result["source_text"] == "Page one.\n\n" + quote
    assert result["evidence"][0]["classification"] == "inferred"
    assert result["evidence"][0]["provenance"]["locations"][0]["start_line"] == 3


def test_network_errors_have_context_without_server_details(workspace, monkeypatch):
    source, output = workspace
    path = source / "chapter.txt"
    path.write_text("Élise attend.", encoding="utf-8")
    chat = Mock(side_effect=APIConnectionError(message="private server detail",
                                             request=httpx.Request("POST", "http://localhost")))
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    with pytest.raises(RuntimeError, match="chapter.txt, passage 1") as caught:
        evidence.extract_chapter(path, output, None, "qwen3.5", settings(), 1)
    assert "private server detail" not in str(caught.value)
    assert chat.call_count == 1


def test_novel_index_finds_later_name_and_retains_pronoun_paragraphs():
    first = "élise attend.\n\nElle a les yeux verts.\n\nSes cheveux sont noirs.\n\nElle porte une bague."
    last = "Élise, appelée Lise, avait les yeux bleus."
    occurrences = evidence.index_names([(first, "text"), (last, "text")])
    assert occurrences[0] == [(0, 5)]
    selected = evidence.selected_passages(first, 1000, occurrences[0])
    assert selected == [(0, first)]
    assert evidence.selected_passages(last, 1000, occurrences[1]) == [(0, last)]


def test_selection_merges_windows_and_reduces_requests():
    paragraphs = ["the rain fell on the empty road. " * 12 for _ in range(60)]
    paragraphs[5] = "Élise had green eyes."
    paragraphs[6] = "Alice had dark hair."
    paragraphs[50] = "Élise had blue eyes."
    text = "\n\n".join(paragraphs)
    occurrences = evidence.index_names([(text, "text")])[0]
    selected = evidence.selected_passages(text, 1000, occurrences)
    assert sum(len(p) for _, p in selected) < len(text) / 4
    assert len(list(evidence.batch_passages(selected, 1000))) < len(list(evidence.passages(text, 1000))) / 4
    assert all(text[offset:offset + len(p)] == p for offset, p in selected)
    assert any("green eyes" in p for _, p in selected)
    assert any("blue eyes" in p for _, p in selected)


def test_unnamed_scope_and_validation(workspace, monkeypatch):
    source, output = workspace
    path = source / "chapter.txt"
    path.write_text("The old man had grey eyes.", encoding="utf-8")
    chat = Mock(return_value={"evidence": []})
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    evidence.extract_chapter(path, output, None, "qwen3.5", settings(character_scope="proper_names"), 1)
    assert chat.call_count == 0
    evidence.extract_chapter(path, output, None, "qwen3.5", settings(character_scope="all_characters"), 1)
    assert chat.call_count == 1
    assert "Exclude locations" in chat.call_args.args[2]
    with pytest.raises(ValueError, match="character_scope"):
        node.CharacterEvidenceExtractorNode().run({}, {}, character_scope="invalid")
    assert node.CharacterEvidenceExtractorNode.INPUT_TYPES()["required"]["character_scope"][0] == list(evidence.SCOPES)


def test_disjoint_batch_offsets_cache_and_conflicts(workspace, monkeypatch):
    source, output = workspace
    path = source / "chapter.txt"
    green, blue = "Élise had green eyes.", "Élise had blue eyes."
    middle = "\n\n".join(["the rain fell." * 20] * 8)
    path.write_bytes((green + "\n\n" + middle + "\n\n" + blue).encode("utf-8"))

    def chat(*args):
        passages = json.loads(args[3])["passages"]
        return {"evidence": [observation(q, value=q) for q in (green, blue) if any(q in p for p in passages)]}

    mocked = Mock(side_effect=chat)
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", mocked)
    diagnostics = []
    args = settings(chunk_chars=1000)
    first = evidence.extract_chapter(path, output, None, "qwen3.5", args, 1, diagnostics=diagnostics)
    assert mocked.call_count == 1  # Both disjoint windows fit one request.
    assert len(first["evidence"]) == 2
    assert diagnostics[0]["selected_chars"] < len(first["source_text"])
    for item in first["evidence"]:
        loc = item["provenance"]["locations"][0]
        assert first["source_text"][loc["start_char"]:loc["end_char"]] == item["quotation"]
    # An edit outside the selected context shifts offsets without invalidating the batch.
    text = first["source_text"]
    cut = text.index("\n\n", len(green) + 3) + 2
    path.write_bytes((text[:cut] + "the extra rain.\n\n" + text[cut:]).encode("utf-8"))
    second = evidence.extract_chapter(path, output, None, "qwen3.5", args, 1, diagnostics=diagnostics)
    assert mocked.call_count == 1
    assert diagnostics[-1]["cache_hits"] == 1
    assert second["evidence"][1]["provenance"]["locations"][0]["start_char"] > first["evidence"][1]["provenance"]["locations"][0]["start_char"]


def test_node_cache_isolated_per_run_and_reused_within_run(workspace, monkeypatch):
    source, output = workspace
    (source / "chapter.txt").write_text("Élise attend.", encoding="utf-8")
    chat = Mock(return_value={"evidence": []})
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model", lambda *args: (nullcontext(), "qwen3.5"))
    for stamp, expected_calls, expected_hits in (
        ("20261024120000", 1, 0),
        ("20261024120000", 1, 1),
        ("20261024130000", 2, 0),
    ):
        node.CharacterEvidenceExtractorNode().run(
            {"api_url": "http://127.0.0.1:1234/v1", "run_folder": stamp}, {"chapter_paths": [str(source)]})
        assert chat.call_count == expected_calls
        assert len(list((output / stamp / ".cache/character_evidence").glob("*.json"))) == 1
        stats = json.loads((output / stamp / "character_evidence/character_evidence_diagnostics.json").read_text())
        assert stats["chapters"][0]["cache_hits"] == expected_hits
        assert stats["chapters"][0]["llm_calls"] == 1 - expected_hits
        assert stats["llm_requests"] == 0  # The mocked transport emits no request metrics.
    assert not (output / ".cache").exists()
    assert evidence.lmstudio_json.REQUEST_METRICS.get() is None


def test_transport_diagnostics_include_retries(monkeypatch):
    monkeypatch.setattr(evidence.lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    client, create = client_for(Stream(['{"evidence":']), Stream(['{"evidence": []}']))
    metrics = dict(llm_requests=0, estimated_input_tokens=0, estimated_output_tokens=0)
    token = evidence.lmstudio_json.REQUEST_METRICS.set(metrics)
    try:
        evidence.lmstudio_json.chat_json(client, "qwen3.5", evidence.SYSTEM, "Élise attend.", evidence.SCHEMA, 0.15, 8192)
    finally:
        evidence.lmstudio_json.REQUEST_METRICS.reset(token)
    assert metrics["llm_requests"] == create.call_count == 2
    assert metrics["estimated_input_tokens"] > 0
    assert metrics["estimated_output_tokens"] > 0


def test_completed_batch_survives_interruption(workspace, monkeypatch):
    source, output = workspace
    path = source / "chapter.txt"
    path.write_bytes(("Élise attend. " * 180).encode())

    class InterruptProcessingException(RuntimeError):
        pass

    chat = Mock(side_effect=[{"evidence": []}, InterruptProcessingException()])
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    args = settings(character_scope="all_characters")
    with pytest.raises(InterruptProcessingException):
        evidence.extract_chapter(path, output, None, "qwen3.5", args, 1)
    assert len(list((output / ".cache").glob("*.json"))) == 1
    chat.side_effect = None
    chat.return_value = {"evidence": []}
    diagnostics = []
    evidence.extract_chapter(path, output, None, "qwen3.5", args, 1, diagnostics=diagnostics)
    assert diagnostics[0]["cache_hits"] >= 1


def test_overlap_dedup_preserves_conflicting_observations(workspace, monkeypatch):
    source, output = workspace
    path = source / "chapter.txt"
    quote = "Élise had green eyes."
    path.write_bytes(("x" * 800 + quote + "x" * 900).encode())
    observations = [observation(quote, value="green"), observation(quote, value="blue", classification="inferred")]

    def chat(*args):
        return {"evidence": observations if quote in args[3] else []}

    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    result = evidence.extract_chapter(path, output, None, "qwen3.5", settings(character_scope="all_characters"), 1)
    assert len(result["evidence"]) == 2
    assert {item["value"] for item in result["evidence"]} == {"green", "blue"}
    assert all(len(item["provenance"]["locations"]) == 1 for item in result["evidence"])


def test_targeted_repair_preserves_facts_provenance_and_cache(workspace, monkeypatch):
    source, output = workspace
    path = source / "chapter.txt"
    quote = "Élise portait une robe rouge."
    path.write_text(quote + "\n" + observation()["quotation"], encoding="utf-8")
    invalid = observation("Elle portait une robe rouge.", attribute="clothing", value="robe rouge")
    chat = Mock(side_effect=[{"evidence": [invalid, observation()]}, {"corrections": [
        {"observation_index": 0, "quotation": quote, "persistence": "temporary"}]}])
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    result = evidence.extract_chapter(path, output, None, "qwen3.5", settings(), 1)
    assert len(result["evidence"]) == 2
    fixed, valid = result["evidence"]
    assert fixed["quotation"] == quote
    assert fixed["persistence"] == "temporary"
    assert fixed["extraction_correction"]["original_quotation"] == invalid["quotation"]
    assert all(fixed[k] == invalid[k] for k in evidence.FIELDS - {"quotation", "persistence"})
    assert all(valid[k] == observation()[k] for k in evidence.FIELDS)
    loc = fixed["provenance"]["locations"][0]
    assert result["source_text"][loc["start_char"]:loc["end_char"]] == quote
    payload = json.loads(chat.call_args_list[1].args[3])
    assert [item["observation_index"] for item in payload["observations"]] == [0]
    assert evidence.extract_chapter(path, output, None, "qwen3.5", settings(), 1) == result
    assert chat.call_count == 2
    report = evidence.review_report({"chapters": [result]})
    assert report["corrected_observations"] == 1
    assert report["observations_for_review"] == 0


def test_repair_never_joins_disjoint_passages_or_drops_missing_indices(monkeypatch):
    raw = {"evidence": [observation("bad one"), observation("bad two"), observation("bad three")]}
    correction = {"observation_index": 0, "quotation": "left\n\nright", "persistence": "persistent"}
    chat = Mock(return_value={"corrections": [correction,
        {"observation_index": 99, "quotation": "left", "persistence": "persistent"}]})
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    result = evidence.repair_observations(raw, ["left", "right"], None, "qwen", settings(), {"llm_calls": 0})
    assert [i["quotation"] for i in result] == [i["quotation"] for i in raw["evidence"]]
    assert all(i["warnings"] for i in result)
    assert len(json.loads(chat.call_args.args[3])["observations"]) == 3


def test_repair_duplicate_indices_and_empty_quote_remain_for_review(monkeypatch):
    correction = {"observation_index": 0, "quotation": "exact", "persistence": "persistent"}
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", Mock(return_value={"corrections": [
        correction, correction, {"observation_index": 1, "quotation": "", "persistence": "persistent"}]}))
    result = evidence.repair_observations({"evidence": [observation("bad"), observation("missing")]},
                                        ["exact"], None, "qwen", settings(), {"llm_calls": 0})
    assert [i["quotation"] for i in result] == ["bad", "missing"]
    assert all(i["warnings"] for i in result)


def test_repair_failure_does_not_block_but_cancellation_propagates(monkeypatch):
    stats = {"llm_calls": 0}
    chat = Mock(side_effect=APIConnectionError(request=httpx.Request("POST", "http://localhost/v1")))
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    raw = {"evidence": [observation("bad")]}
    result = evidence.repair_observations(raw, ["exact"], None, "qwen", settings(), stats)
    assert result[0]["quotation"] == "bad"
    assert stats["repair_failures"] == 1

    class InterruptProcessingException(RuntimeError):
        pass

    chat.side_effect = InterruptProcessingException()
    with pytest.raises(InterruptProcessingException):
        evidence.repair_observations(raw, ["exact"], None, "qwen", settings(), stats)


def test_repair_transport_schema_and_compact_retry_keep_all_pending(monkeypatch):
    monkeypatch.setattr(evidence.lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    corrections = {"corrections": [
        {"observation_index": i, "quotation": "exact", "persistence": "persistent"} for i in range(3)]}
    client, create = client_for(Stream(['{"corrections":']), Stream([json.dumps(corrections)]))
    raw = {"evidence": [observation(f"bad {i}") for i in range(3)]}
    result = evidence.repair_observations(raw, ["exact"], client, "qwen3.5", settings(), {"llm_calls": 0})
    assert len(result) == 3 and all(i["quotation"] == "exact" for i in result)
    assert create.call_count == 2
    for call in create.call_args_list:
        assert call.kwargs["response_format"] == {"type": "json_schema", "json_schema": evidence.REPAIR_SCHEMA}
        messages = json.dumps(call.kwargs["messages"])
        assert all(f"bad {i}" in messages for i in range(3))

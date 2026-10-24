"""Source fidelity, chapter isolation, caching and structured shared transport."""
import argparse
from contextlib import nullcontext
import json
from unittest.mock import Mock

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


@pytest.mark.parametrize("overrides", [
    {"quotation": "Elise had green eyes."}, {"classification": "fact"},
    {"confidence": True}, {"confidence": float("nan")}, {"confidence": 1.1},
    {"attribute": "clothing", "persistence": "persistent"},
    {"attribute": "accessory", "persistence": "unknown"}, {"rationale": ""},
])
def test_invalid_evidence_is_rejected(overrides):
    with pytest.raises(ValueError):
        evidence.validate_result({"evidence": [observation(**overrides)]}, "Élise avait les yeux verts.")


def test_quote_correction_and_actionable_failure(workspace, monkeypatch):
    source, output = workspace
    path = source / "chapter.txt"
    path.write_text("Élise avait les yeux verts.", encoding="utf-8")
    chat = Mock(side_effect=[{"evidence": [observation("translated")]},
                             {"evidence": [observation()]}])
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    evidence.extract_chapter(path, output, None, "qwen3.5", settings(), 1)
    assert "Previous response rejected" in chat.call_args.args[3]
    chat.side_effect = None
    chat.return_value = {"evidence": [observation("invented")]}
    with pytest.raises(RuntimeError, match="chapter.txt, passage 1, character offset 0.*Reduce chunk_chars"):
        evidence.extract_chapter(path, output, None, "qwen3.5", settings(force=True), 1)


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

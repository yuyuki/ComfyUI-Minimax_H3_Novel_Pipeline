"""Semantic review gates, independent from the model's narrative accuracy."""
import json
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import character_evidence as evidence
from minimax_h3_novel_pipeline.core import character_profiles, path_access
from tests.test_character_evidence import observation, settings
from tests.test_character_profiles import document
from tests.test_structured_json import Stream, client_for


def verdicts(*issues):
    return {"verdicts": [{"observation_index": i, "issues": issue} for i, issue in enumerate(issues)]}


@pytest.mark.parametrize("quote,value,issue", [
    ("Vous êtes séduisant, dit Élise à Paul.", "attractive", "Describes Paul, not the speaker."),
    ("Élise ne se trouvait pas au bar.", "present in bar", "Negation and nonphysical fact."),
    ("Élise était assise.", "sitting", "Pose is not persistent anatomy."),
    ("Élise portait une chemise blanche.", "blonde hair", "Quotation does not support hair color."),
])
def test_rejected_exact_quote_cannot_enter_consolidation(monkeypatch, quote, value, issue):
    chat = Mock(return_value=verdicts([issue]))
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    items = evidence.audit_observations([observation(quote, value=value)], [quote],
                                       None, "qwen", settings(), {"llm_calls": 0})
    assert items[0]["warnings"]
    records = character_profiles.prepare_evidence(document(*items))
    assert all(r["validation_confidence"] == "low" for r in records.values())
    assert evidence._batch_observations({"evidence": [
        {k: v for k, v in items[0].items() if k != "warnings"}]}, [quote])[0]["warnings"]


def test_audit_transport_retry_preserves_all_observations_and_schema():
    items = [observation(), observation(value="green eyes")]
    client, create = client_for(Stream([json.dumps(verdicts([]))]),
                                Stream([json.dumps(verdicts([], ["Unsupported value"]))]))
    result = evidence.audit_observations(items, [items[0]["quotation"]], client, "qwen3.5",
                                         settings(), {"llm_calls": 0})
    assert not result[0].get("warnings") and result[1]["warnings"]
    assert create.call_count == 2
    for call in create.call_args_list:
        assert call.kwargs["response_format"] == {"type": "json_schema", "json_schema": evidence.AUDIT_SCHEMA}
        payload = json.JSONDecoder().raw_decode(call.kwargs["messages"][-1]["content"])[0]
        assert len(payload["observations"]) == 2


def test_failed_audit_retains_every_observation_for_review(monkeypatch):
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", Mock(return_value=verdicts([], [])))
    result = evidence.audit_observations([observation()], ["source"], None, "qwen",
                                         settings(correction_attempts=0), {"llm_calls": 0})
    assert len(result) == 1
    assert result[0]["semantic_review"]["issues"] == ["Semantic audit unavailable or invalid."]


def test_repair_audit_and_cache_integration(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    quote = "Élise portait une chemise blanche."
    invalid = observation("invented", value="blonde hair")
    chat = Mock(side_effect=[{"evidence": [invalid]}, {"corrections": [
        {"observation_index": 0, "quotation": quote, "persistence": "persistent"}]},
        verdicts(["Shirt does not support hair color"])])
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    args = settings(character_scope="all_characters")
    path = tmp_path / "chapter.txt"
    result = evidence.extract_chapter(path, tmp_path, None, "qwen", args, 1,
                                      source=(quote, "utf8_decoded_text"))
    assert result["evidence"][0]["extraction_correction"]
    assert result["evidence"][0]["warnings"]
    assert evidence.extract_chapter(path, tmp_path, None, "qwen", args, 1,
                                    source=(quote, "utf8_decoded_text")) == result
    assert chat.call_count == 3
    assert next(iter(character_profiles.prepare_evidence({"format_version": 1,
        "location_convention": "offsets", "chapters": [result]}).values()))["validation_confidence"] == "low"


def test_audit_cancellation_propagates(monkeypatch):
    class InterruptProcessingException(RuntimeError):
        pass
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", Mock(side_effect=InterruptProcessingException()))
    with pytest.raises(InterruptProcessingException):
        evidence.audit_observations([observation()], ["source"], None, "qwen", settings(), {"llm_calls": 0})


def test_audit_compact_transport_retry_keeps_all_observations(monkeypatch):
    monkeypatch.setattr(evidence.lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    client, create = client_for(Stream(['{"verdicts":']), Stream([json.dumps(verdicts([], [], []))]))
    items = [observation(value=str(i)) for i in range(3)]
    result = evidence.audit_observations(items, ["source"], client, "qwen3.5", settings(), {"llm_calls": 0})
    assert len(result) == 3 and not any(i.get("warnings") for i in result)
    assert create.call_count == 2
    for call in create.call_args_list:
        assert call.kwargs["response_format"] == {"type": "json_schema", "json_schema": evidence.AUDIT_SCHEMA}
        assert all(f'"observation_index": {i}' in call.kwargs["messages"][-1]["content"] for i in range(3))

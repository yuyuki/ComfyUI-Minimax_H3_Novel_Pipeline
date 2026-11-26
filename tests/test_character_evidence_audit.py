"""Semantic review gates, independent from the model's narrative accuracy."""
import json
from unittest.mock import Mock

import pytest

from tests.passages_helpers import with_passage_ids

from minimax_h3_novel_pipeline.core import character_evidence as evidence
from minimax_h3_novel_pipeline.core import path_access
from tests.test_character_evidence import observation, settings
from tests.test_structured_json import Stream, client_for


def verdicts(*issues):
    return {"verdicts": [{"observation_index": i, "verdict": "invalid" if issue else "valid",
                          "issue": "unsupported" if issue else "none"} for i, issue in enumerate(issues)]}


def keyed_verdicts(*codes):
    return {"verdicts": {str(i): code for i, code in enumerate(codes)}}


@pytest.mark.parametrize("quote,value,issue", [
    ("Vous êtes séduisant, dit Élise à Paul.", "attractive", "Describes Paul, not the speaker."),
    ("Élise ne se trouvait pas au bar.", "present in bar", "Negation and nonphysical fact."),
    ("Élise était assise.", "sitting", "Pose is not persistent anatomy."),
    ("Élise portait une chemise blanche.", "blonde hair", "Quotation does not support hair color."),
])
def test_semantic_rejection_is_distinct_from_source_integrity(monkeypatch, quote, value, issue):
    chat = Mock(return_value=verdicts([issue]))
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", with_passage_ids(chat))
    items = evidence.audit_observations([observation(quote, value=value)], [quote],
                                       None, "qwen", settings(), {"llm_calls": 0})
    assert items[0]["semantic_review"]["verdict"] == "invalid"
    assert not items[0].get("warnings")


def test_audit_transport_retry_preserves_all_observations_and_schema():
    items = [observation(), observation(value="green eyes")]
    client, create = client_for(Stream([json.dumps(verdicts([]))]),
                                Stream([json.dumps(verdicts([], ["Unsupported value"]))]))
    result = evidence.audit_observations(items, [items[0]["quotation"]], client, "qwen3.5",
                                         settings(), {"llm_calls": 0})
    assert [row["semantic_review"]["verdict"] for row in result] == ["valid", "invalid"]
    assert create.call_count == 2
    for call in create.call_args_list:
        assert call.kwargs["response_format"] == {"type": "json_schema", "json_schema": evidence.AUDIT_SCHEMA}
        payload = json.JSONDecoder().raw_decode(call.kwargs["messages"][-1]["content"])[0]
        assert len(payload["observations"]) == 2


def test_failed_audit_retains_every_observation_for_review(monkeypatch):
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", with_passage_ids(Mock(return_value=verdicts([], []))))
    result = evidence.audit_observations([observation()], ["source"], None, "qwen",
                                         settings(correction_attempts=0), {"llm_calls": 0})
    assert len(result) == 1
    assert result[0]["semantic_review"]["issues"] == ["Semantic audit unavailable or invalid."]


def test_neutral_review_trims_only_literal_values_and_retries_unresolved_rows(monkeypatch):
    mixed = observation("Élise avait les cheveux noirs en désordre.", value="cheveux noirs en désordre")
    scar = observation("Elle avait une cicatrice au menton.", value="cicatrice au menton")
    dirt = observation("Elle avait de la boue sur les mains.", value="mains boueuses")
    injury = observation("Elle s'écorcha le bras.", value="bras écorché", persistence="temporary")
    identity = observation("Élise, aussi appelée Lise.", value="Lise", attribute="identity", persistence="unknown")
    chapters = [{"chapter": 1, "evidence": [mixed, scar, dirt, injury, identity]}]
    chat = Mock(side_effect=[
        {"values": {"1": "cheveux noirs", "2": "cicatrice profonde", "3": ""}},
        {"values": {"2": "cicatrice au menton"}},
    ])
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", chat)
    findings = evidence.review_neutral_appearance(chapters, None, "gemma-4", settings())
    assert [row["value"] for row in chapters[0]["evidence"]] == ["cheveux noirs", "cicatrice au menton", "Lise"]
    assert [row["status"] for row in findings] == ["excluded", "trimmed", "excluded"]
    assert mixed["quotation"] == "Élise avait les cheveux noirs en désordre."
    for call, keys in zip(chat.call_args_list, (["1", "2", "3"], ["2"])):
        schema = call.args[4]
        assert schema["strict"] and schema["name"] == "character_neutral_appearance"
        assert schema["schema"]["properties"]["values"]["required"] == keys
        assert list(json.loads(call.args[3])) == keys


def test_gemma_audit_constrains_coverage_and_corrects_invalid_verdicts():
    invalid = keyed_verdicts("valid:none")
    corrected = keyed_verdicts("valid:none", "invalid:set_persistent")
    client, create = client_for(Stream([json.dumps(invalid)]), Stream([json.dumps(corrected)]))
    scar = observation("Élise avait une cicatrice au menton.", value="cicatrice au menton",
                       persistence="temporary")
    items = evidence.audit_observations([observation(), scar], ["source"], client, "google/gemma-3-12b-it",
                                        settings(), {"llm_calls": 0})
    assert all(row["semantic_review"]["verdict"] == "valid" for row in items)
    assert items[1]["persistence"] == "persistent"
    assert not items[0].get("warnings")
    assert create.call_count == 2
    for call in create.call_args_list:
        response = call.kwargs["response_format"]
        assert response["type"] == "json_schema"
        rows = response["json_schema"]["schema"]["properties"]["verdicts"]
        assert rows["required"] == ["0", "1"]
        assert not rows["additionalProperties"]
        assert '"observation_index":0' in call.kwargs["messages"][-1]["content"]
    assert create.call_args_list[0].kwargs["max_tokens"] == 1024
    assert create.call_args_list[1].kwargs["max_tokens"] == settings().max_tokens
    assert "Validation error:" in create.call_args_list[1].kwargs["messages"][-1]["content"]
    assert "maxItems" not in evidence.AUDIT_SCHEMA["schema"]["properties"]["verdicts"]


def test_exact_unsupported_observation_is_rechecked_and_stays_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    quote = "Élise portait une chemise blanche."
    invalid = observation(quote, value="blonde hair")
    chat = Mock(side_effect=[{"evidence": [invalid]}, verdicts(["Shirt does not support hair color"])] * 2)
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", with_passage_ids(chat))
    args = settings(character_scope="all_characters")
    diagnostics = []
    result = evidence.extract_chapter(tmp_path / "chapter.txt", tmp_path, None, "qwen", args, 1,
                                      source=(quote, "normalized_utf8_text"), diagnostics=diagnostics)
    assert result["evidence"] == []
    assert diagnostics[0]["observations_for_review"][0]["status"] == "excluded"
    assert diagnostics[0]["empty_reason"]
    assert diagnostics[0]["status"] == "incomplete"
    assert diagnostics[0]["semantic_support_checks"][0]["status"] == "incomplete"
    assert not list((tmp_path / ".cache").glob("*.json"))
    assert chat.call_count == 4



def test_audit_cancellation_propagates(monkeypatch):
    class InterruptProcessingException(RuntimeError):
        pass
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", with_passage_ids(Mock(side_effect=InterruptProcessingException())))
    with pytest.raises(InterruptProcessingException):
        evidence.audit_observations([observation()], ["source"], None, "qwen", settings(), {"llm_calls": 0})
    assert evidence.lmstudio_json.REQUEST_THINKING.get() is None


def test_audit_compact_transport_retry_keeps_all_observations(monkeypatch):
    monkeypatch.setattr(evidence.lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    client, create = client_for(Stream(['{"verdicts":']), Stream([json.dumps(verdicts([], [], []))]))
    items = [observation(value=str(i)) for i in range(3)]
    result = evidence.audit_observations(items, ["source"], client, "qwen3.5", settings(), {"llm_calls": 0})
    assert len(result) == 3 and not any(i.get("warnings") for i in result)
    assert create.call_count == 2
    for call in create.call_args_list:
        assert call.kwargs["response_format"] == {"type": "json_schema", "json_schema": evidence.AUDIT_SCHEMA}
        assert all(f'"observation_index":{i}' in call.kwargs["messages"][-1]["content"] for i in range(3))


@pytest.mark.parametrize("status,issue", [("valid", "unsupported"), ("invalid", "none"),
                                         ("valid", "This observation passes all checks.")])
def test_audit_rejects_inconsistent_verdicts(monkeypatch, status, issue):
    response = {"verdicts": [{"observation_index": 0, "verdict": status, "issue": issue}]}
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", with_passage_ids(Mock(return_value=response)))
    item = evidence.audit_observations([observation()], ["source"], None, "qwen",
                                       settings(correction_attempts=0), {"llm_calls": 0})[0]
    assert item["semantic_review"]["verdict"] == "uncertain"
    assert item["semantic_review"]["issue"] == "unavailable"


def test_audit_thinking_override_is_scoped_and_restored(monkeypatch):
    monkeypatch.setattr(evidence.lmstudio_json, "THINKING_ENABLED", True)
    def chat(client, model, system, prompt, schema, temperature, budget):
        assert evidence.lmstudio_json.model_settings(client, model)[1]["thinking"] is False
        assert budget == 512
        return verdicts([])
    monkeypatch.setattr(evidence.lmstudio_json, "chat_json", with_passage_ids(chat))
    evidence.audit_observations([observation()], ["source"], None, "qwen", settings(), {"llm_calls": 0})
    assert evidence.lmstudio_json.model_settings(None, "qwen")[1]["thinking"] is True
    assert evidence.lmstudio_json.REQUEST_THINKING.get() is None


@pytest.mark.parametrize("audit_thinking", [False, True])
@pytest.mark.parametrize("family,model", [("Qwen", "qwen3.5"), ("Gemma", "gemma-4"), ("Mistral", "mistral-small-4")])
def test_audit_transport_uses_separate_thinking_and_restores_client(audit_thinking, family, model):
    from minimax_h3_novel_pipeline.core import lmstudio_models
    response = keyed_verdicts("valid:none") if family == "Gemma" else verdicts([])
    client, create = client_for(Stream([json.dumps(response)]), Stream(['{"evidence": []}']))
    client._minimax_h3_profile = lmstudio_models.get_profile(family)
    client._minimax_h3_settings = client._minimax_h3_profile.settings_from_config({"thinking": True})
    args = settings(audit_thinking=audit_thinking)
    evidence.audit_observations([observation()], [observation()["quotation"]], client, model, args,
                                {"llm_calls": 0})
    evidence.lmstudio_json.chat_json(client, model, evidence.SYSTEM, "source", evidence.SCHEMA, 0.15, 8192)
    if family == "Mistral":
        assert [call.kwargs["extra_body"]["reasoning_effort"] for call in create.call_args_list] == [
            "high" if audit_thinking else "none", "high"]
    else:
        assert [call.kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"]
                for call in create.call_args_list] == [audit_thinking, True]
    assert client._minimax_h3_settings["thinking"] is True

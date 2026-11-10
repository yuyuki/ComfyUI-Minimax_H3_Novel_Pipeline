"""Node thinking controls agree with saved settings, requests and stream logs."""
import argparse
from types import SimpleNamespace

import pytest

from minimax_h3_novel_pipeline.core import configuration_snapshot, lmstudio_json, lmstudio_models, util
from tests.test_structured_json import SCHEMA, Stream, client_for


@pytest.mark.parametrize("family,model", [
    ("Qwen", "qwen3.5-9b"),
    ("Mistral", "mistral-small-4"),
    ("Gemma", "gemma-4-31b-it"),
])
@pytest.mark.parametrize("thinking", [True, False])
def test_thinking_matches_snapshot_requests_and_logs(tmp_path, monkeypatch, capsys, family, model, thinking):
    monkeypatch.setattr(util, "save_json", lambda path, data: None)
    config = {"model_family": family, "thinking": thinking,
              "api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261106120000"}
    profile = lmstudio_models.get_profile(family)
    settings = profile.settings_from_config(config)
    _, snapshot = configuration_snapshot.start(
        tmp_path, "test", config, model, argparse.Namespace(thinking=thinking),
        out_dir="test", inputs={})
    assert snapshot["node_settings"]["thinking"] is thinking
    assert snapshot["model_controls"]["settings"]["thinking"] is thinking
    reasoning = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None, reasoning_content="Planning"))])
    pieces = [reasoning] if thinking else []
    client, create = client_for(Stream([*pieces, "{"]), Stream([*pieces, '{"value":"ok"}']))
    client._minimax_h3_profile = profile
    client._minimax_h3_settings = settings
    assert lmstudio_json.chat_json(client, model, "system", "user", SCHEMA, 0.12, 8192) == {"value": "ok"}
    assert create.call_count == 2
    for call in create.call_args_list:
        request = call.kwargs
        assert request["response_format"] == {"type": "json_schema", "json_schema": SCHEMA}
        if family == "Mistral":
            effort = "high" if thinking else "none"
            assert request["extra_body"] == {"reasoning_effort": effort,
                                             "chat_template_kwargs": {"reasoning_effort": effort}}
        else:
            assert request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is thinking
        assert request["extra_body"] == snapshot["model_controls"]["request_extra_body"]
    log = capsys.readouterr().out
    assert log.count(f"thinking={thinking}") == 2
    assert log.count(f"reasoning_chars={8 if thinking else 0}") == 2


@pytest.mark.parametrize("family", ["Gemma", "Mistral"])
def test_disabled_thinking_rejects_unwanted_reasoning(family, capsys):
    stream = Stream([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None, reasoning_content="Still reasoning"))]), '{"value":"ok"}'])
    client, create = client_for(stream)
    client._minimax_h3_profile = lmstudio_models.get_profile(family)
    client._minimax_h3_settings = {"thinking": False}
    with pytest.raises(RuntimeError, match="returned reasoning with thinking disabled"):
        lmstudio_json.chat_json(client, "model-alias", "system", "user", SCHEMA, 0.12, 8192)
    assert stream.closed
    assert create.call_count == 1
    assert "reasoning_chars=15" in capsys.readouterr().out


@pytest.mark.parametrize("family", ["Gemma", "Mistral"])
@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_disabled_thinking_accepts_empty_thought_block(family, field, capsys):
    pieces = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None, **{field: whitespace}))]) for whitespace in ("\n", " ", "\t")]
    stream = Stream([*pieces, '{"value":"ok"}'])
    client, create = client_for(stream)
    client._minimax_h3_profile = lmstudio_models.get_profile(family)
    client._minimax_h3_settings = {"thinking": False}
    assert lmstudio_json.chat_json(client, "model-alias", "system", "user", SCHEMA, .12, 6500) == {"value": "ok"}
    assert stream.closed and create.call_count == 1
    assert create.call_args.kwargs["response_format"] == {"type": "json_schema", "json_schema": SCHEMA}
    log = capsys.readouterr().out
    assert "reasoning_chars=3" in log
    assert "reasoning_non_whitespace_chars=0" in log


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_real_reasoning_after_empty_block_still_fails(field, capsys):
    pieces = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None, **{field: text}))]) for text in ("\n", " ", "Private reasoning")]
    stream = Stream([*pieces, '{"value":"ok"}'])
    client, create = client_for(stream)
    client._minimax_h3_profile = lmstudio_models.get_profile("Gemma")
    client._minimax_h3_settings = {"thinking": False}
    with pytest.raises(RuntimeError, match="returned reasoning with thinking disabled"):
        lmstudio_json.chat_json(client, "gemma", "system", "user", SCHEMA, .12, 6500)
    assert stream.closed and stream.consumed == 3 and create.call_count == 1
    log = capsys.readouterr().out
    assert "reasoning_non_whitespace_chars=16" in log
    assert "Private reasoning" not in log


def test_gpt_oss_logs_native_reasoning_even_during_disabled_audit(capsys):
    client, _ = client_for(Stream(['{"value":"ok"}']))
    token = lmstudio_json.REQUEST_THINKING.set(False)
    try:
        lmstudio_json.chat_json(client, "gpt-oss-20b", "system", "user", SCHEMA, 0.12, 8192)
    finally:
        lmstudio_json.REQUEST_THINKING.reset(token)
    assert "thinking=True" in capsys.readouterr().out


def test_hosted_openai_requests_have_no_local_template_controls():
    from minimax_h3_novel_pipeline.core import lmstudio_model_openai

    client, create = client_for(Stream(['{"value":"ok"}']))
    client._minimax_h3_profile = lmstudio_model_openai
    client._minimax_h3_settings = lmstudio_model_openai.settings_from_config({"thinking": True})
    token = lmstudio_json.REQUEST_THINKING.set(True)
    try:
        lmstudio_json.chat_json(client, "gpt-4.1", "system", "user", SCHEMA, 0.12, 8192)
    finally:
        lmstudio_json.REQUEST_THINKING.reset(token)
    assert create.call_args.kwargs["extra_body"] == {}
    assert create.call_args.kwargs["response_format"]["type"] == "json_schema"

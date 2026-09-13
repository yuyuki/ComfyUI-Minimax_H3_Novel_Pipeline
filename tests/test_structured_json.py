"""The current backend always streams schema-constrained JSON."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import APIError, OpenAI

from minimax_h3_novel_pipeline import lmstudio_json, lmstudio_pipeline, util


SCHEMA = {"name": "test", "strict": True, "schema": {
    "type": "object", "properties": {"value": {"type": "string"}},
    "required": ["value"], "additionalProperties": False,
}}


class Stream:
    def __init__(self, pieces):
        self.pieces = pieces
        self.closed = False
        self.consumed = 0

    def __iter__(self):
        for piece in self.pieces:
            self.consumed += 1
            yield piece if not isinstance(piece, str) else SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])

    def close(self):
        self.closed = True


def client_for(*streams):
    create = Mock(side_effect=streams)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), create


@pytest.mark.parametrize("schema_name", ["CHUNK_SCHEMA", "MERGE_SCHEMA"])
def test_compact_retry_preserves_visual_detail_and_entity_capacity(schema_name):
    step = lmstudio_pipeline.load("extract")
    schema = getattr(step, schema_name)
    original = copy.deepcopy(schema)
    compact = lmstudio_json._qwen35_compact_schema(schema)
    props = compact["schema"]["properties"]
    for kind in ("characters", "locations", "objects"):
        assert props[kind].get("maxItems") == schema["schema"]["properties"][kind].get("maxItems")
        fields = props[kind]["items"]["properties"]
        assert fields["stable_visual_description"]["maxLength"] == 500
        state = "chapter_appearance" if kind == "characters" else "chapter_state"
        assert fields[state]["maxLength"] == 350
        assert fields["distinguishing_features"]["maxItems"] == 6
        assert fields["distinguishing_features"]["items"]["maxLength"] == 120
        assert fields["evidence"]["maxItems"] == 2
        assert fields["evidence"]["items"]["maxLength"] == 120
        assert fields["reference_view_hints"] == schema["schema"]["properties"][kind]["items"]["properties"]["reference_view_hints"]
    summary = "chunk_summary" if schema_name == "CHUNK_SCHEMA" else "chapter_summary"
    assert props[summary]["maxLength"] == (240 if summary == "chunk_summary" else 400)
    assert schema == original


def test_compact_retry_leaves_unrelated_fields_and_stricter_limits_intact():
    schema = copy.deepcopy(SCHEMA)
    schema["schema"]["properties"] = {
        "generation_prompt": {"type": "string", "maxLength": 300},
        "canonical_name": {"type": "string", "maxLength": 20},
        "description": {"type": "string"},
    }
    assert lmstudio_json._qwen35_compact_schema(schema) == schema


def test_merge_retry_can_return_more_than_three_entities(monkeypatch):
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    step = lmstudio_pipeline.load("extract")
    result = {"chapter_summary": "A gathering.", "characters": [
        {"canonical_name": f"Person {i}"} for i in range(8)
    ], "locations": [], "objects": []}
    client, create = client_for(Stream(['{"characters":']), Stream([json.dumps(result)]))
    assert lmstudio_json.chat_json(client, "qwen3.5", "system", "user", step.MERGE_SCHEMA, 0.2, 8192) == result
    retry = create.call_args.kwargs["response_format"]["json_schema"]
    assert "maxItems" not in retry["schema"]["properties"]["characters"]


@pytest.mark.parametrize("schema_name", ["CHUNK_SCHEMA", "MERGE_SCHEMA"])
@pytest.mark.parametrize("kind", ["characters", "locations", "objects"])
@pytest.mark.parametrize("invalid", [{}, {"canonical_name": ""}, {"canonical_name": " \t\n"}, {"canonical_name": None}])
def test_blank_name_requests_correction(monkeypatch, schema_name, kind, invalid):
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    step = lmstudio_pipeline.load("extract")
    schema = getattr(step, schema_name)
    valid = {kind: [{"canonical_name": "la gardienne"}]}
    first = Stream([json.dumps({kind: [invalid]})])
    second = Stream([json.dumps(valid)])
    client, create = client_for(first, second)
    assert lmstudio_json.chat_json(client, "qwen3.5", "system", "passage", schema, 0.2, 8192) == valid
    request = create.call_args.kwargs
    assert "missing or blank canonical_name" in request["messages"][-1]["content"]
    fields = request["response_format"]["json_schema"]["schema"]["properties"][kind]["items"]["properties"]
    assert fields["canonical_name"]["minLength"] == 1
    assert first.closed and second.closed


def test_blank_name_fails_after_retry_budget(monkeypatch):
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    step = lmstudio_pipeline.load("extract")
    raw = json.dumps({"characters": [{"canonical_name": " "}]})
    client, create = client_for(Stream([raw]), Stream([raw]))
    with pytest.raises(RuntimeError, match="after 2 attempt"):
        lmstudio_json.chat_json(client, "qwen3.5", "system", "passage", step.CHUNK_SCHEMA, 0.2, 8192)
    assert create.call_count == 2


def test_evidence_cleanup_preserves_original_excerpt():
    step = lmstudio_pipeline.load("extract")
    excerpt = "« La gardienne\n  portait une cape rouge. »"
    entity = step.clean_entity({"canonical_name": "la gardienne", "evidence": [excerpt, excerpt]}, "characters")
    assert entity["evidence"] == [excerpt]


@pytest.mark.parametrize("profiled", [False, True])
def test_cache_fingerprint_tracks_compact_policy(monkeypatch, profiled):
    from minimax_h3_novel_pipeline import prompt_cache

    client = SimpleNamespace()
    if profiled:
        client._minimax_h3_profile = SimpleNamespace(NAME="Test")
        client._minimax_h3_settings = {"thinking": False}
    args = SimpleNamespace(max_tokens=8192)
    before = prompt_cache.fingerprint("model", args, SCHEMA, client=client)
    monkeypatch.setattr(lmstudio_json, "COMPACT_SCHEMA_VERSION", lmstudio_json.COMPACT_SCHEMA_VERSION + 1)
    assert prompt_cache.fingerprint("model", args, SCHEMA, client=client) != before


def test_stream_stops_at_complete_json():
    stream = Stream(['{"value":', '"ok"}', 'unneeded'])
    client, create = client_for(stream)
    assert lmstudio_json.chat_json(client, "model", "system", "user", SCHEMA, 0.2, 200) == {"value": "ok"}
    assert stream.closed and stream.consumed == 2
    assert create.call_args.kwargs["response_format"]["type"] == "json_schema"
    assert create.call_args.kwargs["stream"] is True


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("model", ["qwen3.5-9b-uncensored-hauhaucs-aggressive@q6_k", "other-model"])
def test_thinking_control_and_qwen_prefill_survive_retries(monkeypatch, thinking, model):
    monkeypatch.setattr(lmstudio_json, "THINKING_ENABLED", thinking)
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    first, second = Stream(['{"value":']), Stream(['{"value":"ok"}'])
    client, create = client_for(first, second)
    assert lmstudio_json.chat_json(client, model, "system", "user", SCHEMA, 0.2, 200) == {"value": "ok"}
    assert first.closed and second.closed
    assert create.call_count == 2
    for call in create.call_args_list:
        request = call.kwargs
        assert request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is thinking
        messages = request["messages"]
        assert messages[0]["content"].endswith("\n\nsystem")
        assert messages[1]["content"].startswith("user")
        if model.startswith("qwen") and not thinking:
            assert messages[2:] == [{"role": "assistant", "content": "<think>\n\n</think>\n\n"}]
        else:
            assert len(messages) == 2
        assert request["response_format"]["type"] == "json_schema"
        assert request["max_tokens"] == 200


def test_qwen_retries_invalid_output_with_schema_and_closes_streams(monkeypatch):
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    first, second = Stream(['{"value":']), Stream(['{"value":"ok"}'])
    client, create = client_for(first, second)
    assert lmstudio_json.chat_json(client, "qwen3.5", "system", "user", SCHEMA, 0.2, 200) == {"value": "ok"}
    assert first.closed and second.closed
    assert create.call_count == 2
    assert all(c.kwargs["response_format"]["type"] == "json_schema" for c in create.call_args_list)


def test_cancellation_closes_stream_without_retry(monkeypatch):
    class InterruptProcessingException(Exception):
        pass
    check = Mock(side_effect=[None, InterruptProcessingException()])
    monkeypatch.setattr(lmstudio_pipeline, "comfy_interrupt_check", check)
    stream = Stream(['{"value":"ok"}'])
    client, create = client_for(stream)
    with pytest.raises(InterruptProcessingException):
        lmstudio_json.chat_json(client, "qwen3.5", "system", "user", SCHEMA, 0.2, 200)
    assert stream.closed and create.call_count == 1


def test_unsupported_structured_output_propagates_without_fallback():
    client, create = client_for(RuntimeError("unsupported schema"))
    with pytest.raises(RuntimeError, match="unsupported schema"):
        lmstudio_json.chat_json(client, "qwen3.5", "system", "user", SCHEMA, 0.2, 200)
    assert create.call_count == 1


@pytest.mark.parametrize("schema", [util.CHAPTER_SCHEMA, util.REGISTRY_SCHEMA])
def test_old_schemas_are_rejected(schema):
    with pytest.raises(ValueError, match="regenerate"):
        util.require_schema({"schema_version": schema.replace(".v3", ".v2")}, schema)
    util.require_schema({"schema_version": schema}, schema)


@pytest.mark.parametrize("model", ["qwen3.5-9b", "other-model"])
def test_node_token_budget_is_used_for_every_attempt(monkeypatch, model):
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    client, create = client_for(Stream([""]), Stream(['{"value":"ok"}']))
    lmstudio_json.chat_json(client, model, "system", "user", SCHEMA, 0.2, 8000)
    assert [call.kwargs["max_tokens"] for call in create.call_args_list] == [8000, 8000]


def test_reasoning_only_length_failure_has_safe_diagnostics(monkeypatch, capsys):
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 0)
    event = SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=None, reasoning_content="secret", reasoning="private"),
        finish_reason="length",
    )])
    stream = Stream([SimpleNamespace(choices=[]), event])
    client, _ = client_for(stream)
    with pytest.raises(RuntimeError) as error:
        lmstudio_json.chat_json(client, "qwen3.5", "system", "user", SCHEMA, 0.2, 8000)
    output = capsys.readouterr().out + str(error.value)
    for expected in ("content_chars=0", "reasoning_chars=13", "finish_reason=length", "max_tokens=8000"):
        assert expected in output
    assert "secret" not in output and "private" not in output
    assert stream.closed


def test_complete_json_reports_client_stop_without_consuming_finish(capsys):
    client, _ = client_for(Stream(['{"value":"private"}', "unused"]))
    lmstudio_json.chat_json(client, "model", "system", "user", SCHEMA, 0.2, 500)
    output = capsys.readouterr().out
    assert "content_chars=19" in output
    assert "finish_reason=not_received" in output
    assert "local_stop=json_complete" in output
    assert "private" not in output


GRAMMAR_ERROR = (
    "Failed to initialize samplers: Unexpected empty grammar stack "
    "after accepting piece: <think> (248068)"
)


@pytest.mark.parametrize("streamed_error", [False, True])
def test_thinking_grammar_failure_uses_schema_constrained_chatml(monkeypatch, streamed_error):
    import json

    monkeypatch.setattr(lmstudio_json, "THINKING_ENABLED", False)
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 1)
    requests = []
    responses = []

    def handle(request):
        requests.append((request.url.path, json.loads(request.content)))
        if len(requests) == 1:
            error = {"error": {"message": GRAMMAR_ERROR, "type": "invalid_request_error", "code": 400}}
            response = (
                httpx.Response(200, text="data: " + json.dumps(error) + "\n\n",
                               headers={"content-type": "text/event-stream"})
                if streamed_error else httpx.Response(400, json=error)
            )
        else:
            # Malformed raw output must compact-retry using the same backend.
            content = '{"value":' if len(requests) == 2 else '{"value":"ok"}'
            event = {"id": "test", "object": "text_completion", "created": 0,
                     "model": "qwen3.5", "choices": [{"index": 0, "text": content, "finish_reason": "length"}]}
            response = httpx.Response(200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n",
                                      headers={"content-type": "text/event-stream"})
        responses.append(response)
        return response

    with OpenAI(api_key="test", base_url="http://localhost:1234/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        assert lmstudio_json.chat_json(client, "qwen3.5", "system", "user", SCHEMA, 0.2, 200) == {"value": "ok"}
        assert lmstudio_json.chat_json(client, "qwen3.5", "system", "next user", SCHEMA, 0.2, 200) == {"value": "ok"}
    assert [path for path, _ in requests] == ["/v1/chat/completions", "/v1/completions", "/v1/completions", "/v1/completions"]
    assert "next user" in requests[-1][1]["prompt"]
    for _, body in requests:
        assert body["response_format"]["type"] == "json_schema"
        assert body["max_tokens"] == 200 and body["stream"] is True
    prompt = requests[1][1]["prompt"]
    assert "<|im_start|>system\n/no_think\n\nsystem<|im_end|>" in prompt
    assert "<|im_start|>user\nuser<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert all(response.is_closed for response in responses)


@pytest.mark.parametrize("model,thinking,message", [
    ("other-model", False, GRAMMAR_ERROR),
    ("qwen3.5", True, GRAMMAR_ERROR),
    ("qwen3.5", False, "Connection failed"),
])
def test_unrelated_api_failures_do_not_switch_backend(monkeypatch, model, thinking, message):
    monkeypatch.setattr(lmstudio_json, "THINKING_ENABLED", thinking)
    error = APIError(message, httpx.Request("POST", "http://localhost/v1/chat/completions"), body=None)
    client, create = client_for(error)
    with pytest.raises(APIError):
        lmstudio_json.chat_json(client, model, "system", "user", SCHEMA, 0.2, 200)
    assert create.call_count == 1


def test_chatml_fallback_is_bounded_and_does_not_need_length_retries(monkeypatch):
    monkeypatch.setattr(lmstudio_json, "THINKING_ENABLED", False)
    monkeypatch.setattr(lmstudio_json, "QWEN35_LENGTH_RETRIES", 0)
    error = APIError(GRAMMAR_ERROR, httpx.Request("POST", "http://localhost/v1/chat/completions"), body=None)
    client, create = client_for(error)
    fallback = Mock(side_effect=error)
    client.completions = SimpleNamespace(create=fallback)
    with pytest.raises(APIError):
        lmstudio_json.chat_json(client, "qwen3.5", "system", "user", SCHEMA, 0.2, 200)
    assert create.call_count == fallback.call_count == 1
    assert not vars(client).get("_minimax_h3_chatml_models")


@pytest.mark.parametrize("change", ["model", "endpoint", "client", "thinking"])
def test_successful_chatml_preference_is_scoped(monkeypatch, change):
    monkeypatch.setattr(lmstudio_json, "THINKING_ENABLED", False)
    error = APIError(GRAMMAR_ERROR, httpx.Request("POST", "http://localhost/v1/chat/completions"), body=None)
    client, create = client_for(error, Stream(['{"value":"chat"}']))
    client.base_url = "http://localhost/v1"
    client.completions = SimpleNamespace(create=Mock(return_value=Stream([
        SimpleNamespace(choices=[SimpleNamespace(text='{"value":"raw"}')]),
    ])))
    assert lmstudio_json.chat_json(client, "qwen3.5", "system", "user", SCHEMA, 0.2, 200) == {"value": "raw"}
    model = "qwen3.5"
    if change == "model":
        model = "qwen3.5-other"
    elif change == "endpoint":
        client.base_url = "http://localhost:1235/v1"
    elif change == "client":
        client, create = client_for(Stream(['{"value":"chat"}']))
        client.base_url = "http://localhost/v1"
    else:
        monkeypatch.setattr(lmstudio_json, "THINKING_ENABLED", True)
    assert lmstudio_json.chat_json(client, model, "system", "user", SCHEMA, 0.2, 200) == {"value": "chat"}
    assert create.call_count == (1 if change == "client" else 2)


def test_cancellation_closes_chatml_stream_without_retry(monkeypatch):
    class InterruptProcessingException(Exception):
        pass

    monkeypatch.setattr(lmstudio_json, "THINKING_ENABLED", False)
    monkeypatch.setattr(lmstudio_pipeline, "comfy_interrupt_check",
                        Mock(side_effect=[None, None, InterruptProcessingException()]))
    error = APIError(GRAMMAR_ERROR, httpx.Request("POST", "http://localhost/v1/chat/completions"), body=None)
    client, create = client_for(error)
    stream = Stream([SimpleNamespace(choices=[SimpleNamespace(text='{"value":"ok"}')])])
    fallback = Mock(return_value=stream)
    client.completions = SimpleNamespace(create=fallback)
    with pytest.raises(InterruptProcessingException):
        lmstudio_json.chat_json(client, "qwen3.5", "system", "user", SCHEMA, 0.2, 200)
    assert stream.closed
    assert create.call_count == fallback.call_count == 1


@pytest.mark.parametrize("thinking", [False, True])
def test_mistral_retries_without_qwen_options(monkeypatch, thinking):
    monkeypatch.setattr(lmstudio_json, "THINKING_ENABLED", thinking)
    first, second = Stream(['{"value":']), Stream(['{"value":"ok"}'])
    client, create = client_for(first, second)
    result = lmstudio_json.chat_json(client, "Mistral-Small-3.2-24B-Instruct-Q4_K_M", "system", "user", SCHEMA, 0.15, 8192)
    assert result == {"value": "ok"}
    assert first.closed and second.closed
    assert create.call_count == 2
    for call in create.call_args_list:
        request = call.kwargs
        assert request["extra_body"] == {}
        assert request["messages"][0] == {"role": "system", "content": "system"}
        assert len(request["messages"]) == 2
        assert request["top_p"] == 0.9
        assert request["max_tokens"] == 8192
        assert request["response_format"]["type"] == "json_schema"


def test_mistral_does_not_use_qwen_grammar_fallback():
    error = APIError("Failed to initialize samplers: unexpected empty grammar stack <think>",
                     request=httpx.Request("POST", "http://localhost/v1/chat/completions"), body=None)
    client, create = client_for(error)
    with pytest.raises(APIError):
        lmstudio_json.chat_json(client, "mistral-small", "system", "user", SCHEMA, 0.15, 8192)
    assert create.call_count == 1


@pytest.mark.parametrize("stage", ["extract", "consolidate", "generate"])
@pytest.mark.parametrize("family,expected", [("Mistral", "mistral-small-3.2"), ("Qwen", "qwen3.5-9b")])
def test_stage_clients_select_family_and_keep_settings(monkeypatch, stage, family, expected):
    from minimax_h3_novel_pipeline import lmstudio_settings

    def respond(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [
            {"id": "unrelated", "object": "model"},
            {"id": "qwen3.5-9b", "object": "model"},
            {"id": "mistral-small-3.2", "object": "model"},
        ]})

    real_client = httpx.Client
    class MockClient(real_client):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "Client", MockClient)
    monkeypatch.setattr(lmstudio_settings, "get_api_key", lambda: "test-key")
    config = {"model_family": family, "thinking": False, "qwen35_top_k": 37}
    client, model = lmstudio_pipeline.make_client_and_model(
        lmstudio_pipeline.load(stage), "http://127.0.0.1:1234/v1", config)
    with client:
        assert model == expected
        assert client._minimax_h3_profile.NAME == family
        if family == "Qwen":
            assert client._minimax_h3_settings["top_k"] == 37
        else:
            assert client._minimax_h3_settings == {"thinking": False}
        config["qwen35_top_k"] = 1
        monkeypatch.setattr(lmstudio_json, "QWEN35_TOP_K", 2)
        create = Mock(return_value=Stream(['{"value":"ok"}']))
        monkeypatch.setattr(client.chat.completions, "create", create)
        lmstudio_json.chat_json(client, model, "system", "user", SCHEMA, 0.15, 8192)
        extra = create.call_args.kwargs["extra_body"]
        if family == "Mistral":
            assert extra == {}
        else:
            assert extra["top_k"] == 37


@pytest.mark.parametrize("family", ["Mistral", "Qwen"])
def test_family_selection_requires_matching_model(family):
    from minimax_h3_novel_pipeline import lmstudio_models
    client = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="other")])))
    with pytest.raises(RuntimeError, match=f"no matching {family}"):
        lmstudio_models.select_family_model(client, family)


def test_unknown_family_rejected_before_credentials(monkeypatch):
    from minimax_h3_novel_pipeline import lmstudio_config, lmstudio_settings
    key = Mock(side_effect=AssertionError("credentials accessed"))
    monkeypatch.setattr(lmstudio_settings, "get_api_key", key)
    with pytest.raises(ValueError, match="Unsupported model family"):
        lmstudio_config.LMStudioConfigurationNode().run("http://127.0.0.1:1234/v1", model_family="unknown")
    key.assert_not_called()


@pytest.mark.parametrize("family,attempts", [("Mistral", 2), ("Qwen", 4)])
def test_semantic_retries_follow_client_profile(family, attempts):
    from minimax_h3_novel_pipeline import lmstudio_models, reference_requests
    profile = lmstudio_models.get_profile(family)
    client = SimpleNamespace(_minimax_h3_profile=profile,
                             _minimax_h3_settings=profile.settings_from_config({"qwen35_length_retries": 3}))
    chat = Mock(return_value={})
    validate = Mock(side_effect=ValueError("missing field"))
    with pytest.raises(ValueError, match="bounded retries"):
        reference_requests.validated_request(chat, client, "qwen3.5" if family == "Qwen" else "mistral-small",
                                             "system", "user", SCHEMA,
                                             SimpleNamespace(temperature=0.15, max_tokens=500), validate)
    assert chat.call_count == attempts


def test_cache_uses_client_settings_and_ignores_other_clients(monkeypatch):
    from minimax_h3_novel_pipeline import lmstudio_models, prompt_cache
    profile = lmstudio_models.get_profile("Qwen")
    def client(top_k):
        return SimpleNamespace(_minimax_h3_profile=profile,
                               _minimax_h3_settings=profile.settings_from_config({"qwen35_top_k": top_k}))
    first, second = client(20), client(37)
    args = SimpleNamespace(temperature=0.15, max_tokens=500)
    initial = prompt_cache.fingerprint("qwen3.5", args, "prompt", client=first)
    assert initial != prompt_cache.fingerprint("qwen3.5", args, "prompt", client=second)
    monkeypatch.setattr(lmstudio_json, "QWEN35_TOP_K", 99)
    assert initial == prompt_cache.fingerprint("qwen3.5", args, "prompt", client=first)

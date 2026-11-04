import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import lmstudio_json, lmstudio_models
from tests.test_structured_json import Stream, client_for


@pytest.mark.parametrize('model', ['openai/gpt-oss-20b', 'gpt-oss-20b-MXFP4.gguf', 'GPT_OSS_120B'])
def test_gpt_oss_detection(model):
    profile = lmstudio_models.profile_for_model(model)
    assert profile.NAME == 'GPT-OSS'
    assert not profile.allows_chatml(model, {})


@pytest.mark.parametrize('thinking', [False, True])
def test_gpt_oss_reasoning_and_retry_keep_full_schema_and_evidence(thinking):
    schema = {'name': 'evidence', 'strict': True, 'schema': {
        'type': 'object', 'properties': {'evidence': {'type': 'array', 'items': {
            'type': 'object', 'properties': {'id': {'type': 'string'}},
            'required': ['id'], 'additionalProperties': False}}},
        'required': ['evidence'], 'additionalProperties': False}}
    result = {'evidence': [{'id': str(i)} for i in range(5)]}
    reasoning = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None, reasoning='Reasoning is separate from the final JSON.'))])
    first = Stream([reasoning, '{'])
    second = Stream([reasoning, json.dumps(result)])
    client, create = client_for(first, second)
    client.completions = SimpleNamespace(create=Mock(side_effect=AssertionError('No ChatML fallback')))
    token = lmstudio_json.REQUEST_THINKING.set(thinking)
    try:
        assert lmstudio_json.chat_json(client, 'gpt-oss-20b', 'system', json.dumps(result),
                                       schema, 0.12, 8192) == result
    finally:
        lmstudio_json.REQUEST_THINKING.reset(token)
    assert create.call_count == 2
    assert first.closed and second.closed
    for call in create.call_args_list:
        request = call.kwargs
        assert request['response_format'] == {'type': 'json_schema', 'json_schema': schema}
        assert request['extra_body'] == {}
        assert request['messages'][0] == {'role': 'system', 'content': 'system'}
        assert request['messages'][1]['content'].startswith(json.dumps(result))
        assert request['top_p'] == 1.0
    client.completions.create.assert_not_called()

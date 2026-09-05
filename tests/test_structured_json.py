"""The current backend always streams schema-constrained JSON."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

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
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])

    def close(self):
        self.closed = True


def client_for(*streams):
    create = Mock(side_effect=streams)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), create


def test_stream_stops_at_complete_json():
    stream = Stream(['{"value":', '"ok"}', 'unneeded'])
    client, create = client_for(stream)
    assert lmstudio_json.chat_json(client, "model", "system", "user", SCHEMA, 0.2, 200) == {"value": "ok"}
    assert stream.closed and stream.consumed == 2
    assert create.call_args.kwargs["response_format"]["type"] == "json_schema"
    assert create.call_args.kwargs["stream"] is True


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

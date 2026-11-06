import json

from minimax_h3_novel_pipeline.core import lmstudio_json, lmstudio_models
from tests.test_structured_json import SCHEMA, Stream, client_for


def test_gemma_profile_matches_loaded_models_and_configuration():
    profile = lmstudio_models.get_profile("Gemma")
    assert profile.NAME == "Gemma"
    assert profile.matches("google/gemma-3-12b-it")
    assert profile.matches("Gemma-2-9b-it-GGUF")
    assert not profile.matches("qwen3.5-9b")
    assert profile.settings_from_config({"qwen35_top_k": 42}) == {"thinking": False}
    assert profile.retries("gemma-3-12b-it", {}) == 1
    assert not profile.allows_chatml("gemma-3-12b-it", {})


def test_gemma_uses_json_schema_on_initial_and_compact_retry():
    result = {"value": "ok"}
    first = Stream(["{"])
    second = Stream([json.dumps(result)])
    client, create = client_for(first, second)

    assert lmstudio_json.chat_json(
        client, "google/gemma-3-12b-it", "system", "user", SCHEMA, 0.12, 8192
    ) == result

    assert create.call_count == 2
    assert first.closed and second.closed
    for call in create.call_args_list:
        assert call.kwargs["response_format"] == {"type": "json_schema", "json_schema": SCHEMA}
        assert call.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

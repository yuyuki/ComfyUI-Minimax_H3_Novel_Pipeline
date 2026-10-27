"""Node persistence selection reaches every request and saved source coverage."""
from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline.core import character_profile_batches as batches, path_access
from minimax_h3_novel_pipeline.core.character_profile_document import validate_document
from minimax_h3_novel_pipeline.nodes import consolidate_character_profiles as node
from tests.test_character_evidence import observation
from tests.test_character_profiles import document
from tests.test_character_profile_batches import fake_model, request_payload


@pytest.mark.parametrize("selection,count", [(None, 1), ("persistent", 1), ("everything", 3)])
def test_selection_requests_saved_coverage_and_defaults(tmp_path, monkeypatch, selection, count):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path / kind)
    chat = Mock(side_effect=fake_model)
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    client = Mock(return_value=(nullcontext(), "qwen"))
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model", client)
    data = document(observation(), observation("Temporary.", persistence="temporary"),
                    observation("Unknown.", persistence="unknown"))
    original = deepcopy(data)
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261025120000"}
    kwargs = {} if selection is None else {"persistence_filter": selection}
    instance = node.ConsolidateCharacterProfilesNode()
    result, _ = instance.run(data, config, **kwargs)["result"]
    assert validate_document(result) is result
    assert sum(len(p["supporting_evidence"]) for p in result["profiles"]) == count
    assert sum(len(c["evidence"]) for c in result["source_document"]["chapters"]) == count
    assert data == original
    assert client.call_args.args[2]["thinking"] is False
    assert "thinking" not in config
    assert instance.INPUT_TYPES()["required"]["thinking"][1]["default"] is False
    assert instance.INPUT_TYPES()["required"]["correction_attempts"][1]["default"] == 0
    for call in chat.call_args_list:
        for row in request_payload(call.args[3]).get("observations", []):
            if count == 1:
                assert row["observation"]["persistence"] == "persistent"
    # Switching modes in the same output folder must not reuse incompatible batches.
    other, _ = instance.run(data, config, persistence_filter="everything", thinking=True)["result"]
    assert sum(len(p["supporting_evidence"]) for p in other["profiles"]) == 3
    assert client.call_args.args[2]["thinking"] is True


def test_no_persistent_observations_skips_model(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path / kind)
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "qwen"))
    chat = Mock(side_effect=AssertionError("No model request expected"))
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", chat)
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261025120000"}
    result, _ = node.ConsolidateCharacterProfilesNode().run(
        document(observation(persistence="unknown")), config)["result"]
    assert validate_document(result) is result
    assert result["profiles"] == []
    chat.assert_not_called()
    with pytest.raises(ValueError, match="persistence_filter"):
        node.ConsolidateCharacterProfilesNode().run(document(), config, persistence_filter="invalid")


def test_invalid_identity_pairs_are_not_retried_or_subdivided(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path / kind)
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "qwen"))
    rejected = []
    def respond(*args):
        result = fake_model(*args)
        if args[4] == batches.IDENTITY_PAIR:
            rejected.append(request_payload(args[3]))
            for decision in result["decisions"]:
                decision["right_support"] = []  # "same" without supporting both sides
        return result
    monkeypatch.setattr(batches.lmstudio_json, "chat_json", respond)
    updates = []
    monkeypatch.setattr(batches.progress, "report", updates.append)
    data = document(observation(), observation("Second."), observation("Third."))
    config = {"api_url": "http://127.0.0.1:1234/v1", "run_folder": "20261025120000"}
    result, _ = node.ConsolidateCharacterProfilesNode().run(data, config)["result"]
    assert len(rejected) == 1
    assert len(rejected[0]["jobs"]) == 3
    assert len(result["profiles"]) == 3
    assert validate_document(result) is result
    assert any(0 < fraction < 1 for fraction in updates)

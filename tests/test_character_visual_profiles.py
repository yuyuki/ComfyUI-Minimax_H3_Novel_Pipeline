"""Stable visual designs, evidence retention and shared structured transport."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from minimax_h3_novel_pipeline import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from minimax_h3_novel_pipeline.core import character_profiles as cp, character_visual_profiles as vp
from minimax_h3_novel_pipeline.nodes import character_visual_profile_generator as node
from tests.test_character_evidence import observation
from tests.test_character_profiles import document, plan, settings
from minimax_h3_novel_pipeline.core import path_access
from tests.test_structured_json import Stream, client_for


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path / kind)
    return tmp_path / "output"


def consolidated(*items):
    data = document(*(items or (observation(),)))
    records = cp.prepare_evidence(data)
    from minimax_h3_novel_pipeline.core.character_profile_document import make_document
    profiles = cp.build_profiles(plan(records), records)
    for profile in profiles:
        profile['global_traits'] = [{'trait': 'eye_color', 'value': r['observation']['value'],
                                     'evidence_ids': [r['evidence_id']]}
                                    for r in profile['supporting_evidence']
                                    if r['observation']['persistence'] == 'persistent']
    return make_document(data, records, profiles)


def cinematic(data):
    source = data['source_document']['chapters'][0]
    return [{'source_file': source['source_file'], 'chapter_name': source['chapter_name'],
             'sequences': [{'sequence': 1, 'source': source['source_text'],
                            'adaptation': {p: source['source_text'] for p in ('initialState', 'event', 'endingState')}}]}]


def design(data):
    result = {field: [] for field in vp.FIELDS}
    result["eyes"] = [{"description": "Green eyes.", "classification": "canonical",
                       "evidence": [{"evidence_id": data["profiles"][0]["supporting_evidence"][0]["evidence_id"]}]}]
    return result


def chat_mock(monkeypatch, value):
    chat = Mock(side_effect=lambda *a: {"issues": []} if a[4] == vp.REVIEW_SCHEMA else deepcopy(value))
    monkeypatch.setattr(vp.lmstudio_json, "chat_json", chat)
    return chat


@pytest.mark.parametrize("mode", vp.MODES)
def test_node_modes_english_exact_evidence_and_json(workspace, monkeypatch, mode):
    data = consolidated()
    original = deepcopy(data)
    chat = chat_mock(monkeypatch, design(data))
    monkeypatch.setattr(node.character_chapter_enrichment, 'enrich', lambda *a: {
        '$schema': './character_chapter_enrichment.schema.json', 'format_version': 1,
        'source_file': 'chapter1.txt', 'chapter_name': 'chapter1', 'chapter_digest': 'test',
        'profiles_digest': 'test', 'sequences': [{'sequence': 1, 'characters': [{
            'local_id': data['profiles'][0]['local_id'], 'presence': 'visible', 'source_quote': 'Élise', 'label': 'Élise'}],
            'states': [], 'unresolved': []}]})
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "qwen"))
    monkeypatch.setattr(node, "stage_output", lambda *a: workspace)
    result, saved = node.CharacterVisualProfileGeneratorNode().run(
        data, {"api_url": "http://127.0.0.1:1234/v1"}, mode=mode, cinematic_chapters=cinematic(data))["result"]
    assert data == original
    assert result["source_document"] == original
    assert result["mode"] == mode
    p = result["profiles"][0]
    assert p["local_id"] == data["profiles"][0]["local_id"]
    assert p["supporting_evidence"] == data["profiles"][0]["supporting_evidence"]
    assert p["eyes"][0]["description"] == "Green eyes."
    assert p["hair"] == []
    text = Path(saved).read_text(encoding="utf-8")
    assert '\n  ' in text and "Élise avait les yeux verts." in text
    assert json.loads(text) == result
    global_designs = json.loads((Path(saved).parent / "visual_designs.json").read_text(encoding="utf-8"))
    assert [e["global_id"] for e in global_designs["entities"]] == [p["local_id"]]
    assert global_designs["entities"][0]["source_facts"]["stable_visual_description"] == "Green eyes."
    assert chat.call_count == 2
    assert json.loads(chat.call_args_list[0].args[3])["mode"] == mode
    assert "English" in chat.call_args_list[0].args[2]
    assert "English" in chat.call_args_list[1].args[2]
    assert NODE_CLASS_MAPPINGS["CharacterVisualProfileGeneratorNode"] is node.CharacterVisualProfileGeneratorNode
    assert NODE_DISPLAY_NAME_MAPPINGS["CharacterVisualProfileGeneratorNode"] == "Character Visual Profile Generator"


@pytest.mark.parametrize("mode,kind,valid", [("strict_canon", "inferred", False),
    ("strict_canon", "creative", False), ("balanced", "inferred", True),
    ("balanced", "creative", False), ("cinematic", "creative", True)])
def test_mode_classifications(mode, kind, valid):
    data = consolidated()
    response = design(data)
    response["hair"] = [{"description": "Neatly cropped hair.", "classification": kind,
                         "evidence": response["eyes"][0]["evidence"]}]
    if valid:
        vp.validate_design(response, vp.prepare_profiles(data)[0], mode)
    else:
        with pytest.raises(ValueError, match="generation mode"):
            vp.validate_design(response, vp.prepare_profiles(data)[0], mode)


@pytest.mark.parametrize('selected', [(True, False), (True, True), (False, False)])
def test_edited_conflict_selections(workspace, monkeypatch, selected):
    data = consolidated(observation(), observation("Élise avait les yeux bleus.", value="yeux bleus"))
    profile = data['profiles'][0]
    ids = [r['evidence_id'] for r in profile['supporting_evidence']]
    profile['unresolved_conflicts'] = [{'conflict_id': 'eyes', 'trait': 'eye_color', 'scope': 'same period',
        'evidence_ids': ids, 'choices': [{'id': str(i), 'value': str(i), 'evidence_ids': [key],
                                        'selected': selected[i]} for i, key in enumerate(ids)]}]
    projected = vp.prepare_profiles(data)[0]
    expected = design(data) if any(selected) else {f: [] for f in vp.FIELDS}
    chat_mock(monkeypatch, expected)
    result, _ = vp.generate(data, [projected], workspace, None, 'qwen', settings(mode='cinematic'))
    assert result['profiles'][0]['unresolved_conflicts'] == profile['unresolved_conflicts']
    assert result['profiles'][0]['eyes'] == expected['eyes']
    assert bool(projected['ignored_traits']) == (not any(selected))


def test_canonical_coverage_refs_and_temporary_traits():
    data = consolidated()
    response = design(data)
    response["eyes"] = []
    with pytest.raises(ValueError, match="every unblocked"):
        vp.validate_design(response, vp.prepare_profiles(data)[0], "strict_canon")
    response = design(data)
    response["eyes"][0]["evidence"][0]["evidence_id"] = "invented"
    with pytest.raises(ValueError, match="unknown evidence"):
        vp.validate_design(response, vp.prepare_profiles(data)[0], "strict_canon")
    for attribute in ("clothing", "accessory"):
        data = consolidated(observation(attribute=attribute, persistence="temporary"))
        response = {f: [] for f in vp.FIELDS}
        response["accessories"] = design(data)["eyes"]
        with pytest.raises(ValueError, match="unknown evidence"):
            vp.validate_design(response, vp.prepare_profiles(data)[0], "strict_canon")


@pytest.mark.parametrize("issue,bad", [("Description is not English", "Les yeux verts."),
                                      ("Canonical eye colour contradicts cited evidence", "Blue eyes.")])
def test_semantic_review_retries_english_and_fidelity(workspace, monkeypatch, issue, bad):
    data = consolidated()
    wrong = design(data)
    wrong["eyes"][0]["description"] = bad
    chat = Mock(side_effect=[wrong, {"issues": [issue]}, design(data), {"issues": []}])
    monkeypatch.setattr(vp.lmstudio_json, "chat_json", chat)
    result, _ = vp.generate(data, vp.prepare_profiles(data), workspace, None, "qwen", settings(mode="strict_canon"))
    assert result["profiles"][0]["eyes"][0]["description"] == "Green eyes."
    assert issue in chat.call_args_list[2].args[3]
    for call in chat.call_args_list:
        assert call.args[4]["strict"] is True
        assert data["profiles"][0]["supporting_evidence"][0]["evidence_id"] in call.args[3]


def test_cache_mode_invalidation_and_invalid_cache(workspace, monkeypatch):
    data = consolidated()
    chat = chat_mock(monkeypatch, design(data))
    args = settings(mode="balanced")
    for _ in range(2):
        vp.generate(data, vp.prepare_profiles(data), workspace, None, "qwen", args)
    assert chat.call_count == 2
    cache = next((workspace / ".cache").glob("*.json"))
    cache.write_text('{}')
    vp.generate(data, vp.prepare_profiles(data), workspace, None, "qwen", args)
    assert chat.call_count == 3  # The independently validated review remains cached.
    args.mode = "cinematic"
    vp.generate(data, vp.prepare_profiles(data), workspace, None, "qwen", args)
    assert chat.call_count == 5
    args.force = True
    vp.generate(data, vp.prepare_profiles(data), workspace, None, "qwen", args)
    assert chat.call_count == 7


def test_invalid_input_before_client(monkeypatch):
    client = Mock()
    monkeypatch.setattr(node.lmstudio_pipeline, "make_client_and_model", client)
    data = consolidated()
    data["profiles"][0]["supporting_evidence"][0]["observation"]["quotation"] = "changed"
    with pytest.raises(ValueError, match="differs from its validated source"):
        node.CharacterVisualProfileGeneratorNode().run(data, {"api_url": "http://127.0.0.1:1234/v1"})
    client.assert_not_called()


def test_real_transport_compact_retry_preserves_schema_and_all_evidence(workspace):
    data = consolidated()
    response = design(data)
    client, create = client_for(Stream(['{"eyes":']), Stream([json.dumps(response)]), Stream(['{"issues": []}']))
    vp.generate(data, vp.prepare_profiles(data), workspace, client, "qwen", settings(mode="strict_canon"))
    requests = create.call_args_list
    assert len(requests) == 3
    for call in requests:
        assert call.kwargs["response_format"]["type"] == "json_schema"
    for call in requests[:2]:
        schema = call.kwargs["response_format"]["json_schema"]
        assert set(schema["schema"]["properties"]) == set(vp.FIELDS)
        assert "maxItems" not in schema["schema"]["properties"]["eyes"]["items"]["properties"]["evidence"]
        assert data["profiles"][0]["supporting_evidence"][0]["evidence_id"] in str(call.kwargs["messages"])


def test_rejected_review_does_not_save_output(workspace, monkeypatch):
    data = consolidated()
    chat = Mock(side_effect=[design(data), {"issues": ["Unsupported canonical detail"]}])
    monkeypatch.setattr(vp.lmstudio_json, "chat_json", chat)
    with pytest.raises(ValueError, match="no valid decision"):
        vp.generate(data, vp.prepare_profiles(data), workspace, None, "qwen",
                    settings(mode="strict_canon", correction_attempts=0))
    assert not (workspace / "character_visual_profiles.json").exists()
    assert not list((workspace / ".cache").glob("*character_visual_profile*"))


def test_empty_input_and_precise_values(workspace, monkeypatch):
    source = document()
    from minimax_h3_novel_pipeline.core.character_profile_document import make_document
    data = make_document(source, {}, [])
    chat = Mock()
    monkeypatch.setattr(vp.lmstudio_json, "chat_json", chat)
    result, saved = vp.generate(data, vp.prepare_profiles(data), workspace, None, "qwen", settings(mode="balanced"))
    assert result["profiles"] == [] and Path(saved).is_file()
    chat.assert_not_called()
    data = consolidated()
    value = design(data)
    value["eyes"][0]["description"] = "Green or blue eyes."
    with pytest.raises(ValueError, match="alternatives"):
        vp.validate_design(value, vp.prepare_profiles(data)[0], "balanced")

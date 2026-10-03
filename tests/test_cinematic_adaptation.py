"""Public cinematic JSON shape, source coverage and saved node output."""
from contextlib import nullcontext
import json

import pytest

from minimax_h3_novel_pipeline import cinematic_adaptation as adaptation
from minimax_h3_novel_pipeline import cinematic_chapter_adapter as adapter
from minimax_h3_novel_pipeline import path_access


def _sequence(source, initial="Before.", final="After."):
    return {"source": source, "initial_state": initial, "events": ["Action."], "final_state": final}


def test_long_passage_recovers_after_corrections_with_lossless_continuity(monkeypatch):
    source = "Il attend. Il crie.\n\n" * 300
    requests = []
    accepted = []

    def chat(client, model, system, prompt, schema, *args):
        payload, _ = json.JSONDecoder().raw_decode(prompt)
        requests.append((payload, prompt))
        passage = payload["current_passage"]
        if len(passage) > 2000:
            return {"sequences": [_sequence(passage.replace("crie", "parle", 1))]}
        final = f"State {len(accepted) + 1}."
        accepted.append((payload, final))
        return {"sequences": [_sequence(passage, payload["previous_final_state"] or "Before.", final)]}

    monkeypatch.setattr(adaptation.lmstudio_json, "chat_json", chat)
    result = adaptation.adapt_chapter(None, "mock", source, chunk_chars=4000,
                                      temperature=0.15, max_tokens=8192, correction_attempts=1)
    assert "".join(item["source"] for item in result["sequences"]) == source
    assert [item["sequence"] for item in result["sequences"]] == list(range(1, len(result["sequences"]) + 1))
    assert len(accepted) > 2
    assert requests[0][0] == requests[1][0]
    assert "Previous response rejected:" in requests[1][1]
    assert accepted[0][0]["previous_final_state"] == ""
    for (previous, final), (following, _) in zip(accepted, accepted[1:]):
        assert following["previous_final_state"] == final


def test_correction_success_does_not_split_passage(monkeypatch):
    source = "Il attend. " * 200
    prompts = []

    def chat(*args):
        prompts.append(args[3])
        return {"sequences": [_sequence("Wrong." if len(prompts) == 1 else source)]}

    monkeypatch.setattr(adaptation.lmstudio_json, "chat_json", chat)
    result = adaptation.adapt_chapter(None, "mock", source, chunk_chars=4000,
                                      temperature=0.15, max_tokens=8192, correction_attempts=1)
    assert len(prompts) == 2
    assert len(result["sequences"]) == 1
    assert result["sequences"][0]["source"] == source


@pytest.mark.parametrize("invalid_source", ["Il part.", "Il attend."])
def test_recovery_remains_bounded_and_rejects_invalid_fallback(monkeypatch, invalid_source):
    requests = []

    def chat(*args):
        requests.append(args[3])
        if args[4]["name"] == "cinematic_single_passage":
            return {"initial_state": "Before.", "events": [""], "final_state": "After."}
        return {"sequences": [_sequence(invalid_source)]}

    monkeypatch.setattr(adaptation.lmstudio_json, "chat_json", chat)
    with pytest.raises(ValueError, match="Cinematic adaptation passage 1/1: events"):
        adaptation.adapt_chapter(None, "mock", "Il attend. " * 300, chunk_chars=4000,
                                 temperature=0.15, max_tokens=8192, correction_attempts=1)
    # Two attempts each for the original, its half, the small child and fallback.
    assert len(requests) == 8


@pytest.mark.parametrize("invalid_source", ["Translated source.", "Il attend."])
def test_copy_failure_regenerates_whole_passage_and_preserves_continuity(monkeypatch, invalid_source):
    source = "Il attend. Il crie.\n\nIl part."
    requests = []

    def chat(client, model, system, prompt, schema, *args):
        payload, _ = json.JSONDecoder().raw_decode(prompt)
        requests.append((schema["name"], payload))
        if payload["current_passage"].strip() == "Il part.":
            assert payload["previous_final_state"] == "Il a crié."
            return {"sequences": [_sequence("Il part.")]}
        if schema["name"] == "cinematic_single_passage":
            assert "source" not in schema["schema"]["properties"]
            return {"initial_state": "Il est debout.", "events": ["Il attend.", "Il crie."],
                    "final_state": "Il a crié."}
        return {"sequences": [_sequence(invalid_source)]}

    monkeypatch.setattr(adaptation.lmstudio_json, "chat_json", chat)
    result = adaptation.adapt_chapter(None, "mock", source, chunk_chars=20,
                                      temperature=0.15, max_tokens=8192, correction_attempts=1)
    assert "".join(item["source"] for item in result["sequences"]) == source
    assert [item["sequence"] for item in result["sequences"]] == [1, 2]
    assert result["sequences"][0]["adaptation"]["event"] == "1. Il attend.\n2. Il crie."
    assert [name for name, _ in requests] == [
        "cinematic_chapter_adaptation", "cinematic_chapter_adaptation",
        "cinematic_single_passage", "cinematic_chapter_adaptation",
    ]


@pytest.mark.parametrize("error", [RuntimeError("network failure"), KeyboardInterrupt()])
def test_request_errors_do_not_trigger_subdivision(monkeypatch, error):
    requests = []

    def chat(*args):
        requests.append(args[3])
        raise error

    monkeypatch.setattr(adaptation.lmstudio_json, "chat_json", chat)
    with pytest.raises(type(error)):
        adaptation.adapt_chapter(None, "mock", "Il attend. " * 300, chunk_chars=4000,
                                 temperature=0.15, max_tokens=8192, correction_attempts=1)
    assert len(requests) == 1


def test_structured_adaptation_preserves_events_source_and_chunk_continuity(monkeypatch):
    source = "Il attend. Il crie. Il part."
    requests = []

    def chat(client, model, system, prompt, schema, *args):
        payload = json.loads(prompt)
        requests.append(payload)
        if len(requests) == 1:
            return {"sequences": [
                {"source": "Il attend.", "initial_state": "Il est debout.",
                 "events": ["Il attend."], "final_state": "Il reste debout."},
                {"source": "Il crie.", "initial_state": "Il reste debout.",
                 "events": ["Il inspire.", "Il crie : « À l’aide ! »"], "final_state": "Il a crié."},
            ]}
        return {"sequences": [{"source": "Il part.", "initial_state": "Il a crié.",
                               "events": ["Il part."], "final_state": "Il est parti."}]}

    monkeypatch.setattr(adaptation.lmstudio_json, "chat_json", chat)
    result = adaptation.adapt_chapter(None, "mock", source, chunk_chars=20,
                                      temperature=0.15, max_tokens=8192, correction_attempts=0,
                                      chapter_name="Chapitre 1 — Le départ")
    assert [item["sequence"] for item in result["sequences"]] == [1, 2, 3]
    assert "".join(item["source"] for item in result["sequences"]) == source
    assert set(result) == {"chapter_name", "sequences"}
    assert result["chapter_name"] == "Chapitre 1 — Le départ"
    assert all(set(item) == {"sequence", "source", "adaptation"} for item in result["sequences"])
    assert result["sequences"][1]["adaptation"] == {
        "initialState": "Il reste debout.",
        "event": "1. Il inspire.\n2. Il crie : « À l’aide ! »",
        "endingState": "Il a crié.",
    }
    assert requests[0]["previous_final_state"] == ""
    assert requests[1]["previous_final_state"] == "Il a crié."
    assert all(set(item["adaptation"]) == {"initialState", "event", "endingState"} for item in result["sequences"])


@pytest.mark.parametrize("copy_failure", [False, True])
def test_node_saves_structured_adaptation(tmp_path, monkeypatch, copy_failure):
    chapter = tmp_path / "chapter.txt"
    source = " ".join(["Il part."] * 15)
    chapter.write_text(source, encoding="utf-8")
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(adapter, "stage_output", lambda *args: tmp_path / "output")
    monkeypatch.setattr(adapter.lmstudio_pipeline, "make_client_and_model", lambda *args: (nullcontext(), "mock"))
    def chat(*args):
        fields = {"initial_state": "Il est ici.", "events": ["Il part."], "final_state": "Il est parti."}
        if args[4]["name"] == "cinematic_single_passage":
            return fields
        return {"sequences": [{"source": "Incorrect." if copy_failure else source, **fields}]}

    monkeypatch.setattr(adaptation.lmstudio_json, "chat_json", chat)

    chapters, saved_files = adapter.CinematicChapterAdapterNode().run(
        {"api_url": "unused", "run_folder": "test"}, {"chapter_paths": [str(chapter)]},
    )
    saved = json.loads((tmp_path / "output/001_chapter.cinematic.json").read_text(encoding="utf-8"))
    assert saved == {"chapter_name": "chapter", "sequences": chapters[0]["sequences"]}
    assert chapters[0]["chapter_name"] == "chapter"
    assert saved["sequences"] == [{
        "sequence": 1, "source": source, "adaptation": {
            "initialState": "Il est ici.", "event": "1. Il part.", "endingState": "Il est parti.",
        },
    }]
    assert saved_files == chapters[0]["saved_file"]


@pytest.mark.parametrize("cancel_after_first", [False, True])
def test_completed_chapter_saved_before_later_failure_or_cancellation(tmp_path, monkeypatch, cancel_after_first):
    paths = [tmp_path / "chapter1.txt", tmp_path / "chapter2.txt"]
    for path in paths:
        path.write_text("Il part. " * 15, encoding="utf-8")
    output = tmp_path / "output"
    saved = output / "001_chapter1.cinematic.json"
    sequences = [{"sequence": 1, "source": "Il part.", "adaptation": {
        "initialState": "Before.", "event": "1. Action.", "endingState": "After.",
    }}]
    chapter = {"chapter_name": "chapter1", "sequences": sequences}
    completed = False
    calls = []

    def adapt(*args, **kwargs):
        nonlocal completed
        calls.append(args[2])
        if len(calls) == 2:
            assert json.loads(saved.read_text(encoding="utf-8")) == chapter
            raise RuntimeError("second chapter failed")
        completed = True
        return chapter

    def interrupt_check():
        if cancel_after_first and completed:
            assert json.loads(saved.read_text(encoding="utf-8")) == chapter
            raise KeyboardInterrupt()

    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(adapter, "stage_output", lambda *args: output)
    monkeypatch.setattr(adapter.lmstudio_pipeline, "make_client_and_model", lambda *args: (nullcontext(), "mock"))
    monkeypatch.setattr(adapter.lmstudio_pipeline, "comfy_interrupt_check", interrupt_check)
    monkeypatch.setattr(adaptation, "adapt_chapter", adapt)

    with pytest.raises(KeyboardInterrupt if cancel_after_first else RuntimeError):
        adapter.CinematicChapterAdapterNode().run(
            {"api_url": "unused", "run_folder": "test"}, {"chapter_paths": [str(path) for path in paths]},
        )
    assert json.loads(saved.read_text(encoding="utf-8")) == chapter
    assert not (output / "002_chapter2.cinematic.json").exists()
    assert len(calls) == (1 if cancel_after_first else 2)

"""Public cinematic JSON shape, source coverage and saved node output."""
from contextlib import nullcontext
import json

from minimax_h3_novel_pipeline import cinematic_adaptation as adaptation
from minimax_h3_novel_pipeline import cinematic_chapter_adapter as adapter
from minimax_h3_novel_pipeline import path_access


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
                                      temperature=0.15, max_tokens=8192, correction_attempts=0)
    assert [item["sequence"] for item in result] == [1, 2, 3]
    assert "".join(item["source"] for item in result) == source
    assert all(set(item) == {"sequence", "source", "adaptation"} for item in result)
    assert result[1]["adaptation"] == {
        "initialState": "Il reste debout.",
        "event": "1. Il inspire.\n2. Il crie : « À l’aide ! »",
        "endingState": "Il a crié.",
    }
    assert requests[0]["previous_final_state"] == ""
    assert requests[1]["previous_final_state"] == "Il a crié."
    assert all(set(item["adaptation"]) == {"initialState", "event", "endingState"} for item in result)


def test_node_saves_structured_adaptation(tmp_path, monkeypatch):
    chapter = tmp_path / "chapter.txt"
    source = " ".join(["Il part."] * 15)
    chapter.write_text(source, encoding="utf-8")
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(adapter, "stage_output", lambda *args: tmp_path / "output")
    monkeypatch.setattr(adapter.lmstudio_pipeline, "make_client_and_model", lambda *args: (nullcontext(), "mock"))
    monkeypatch.setattr(adaptation.lmstudio_json, "chat_json", lambda *args: {"sequences": [{
        "source": source, "initial_state": "Il est ici.", "events": ["Il part."], "final_state": "Il est parti.",
    }]})

    chapters, saved_files = adapter.CinematicChapterAdapterNode().run(
        {"api_url": "unused", "run_folder": "test"}, {"chapter_paths": str(chapter)},
    )
    saved = json.loads((tmp_path / "output/001_chapter.cinematic.json").read_text(encoding="utf-8"))
    assert saved == chapters[0]["sequences"] == [{
        "sequence": 1, "source": source, "adaptation": {
            "initialState": "Il est ici.", "event": "1. Il part.", "endingState": "Il est parti.",
        },
    }]
    assert saved_files == chapters[0]["saved_file"]

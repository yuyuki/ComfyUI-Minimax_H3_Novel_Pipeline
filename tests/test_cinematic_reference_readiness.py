"""Check the adapter's reference handoff and source-scoped repair offline."""
from copy import deepcopy
import json

import pytest

from minimax_h3_novel_pipeline.core import cinematic_adaptation as ca, cinematic_diagnostics as diagnostics
from minimax_h3_novel_pipeline.core import narrative_validation as nv
from tests.passages_helpers import with_passage_ids

pytestmark = pytest.mark.usefixtures("mock_scene_planning")


def test_reference_repairs_preserve_other_fields_and_check_the_actual_neighbor(monkeypatch):
    first = ca._normalize_text("Suspendue à sa corde de sécurité dans une crevasse, Élise saisit le filin de la lampe. "
             "Elle approche de la paroi. Des signes sont gravés sur la tablette. "
             "Elle insère la lampe dans son support. « Attends ! » crie Luc d'en haut.")
    second = (" La corde de sécurité s'effiloche sous son poids. Elle appelle Nora. "
              "Luc actionne le treuil hors de vue et un cri anonyme retentit.")
    first_draft = {
        "source": first, "initial_state": "Élise est suspendue dans la crevasse, lampe portative en main.",
        "events": ["Élise saisit la corde près de la paroi gravée.", "Elle pose la lampe. Nora crie."],
        "final_state": "La lampe portative est insérée dans son support.",
    }
    second_draft = {
        "source": second, "initial_state": "Élise est suspendue dans la crevasse.",
        "events": ["Le filin de la lampe s'effiloche. Nora et Luc parlent."],
        "final_state": "La corde de sécurité s'effiloche sous le poids d'Élise.",
    }
    fixes = [
        ["Élise saisit le filin de la lampe, distinct de la corde de sécurité qui la soutient.",
         "Dans la crevasse, elle approche de la paroi et voit les signes gravés sur la tablette.",
         "Elle insère la lampe portative dans son support.", ca._normalize_text('Hors champ, Luc crie : « Attends ! »')],
        ["La corde de sécurité s'effiloche sous son poids. Élise appelle Nora.",
         "Luc actionne le treuil hors champ, sans parole. Un cri anonyme retentit."],
    ]
    outputs, repairs = [], []

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        assert schema["strict"] is True
        if schema["name"] == nv.SCHEMA["name"]:
            rendered = data["exported_adaptation"]
            outputs.append(rendered)
            assert all(rule in system for rule in ("coexisting physical instances", "marks and damage",
                                                    "anonymous", "addressee", "boundary context"))
            slot = int(data["sequences"][0]["source"] == second)
            if slot:
                assert data["previous_sequence"]["events"] == [outputs[1]["event"]]
            else:
                assert data["following_source"] == second
                assert any(p["context"] == "following_sequence" for p in data["review_context_passages"])
            if data["sequences"][0]["events"] == fixes[slot]:
                return {"issues": []}
            passage_ids = [row["id"] for row in data["review_passages"]]
            return {"issues": [{"target": 1, "message": "Incorrect ownership or attribution in the action.",
                                "field": "event", "disposition": "repair", "check": "adaptation",
                                "passage_ids": passage_ids}]}
        if "rejected_adaptation" in data:
            repairs.append(data)
            assert data["fields_to_correct"] == ["event"]
            for passage in data["cited_defects"][0]["source_passages"]:
                assert data["current_passage"][passage["start"]:passage["end"]] == passage["text"]
            draft = deepcopy((first_draft, second_draft)[len(repairs) - 1])
            draft.update(initial_state="Élise apparaît ailleurs.", events=fixes[len(repairs) - 1],
                         final_state="Une autre personne apparaît.")
            return {"sequences": [draft]}
        return {"sequences": [first_draft, second_draft]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    chapter = ca.adapt_chapter(None, "mock", first + second, chunk_chars=4000,
                               temperature=0.15, max_tokens=8192, correction_attempts=1)
    assert len(repairs) == 2
    for scene, draft, reviewed in zip(chapter["sequences"], (first_draft, second_draft), outputs[1::2]):
        assert scene["source"] == draft["source"]
        assert scene["adaptation"] == reviewed
        assert reviewed["initialState"] == draft["initial_state"]
        assert reviewed["endingState"] == draft["final_state"]
        assert "review" not in scene


def test_uncertain_identity_is_diagnosed_without_guessing_or_rewriting(monkeypatch):
    source = "Dans le noir, une corde frotte contre la paroi."
    draft = {"source": source, "initial_state": "L'obscurité masque la paroi.",
             "events": [source], "final_state": "Un frottement se fait entendre."}
    snapshots = []

    def chat(client, model, system, prompt, schema, *args):
        data = json.loads(prompt)
        if schema["name"] == nv.SCHEMA["name"]:
            return {"issues": [{"target": 1, "message": "The rope's identity cannot be established.",
                                "field": "event", "disposition": "uncertain", "check": "adaptation",
                                "passage_ids": [data["review_passages"][0]["id"]]}]}
        assert "rejected_adaptation" not in data
        assert schema["name"] == ca.SCHEMA["name"]
        return {"sequences": [draft]}

    monkeypatch.setattr(ca.lmstudio_json, "chat_json", with_passage_ids(chat))
    chapter = ca.adapt_chapter(None, "mock", source, chunk_chars=4000,
                               temperature=0.15, max_tokens=8192, correction_attempts=1,
                               on_checkpoint=lambda state: snapshots.append(deepcopy(state)))
    assert chapter["sequences"][0]["adaptation"]["event"] == f"1. {source}"
    report = diagnostics.report(snapshots[-1], chapter_name="Test", source_file="Test", maximum_seconds=8, completed=True)
    assert report["sequences"][0]["status"] == "incomplete"
    assert report["warnings"][0]["reason"] == "unresolved_source_evidence"
    assert "review" not in diagnostics.downstream_chapter(chapter)["sequences"][0]

"""Source-backed continuity regression; LM Studio responses remain deterministic mocks."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path

import pytest

from minimax_h3_novel_pipeline import narrative_nodes as nodes, narrative_state as ns, path_access
from .test_narrative_state import contract, entity


SOURCE = Path(__file__).parent / "fixtures" / "00_PROLOGUE.md"


def source_contracts():
    state = {"entities": [
        entity("indy", "character", location="crevasse", position="suspended_on_rope", posture="hanging", visible=True),
        entity("crevasse", "location", environment="dark", visible=True),
        entity("backpack", owner="indy", location="indy", relationship="worn", visible=True),
        entity("main_rope", status="intact", location="crevasse", relationship="under_indy_arms", visible=True),
        entity("torch_1", status="lost_below"),
        *[entity(key, status="not_introduced") for key in ("torch_2", "torch_line", "tablet", "holder")],
    ]}
    specs = [
        ("— Doriane ! hurla-t-il. Envoyez une autre torche !", [], "action"),
        ("Il leva les yeux et vit une lueur vacillante descendre vers lui en tressautant.", [
            ("torch_2", "status", "lit"), ("torch_2", "location", "crevasse"), ("torch_2", "visible", True),
            ("torch_line", "status", "intact"), ("torch_line", "location", "crevasse"), ("torch_line", "visible", True),
        ], "introduction"),
        ("Indy se baissa pour éviter la torche, qui passa près de sa tête", [("indy", "posture", "ducking")], "action"),
        ("puis il s’en empara après avoir attrapé le filin.", [
            ("torch_line", "owner", "indy"), ("torch_line", "location", "indy"), ("torch_line", "relationship", "held"),
        ], "action"),
        ("puis il s’en empara après avoir attrapé le filin.", [
            ("torch_2", "owner", "indy"), ("torch_2", "location", "indy"), ("torch_2", "relationship", "held"),
        ], "action"),
        ("Il tira sur la corde à deux reprises et Doumas, l’assistant de Doriane, le fit descendre de soixante centimètres.", [
            ("indy", "position", "lowered_60cm"),
        ], "action"),
        ("Il se trouva alors juste devant la tablette.", [
            ("tablet", "status", "embedded_in_wall"), ("tablet", "location", "crevasse"), ("tablet", "visible", True),
        ], "introduction"),
        ("Indy tira de son sac à dos une torchère munie de quatre pointes", [
            ("holder", "status", "intact"), ("holder", "visible", True),
            ("holder", "owner", "indy"), ("holder", "location", "indy"), ("holder", "relationship", "held"),
        ], "introduction"),
        ("qu’il planta dans le roc à l’aide de son maillet.", [
            ("holder", "owner", None), ("holder", "location", "crevasse"), ("holder", "relationship", "fixed_in_rock"),
        ], "action"),
        ("Éclairant la tablette, il se pencha en avant pour mieux voir.", [("indy", "posture", "leaning_forward")], "action"),
        ("Insérant la torche dans son support", [
            ("torch_2", "owner", "holder"), ("torch_2", "location", "holder"), ("torch_2", "relationship", "stored"),
        ], "action"),
        ("Il descendit de quelques centimètres. Sous ses bras, la corde se resserra.", [
            ("indy", "position", "below_tablet"), ("main_rope", "relationship", "tight_under_indy_arms"),
        ], "action"),
        ("Un craquement sinistre retentit dans la crevasse tandis qu’une nouvelle secousse animait la corde.", [
            ("main_rope", "status", "damaged"),
        ], "action"),
        ("Il récupéra la torche et la brandit.", [
            ("torch_2", "owner", "indy"), ("torch_2", "location", "indy"), ("torch_2", "relationship", "held"),
        ], "action"),
        ("Tenant la torche entre les dents", [("torch_2", "relationship", "in_mouth")], "action"),
        ("il tendit la main pour saisir le filin au-dessus de l’endroit où il commençait à s’effilocher.", [
            ("indy", "posture", "reaching_above_fray"),
        ], "action"),
        ("Il y eut un claquement – un bruit terrible, qui résonna longuement.", [("main_rope", "status", "broken")], "action"),
        ("Ses doigts se refermèrent sur la corde.", [("main_rope", "relationship", "gripped_above_break")], "action"),
        ("Il demeura suspendu par une main", [("indy", "position", "hanging_by_one_hand"), ("indy", "posture", "one_hand_overhead")], "action"),
        ("Sous l’effet d’une forte traction venue d’en haut, la corde glissa entre ses doigts.", [
            ("main_rope", "relationship", "slipping_from_grip"),
        ], "action"),
        ("sa main se referma sur le vide.", [("main_rope", "relationship", "no_contact")], "action"),
        ("Il tomba.", [("indy", "position", "falling"), ("indy", "posture", "falling")], "action"),
    ]
    beats = []
    for index, (evidence, changes, kind) in enumerate(specs):
        beat = contract(state, evidence, changes, kind)
        beat["events"][0]["id"] = f"prologue_{index + 1:02d}"
        beats.append(beat)
        state = beat["state_after"]
    return beats


def test_actual_prologue_evidence_and_critical_states():
    source = SOURCE.read_text(encoding="utf-8-sig")
    beats = source_contracts()
    previous = None
    for beat in beats:
        assert not ns.validate_contract(beat, previous, source)
        previous = beat["state_after"]
    combined = {**beats[0], "events": [b["events"][0] for b in beats], "state_after": previous}
    assert not ns.validate_contract(combined, source=source)
    states = [ns.state_map(b["state_before"]) for b in beats]
    assert not states[0]["holder"]["visible"]
    assert states[4]["torch_line"]["relationship"] == "held"
    assert states[4]["torch_2"]["owner"] is None
    assert states[10]["torch_2"]["relationship"] == "held"  # Intention to store it is not storage.
    assert states[13]["main_rope"]["status"] == "damaged"  # Damage precedes retrieval.
    assert states[13]["torch_2"]["owner"] == "holder"
    assert states[14]["torch_2"]["relationship"] == "held"
    assert states[19]["main_rope"]["status"] == "broken"
    assert states[19]["indy"]["position"] == "hanging_by_one_hand"
    assert all(state["indy"]["position"] != "falling" for state in states)


@pytest.mark.parametrize("index,key,field,value", [
    (0, "indy", "position", "falling"),
    (10, "torch_2", "relationship", "stored"),
    (14, "torch_2", "relationship", "in_mouth"),
    (18, "indy", "position", "falling"),
])
def test_source_backed_initial_frame_contamination(index, key, field, value):
    beat = deepcopy(source_contracts()[index])
    ns.state_map(beat["initial_frame"])[key][field] = value
    errors = ns.validate_contract(beat, source=SOURCE.read_text(encoding="utf-8-sig"))
    assert any(error["entity"] == key and error["suggested_correction"] for error in errors)


def test_actual_prologue_preprocessing_with_mocked_lmstudio(tmp_path, monkeypatch):
    source = "\n\n".join(p.strip() for p in SOURCE.read_text(encoding="utf-8-sig").split("\n\n") if p.strip())
    chapter = tmp_path / SOURCE.name
    chapter.write_text(source, encoding="utf-8")
    beats = source_contracts()
    combined = {**beats[0], "events": [b["events"][0] for b in beats], "state_after": beats[-1]["state_after"]}
    # Preserve the entire source and dialogue; normalize the opening metaphor for this mock.
    cinematic = source.replace("suspendu tel un croissant de lune à une corde qui lui meurtrissait le torse et les aisselles",
                               "suspendu à une corde passant étroitement sous ses bras et autour du haut de son torse")
    calls = []
    def chat(client, model, system, user, schema, *args):
        calls.append(json.loads(user))
        if schema == ns.SIMPLIFY_SCHEMA:
            return {"cinematic_text": cinematic}
        if schema == ns.CONTRACT_SCHEMA:
            return deepcopy(combined)
        return {"errors": []}
    monkeypatch.setattr(ns, "chat_json", chat)
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    monkeypatch.setattr(nodes, "stage_output", lambda *a: tmp_path / "output")
    monkeypatch.setattr(nodes.lmstudio_pipeline, "make_client_and_model", lambda *a: (nullcontext(), "mock-qwen"))
    result = nodes.NovelCinematicSimplifierNode().run(str(chapter), {"api_url": "unused"})
    assert calls[0]["original_scene"] == source
    assert "croissant de lune" not in result[1]
    assert "— Doriane ! hurla-t-il. Envoyez une autre torche !" in result[1]
    saved = json.loads((tmp_path / "output/cinematic_narrative.json").read_text(encoding="utf-8"))
    record = saved["chapters"][str(chapter.resolve())]
    assert record["segments"][0]["original_text"] == source
    assert record["segments"][0]["contract"] == combined

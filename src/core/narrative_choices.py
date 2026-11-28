"""Editable cinematic alternatives, resolved before any downstream model work."""
from copy import deepcopy
import re

PHASES = ("initialState", "event", "endingState")
_TEXT = {"type": "string", "minLength": 1}
ADAPTATION_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": list(PHASES),
    "properties": {phase: _TEXT for phase in PHASES},
}


def validate_scene_text(text):
    """Reject source metadata embedded as a scene action or state."""
    if re.search(r'''^\s*(?:\d+[.)]\s*)?["']?source_excerpt["']?\s*:''', text, re.MULTILINE | re.IGNORECASE):
        raise ValueError("source_excerpt metadata must not appear in scene descriptions; keep excerpts in source.")


def _validate_adaptation(value):
    if (not isinstance(value, dict) or set(value) != set(PHASES)
            or any(not isinstance(value[p], str) or not value[p].strip() for p in PHASES)):
        raise ValueError("Narrative proposal requires nonempty initialState, event and endingState strings.")
    for phase in PHASES:
        validate_scene_text(value[phase])


def resolve_chapter(chapter):
    """Return a copy with exactly one selected proposal applied per affected sequence."""
    chapter = deepcopy(chapter)
    review = chapter.get("narrative_review", {"issues": []})
    if not isinstance(review, dict) or not isinstance(review.get("issues"), list):
        raise ValueError("narrative_review requires an issues array.")
    sequences = {s["sequence"]: s for s in chapter["sequences"]}
    seen = set()
    for issue in review["issues"]:
        if not isinstance(issue, dict):
            raise ValueError("Narrative issue must be an object.")
        seq = issue.get("sequence")
        if type(seq) is not int or seq not in sequences or seq in seen:
            raise ValueError("Narrative issues require one valid, unique sequence address per issue.")
        seen.add(seq)
        for field in ("id", "description", "source_excerpt"):
            if not isinstance(issue.get(field), str) or not issue[field].strip():
                raise ValueError(f"Narrative issue requires {field}.")
        proposals = issue.get("proposals")
        if not isinstance(proposals, list) or len(proposals) < 2:
            raise ValueError("Narrative issue requires at least two proposals.")
        ids = set()
        for proposal in proposals:
            if not isinstance(proposal, dict) or type(proposal.get("selected")) is not bool:
                raise ValueError("Narrative proposal selected must be a boolean.")
            pid = proposal.get("id")
            if not isinstance(pid, str) or not pid.strip() or pid in ids:
                raise ValueError("Narrative proposal IDs must be nonempty and unique.")
            ids.add(pid)
            if not isinstance(proposal.get("reason"), str) or not proposal["reason"].strip():
                raise ValueError("Narrative proposal requires a reason.")
            _validate_adaptation(proposal.get("adaptation"))
        selected = [p for p in proposals if p["selected"]]
        choice = selected[0] if selected else proposals[0]
        if len(selected) != 1:
            print(f"Narrative choice: sequence {seq}: normalized selection to {choice['id']}.", flush=True)
        for proposal in proposals:
            proposal["selected"] = proposal is choice
        sequences[seq]["adaptation"] = deepcopy(choice["adaptation"])
    return chapter

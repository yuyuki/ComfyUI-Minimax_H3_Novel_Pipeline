"""Current structured continuity-review fixtures."""


def issue(message="Wrong eyeline", target="prompt", path="/current_prompt"):
    return {"target": target, "message": message, "evidence_paths": [path]}

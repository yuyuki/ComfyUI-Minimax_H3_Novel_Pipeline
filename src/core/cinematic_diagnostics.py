"""Report adapter review and execution metrics separately from story data."""
from copy import deepcopy


def downstream_chapter(chapter):
    """Retain chapter data and editable choices; all review execution metadata stays aside."""
    result = deepcopy(chapter)
    for scene in result["sequences"]:
        scene.pop("review", None)
    review = result.get("narrative_review")
    if review is not None:
        result["narrative_review"] = {"issues": review.get("issues", [])}
        for issue in result["narrative_review"]["issues"]:
            for proposal in issue.get("proposals", []):
                proposal.pop("review", None)
    return result


def report(state, *, chapter_name, source_file, maximum_seconds, completed=False):
    scenes = [
        {**deepcopy(review), "estimated_seconds": scene["estimated_seconds"]}
        for scene, review in zip(state["sequences"], state["reviews"], strict=True)
    ]
    incomplete = sum(scene["status"] == "incomplete" for scene in scenes)
    return {
        "schema_version": "minimax-cinematic-diagnostics.v2",
        "chapter_name": chapter_name, "source_file": source_file, "maximum_seconds": maximum_seconds,
        "status": "completed" if completed and not incomplete else "incomplete",
        "generation_status": "completed" if completed else "in_progress",
        "summary": {"committed_scenes": len(scenes), "incomplete_scenes": incomplete,
                    "consumed_chars": state["consumed_chars"], "source_chars": state["source_chars"]},
        "sequences": scenes, "performance": deepcopy(state["performance"]),
    }

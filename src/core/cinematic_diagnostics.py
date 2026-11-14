"""Keep adapter diagnostics inspectable without sending them to downstream nodes."""
from copy import deepcopy
import json

from . import cinematic_scene_planning as planning


_EXPLANATIONS = {
    "duplicate_entity_aspect": (
        "Two updates describe the same object/person and the same physical aspect.",
        "Keep one source-supported final state for this pair without removing independent facts.",
    ),
    "source_quote_mismatch": (
        "The supporting quotation is not an exact substring of this scene's source.",
        "Copy a contiguous quotation from the source without paraphrasing or changing punctuation.",
    ),
    "event_text_mismatch": (
        "A fragment differs from the original event at the indicated position.",
        "Copy the remaining event text exactly; compare the Unicode code points at the first difference.",
    ),
    "event_text_incomplete": (
        "The end of an event is not covered by any planning fragment.",
        "Add the remaining text using the same event identifier without removing any action.",
    ),
    "event_coverage": ("The plan does not cover all events.", "Reference every original event."),
    "event_order": ("Fragments do not follow the event order.", "Preserve the original order without returning to earlier events."),
    "event_reference": ("An event identifier is invalid.", "Use the original event identifiers, starting at 1."),
    "timing_total_mismatch": ("The reported total differs from the sum of beat durations and pauses.", "Recalculate the total without shortening actions."),
    "timing_nonfinite": ("A duration is not a finite number.", "Return finite numeric durations."),
    "reset_quote_mismatch": ("The context reset has no valid source quotation.", "Support the reset with the current source or retain the context."),
    "blank_fact_field": ("A continuity field is blank.", "Provide the fact and its source quotation."),
    "output_limit": ("The response reached the output limit.", "Check the budget shared by reasoning and JSON."),
    "request_failed": ("The review request failed; the precise cause is unavailable.", "Check the server and its logs; this code does not prove a token limit was reached."),
    "duration_exceeded": ("The timing estimate exceeds the allowed maximum, including any allowance.", "Split between actions without rushing or removing them."),
}


def explain(record):
    """Translate local codes only; never interpolate arbitrary exception/backend text."""
    result = deepcopy(record)
    reason = record.get("reason", "")
    if record.get("status") == "completed":
        message, action = "The plan passed structural validation; this does not imply narrative approval.", ""
    elif reason.startswith("schema_"):
        message, action = "The response does not match the requested JSON schema.", "Correct the indicated field according to the strict schema."
    else:
        message, action = _EXPLANATIONS.get(reason, (
            "The check could not be completed.", "Inspect the indicated check and field before treating the scene as approved.",
        ))
    result["explanation"] = message
    if reason == "event_text_mismatch" and "expected_codepoint" in record:
        def character(value):
            if value is None:
                return "end of text"
            return f"{json.dumps(chr(value), ensure_ascii=True)} (U+{value:04X})"

        result["difference"] = (
            f"Index {record['mismatch_offset']} (0-based): "
            f"expected {character(record['expected_codepoint'])}, "
            f"received {character(record['received_codepoint'])}."
        )
    if action:
        result["suggested_action"] = action
    return result


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
    """Build a sidecar from a checkpoint, including progress before a run completes."""
    review = state.get("narrative_review", {})
    attempts = [explain(record) for record in review.get("planning_diagnostics", [])]
    recoveries = {record["assessment_id"]: record["attempt"] for record in attempts if record["status"] == "completed"}
    for record in attempts:
        if record["status"] == "failed" and record["assessment_id"] in recoveries:
            record["resolved_by_attempt"] = recoveries[record["assessment_id"]]
    warnings = [explain(record) for record in review.get("warnings", [])]
    scenes = []
    committed = state.get("sequences", [])
    plans = state.get("scene_plans", [])
    for index, scene in enumerate(committed):
        saved_plan = plans[index] if index < len(plans) else None
        plan = saved_plan["plan"] if saved_plan else None
        incomplete = scene.get("review", {}).get("status") == "incomplete"
        # A missing plan must never be presented as approval, even in a partial report.
        status = "incomplete" if incomplete or not plan else "completed"
        scenes.append({
            "sequence": scene["sequence"], "status": status,
            "source_excerpt": scene["source"],
            "events": deepcopy(saved_plan["events"]) if saved_plan else scene["adaptation"]["event"],
            "timing": {"status": ("exceeded" if planning.duration_issues(plan, maximum_seconds) else "available") if plan else "incomplete",
                       "maximum_seconds": maximum_seconds,
                       "estimated_seconds": plan["timing"]["estimated_seconds"] if plan else None,
                       "seconds_with_allowance": (round(plan["timing"]["estimated_seconds"] * (
                           planning.SEQUENTIAL_TIMING_FACTOR if len(plan["timing"]["beats"]) > 1 else 1), 3)
                           if plan else None),
                       "beats": deepcopy(plan["timing"]["beats"]) if plan else [],
                       "pause_seconds": plan["timing"]["pause_seconds"] if plan else None},
            "review": deepcopy(scene.get("review", {"status": status})),
            "failed_checks": [w for w in warnings if w["sequence"] == scene["sequence"] and w["status"] == "incomplete"],
        })
    incomplete_count = sum(scene["status"] == "incomplete" for scene in scenes)
    return {
        "schema_version": "minimax-cinematic-diagnostics.v1",
        "chapter_name": chapter_name, "source_file": source_file,
        "stage": state.get("stage", "completed" if completed else "unknown"),
        "status": "incomplete" if not completed or incomplete_count else "completed",
        "generation_status": "completed" if completed else "in_progress",
        "summary": {"committed_scenes": len(scenes), "pending_scenes": len(state.get("pending_sequences", [])),
                    "incomplete_scenes": incomplete_count, "planning_attempts": len(attempts),
                    "failed_planning_attempts": sum(a["status"] == "failed" for a in attempts)},
        "sequences": scenes, "planning_attempts": attempts, "warnings": warnings,
        "narrative_choices": deepcopy(review.get("issues", [])),
    }

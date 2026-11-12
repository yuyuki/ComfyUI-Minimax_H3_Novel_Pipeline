"""Explicit cinematic inputs for generation tests."""
from copy import deepcopy


def chapter_input(refs, chapter_id="chapter"):
    record = next(c for c in refs["chapters"] if c["chapter_id"] == chapter_id)
    return {"chapter_name": chapter_id, "duration": 8.0, "source_file": record.get("source_file", ""),
            "sequences": deepcopy(refs["chapter_timelines"][chapter_id]["sequences"])}


def metadata_scene(client, model, row, period, args):
    from minimax_h3_novel_pipeline.core.pipeline_step3_generate import Scene
    seq = row["sequence"]
    return Scene(f"Sequence {seq['sequence']:03d}", seq["source"], seq["adaptation"]["event"],
                 "", [], [], [], False, str(seq["adaptation"]) + period, seq["sequence"])

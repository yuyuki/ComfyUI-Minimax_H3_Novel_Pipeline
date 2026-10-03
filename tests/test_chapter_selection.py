"""Tests for the shared Select Chapters node payload."""

from minimax_h3_novel_pipeline.chapter_selection import SelectChaptersNode, chapter_paths


def test_selection_output_contains_a_trimmed_path_list():
    result = SelectChaptersNode().run(" minimax_h3_novel/test.md\n\nminimax_h3_novel/test2.md ", "")

    assert result == ({"chapter_paths": ["minimax_h3_novel/test.md", "minimax_h3_novel/test2.md"]},)


def test_saved_chapter_is_returned_as_a_single_item_list():
    assert SelectChaptersNode().run("", "minimax_h3_novel/test.md") == (
        {"chapter_paths": ["minimax_h3_novel/test.md"]},
    )


def test_consumers_accept_the_structured_selection():
    assert chapter_paths({"chapter_paths": ["first.md", "second.md"]}) == "first.md\nsecond.md"

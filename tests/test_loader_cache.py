"""Global cache inspection must tolerate unused, unavailable file imports."""
import math

import pytest

from minimax_h3_novel_pipeline.core import configuration_snapshot, path_access
from minimax_h3_novel_pipeline.nodes.load_chapter_catalogs import LoadChapterCatalogsNode
from minimax_h3_novel_pipeline.nodes.load_cinematic_chapters import LoadCinematicChaptersNode
from minimax_h3_novel_pipeline.nodes.load_consolidated_references import LoadConsolidatedReferencesNode


@pytest.fixture(params=[
    (LoadChapterCatalogsNode, "chapter_references.json"),
    (LoadCinematicChaptersNode, "chapter.cinematic.json"),
    (LoadConsolidatedReferencesNode, "consolidated_references.json"),
])
def loader(request, tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, "storage_root", lambda kind: tmp_path)
    return request.param


@pytest.mark.parametrize("value", ["missing/references", "", "../outside"])
def test_unavailable_import_only_fails_when_executed(loader, value):
    cls, _ = loader
    assert math.isnan(cls.IS_CHANGED(value))
    with pytest.raises(ValueError):
        cls().run(value)


def test_cache_tracks_creation_edits_and_deletion(loader, tmp_path):
    cls, filename = loader
    folder = tmp_path / "references"
    assert math.isnan(cls.IS_CHANGED(str(folder)))
    folder.mkdir()
    file = folder / filename
    file.write_text("{}", encoding="utf-8")
    first = cls.IS_CHANGED(str(folder))
    assert first == cls.IS_CHANGED(str(folder))
    file.write_text('{"edited": true}', encoding="utf-8")
    assert first != cls.IS_CHANGED(str(folder))
    file.unlink()
    assert math.isnan(cls.IS_CHANGED(str(folder)))


def test_file_disappears_during_cache_read(loader, tmp_path, monkeypatch):
    cls, filename = loader
    file = tmp_path / filename
    file.write_text("{}", encoding="utf-8")

    def disappeared(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(configuration_snapshot, "file_digest", disappeared)
    assert math.isnan(cls.IS_CHANGED(str(file)))

"""Global cache inspection must tolerate unused, unavailable file imports."""
import hashlib
import math
import os

import pytest

from minimax_h3_novel_pipeline.core import configuration_snapshot, path_access
from minimax_h3_novel_pipeline.nodes.load_chapter_catalogs import LoadChapterCatalogsNode
from minimax_h3_novel_pipeline.nodes.load_cinematic_chapters import LoadCinematicChaptersNode
from minimax_h3_novel_pipeline.nodes.load_consolidated_references import LoadConsolidatedReferencesNode
from minimax_h3_novel_pipeline.nodes.load_character_evidence import LoadCharacterEvidenceNode
from minimax_h3_novel_pipeline.nodes.load_character_profiles import LoadCharacterProfilesNode
from minimax_h3_novel_pipeline.nodes.load_character_visual_profiles import LoadCharacterVisualProfilesNode
from minimax_h3_novel_pipeline.nodes.load_spatial_continuity import LoadSpatialContinuityNode
from tests.test_path_access import link_to


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


@pytest.fixture(params=[
    LoadCharacterEvidenceNode,
    LoadCharacterProfilesNode,
    LoadCharacterVisualProfilesNode,
    LoadSpatialContinuityNode,
])
def editable_loader(request, tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(path_access, "storage_root", lambda kind: output)
    return request.param, output


def test_editable_loader_hashes_all_bytes_and_detects_preserved_metadata(editable_loader):
    cls, output = editable_loader
    file = output / "reviewed.json"
    # Cross the shared hasher's block boundary and include BOM/CRLF bytes.
    contents = b'\xef\xbb\xbf{"notes":"' + b"a" * (1024 * 1024) + b'"}\r\n'
    file.write_bytes(contents)
    first = cls.IS_CHANGED(str(file))
    assert first == hashlib.sha256(contents).hexdigest()
    assert first == cls.IS_CHANGED(file.name)
    stat = file.stat()
    edited = contents[:-4] + b" \r\n "
    file.write_bytes(edited)
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert file.stat().st_size == stat.st_size
    assert cls.IS_CHANGED(str(file)) == hashlib.sha256(edited).hexdigest() != first
    file.unlink()
    assert math.isnan(cls.IS_CHANGED(str(file)))


@pytest.mark.parametrize("kind", ["absolute", "traversal", "directory_link"])
def test_editable_loader_rejects_escape_before_hashing(editable_loader, tmp_path, monkeypatch, kind):
    cls, output = editable_loader
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "reviewed.json").write_text("{}", encoding="utf-8")
    if kind == "directory_link":
        link_to(output / "linked", outside)
        value = str(output / "linked/reviewed.json")
    elif kind == "traversal":
        value = "../private/reviewed.json"
    else:
        value = str(outside / "reviewed.json")

    def forbidden(path):
        pytest.fail(f"External path reached file hashing: {path}")

    monkeypatch.setattr(configuration_snapshot, "file_digest", forbidden)
    assert math.isnan(cls.IS_CHANGED(value))


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
def test_editable_loader_tolerates_cache_read_errors(editable_loader, monkeypatch, error):
    cls, output = editable_loader
    file = output / "reviewed.json"
    file.write_text("{}", encoding="utf-8")

    def unavailable(path):
        raise error(path)

    monkeypatch.setattr(configuration_snapshot, "file_digest", unavailable)
    assert math.isnan(cls.IS_CHANGED(str(file)))

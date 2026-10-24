"""Manual editing, validation and cache invalidation without a model server."""
import json
import math
import os

import pytest

from minimax_h3_novel_pipeline.core import path_access, util
from minimax_h3_novel_pipeline.nodes.load_spatial_continuity import LoadSpatialContinuityNode as Loader
from minimax_h3_novel_pipeline.nodes.spatial_continuity import _FIELDS


@pytest.fixture
def saved_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(path_access, 'storage_root', lambda kind: tmp_path)
    fields = {k: [] if v['type'] == 'array' else 'unknown' for k, v in _FIELDS.items()}
    camera = {k: fields.pop(k) for k in list(fields) if k.startswith('camera_')}
    plan = {'schema_version': 'minimax-spatial-continuity.v2', 'source_digest': 'original',
            'chapters': {'chapter': {'sequences': [{'sequence': 1, 'camera': camera, **fields}]}}}
    registry = {'schema_version': util.REGISTRY_SCHEMA, 'source_digest': 'original',
                'chapter_timelines': {'chapter': {'sequences': [{'sequence': 1}]}}}
    path = tmp_path / 'spatial_continuity.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    return path, plan, registry


def test_file_folder_and_renamed_copy_preserve_edits(saved_plan):
    path, plan, registry = saved_plan
    plan['chapters']['chapter']['sequences'][0]['camera']['camera_position'] = 'À gauche'
    path.write_text(json.dumps(plan, ensure_ascii=False), encoding='utf-8-sig')
    for value in (str(path), str(path.parent), path.name):
        loaded, text, summary = Loader().run(value, registry)
        assert loaded == json.loads(text) == plan
        assert '1 sequences' in summary
    renamed = path.with_name('corrected.json')
    path.rename(renamed)
    assert Loader().run(str(renamed), registry)[0] == plan


def test_cache_hash_detects_same_size_same_timestamp_edits(saved_plan):
    path, _, registry = saved_plan
    old = Loader.IS_CHANGED(str(path), registry)
    assert old == Loader.IS_CHANGED(str(path), registry)
    stat = path.stat()
    path.write_text(path.read_text().replace('unknown', 'changed'), encoding='utf-8')
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert Loader.IS_CHANGED(str(path), registry) != old
    path.unlink()
    assert math.isnan(Loader.IS_CHANGED(str(path), registry))
    with pytest.raises(ValueError, match='does not exist'):
        Loader().run(str(path), registry)


def test_invalid_json_and_stale_plan_are_rejected(saved_plan):
    path, plan, registry = saved_plan
    path.write_text('{\n broken')
    with pytest.raises(ValueError, match='line 2, column'):
        Loader().run(str(path), registry)
    path.write_text(json.dumps(plan))
    registry['source_digest'] = 'new'
    with pytest.raises(ValueError, match='stale'):
        Loader().run(str(path), registry)
    registry['source_digest'] = 'original'
    plan['chapters']['chapter']['sequences'][0]['camera']['camera_position'] = 42
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='invalid spatial output'):
        Loader().run(str(path), registry)


def test_paths_are_confined(saved_plan):
    path, _, registry = saved_plan
    outside = path.parent.parent / (path.parent.name + '-outside.json')
    outside.write_text(path.read_text())
    link = path.parent / 'link.json'
    link.symlink_to(outside)
    for value in (str(outside), '../outside.json', str(link)):
        with pytest.raises(ValueError):
            Loader().run(value, registry)
        assert math.isnan(Loader.IS_CHANGED(value, registry))


@pytest.mark.parametrize("change", ["root_extra", "chapter_extra", "camera_missing", "sequence_type", "entity_enum"])
def test_bundled_schema_rejects_invalid_manual_edits(saved_plan, change):
    from jsonschema import Draft202012Validator
    from minimax_h3_novel_pipeline.nodes.spatial_continuity import SPATIAL_SCHEMA, _ENTITY
    path, plan, registry = saved_plan
    sequence = plan['chapters']['chapter']['sequences'][0]
    if change == 'root_extra':
        plan['typo'] = 1
    elif change == 'chapter_extra':
        plan['chapters']['chapter']['typo'] = 1
    elif change == 'camera_missing':
        del sequence['camera']['camera_axis']
    elif change == 'sequence_type':
        sequence['sequence'] = True
    else:
        entity = {k: 'unknown' for k in _ENTITY}
        entity.update(kind='invalid', source_sequence=1)
        sequence['initial_entities'] = [entity]
    assert not Draft202012Validator(SPATIAL_SCHEMA).is_valid(plan)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='invalid spatial output'):
        Loader().run(str(path), registry)


def test_document_schema_url_is_not_fetched(saved_plan):
    path, plan, registry = saved_plan
    plan['$schema'] = 'https://invalid.example/no-network'
    path.write_text(json.dumps(plan))
    assert Loader().run(str(path), registry)[0] == plan

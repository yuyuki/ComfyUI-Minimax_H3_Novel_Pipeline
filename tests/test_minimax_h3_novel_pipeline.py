"""Regression checks for scaffold layout, node loading, and bundled scripts."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plugin():
    # Match ComfyUI's file-based loading, including arbitrary checkout names.
    spec = importlib.util.spec_from_file_location(
        "scaffold_test_plugin", ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module}):
        spec.loader.exec_module(module)
        yield module


def test_root_registration_and_frontend(plugin):
    assert set(plugin.NODE_CLASS_MAPPINGS) == {
        "LMStudioConfigurationNode", "SelectChaptersNode", "ExtractChapterReferencesNode",
        "LoadChapterCatalogsNode", "LoadConsolidatedReferencesNode",
        "ConsolidateReferencesNode", "GenerateH3PromptsNode",
    }
    assert set(plugin.NODE_DISPLAY_NAME_MAPPINGS) == set(plugin.NODE_CLASS_MAPPINGS)
    assert (ROOT / plugin.WEB_DIRECTORY / "minimax_h3_novel.js").is_file()
    for cls in plugin.NODE_CLASS_MAPPINGS.values():
        assert callable(getattr(cls(), cls.FUNCTION))
        assert "required" in cls.INPUT_TYPES()


@pytest.mark.parametrize("step", ["extract", "consolidate", "generate"])
def test_bundled_steps_and_real_client_transport(plugin, step):
    package = sys.modules[plugin.NODE_CLASS_MAPPINGS["LMStudioConfigurationNode"].__module__]
    pipeline = sys.modules[package.__package__ + ".lmstudio_pipeline"]
    module = pipeline.load(step)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as transport:
        client = module.make_client("http://127.0.0.1:1234/v1", "test-key", http_client=transport)
        assert client._client is transport
        client.close()


def test_installed_package_import_outside_checkout(tmp_path):
    # Run independently of pytest's treatment of the root __init__.py.
    result = subprocess.run(
        [sys.executable, "-c",
         "from pathlib import Path; import minimax_h3_novel_pipeline as p; "
         "assert len(p.NODE_CLASS_MAPPINGS) == 7; "
         "assert (Path(p.WEB_DIRECTORY) / 'minimax_h3_novel.js').is_file(); "
         "from minimax_h3_novel_pipeline.lmstudio_pipeline import load; "
         "[load(s) for s in ('extract', 'consolidate', 'generate')]"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

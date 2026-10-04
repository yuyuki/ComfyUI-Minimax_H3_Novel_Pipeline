from pathlib import Path
import runpy
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = runpy.run_path(str(ROOT / "tools" / "generate_architecture.py"))


def test_generated_architecture_map_is_current() -> None:
    result = subprocess.run(
        [sys.executable, "tools/generate_architecture.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_dependency_graph_omits_isolated_modules_but_keeps_leaf_modules() -> None:
    module_info = GENERATOR["ModuleInfo"]
    modules = [
        module_info("source", "src/source.py", ("leaf",), ()),
        module_info("leaf", "src/leaf.py", (), ()),
        module_info("isolated", "src/isolated.py", (), ()),
    ]
    rendered = GENERATOR["render_architecture"](modules)
    graph = rendered.split("## Internal module dependencies", 1)[1].split("```mermaid", 1)[1].split("```", 1)[0]
    assert 'm_source["source"]' in graph
    assert 'm_leaf["leaf"]' in graph
    assert "m_source --> m_leaf" in graph
    assert "m_isolated" not in graph
    assert "`src/isolated.py`" in rendered
    assert "Modules: **3**. Internal dependency edges: **1**." in rendered


def test_scan_reads_registered_names_without_importing_plugin(tmp_path, monkeypatch) -> None:
    scan_modules = GENERATOR["scan_modules"]
    monkeypatch.setitem(scan_modules.__globals__, "ROOT", tmp_path)
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "__init__.py").write_text(
        'raise RuntimeError("Must not import plugin")\n'
        'from . import nodes\n'
        'NODE_CLASS_MAPPINGS = {}\n'
        'NODE_DISPLAY_NAME_MAPPINGS = {}\n'
        'NODE_CLASS_MAPPINGS.update({"FirstId": nodes.FirstNode, "SecondId": nodes.SecondNode})\n'
        'NODE_DISPLAY_NAME_MAPPINGS.update({"FirstId": "First UI name", "SecondId": \'Second "UI" name\'})\n',
        encoding="utf-8",
    )
    (source_dir / "nodes.py").write_text(
        "class FirstNode: pass\nclass SecondNode: pass\nclass UnregisteredNode: pass\n", encoding="utf-8"
    )
    modules = scan_modules(source_dir)
    node_module = next(module for module in modules if module.name == "nodes")
    assert node_module.comfyui_nodes == (("FirstNode", "First UI name"), ("SecondNode", 'Second "UI" name'))
    graph = GENERATOR["render_architecture"](modules)
    assert 'm_nodes["nodes<br/>First UI name<br/>Second #quot;UI#quot; name"]' in graph


def test_pipeline_labels_follow_comfyui_registration() -> None:
    modules = GENERATOR["scan_modules"]()
    rendered = GENERATOR["render_architecture"](modules)
    pipeline = rendered.split("## Pipeline flow", 1)[1].split("```mermaid", 1)[1].split("```", 1)[0]
    assert 'config["LM Studio Configuration"]' in pipeline
    assert 'extract["Extract Chapter References"]' in pipeline
    assert 'consolidate["Consolidate References"]' in pipeline
    assert 'generate["Generate H3 Prompts"]' in pipeline
    assert 'normalize["Novel Cinematic Simplifier"]' in pipeline
    assert 'state["Narrative Continuity<br/>' in pipeline
    assert 'm_narrative_nodes["narrative_nodes<br/>Narrative Continuity<br/>Novel Cinematic Simplifier"]' in rendered
    module_info = GENERATOR["ModuleInfo"]
    renamed = module_info("extract", "src/extract.py", (), (), (("ExtractChapterReferencesNode", "Renamed Extract"),))
    assert 'extract["Renamed Extract"]' in GENERATOR["render_architecture"]([renamed])

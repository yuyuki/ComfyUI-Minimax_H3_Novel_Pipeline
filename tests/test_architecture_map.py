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
    assert 'evidence_json --> evidence_loader["Load Character Evidence"]' in pipeline
    assert 'evidence --> profiles["Consolidate Character Profiles"]' in pipeline
    assert 'evidence_loader --> profiles' in pipeline
    assert 'profiles_json --> profiles_loader["Load Character Profiles"]' in pipeline
    assert 'profiles_loader --> visual_profiles' in pipeline
    assert 'profiles --> visual_profiles["Character Visual Profile Generator"]' in pipeline
    assert 'visual_json --> visual_loader["Load Character Visual Profiles"]' in pipeline
    assert "Novel Cinematic Simplifier" not in rendered
    assert "Narrative Continuity" not in rendered
    assert "narrative_nodes" not in rendered
    module_info = GENERATOR["ModuleInfo"]
    renamed = module_info("extract", "src/extract.py", (), (), (("ExtractChapterReferencesNode", "Renamed Extract"),))
    assert 'extract["Renamed Extract"]' in GENERATOR["render_architecture"]([renamed])


def test_scan_nested_packages_resolves_relative_imports(tmp_path, monkeypatch) -> None:
    scan = GENERATOR["scan_modules"]
    monkeypatch.setitem(scan.__globals__, "ROOT", tmp_path)
    source = tmp_path / "src"
    for package in (source, source / "nodes", source / "core"):
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
    (source / "__init__.py").write_text("from . import nodes\n", encoding="utf-8")
    (source / "nodes" / "__init__.py").write_text("from .worker import Worker\n", encoding="utf-8")
    (source / "nodes" / "worker.py").write_text(
        "from ..core import worker\nfrom ..core.util import helper\nclass Worker: pass\n", encoding="utf-8"
    )
    (source / "core" / "worker.py").write_text("from . import util\n", encoding="utf-8")
    (source / "core" / "util.py").write_text("def helper(): pass\n", encoding="utf-8")
    modules = {module.name: module for module in scan(source)}
    assert modules["package_init"].dependencies == ("nodes",)
    assert modules["nodes"].dependencies == ("nodes.worker",)
    assert modules["nodes.worker"].dependencies == ("core.util", "core.worker")
    assert modules["core.worker"].dependencies == ("core.util",)
    assert modules["core.util"].path == "src/core/util.py"

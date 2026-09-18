from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_generated_architecture_map_is_current() -> None:
    result = subprocess.run(
        [sys.executable, "tools/generate_architecture.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

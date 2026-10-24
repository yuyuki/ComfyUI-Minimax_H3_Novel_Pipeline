"""Filesystem boundaries for workflow paths and generated pipeline files."""
from __future__ import annotations

from pathlib import Path
import re


def confined_path(value: str | Path, root: Path) -> Path:
    """Resolve a path inside root, rejecting traversal and links that escape it."""
    path = Path(value)
    if not str(value).strip() or ".." in path.parts:
        raise ValueError("Path must be non-empty and must not contain '..'.")
    if (path.drive or path.root) and not path.is_absolute():
        raise ValueError("Drive-relative and rooted-relative paths are not allowed.")
    # Apply Win32 restrictions on every platform so workflows remain portable.
    # In particular, colons must never select an NTFS alternate data stream.
    parts = path.parts[1:] if path.anchor else path.parts
    for part in parts:
        name = part.split(".")[0].upper()
        if (re.search(r'[<>:"|?*\\\x00-\x1f]', part)
                or part.endswith((".", " "))
                or name in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
                or re.fullmatch(r"(?:COM|LPT)[1-9¹²³]", name)):
            raise ValueError("Invalid filesystem path component.")
    root = root.resolve()
    candidate = path if path.is_absolute() else root / path
    # Reject external absolute/UNC paths before resolving or probing them.
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path must stay inside {root}.")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path must stay inside {root}.")
    return resolved


_FALLBACK_BASE = Path.cwd().resolve()


def storage_root(kind: str) -> Path:
    """Get a server-owned root; workflow inputs cannot override it."""
    if kind not in {"input", "output"}:
        raise ValueError("Unknown storage root.")
    try:
        import folder_paths
    except ImportError:
        root = _FALLBACK_BASE / kind
    else:
        root = Path(getattr(folder_paths, f"get_{kind}_directory")())
    root = root.resolve()
    return confined_path(root / "minimax_h3_novel", root) if kind == "output" else root


def input_path(value: str | Path) -> Path:
    return confined_path(value, storage_root("input"))


def output_path(value: str | Path) -> Path:
    return confined_path(value, storage_root("output"))

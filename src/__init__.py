"""ComfyUI plugin registration for MiniMax H3 novel pipeline nodes.

This module exposes `NODE_CLASS_MAPPINGS` for ComfyUI to import and register. The actual
node implementations live in `nodes.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from . import nodes as _nodes
from .route_access import require_local_request
from .path_access import confined_path

# Provide mappings expected by ComfyUI custom node loader:
# - NODE_CLASS_MAPPINGS: dict name -> class
# - NODE_DISPLAY_NAME_MAPPINGS: optional human-friendly labels
NODE_CLASS_MAPPINGS: Dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}
# Source checkouts keep web assets at the root; wheels bundle them here.
_web_root = Path(__file__).resolve().parent / "web"
if not _web_root.is_dir():
    _web_root = Path(__file__).resolve().parents[1] / "web"
WEB_DIRECTORY = str(_web_root / "js")


def _build_mappings() -> None:
    """Populate mapping dicts and log diagnostics for debugging ComfyUI loading."""
    try:
        NODE_CLASS_MAPPINGS.update(
            {
                "ExtractChapterReferencesNode": _nodes.ExtractChapterReferencesNode,
                "LMStudioConfigurationNode": _nodes.LMStudioConfigurationNode,
                "LoadChapterCatalogsNode": _nodes.LoadChapterCatalogsNode,
                "LoadConsolidatedReferencesNode": _nodes.LoadConsolidatedReferencesNode,
                "ConsolidateReferencesNode": _nodes.ConsolidateReferencesNode,
                "GenerateH3PromptsNode": _nodes.GenerateH3PromptsNode,
            }
        )
        NODE_DISPLAY_NAME_MAPPINGS.update(
            {
                "ExtractChapterReferencesNode": "Extract Chapter References",
                "LMStudioConfigurationNode": "LM Studio Configuration",
                "LoadChapterCatalogsNode": "Load Chapter Catalogs",
                "LoadConsolidatedReferencesNode": "Load Consolidated References",
                "ConsolidateReferencesNode": "Consolidate References",
                "GenerateH3PromptsNode": "Generate H3 Prompts",
            }
        )
    except Exception as e:
        print(f"[minimax_h3_novel] ERROR building NODE_CLASS_MAPPINGS: {e}")


_build_mappings()


def _register_upload_route() -> None:
    """Register the chapter upload endpoint used by the browser picker."""
    try:
        from aiohttp import web
        from server import PromptServer
        import folder_paths

        allowed = {".txt", ".md", ".markdown", ".pdf"}
        def chapter_root() -> Path:
            return confined_path("minimax_h3_novel", Path(folder_paths.get_input_directory()))

        chapter_root().mkdir(parents=True, exist_ok=True)

        @PromptServer.instance.routes.post("/minimax_h3_novel/upload")
        async def upload_chapters(request):
            require_local_request(request)
            input_root = chapter_root()
            reader = await request.multipart()
            uploaded = []
            while True:
                part = await reader.next()
                if part is None:
                    break
                if not part.filename:
                    continue
                filename = Path(part.filename).name
                if Path(filename).suffix.lower() not in allowed:
                    return web.json_response(
                        {"error": f"Unsupported chapter type: {filename}"}, status=400
                    )
                stem, suffix = Path(filename).stem, Path(filename).suffix
                counter = 0
                while True:
                    candidate = filename if counter == 0 else f"{stem}_{counter}{suffix}"
                    try:
                        target = confined_path(candidate, input_root)
                    except ValueError as exc:
                        return web.json_response({"error": str(exc)}, status=400)
                    try:
                        handle = target.open("xb")
                        break
                    except FileExistsError:
                        counter += 1
                filename = target.name
                with handle:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        handle.write(chunk)
                uploaded.append(f"minimax_h3_novel/{filename}")
            return web.json_response({"files": uploaded})

        @PromptServer.instance.routes.get("/minimax_h3_novel/chapters")
        async def list_chapters(request):
            require_local_request(request)
            input_root = chapter_root()
            files = sorted(
                (path for path in input_root.iterdir()
                 if path.is_file() and path.suffix.lower() in allowed),
                key=lambda path: path.name.lower(),
            )
            return web.json_response({
                "files": [f"minimax_h3_novel/{path.name}" for path in files]
            })

        @PromptServer.instance.routes.delete("/minimax_h3_novel/chapters")
        async def delete_chapter(request):
            """Delete one file previously uploaded through the chapter picker."""
            require_local_request(request)
            try:
                input_root = chapter_root()
                data = await request.json()
                saved_path = str(data.get("file", ""))
                prefix = "minimax_h3_novel/"
                if not saved_path.startswith(prefix):
                    raise ValueError("Invalid saved chapter path")
                filename = Path(saved_path[len(prefix):]).name
                if filename != saved_path[len(prefix):] or Path(filename).suffix.lower() not in allowed:
                    raise ValueError("Invalid saved chapter path")
                target = input_root / filename
                confined_path(target, input_root)
                # Resolving prevents a crafted filename from escaping the upload folder.
                if target.resolve().parent != input_root.resolve() or not target.is_file():
                    return web.json_response({"error": "Saved chapter not found"}, status=404)
                target.unlink()
                return web.json_response({"deleted": f"minimax_h3_novel/{filename}"})
            except (ValueError, json.JSONDecodeError) as exc:
                return web.json_response({"error": str(exc)}, status=400)

        @PromptServer.instance.routes.post("/minimax_h3_novel/lmstudio-settings")
        async def save_lmstudio_settings(request):
            """Receive the local ComfyUI setting without ever returning/logging it."""
            require_local_request(request)
            try:
                from . import lmstudio_settings

                data = await request.json()
                api_key = data.get("api_key", "")
                if not isinstance(api_key, str):
                    raise ValueError("api_key must be a string")
                lmstudio_settings.set_api_key(api_key)
                return web.json_response({"configured": bool(api_key.strip())})
            except (ValueError, json.JSONDecodeError) as exc:
                return web.json_response({"error": str(exc)}, status=400)

        print("[minimax_h3_novel] chapter upload route registered")
    except Exception as exc:
        # Importing the package outside ComfyUI (for tests/tools) should still
        # work; the picker simply will not be available in that environment.
        print(f"[minimax_h3_novel] chapter upload route unavailable: {exc}")


_register_upload_route()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

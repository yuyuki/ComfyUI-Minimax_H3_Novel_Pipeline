"""Run with python -m unittest discover -s tests (no ComfyUI required)."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch


ROOT = Path(__file__).resolve().parents[1] / "src" / "minimax_h3_novel_pipeline"


class Forbidden(Exception):
    def __init__(self, *, reason):
        super().__init__(reason)


class RouteAccessTests(unittest.TestCase):
    def setUp(self):
        self.routes = {}

        def register(method):
            def route(path):
                def decorate(handler):
                    self.routes[method, path] = handler
                    return handler
                return decorate
            return route

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        fake_nodes = ModuleType("access_test_plugin.nodes")
        for name in ("ExtractChapterReferencesNode", "LMStudioConfigurationNode",
                     "LoadChapterCatalogsNode", "LoadConsolidatedReferencesNode",
                     "ConsolidateReferencesNode", "GenerateH3PromptsNode", "SelectH3SceneNode"):
            setattr(fake_nodes, name, type(name, (), {}))
        web = SimpleNamespace(HTTPForbidden=Forbidden, json_response=lambda data, **kwargs: data)
        routes = SimpleNamespace(**{method: register(method) for method in ("post", "get", "delete")})
        modules = {
            "aiohttp": SimpleNamespace(web=web),
            "server": SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(routes=routes))),
            "folder_paths": SimpleNamespace(get_input_directory=lambda: self.temp.name),
            "access_test_plugin.nodes": fake_nodes,
        }
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        spec = importlib.util.spec_from_file_location("access_test_plugin", ROOT / "__init__.py",
                                                     submodule_search_locations=[str(ROOT)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        with patch("builtins.print"):
            spec.loader.exec_module(package)
        self.guard = package.require_local_request

    def request(self, peer="127.0.0.1", host="localhost:8188", **headers):
        return SimpleNamespace(
            transport=SimpleNamespace(get_extra_info=lambda name: (peer, 12345)),
            scheme="http", host=host,
            headers={"X-MiniMax-H3-Request": "1", **headers},
            json=Mock(side_effect=AssertionError("Body must not be read")),
            multipart=Mock(side_effect=AssertionError("Body must not be read")),
        )

    def test_direct_local_browser(self):
        for peer, host in [("127.0.0.1", "localhost:8188"), ("::1", "[::1]:8188"),
                           ("::ffff:127.0.0.1", "127.0.0.1:8188")]:
            with self.subTest(peer=peer):
                self.guard(self.request(peer, host, Origin=f"http://{host}",
                                        **{"Sec-Fetch-Site": "same-origin"}))
        self.guard(self.request())  # Same-origin GET may omit Origin.

    def test_blocked_requests_have_no_route_side_effects(self):
        requests = [
            self.request("192.168.1.2"), self.request("8.8.8.8"),
            self.request("::ffff:192.168.1.2"), self.request(host="evil.example:8188"),
            self.request(Origin="http://evil.example"), self.request(Origin="null"),
            self.request(Origin="http://localhost:9999"),
            self.request(Origin="http://localhost:8188@evil.example"),
            self.request(**{"X-Forwarded-For": "127.0.0.1"}),
            self.request(**{"Forwarded": "for=127.0.0.1"}),
            self.request(**{"X-Real-IP": "127.0.0.1"}),
            self.request(**{"Sec-Fetch-Site": "cross-site"}),
            self.request(**{"X-MiniMax-H3-Request": ""}),
        ]
        missing_transport = self.request()
        missing_transport.transport = None
        requests.append(missing_transport)
        chapter = Path(self.temp.name) / "minimax_h3_novel" / "chapter.txt"
        chapter.write_text("keep me", encoding="utf-8")
        self.assertEqual(len(self.routes), 4)
        for route, handler in self.routes.items():
            for request in requests:
                with self.subTest(route=route, headers=request.headers, host=request.host):
                    with self.assertRaises(Forbidden):
                        asyncio.run(handler(request))
                    request.json.assert_not_called()
                    request.multipart.assert_not_called()
                    self.assertEqual(chapter.read_text(encoding="utf-8"), "keep me")

    def test_local_listing_still_works(self):
        chapter = Path(self.temp.name) / "minimax_h3_novel" / "chapter.txt"
        chapter.write_text("chapter", encoding="utf-8")
        result = asyncio.run(self.routes["get", "/minimax_h3_novel/chapters"](self.request()))
        self.assertEqual(result, {"files": ["minimax_h3_novel/chapter.txt"]})

    def test_local_mutations_still_work(self):
        request = self.request(Origin="http://localhost:8188")
        part = SimpleNamespace(filename="chapter.txt", read_chunk=AsyncMock(side_effect=[b"chapter", b""]))
        request.multipart = AsyncMock(return_value=SimpleNamespace(next=AsyncMock(side_effect=[part, None])))
        result = asyncio.run(self.routes["post", "/minimax_h3_novel/upload"](request))
        self.assertEqual(result, {"files": ["minimax_h3_novel/chapter.txt"]})
        chapter = Path(self.temp.name) / "minimax_h3_novel" / "chapter.txt"
        self.assertEqual(chapter.read_bytes(), b"chapter")

        request.json = AsyncMock(return_value={"file": "minimax_h3_novel/chapter.txt"})
        asyncio.run(self.routes["delete", "/minimax_h3_novel/chapters"](request))
        self.assertFalse(chapter.exists())

        request.json = AsyncMock(return_value={"api_key": "test-only"})
        result = asyncio.run(self.routes["post", "/minimax_h3_novel/lmstudio-settings"](request))
        self.assertEqual(result, {"configured": True})
        settings = sys.modules["access_test_plugin.lmstudio_settings"]
        self.assertEqual(settings.get_api_key(), "test-only")
        settings.set_api_key("")

    def test_upload_collision_preserves_existing_file(self):
        chapter = Path(self.temp.name) / "minimax_h3_novel" / "chapter.txt"
        chapter.write_bytes(b"original")
        request = self.request()
        part = SimpleNamespace(filename="chapter.txt", read_chunk=AsyncMock(side_effect=[b"new", b""]))
        request.multipart = AsyncMock(return_value=SimpleNamespace(next=AsyncMock(side_effect=[part, None])))
        result = asyncio.run(self.routes["post", "/minimax_h3_novel/upload"](request))
        self.assertEqual(result, {"files": ["minimax_h3_novel/chapter_1.txt"]})
        self.assertEqual(chapter.read_bytes(), b"original")
        self.assertEqual(chapter.with_name("chapter_1.txt").read_bytes(), b"new")

    def test_upload_rejects_windows_special_filenames_before_reading_bytes(self):
        for filename in ("NUL.txt", "chapter.txt:stream.txt", "bad?.txt"):
            with self.subTest(filename=filename):
                request = self.request()
                part = SimpleNamespace(filename=filename, read_chunk=AsyncMock())
                request.multipart = AsyncMock(return_value=SimpleNamespace(next=AsyncMock(side_effect=[part, None])))
                result = asyncio.run(self.routes["post", "/minimax_h3_novel/upload"](request))
                self.assertIn("error", result)
                part.read_chunk.assert_not_called()


if __name__ == "__main__":
    unittest.main()

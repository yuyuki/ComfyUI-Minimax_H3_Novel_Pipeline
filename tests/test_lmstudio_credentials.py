"""Credential destination regressions; no ComfyUI or live LM Studio required."""
import ast
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx


ROOT = Path(__file__).resolve().parents[1] / "src"
DEFAULT_URL = "http://127.0.0.1:1234/v1"


class CredentialTests(unittest.TestCase):
    def setUp(self):
        package = ModuleType("credential_test_plugin")
        package.__path__ = [str(ROOT)]
        modules = patch.dict(sys.modules, {package.__name__: package})
        modules.start()
        self.addCleanup(modules.stop)
        self.settings = importlib.import_module("credential_test_plugin.lmstudio_settings")
        self.pipeline = importlib.import_module("credential_test_plugin.lmstudio_pipeline")
        self.config = importlib.import_module("credential_test_plugin.lmstudio_config")
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.settings.set_api_key("operator-secret")

    def test_untrusted_urls_rejected_before_secret_lookup_or_client_creation(self):
        urls = ["https://attacker.example/v1", "http://127.0.0.1:4321/v1",
                "http://127.0.0.1:1234/other", "http://127.0.0.1:1234/v1/../other",
                "http://127.0.0.1:1234@attacker.example/v1",
                "http://127.0.0.1:1234.attacker.example/v1",
                DEFAULT_URL + "?target=evil", DEFAULT_URL + "#fragment",
                "file:///etc/passwd", "http://127.0.0.1:bad/v1",
                "http://127.0.0.1:1234\\@attacker.example/v1",
                "http://127.0.0.1:\n1234/v1", ""]
        module = SimpleNamespace(make_client=Mock(), select_model=Mock())
        with patch.object(self.settings, "get_api_key") as get_key:
            for url in urls:
                with self.subTest(url=url):
                    with self.assertRaises(ValueError):
                        self.config.LMStudioConfigurationNode().run(url)
                    with self.assertRaises(ValueError):
                        self.pipeline.make_client_and_model(module, url)
            get_key.assert_not_called()
            module.make_client.assert_not_called()

    def test_configuration_does_not_expose_secret(self):
        config, status = self.config.LMStudioConfigurationNode().run(DEFAULT_URL + "/")
        self.assertEqual(config["api_url"], DEFAULT_URL)
        self.assertNotIn("model", config)
        self.assertNotIn("model", self.config.LMStudioConfigurationNode.INPUT_TYPES().get("optional", {}))
        self.assertNotIn("model", self.config.LMStudioConfigurationNode.INPUT_TYPES()["required"])
        self.assertNotIn("api_key", config)
        self.assertNotIn("operator-secret", json.dumps([config, status]))

    def test_operator_can_authorize_remote_endpoint(self):
        url = "https://trusted.example/lm/v1"
        os.environ["MINIMAX_H3_LMSTUDIO_BASE_URL"] = url
        self.assertEqual(self.settings.validate_api_url(url + "/"), url)
        with self.assertRaises(ValueError):
            self.settings.validate_api_url(DEFAULT_URL)
        with self.assertRaises(ValueError):
            self.settings.validate_api_url("http://trusted.example/lm/v1")

    def test_cached_configuration_is_revalidated(self):
        config, _ = self.config.LMStudioConfigurationNode().run(DEFAULT_URL)
        os.environ["MINIMAX_H3_LMSTUDIO_BASE_URL"] = "https://new.example/v1"
        module = SimpleNamespace(make_client=Mock())
        with self.assertRaises(ValueError):
            self.pipeline.make_client_and_model(module, config["api_url"])
        module.make_client.assert_not_called()

    def test_authenticated_request_does_not_follow_redirects(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(307, headers={"Location": DEFAULT_URL + "/other"})

        real_client = httpx.Client

        def transport_factory(**kwargs):
            self.assertFalse(kwargs["follow_redirects"])
            self.assertFalse(kwargs["trust_env"])
            return real_client(transport=httpx.MockTransport(respond), **kwargs)

        def make_client(url, key, *, http_client):
            return SimpleNamespace(url=url, key=key, transport=http_client)

        def select_model(client, model):
            self.assertIsNone(model)
            response = client.transport.get(client.url + "/models",
                                            headers={"Authorization": "Bearer " + client.key})
            self.assertEqual(response.status_code, 307)
            return "selected-model"

        module = SimpleNamespace(make_client=make_client, select_model=select_model)
        with patch("httpx.Client", side_effect=transport_factory):
            client, model = self.pipeline.make_client_and_model(module, DEFAULT_URL)
        self.addCleanup(client.transport.close)
        self.assertEqual(model, "selected-model")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].headers["Authorization"], "Bearer operator-secret")

    def test_all_bundled_factories_forward_secure_transport(self):
        # Execute the actual factories without importing optional pipeline dependencies.
        for name in self.pipeline._SCRIPT_FILES.values():
            with self.subTest(step=name):
                tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
                factory = next(node for node in tree.body
                               if isinstance(node, ast.FunctionDef) and node.name == "make_client")
                constructor = Mock()
                namespace = {"OpenAI": constructor}
                exec(compile(ast.Module(body=[factory], type_ignores=[]), name, "exec"), namespace)  # noqa: S102
                transport = object()
                namespace["make_client"](DEFAULT_URL, "test-key", http_client=transport)
                self.assertIs(constructor.call_args.kwargs["http_client"], transport)


if __name__ == "__main__":
    unittest.main()

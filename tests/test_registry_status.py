"""Registry reporting is tested without network requests or publishing."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest


spec = importlib.util.spec_from_file_location(
    "registry_status", Path(__file__).resolve().parents[1] / "tools" / "check_registry_status.py"
)
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


@pytest.mark.parametrize("state,deprecated,expected", [
    ("NodeVersionStatusActive", False, 0),
    ("NodeVersionStatusActive", True, 1),
    ("NodeVersionStatusFlagged", False, 1),
    ("NodeVersionStatusBanned", False, 1),
    ("UnexpectedStatus", False, 1),
])
def test_terminal_status(monkeypatch, state, deprecated, expected):
    fetch = Mock(return_value={"version": "1.0.0", "status": state, "deprecated": deprecated})
    monkeypatch.setattr(status, "fetch_version", fetch)
    assert status.check("node", "1.0.0", 2, 0)[0] == expected
    fetch.assert_called_once()


def test_pending_then_active(monkeypatch):
    monkeypatch.setattr(status, "fetch_version", Mock(side_effect=[
        {"status": "NodeVersionStatusPending"}, {"status": "NodeVersionStatusActive"},
    ]))
    sleep = Mock()
    monkeypatch.setattr(status.time, "sleep", sleep)
    assert status.check("node", "1.0.0", 2, 30)[0] == 0
    sleep.assert_called_once_with(30)


@pytest.mark.parametrize("response", [
    {"status": "NodeVersionStatusPending"},
    HTTPError("url", 404, "missing", {}, None),
    HTTPError("url", 429, "rate limit", {}, None),
    URLError("offline"),
])
def test_unconfirmed_is_not_success(monkeypatch, response):
    fetch = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
    monkeypatch.setattr(status, "fetch_version", fetch)
    assert status.check("node", "1.0.0", 2, 0)[0] == 1
    assert fetch.call_count == 2


def test_fetch_validates_version(monkeypatch):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps({"version": "wrong", "status": "NodeVersionStatusActive"})
    fetch = Mock(return_value=response)
    monkeypatch.setattr(status, "urlopen", fetch)
    with pytest.raises(ValueError, match="unexpected version"):
        status.fetch_version("node", "1.0.0")
    assert fetch.call_args.args[0].endswith("/1.0.0")


def test_fetch_reasons_from_list_for_exact_version(monkeypatch):
    from io import StringIO

    fetch = Mock(side_effect=[
        StringIO(json.dumps({"version": "1.0.0", "status": "NodeVersionStatusFlagged"})),
        StringIO(json.dumps([
            {"version": "2.0.0", "status_reason": "wrong release"},
            {"version": "1.0.0", "status_reason": "test finding"},
        ])),
    ])
    monkeypatch.setattr(status, "urlopen", fetch)
    assert status.fetch_version("node", "1.0.0")["status_reason"] == "test finding"
    assert fetch.call_args.args[0].endswith("/versions?include_status_reason=true")


def test_flag_report_contains_review_draft_and_escapes_findings():
    report = status.report({"name": "node", "version": "1.0.0"}, "publisher", {
        "status": "NodeVersionStatusFlagged", "id": "version-id",
        "status_reason": "<script>untrusted</script>",
    })
    assert "Version ID: version-id" in report
    assert "Publisher: publisher" in report
    assert "issues/new" in report
    assert "<script>" not in report
    assert "&lt;script&gt;" in report

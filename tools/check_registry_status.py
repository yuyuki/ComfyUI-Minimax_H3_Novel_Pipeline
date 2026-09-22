"""Check a published version and prepare a review request without posting it.

Uses only the public registry API; Python 3.11+ is required for tomllib.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


def fetch_version(node: str, version: str) -> dict:
    versions_url = f"https://api.comfy.org/nodes/{quote(node, safe='')}/versions"
    url = f"{versions_url}/{quote(version, safe='')}"
    with urlopen(url, timeout=30) as response:
        result = json.load(response)
    if not isinstance(result, dict) or result.get("version") != version or not result.get("status"):
        raise ValueError("Registry returned an unexpected version response")
    # The detail endpoint does not support include_status_reason; the list does.
    if result["status"] == "NodeVersionStatusFlagged":
        with urlopen(f"{versions_url}?include_status_reason=true", timeout=30) as response:
            versions = json.load(response)
        if not isinstance(versions, list):
            raise ValueError("Registry returned an unexpected version list")
        match = next((item for item in versions if item.get("version") == version), None)
        if match:
            result["status_reason"] = match.get("status_reason")
    return result


def check(node: str, version: str, attempts: int, interval: float) -> tuple[int, dict]:
    result = {"version": version, "status": "Not yet visible"}
    for attempt in range(attempts):
        try:
            result = fetch_version(node, version)
        except HTTPError as error:
            if error.code != 404 and error.code != 429 and error.code < 500:
                raise
            result = {"version": version, "status": f"HTTP {error.code}; status unavailable"}
        except (URLError, TimeoutError):
            result = {"version": version, "status": "Network error; status unavailable"}
        status = result["status"]
        print(f"Registry check {attempt + 1}/{attempts}: {version}: {status}", flush=True)
        if status == "NodeVersionStatusActive":
            return (1 if result.get("deprecated") else 0), result
        if status not in {"NodeVersionStatusPending", "Not yet visible"} and "unavailable" not in status:
            return 1, result
        if attempt + 1 < attempts:
            time.sleep(interval)
    return 1, result


def report(project: dict, publisher: str, result: dict) -> str:
    node, version = project["name"], project["version"]
    reason = result.get("status_reason") or "The registry did not provide a reason."
    if not isinstance(reason, str):
        reason = json.dumps(reason, indent=2)
    text = (f"## Registry status: {html.escape(node)} {html.escape(version)}\n\n"
            f"Status: {html.escape(result['status'])}\n\n"
            f"Deprecated: {bool(result.get('deprecated'))}\n\n"
            f"[Registry version](https://api.comfy.org/nodes/{quote(node, safe='')}/versions/"
            f"{quote(version, safe='')})\n\n"
            f"<pre>{html.escape(reason)}</pre>\n")
    if result["status"] == "NodeVersionStatusFlagged":
        draft = (f"Manual review request: {node} {version}\n\n"
                 f"Publisher: {publisher}\nNode: {node}\nVersion: {version}\n"
                 f"Version ID: {result.get('id', 'unavailable')}\n"
                 f"Repository: {project.get('urls', {}).get('Repository', '')}\n\n"
                 "Publishing completed, but this version is flagged. Please review the findings "
                 "below and explain any required changes.\n\n" + reason)
        text += ("\n[Open a manual review issue](https://github.com/Comfy-Org/registry-backend/issues/new)"
                 "\n\nReview the draft before submitting it:\n\n"
                 f"<pre>{html.escape(draft)}</pre>\n")
    elif result["status"] != "NodeVersionStatusActive":
        text += "\nApproval is unconfirmed. Run the workflow with `check_only` to check again without republishing.\n"
    return text


def main() -> int:
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.attempts < 1 or args.interval < 0:
        parser.error("attempts must be positive and interval must be nonnegative")
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = config["project"]
    code, result = check(project["name"], project["version"], args.attempts, args.interval)
    output = report(project, config["tool"]["comfy"]["PublisherId"], result)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as summary:
            summary.write(output)
    else:
        print(output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

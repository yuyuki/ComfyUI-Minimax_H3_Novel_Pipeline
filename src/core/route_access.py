"""Access policy for the plugin's local browser controls."""
from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import urlsplit


def _is_loopback(host: str) -> bool:
    try:
        address = ip_address(host)
        mapped = getattr(address, "ipv4_mapped", None)
        return (mapped or address).is_loopback
    except ValueError:
        return False


def _local_origin(value: str):
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment):
            return None
        if parsed.hostname != "localhost" and not _is_loopback(parsed.hostname):
            return None
        return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None


def require_local_request(request) -> None:
    """Reject remote peers, proxy requests, DNS rebinding and cross-site calls.

    Use the socket peer, never a client-supplied forwarding header. These
    controls deliberately support direct local access only.
    """
    from aiohttp import web

    peer = request.transport.get_extra_info("peername") if request.transport else None
    if not isinstance(peer, tuple) or not peer or not _is_loopback(str(peer[0])):
        raise web.HTTPForbidden(reason="MiniMax H3 controls require local access")
    if any(name.lower() == "forwarded" or name.lower().startswith("x-forwarded-")
           or name.lower() == "x-real-ip" for name in request.headers):
        raise web.HTTPForbidden(reason="MiniMax H3 controls require direct local access")

    expected = _local_origin(f"{request.scheme}://{request.host}")
    origin = request.headers.get("Origin")
    if expected is None or (origin is not None and _local_origin(origin) != expected):
        raise web.HTTPForbidden(reason="Invalid MiniMax H3 request origin")
    if request.headers.get("Sec-Fetch-Site", "same-origin") != "same-origin":
        raise web.HTTPForbidden(reason="Cross-site MiniMax H3 requests are forbidden")
    # Browsers cannot attach this header to a cross-origin simple/form request.
    if request.headers.get("X-MiniMax-H3-Request") != "1":
        raise web.HTTPForbidden(reason="Missing MiniMax H3 request header")

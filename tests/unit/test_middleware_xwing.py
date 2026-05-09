"""
Multi-suite middleware tests via direct ASGI invocation.

Verifies HPKEMiddleware correctly:
- Accepts X25519 traffic without an X-HPKE-Suite header (back-compat).
- Routes X-Wing traffic via the X-HPKE-Suite header to the matching key.
- Returns 415 when the client requests a KEM the server hasn't registered.
- Returns 400 on a malformed X-HPKE-Suite header.

Uses direct ASGI invocation (no granian subprocess) to keep the test fast.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from hpke_http.constants import (
    HEADER_HPKE_PSK_ID,
    HEADER_HPKE_SUITE,
    KemId,
)
from hpke_http.core import RequestEncryptor
from hpke_http.middleware.fastapi import HPKEMiddleware
from hpke_http.primitives import X25519KEM, XWingKEM

_PSK = b"test-api-key-for-hpke-psk-mode!!"
_PSK_ID = b"tenant-1"


def _b64url_encode(data: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


async def _async_psk_resolver(_scope: Any) -> tuple[bytes, bytes]:
    return _PSK, _PSK_ID


def _build_scope(headers: dict[str, str], body: bytes = b"") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build an ASGI scope + receive sequence for a single request body."""
    scope: dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/api/test",
        "raw_path": b"/api/test",
        "query_string": b"",
        "headers": [(k.lower().encode("ascii"), v.encode("ascii")) for k, v in headers.items()],
    }
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    return scope, messages


def _make_receive(messages: list[dict[str, Any]]) -> Any:
    queue = list(messages)

    async def receive() -> dict[str, Any]:
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    return receive


async def _capture_send() -> tuple[Any, list[dict[str, Any]]]:
    captured: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        captured.append(message)

    return send, captured


def _status(captured: list[dict[str, Any]]) -> int:
    starts = [m for m in captured if m["type"] == "http.response.start"]
    assert starts, "no http.response.start emitted"
    return int(starts[0]["status"])


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _x25519_keys() -> tuple[bytes, bytes]:
    return X25519KEM.generate_keypair()


def _xwing_keys() -> tuple[bytes, bytes]:
    return XWingKEM.generate_keypair()


def _encrypt_request(pk_r: bytes, kem_id: KemId, body: bytes = b'{"hello":"world"}') -> tuple[dict[str, str], bytes]:
    encryptor = RequestEncryptor(public_key=pk_r, psk=_PSK, psk_id=_PSK_ID, kem_id=kem_id)
    ct = encryptor.encrypt_all(body)
    headers = encryptor.get_headers()
    headers[HEADER_HPKE_PSK_ID] = _b64url_encode(_PSK_ID)
    return headers, ct


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiSuiteRouting:
    @pytest.mark.asyncio
    async def test_x25519_no_suite_header_succeeds(self) -> None:
        """Legacy client (no X-HPKE-Suite) → server defaults to X25519."""
        sk_x, pk_x = _x25519_keys()

        app = AsyncMock()
        middleware = HPKEMiddleware(
            app=app,
            private_keys={KemId.DHKEM_X25519_HKDF_SHA256: sk_x},
            psk_resolver=_async_psk_resolver,
        )

        headers, body = _encrypt_request(pk_x, KemId.DHKEM_X25519_HKDF_SHA256)
        # Default suite must NOT emit X-HPKE-Suite (back-compat assertion)
        assert HEADER_HPKE_SUITE not in headers
        scope, messages = _build_scope(headers, body)
        send, _captured = await _capture_send()
        await middleware(scope, _make_receive(messages), send)

        # App was called → middleware accepted decryption
        assert app.called

    @pytest.mark.asyncio
    async def test_xwing_suite_header_routes_correctly(self) -> None:
        """Client with X-HPKE-Suite=kem=0x647a → server routes to X-Wing key."""
        sk_x, pk_x = _x25519_keys()
        sk_w, pk_w = _xwing_keys()

        app = AsyncMock()
        middleware = HPKEMiddleware(
            app=app,
            private_keys={
                KemId.DHKEM_X25519_HKDF_SHA256: sk_x,
                KemId.XWING: sk_w,
            },
            psk_resolver=_async_psk_resolver,
        )

        headers, body = _encrypt_request(pk_w, KemId.XWING)
        assert headers[HEADER_HPKE_SUITE] == "kem=0x647a"
        scope, messages = _build_scope(headers, body)
        send, _captured = await _capture_send()
        await middleware(scope, _make_receive(messages), send)

        assert app.called

        # Sanity: we used X25519 pk_x in scope but it shouldn't matter.
        _ = pk_x

    @pytest.mark.asyncio
    async def test_xwing_request_to_x25519_only_server_returns_415(self) -> None:
        """Server doesn't have X-Wing key → 415 (UnsupportedKEMError path)."""
        sk_x, _pk_x = _x25519_keys()

        app = AsyncMock()
        middleware = HPKEMiddleware(
            app=app,
            private_keys={KemId.DHKEM_X25519_HKDF_SHA256: sk_x},
            psk_resolver=_async_psk_resolver,
        )

        # Encrypt with X-Wing pk we never registered → server rejects on header parse
        _sk_w, pk_w = _xwing_keys()
        headers, body = _encrypt_request(pk_w, KemId.XWING)
        scope, messages = _build_scope(headers, body)
        send, captured = await _capture_send()
        await middleware(scope, _make_receive(messages), send)

        assert _status(captured) == 415
        assert not app.called

    @pytest.mark.asyncio
    async def test_malformed_suite_header_returns_400(self) -> None:
        """Malformed X-HPKE-Suite header → DecryptionError → HTTP 400."""
        sk_x, pk_x = _x25519_keys()

        app = AsyncMock()
        middleware = HPKEMiddleware(
            app=app,
            private_keys={KemId.DHKEM_X25519_HKDF_SHA256: sk_x},
            psk_resolver=_async_psk_resolver,
        )

        headers, body = _encrypt_request(pk_x, KemId.DHKEM_X25519_HKDF_SHA256)
        headers[HEADER_HPKE_SUITE] = "KEM=0x0020"  # uppercase key — rejected
        scope, messages = _build_scope(headers, body)
        send, captured = await _capture_send()
        await middleware(scope, _make_receive(messages), send)

        assert _status(captured) == 400
        assert not app.called


class TestDiscoveryDocAdvertisesBothSuites:
    """When the middleware is given both keys, the discovery doc lists both."""

    @pytest.mark.asyncio
    async def test_discovery_lists_both_kems(self) -> None:
        sk_x, _pk_x = _x25519_keys()
        sk_w, _pk_w = _xwing_keys()

        app = AsyncMock()
        middleware = HPKEMiddleware(
            app=app,
            private_keys={
                KemId.DHKEM_X25519_HKDF_SHA256: sk_x,
                KemId.XWING: sk_w,
            },
            psk_resolver=_async_psk_resolver,
        )

        scope: dict[str, Any] = {
            "type": "http",
            "method": "GET",
            "path": "/.well-known/hpke-keys",
            "raw_path": b"/.well-known/hpke-keys",
            "query_string": b"",
            "headers": [],
        }
        send, captured = await _capture_send()
        await middleware(scope, _make_receive([{"type": "http.request", "body": b"", "more_body": False}]), send)

        body_msgs = [m for m in captured if m["type"] == "http.response.body"]
        assert body_msgs, "no body emitted"

        import json

        payload = json.loads(body_msgs[0]["body"])
        kem_ids = [int(k["kem_id"], 16) for k in payload["keys"]]
        assert KemId.DHKEM_X25519_HKDF_SHA256 in kem_ids
        assert KemId.XWING in kem_ids

        # default_suite reflects DEFAULT_KEM_PRIORITY (X-Wing first when registered).
        assert int(payload["default_suite"]["kem_id"], 16) == KemId.XWING

        # Sanity: pk sizes match
        kem_id_to_b64_len = {int(k["kem_id"], 16): len(k["public_key"]) for k in payload["keys"]}
        assert kem_id_to_b64_len[KemId.DHKEM_X25519_HKDF_SHA256] == 43  # b64url(32B)
        assert kem_id_to_b64_len[KemId.XWING] == 1622  # b64url(1216B)

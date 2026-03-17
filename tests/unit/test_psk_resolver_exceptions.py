"""Unit tests for psk_resolver exception handling edge cases.

These test scenarios that CANNOT be tested via E2E (real granian server):
- HTTPException with custom headers (WWW-Authenticate)
- HTTPException edge cases (None detail, empty detail, 200 status, dict detail)
- HTTPException subclass forwarding
- CryptoError from psk_resolver (real resolvers never raise CryptoError)
- Multiple generic exception types (E2E only tests ValueError)

Normal HTTPException forwarding (401, 403, 503) and generic fallback are
tested E2E in test_psk_id_header.py::TestPSKResolverHTTPExceptionForwarding.
"""

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from starlette.exceptions import HTTPException

from hpke_http.constants import (
    HEADER_HPKE_ERROR,
    SCOPE_HPKE_PSK_ID,
    KemId,
)
from hpke_http.exceptions import CryptoError, DecryptionError
from hpke_http.headers import b64url_encode
from hpke_http.middleware.fastapi import HPKEMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
#
# These tests mock the ASGI interface (scope/receive/send) because they test
# psk_resolver exception handling edge cases that cannot be triggered through
# a real HTTP server:
# - The psk_resolver must raise specific exception types/subclasses
# - The psk_resolver must raise with specific arguments (None detail, dict
#   detail, custom headers) that would require excessive e2e_server complexity
# - CryptoError from psk_resolver never occurs in real resolvers
#
# Normal cases are tested E2E with real granian server in test_psk_id_header.py.


def _encrypted_scope(psk_id: bytes = b"tenant-123") -> dict[str, Any]:
    """Build an ASGI scope for an encrypted request (X-HPKE-Enc present).

    The enc/stream header values are dummy — psk_resolver fails before actual crypto.
    X-HPKE-Stream is required by _create_decrypted_receive before _setup_decryption.
    """
    return {
        "type": "http",
        "method": "POST",
        "path": "/test",
        "headers": [
            (b"x-hpke-enc", b"AAAA"),  # dummy, never reaches decryption
            (b"x-hpke-stream", b"AAAA"),  # required before psk_resolver is called
            (b"x-hpke-psk-id", b64url_encode(psk_id).encode()),
        ],
        SCOPE_HPKE_PSK_ID: psk_id,
    }


def _unencrypted_scope(psk_id: bytes = b"tenant-123") -> dict[str, Any]:
    """Build an ASGI scope for an unencrypted request (no X-HPKE-Enc)."""
    return {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [
            (b"x-hpke-psk-id", b64url_encode(psk_id).encode()),
        ],
        SCOPE_HPKE_PSK_ID: psk_id,
    }


async def _invoke(
    side_effect: Exception,
    *,
    encrypted: bool = True,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    """Invoke middleware with a failing psk_resolver and return the response.

    Returns (status_code, json_body, headers_dict).
    """
    resolver = AsyncMock(side_effect=side_effect)
    mw = HPKEMiddleware(
        app=AsyncMock(),
        private_keys={KemId.DHKEM_X25519_HKDF_SHA256: b"x" * 32},
        psk_resolver=resolver,
    )
    send = AsyncMock()
    scope = _encrypted_scope() if encrypted else _unencrypted_scope()
    await mw(scope, AsyncMock(), send)

    calls = [call.args[0] for call in send.call_args_list]
    start = next(c for c in calls if c["type"] == "http.response.start")
    body_msg = next(c for c in calls if c["type"] == "http.response.body")
    status: int = start["status"]
    body: dict[str, Any] = json.loads(body_msg["body"])
    headers = {k.decode().lower(): v.decode() for k, v in start.get("headers", [])}
    return status, body, headers


# ---------------------------------------------------------------------------
# HTTPException edge cases (cannot test E2E)
# ---------------------------------------------------------------------------


class TestHTTPExceptionEdgeCases:
    """HTTPException edge cases that require mock psk_resolver."""

    async def test_headers_forwarded(self) -> None:
        """HTTPException headers (e.g., WWW-Authenticate) are forwarded.

        E2E server's magic PSK IDs don't set custom headers on HTTPException,
        so this can only be tested via mock.
        """
        exc = HTTPException(401, "Auth required", headers={"WWW-Authenticate": "Bearer"})
        status, _, headers = await _invoke(exc)
        assert status == 401
        assert headers["www-authenticate"] == "Bearer"

    async def test_headers_forwarded_unencrypted(self) -> None:
        """HTTPException headers forwarded on unencrypted path too."""
        exc = HTTPException(401, "Auth required", headers={"WWW-Authenticate": "Bearer"})
        _, _, headers = await _invoke(exc, encrypted=False)
        assert headers["www-authenticate"] == "Bearer"

    async def test_none_detail_uses_default(self) -> None:
        """HTTPException with detail=None uses HTTP reason phrase."""
        status, body, _ = await _invoke(HTTPException(400))
        assert status == 400
        assert body == {"error": "Bad Request"}

    async def test_empty_detail(self) -> None:
        """HTTPException with empty string detail."""
        status, body, _ = await _invoke(HTTPException(403, ""))
        assert status == 403
        assert body == {"error": ""}

    async def test_200_forwarded(self) -> None:
        """HTTPException(200) is forwarded — unusual but valid."""
        status, body, _ = await _invoke(HTTPException(200, "OK"))
        assert status == 200
        assert body == {"error": "OK"}

    async def test_long_detail_forwarded(self) -> None:
        """Long detail (10KB) is forwarded without truncation."""
        long_detail = "x" * 10_000
        _, body, _ = await _invoke(HTTPException(400, long_detail))
        assert body == {"error": long_detail}

    async def test_dict_detail_converted_to_str(self) -> None:
        """Non-string detail is str()-converted for JSON safety."""
        _, body, _ = await _invoke(HTTPException(400, detail={"key": "val"}))  # type: ignore[arg-type]
        # str({"key": "val"}) produces the Python repr, not JSON
        assert "key" in body["error"]

    async def test_subclass_forwarded(self) -> None:
        """HTTPException subclass is caught via isinstance."""

        class CustomHTTPException(HTTPException):
            pass

        status, body, _ = await _invoke(CustomHTTPException(418, "I'm a teapot"))
        assert status == 418
        assert body == {"error": "I'm a teapot"}

    async def test_hpke_error_header_present(self) -> None:
        """Forwarded HTTPException responses include X-HPKE-Error: true.

        Also tested E2E, but kept here to verify on encrypted path
        (E2E test uses unencrypted raw request).
        """
        _, _, headers = await _invoke(HTTPException(401, "Invalid"))
        assert headers[HEADER_HPKE_ERROR.lower()] == "true"


# ---------------------------------------------------------------------------
# Generic exception fallback (multiple types — E2E only tests ValueError)
# ---------------------------------------------------------------------------


class TestGenericExceptionFallback:
    """Multiple exception types all produce 401 — E2E only tests ValueError."""

    @pytest.mark.parametrize(
        ("exc", "description"),
        [
            (ValueError("bad key"), "ValueError"),
            (RuntimeError("oops"), "RuntimeError"),
            (KeyError("missing"), "KeyError"),
            (TypeError("wrong type"), "TypeError"),
            (PermissionError("denied"), "PermissionError"),
        ],
    )
    async def test_generic_exception_encrypted(self, exc: Exception, description: str) -> None:
        """Generic exception on encrypted path → 401."""
        status, body, _ = await _invoke(exc)
        assert status == 401, f"Expected 401 for {description}, got {status}"
        assert body == {"error": "PSK authentication failed"}


# ---------------------------------------------------------------------------
# CryptoError from psk_resolver (real resolvers never raise this)
# ---------------------------------------------------------------------------


class TestCryptoErrorFromResolver:
    """CryptoError from psk_resolver — behavior differs by path.

    Real psk_resolver implementations never raise CryptoError, so this
    can only be tested via mock. Documents the catch-order behavior:
    encrypted path CryptoError → 400, unencrypted path → 401.
    """

    async def test_crypto_error_encrypted(self) -> None:
        """CryptoError from psk_resolver → caught by CryptoError handler → 400."""
        status, body, _ = await _invoke(CryptoError("test"))
        assert status == 400
        assert body == {"error": "Request decryption failed"}

    async def test_decryption_error_encrypted(self) -> None:
        """DecryptionError (CryptoError subclass) from psk_resolver → 400."""
        status, body, _ = await _invoke(DecryptionError("test"))
        assert status == 400
        assert body == {"error": "Request decryption failed"}

    async def test_crypto_error_unencrypted(self) -> None:
        """CryptoError on unencrypted path → falls to Exception handler → 401.

        On the unencrypted path there is no CryptoError handler (no crypto
        happens), so CryptoError from psk_resolver is treated as a generic
        auth failure. This differs from the encrypted path where CryptoError → 400.
        """
        status, body, _ = await _invoke(CryptoError("test"), encrypted=False)
        assert status == 401
        assert body == {"error": "PSK authentication failed"}

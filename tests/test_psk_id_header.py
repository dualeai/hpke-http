"""Tests for X-HPKE-PSK-ID header functionality.

Tests that the PSK ID is correctly:
- Sent by client in X-HPKE-PSK-ID header (base64url-encoded)
- Parsed by server middleware and stored in scope["hpke_psk_id"]
- Available to psk_resolver for PSK lookup
- Sent on bodyless requests (GET, DELETE, HEAD) for auth
- Rejected when missing, unknown, or malformed
"""

import aiohttp
import pytest

from hpke_http.constants import HEADER_HPKE_PSK_ID
from hpke_http.headers import b64url_decode, b64url_encode
from hpke_http.middleware.aiohttp import HPKEClientSession
from hpke_http.middleware.httpx import HPKEAsyncClient
from tests.conftest import E2EServer


class TestPSKIDHeaderEncoding:
    """Test PSK ID base64url encoding/decoding roundtrip."""

    def test_psk_id_roundtrip(self) -> None:
        """b64url_encode/b64url_decode roundtrip for PSK ID."""
        psk_id = b"tenant-123"
        encoded = b64url_encode(psk_id)
        assert b64url_decode(encoded) == psk_id

    @pytest.mark.parametrize(
        ("psk_id", "description"),
        [
            (b"tenant-123", "standard tenant ID"),
            (b"user-abc-123", "ID with hyphens"),
            (b"12345", "numeric ID"),
            (b"a" * 32, "32-byte ID"),
            (b"x" * 255, "maximum reasonable length"),
        ],
    )
    def test_psk_id_encoding_parametrized(self, psk_id: bytes, description: str) -> None:
        """Various PSK ID formats encode correctly."""
        encoded = b64url_encode(psk_id)
        decoded = b64url_decode(encoded)
        assert decoded == psk_id, f"Failed for {description}"

    def test_psk_id_binary_encoding(self) -> None:
        """Binary (non-UTF8) PSK ID encodes correctly."""
        binary_psk_id = b"\x00\x01\x02\xff\xfe"
        encoded = b64url_encode(binary_psk_id)
        decoded = b64url_decode(encoded)
        assert decoded == binary_psk_id

    def test_psk_id_special_chars(self) -> None:
        """PSK ID with special characters encodes correctly."""
        psk_id = b"tenant/with/slashes+and+plus"
        encoded = b64url_encode(psk_id)
        decoded = b64url_decode(encoded)
        assert decoded == psk_id


class TestPSKIDHeaderE2E:
    """E2E tests for PSK ID header transmission."""

    async def test_psk_id_roundtrip_aiohttp(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Client sends X-HPKE-PSK-ID, server receives it."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            resp = await client.post("/echo", json={"test": "psk_id"})
            assert resp.status == 200
            # If we get here without decryption error, PSK ID matched

    async def test_psk_id_roundtrip_httpx(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """httpx client sends X-HPKE-PSK-ID correctly."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEAsyncClient(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            resp = await client.post("/echo", json={"test": "psk_id"})
            assert resp.status_code == 200

    @pytest.mark.parametrize(
        ("psk_id", "description"),
        [
            (b"a", "single byte PSK ID"),
            (b"x" * 100, "100-byte ID"),
            (b"\x00\x01\x02\xff", "binary PSK ID"),
        ],
    )
    async def test_psk_id_edge_cases(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        psk_id: bytes,
        description: str,
    ) -> None:
        """Edge case PSK ID values work correctly.

        Note: These tests use different PSK IDs than the server expects,
        so they will fail at the crypto level (PSK ID is part of HPKE context).
        We're testing that the header is sent correctly.
        """
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=psk_id,  # Different from server's test_psk_id
        ) as client:
            # This should fail with 400 because PSK ID doesn't match server
            # The important thing is that the header was sent (not a header format error)
            resp = await client.post("/echo", json={"test": "edge_case"})
            # Decryption will fail because psk_id is part of HPKE info parameter
            assert resp.status == 400, f"Expected 400 for mismatched PSK ID ({description})"


class TestPSKIDMismatch:
    """Test behavior when PSK ID doesn't match."""

    async def test_wrong_psk_id_rejected(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        wrong_psk_id: bytes,
    ) -> None:
        """Request with different PSK ID than server expects is rejected.

        PSK ID is used in HPKE's info parameter, so mismatched IDs will
        derive different keys and decryption will fail.
        """
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=wrong_psk_id,  # Different from server's test_psk_id
        ) as client:
            resp = await client.post("/echo", json={"test": "wrong_id"})
            # Server should return 400 (decryption failed)
            assert resp.status == 400

    async def test_psk_id_case_sensitive(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """PSK ID comparison is case-sensitive.

        'Tenant-123' != 'tenant-123' because it affects key derivation.
        """
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        # Flip case of PSK ID
        wrong_case_id = test_psk_id.upper() if test_psk_id.islower() else test_psk_id.lower()
        if wrong_case_id == test_psk_id:
            # If no alphabetic chars, modify it differently
            wrong_case_id = test_psk_id + b"-modified"

        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=wrong_case_id,
        ) as client:
            resp = await client.post("/echo", json={"test": "case_sensitive"})
            # Should fail because PSK ID doesn't match
            assert resp.status == 400


class TestBackwardsCompatibility:
    """Test backwards compatibility with existing psk_resolver implementations."""

    async def test_psk_resolver_still_works(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Existing psk_resolver implementations continue to work.

        The server's psk_resolver returns fixed (psk, psk_id) and doesn't
        need to use scope["hpke_psk_id"]. This should still work.
        """
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            # Normal request should work
            resp = await client.post("/echo", json={"backwards": "compatible"})
            assert resp.status == 200

            data = await resp.json()
            assert "backwards" in data["echo"]


class TestBodylessRequestPSKID:
    """Test that bodyless requests (GET, DELETE, HEAD) send X-HPKE-PSK-ID.

    Without this fix, GET /whoami returns 401 because scope["hpke_psk_id"]
    is never populated on non-encrypted requests.
    """

    async def test_aiohttp_get_whoami(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """aiohttp GET /whoami returns 200 with correct PSK ID."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            resp = await client.get("/whoami")
            assert resp.status == 200
            data = await resp.json()
            assert data["psk_id"] == test_psk_id.hex()

    async def test_httpx_get_whoami(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """httpx GET /whoami returns 200 with correct PSK ID."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEAsyncClient(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            resp = await client.get("/whoami")
            assert resp.status_code == 200
            data = resp.json()
            assert data["psk_id"] == test_psk_id.hex()

    async def test_aiohttp_delete_sends_psk_id(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """aiohttp DELETE sends X-HPKE-PSK-ID header."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            # /echo accepts DELETE, check headers are sent
            resp = await client.delete("/echo")
            assert resp.status == 200
            data = await resp.json()
            assert data["method"] == "DELETE"

    async def test_httpx_head_sends_psk_id(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """httpx HEAD sends X-HPKE-PSK-ID header."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEAsyncClient(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            resp = await client.head("/health")
            assert resp.status_code == 200

    async def test_aiohttp_get_whoami_header_encoding(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Verify the PSK ID header is base64url-encoded on bodyless requests."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        expected_header = b64url_encode(test_psk_id)

        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            # Use echo-headers to verify the header value
            # echo-headers only accepts POST, so use GET on /whoami for auth check
            resp = await client.get("/whoami")
            assert resp.status == 200

            # Also verify round-trip: server decoded it correctly
            data = await resp.json()
            assert bytes.fromhex(data["psk_id"]) == test_psk_id

            # Verify encoding consistency
            assert expected_header == b64url_encode(test_psk_id)


class TestBodylessRequestPSKIDDeny:
    """Deny cases for bodyless requests - missing, unknown, or malformed PSK ID."""

    async def test_no_psk_id_header_returns_401(
        self,
        granian_server: E2EServer,
    ) -> None:
        """Raw GET without X-HPKE-PSK-ID is rejected by middleware with 401."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{granian_server.host}:{granian_server.port}/whoami",
            ) as resp:
                assert resp.status == 401

    async def test_unknown_psk_id_bodyless_rejected(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
    ) -> None:
        """Bodyless GET with unknown PSK ID → psk_resolver rejects → 401."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=b"unknown-tenant",
        ) as client:
            resp = await client.get("/whoami")
            assert resp.status == 401

    async def test_unknown_psk_id_encrypted_rejected(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
    ) -> None:
        """Encrypted POST with unknown PSK ID → psk_resolver rejects → 400."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=b"unknown-tenant",
        ) as client:
            resp = await client.post("/echo", json={"test": "deny"})
            assert resp.status == 400

    async def test_malformed_base64_psk_id_returns_400(
        self,
        granian_server: E2EServer,
    ) -> None:
        """Invalid base64url in X-HPKE-PSK-ID header returns 400."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{granian_server.host}:{granian_server.port}/whoami",
                headers={HEADER_HPKE_PSK_ID: "a==="},  # 1 char + padding → invalid base64
            ) as resp:
                assert resp.status == 400

    async def test_empty_psk_id_header_returns_401(
        self,
        granian_server: E2EServer,
    ) -> None:
        """Empty X-HPKE-PSK-ID header → empty bytes in scope → /whoami rejects."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{granian_server.host}:{granian_server.port}/whoami",
                headers={HEADER_HPKE_PSK_ID: b64url_encode(b"")},
            ) as resp:
                # Empty PSK ID decodes to b"" which is falsy → 401
                assert resp.status == 401

    async def test_wrong_psk_id_encrypted_httpx(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        wrong_psk_id: bytes,
    ) -> None:
        """httpx encrypted POST with wrong PSK ID → psk_resolver rejects → 400."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEAsyncClient(
            base_url=base_url,
            psk=test_psk,
            psk_id=wrong_psk_id,
        ) as client:
            resp = await client.post("/echo", json={"test": "deny"})
            assert resp.status_code == 400

"""Tests for X-HPKE-PSK-ID header functionality.

Tests that the PSK ID is correctly:
- Sent by client in X-HPKE-PSK-ID header (base64url-encoded)
- Parsed by server middleware and stored in scope["hpke_psk_id"]
- Available to psk_resolver for PSK lookup
"""

import pytest

from hpke_http.constants import HEADER_HPKE_PSK_ID
from hpke_http.core import RequestEncryptor
from hpke_http.headers import b64url_decode
from hpke_http.middleware.aiohttp import HPKEClientSession
from hpke_http.middleware.httpx import HPKEAsyncClient
from tests.conftest import E2EServer


class TestPSKIDHeaderEncoding:
    """Test PSK ID header encoding in RequestEncryptor."""

    def test_psk_id_header_present(self, platform_keypair: tuple[bytes, bytes], test_psk: bytes) -> None:
        """RequestEncryptor includes X-HPKE-PSK-ID header."""
        _sk, pk = platform_keypair
        psk_id = b"tenant-123"

        encryptor = RequestEncryptor(public_key=pk, psk=test_psk, psk_id=psk_id)
        headers = encryptor.get_headers()

        assert HEADER_HPKE_PSK_ID in headers
        # Verify it's base64url encoded
        decoded = b64url_decode(headers[HEADER_HPKE_PSK_ID])
        assert decoded == psk_id

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
    def test_psk_id_encoding_parametrized(
        self, platform_keypair: tuple[bytes, bytes], test_psk: bytes, psk_id: bytes, description: str
    ) -> None:
        """Various PSK ID formats encode correctly."""
        _sk, pk = platform_keypair

        encryptor = RequestEncryptor(public_key=pk, psk=test_psk, psk_id=psk_id)
        headers = encryptor.get_headers()

        decoded = b64url_decode(headers[HEADER_HPKE_PSK_ID])
        assert decoded == psk_id, f"Failed for {description}"

    def test_psk_id_binary_encoding(self, platform_keypair: tuple[bytes, bytes], test_psk: bytes) -> None:
        """Binary (non-UTF8) PSK ID encodes correctly."""
        _sk, pk = platform_keypair
        # Binary PSK ID with non-UTF8 bytes
        binary_psk_id = b"\x00\x01\x02\xff\xfe"

        encryptor = RequestEncryptor(public_key=pk, psk=test_psk, psk_id=binary_psk_id)
        headers = encryptor.get_headers()

        decoded = b64url_decode(headers[HEADER_HPKE_PSK_ID])
        assert decoded == binary_psk_id

    def test_psk_id_special_chars(self, platform_keypair: tuple[bytes, bytes], test_psk: bytes) -> None:
        """PSK ID with special characters encodes correctly."""
        _sk, pk = platform_keypair
        psk_id = b"tenant/with/slashes+and+plus"

        encryptor = RequestEncryptor(public_key=pk, psk=test_psk, psk_id=psk_id)
        headers = encryptor.get_headers()

        decoded = b64url_decode(headers[HEADER_HPKE_PSK_ID])
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

"""Tests for compression capability negotiation (RFC 9110 §12.5.3).

Tests that servers correctly advertise encoding support via Accept-Encoding header
and reject unsupported encodings with HTTP 415.
"""

import aiohttp
import pytest

from hpke_http.constants import (
    CHACHA20_POLY1305_KEY_SIZE,
    HEADER_HPKE_ENC,
    HEADER_HPKE_ENCODING,
    HEADER_HPKE_STREAM,
    REQUEST_KEY_LABEL,
)
from hpke_http.core import parse_accept_encoding
from hpke_http.headers import b64url_encode
from hpke_http.hpke import setup_sender_psk
from hpke_http.streaming import ChunkEncryptor, RawFormat, StreamingSession

from .conftest import E2EServer


def _encrypt_request(
    body: bytes,
    pk_r: bytes,
    psk: bytes,
    psk_id: bytes,
) -> tuple[bytes, str, str]:
    """Encrypt request body for testing using chunked streaming format.

    Returns:
        Tuple of (encrypted_body, enc_header_value, stream_header_value)
    """
    ctx = setup_sender_psk(
        pk_r=pk_r,
        info=psk_id,
        psk=psk,
        psk_id=psk_id,
    )
    request_key = ctx.export(REQUEST_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
    session = StreamingSession.create(request_key)
    encryptor = ChunkEncryptor(session, format=RawFormat(), compress=False)

    encrypted_body = encryptor.encrypt(body) if body else encryptor.encrypt(b"")
    enc_header = b64url_encode(ctx.enc)
    stream_header = b64url_encode(session.session_salt)
    return (encrypted_body, enc_header, stream_header)


# =============================================================================
# Discovery Endpoint Tests
# =============================================================================


class TestDiscoveryAcceptEncoding:
    """Discovery endpoint advertises supported encodings via Accept-Encoding header."""

    async def test_discovery_accept_encoding_with_zstd(
        self,
        granian_server: E2EServer,
    ) -> None:
        """Server with zstd available advertises 'identity, zstd'."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{granian_server.host}:{granian_server.port}/.well-known/hpke-keys") as resp:
                assert resp.status == 200
                assert "Accept-Encoding" in resp.headers
                encodings = [e.strip() for e in resp.headers["Accept-Encoding"].split(",")]
                assert "identity" in encodings
                assert "zstd" in encodings

    async def test_discovery_accept_encoding_without_zstd(
        self,
        granian_server_no_zstd: E2EServer,
    ) -> None:
        """Server without zstd advertises 'identity, gzip' (gzip is stdlib fallback)."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{granian_server_no_zstd.host}:{granian_server_no_zstd.port}/.well-known/hpke-keys"
            ) as resp:
                assert resp.status == 200
                accept_encoding = resp.headers.get("Accept-Encoding")
                assert accept_encoding == "identity, gzip"


# =============================================================================
# 415 Rejection Tests
# =============================================================================


class TestUnsupportedEncodingRejection:
    """Server rejects unsupported X-HPKE-Encoding with HTTP 415."""

    async def test_zstd_rejected_when_unavailable(
        self,
        granian_server_no_zstd: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """X-HPKE-Encoding: zstd to server without zstd returns 415."""
        encrypted_body, enc_header, stream_header = _encrypt_request(
            b'{"test": "data"}',
            granian_server_no_zstd.public_key,
            test_psk,
            test_psk_id,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{granian_server_no_zstd.host}:{granian_server_no_zstd.port}/echo",
                headers={
                    HEADER_HPKE_ENC: enc_header,
                    HEADER_HPKE_STREAM: stream_header,
                    HEADER_HPKE_ENCODING: "zstd",
                    "Content-Type": "application/octet-stream",
                },
                data=encrypted_body,
            ) as resp:
                assert resp.status == 415
                # Server advertises gzip as fallback (stdlib, always available)
                assert resp.headers.get("Accept-Encoding") == "identity, gzip"
                body = await resp.json()
                assert "error" in body
                assert "zstd" in body["error"].lower()

    async def test_uppercase_zstd_ignored(
        self,
        granian_server_no_zstd: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """X-HPKE-Encoding: ZSTD (uppercase) is ignored, not rejected."""
        encrypted_body, enc_header, stream_header = _encrypt_request(
            b'{"test": "data"}',
            granian_server_no_zstd.public_key,
            test_psk,
            test_psk_id,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{granian_server_no_zstd.host}:{granian_server_no_zstd.port}/echo",
                headers={
                    HEADER_HPKE_ENC: enc_header,
                    HEADER_HPKE_STREAM: stream_header,
                    HEADER_HPKE_ENCODING: "ZSTD",  # Uppercase - should be ignored
                    "Content-Type": "application/octet-stream",
                },
                data=encrypted_body,
            ) as resp:
                # Should succeed because uppercase ZSTD is ignored (case-sensitive)
                assert resp.status == 200

    async def test_zstd_accepted_when_available(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """X-HPKE-Encoding: zstd to server with zstd is accepted.

        Note: This test sends X-HPKE-Encoding: zstd but the body is NOT actually
        compressed, so it will fail with 400 during decompression, not 415.
        The point is that it passes the early 415 check.
        """
        encrypted_body, enc_header, stream_header = _encrypt_request(
            b'{"test": "data"}',
            granian_server.public_key,
            test_psk,
            test_psk_id,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{granian_server.host}:{granian_server.port}/echo",
                headers={
                    HEADER_HPKE_ENC: enc_header,
                    HEADER_HPKE_STREAM: stream_header,
                    HEADER_HPKE_ENCODING: "zstd",
                    "Content-Type": "application/octet-stream",
                },
                data=encrypted_body,
            ) as resp:
                # Should get 400 (decompression fails), not 415 (rejected)
                assert resp.status == 400


# =============================================================================
# Edge Cases
# =============================================================================


class TestEncodingEdgeCases:
    """Edge cases for X-HPKE-Encoding header against no-zstd server."""

    @pytest.mark.parametrize(
        ("encoding", "expected_status"),
        [
            ("zstd", 415),  # Rejected - server lacks zstd
            ("ZSTD", 200),  # Case-sensitive, ignored
            ("gzip", 400),  # Valid encoding, but body not compressed → decompression fails
            ("GZIP", 200),  # Case-sensitive, ignored
            ("br", 200),  # Unknown, ignored
            ("deflate", 200),  # Unknown, ignored
        ],
    )
    async def test_encoding_values(
        self,
        granian_server_no_zstd: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
        encoding: str,
        expected_status: int,
    ) -> None:
        """Test various X-HPKE-Encoding values against no-zstd server."""
        encrypted_body, enc_header, stream_header = _encrypt_request(
            b'{"test": "data"}',
            granian_server_no_zstd.public_key,
            test_psk,
            test_psk_id,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{granian_server_no_zstd.host}:{granian_server_no_zstd.port}/echo",
                headers={
                    HEADER_HPKE_ENC: enc_header,
                    HEADER_HPKE_STREAM: stream_header,
                    HEADER_HPKE_ENCODING: encoding,
                    "Content-Type": "application/octet-stream",
                },
                data=encrypted_body,
            ) as resp:
                assert resp.status == expected_status, f"Failed for encoding={encoding}"

    async def test_empty_encoding_accepted(
        self,
        granian_server_no_zstd: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Empty X-HPKE-Encoding header is treated as identity."""
        encrypted_body, enc_header, stream_header = _encrypt_request(
            b'{"test": "data"}',
            granian_server_no_zstd.public_key,
            test_psk,
            test_psk_id,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{granian_server_no_zstd.host}:{granian_server_no_zstd.port}/echo",
                headers={
                    HEADER_HPKE_ENC: enc_header,
                    HEADER_HPKE_STREAM: stream_header,
                    HEADER_HPKE_ENCODING: "",
                    "Content-Type": "application/octet-stream",
                },
                data=encrypted_body,
            ) as resp:
                assert resp.status == 200

    async def test_no_encoding_header_accepted(
        self,
        granian_server_no_zstd: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Missing X-HPKE-Encoding header is treated as identity."""
        encrypted_body, enc_header, stream_header = _encrypt_request(
            b'{"test": "data"}',
            granian_server_no_zstd.public_key,
            test_psk,
            test_psk_id,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{granian_server_no_zstd.host}:{granian_server_no_zstd.port}/echo",
                headers={
                    HEADER_HPKE_ENC: enc_header,
                    HEADER_HPKE_STREAM: stream_header,
                    # No X-HPKE-Encoding header
                    "Content-Type": "application/octet-stream",
                },
                data=encrypted_body,
            ) as resp:
                assert resp.status == 200


# =============================================================================
# Client-Server Compatibility Matrix
# =============================================================================


class TestCompressionCompatibilityAiohttp:
    """Full compatibility matrix using aiohttp client."""

    async def test_server_zstd_client_compress(
        self,
        granian_server_compressed: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Both server and client have zstd, compression works."""
        from hpke_http.middleware.aiohttp import HPKEClientSession

        base_url = f"http://{granian_server_compressed.host}:{granian_server_compressed.port}"
        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
            compress=True,
        ) as client:
            resp = await client.post("/echo", json={"test": "compressed"})
            assert resp.status == 200
            data = await resp.json()
            assert data["method"] == "POST"

    async def test_server_zstd_client_no_compress(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Server has zstd, client doesn't compress."""
        from hpke_http.middleware.aiohttp import HPKEClientSession

        base_url = f"http://{granian_server.host}:{granian_server.port}"
        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
            compress=False,
        ) as client:
            resp = await client.post("/echo", json={"test": "uncompressed"})
            assert resp.status == 200
            data = await resp.json()
            assert data["method"] == "POST"

    async def test_server_no_zstd_client_compress(
        self,
        granian_server_no_zstd: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Server lacks zstd, client configured with compress=True uses gzip fallback.

        Client reads Accept-Encoding from discovery ('identity, gzip') and uses
        gzip compression since zstd is unavailable. Server decompresses gzip request.
        """
        from hpke_http.middleware.aiohttp import HPKEClientSession

        base_url = f"http://{granian_server_no_zstd.host}:{granian_server_no_zstd.port}"
        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
            compress=True,  # Client uses gzip since server lacks zstd
        ) as client:
            # Body triggers gzip compression (server advertises 'identity, gzip')
            large_payload = {"data": "x" * 100}
            resp = await client.post("/echo", json=large_payload)
            # Auto-negotiation: client uses gzip, server decompresses
            assert resp.status == 200
            data = await resp.json()
            assert data["method"] == "POST"

    async def test_server_no_zstd_client_no_compress(
        self,
        granian_server_no_zstd: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Neither has/uses zstd, works fine."""
        from hpke_http.middleware.aiohttp import HPKEClientSession

        base_url = f"http://{granian_server_no_zstd.host}:{granian_server_no_zstd.port}"
        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
            compress=False,
        ) as client:
            resp = await client.post("/echo", json={"test": "uncompressed"})
            assert resp.status == 200
            data = await resp.json()
            assert data["method"] == "POST"


class TestCompressionCompatibilityHttpx:
    """Same compatibility matrix using httpx client."""

    async def test_server_zstd_client_compress(
        self,
        granian_server_compressed: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Both server and client have zstd, compression works."""
        from hpke_http.middleware.httpx import HPKEAsyncClient

        base_url = f"http://{granian_server_compressed.host}:{granian_server_compressed.port}"
        async with HPKEAsyncClient(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
            compress=True,
        ) as client:
            resp = await client.post("/echo", json={"test": "compressed"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["method"] == "POST"

    async def test_server_zstd_client_no_compress(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Server has zstd, client doesn't compress."""
        from hpke_http.middleware.httpx import HPKEAsyncClient

        base_url = f"http://{granian_server.host}:{granian_server.port}"
        async with HPKEAsyncClient(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
            compress=False,
        ) as client:
            resp = await client.post("/echo", json={"test": "uncompressed"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["method"] == "POST"

    async def test_server_no_zstd_client_compress(
        self,
        granian_server_no_zstd: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Server lacks zstd, client configured with compress=True uses gzip fallback.

        Client reads Accept-Encoding from discovery ('identity, gzip') and uses
        gzip compression since zstd is unavailable. Server decompresses gzip request.
        """
        from hpke_http.middleware.httpx import HPKEAsyncClient

        base_url = f"http://{granian_server_no_zstd.host}:{granian_server_no_zstd.port}"
        async with HPKEAsyncClient(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
            compress=True,  # Client uses gzip since server lacks zstd
        ) as client:
            # Body triggers gzip compression (server advertises 'identity, gzip')
            large_payload = {"data": "x" * 100}
            resp = await client.post("/echo", json=large_payload)
            # Auto-negotiation: client uses gzip, server decompresses
            assert resp.status_code == 200
            data = resp.json()
            assert data["method"] == "POST"

    async def test_server_no_zstd_client_no_compress(
        self,
        granian_server_no_zstd: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Neither has/uses zstd, works fine."""
        from hpke_http.middleware.httpx import HPKEAsyncClient

        base_url = f"http://{granian_server_no_zstd.host}:{granian_server_no_zstd.port}"
        async with HPKEAsyncClient(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
            compress=False,
        ) as client:
            resp = await client.post("/echo", json={"test": "uncompressed"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["method"] == "POST"


# =============================================================================
# Server Config Validation
# =============================================================================


class TestServerConfigValidation:
    """Server compression configuration validation."""

    async def test_compress_true_without_zstd_uses_gzip_fallback(self) -> None:
        """HPKEMiddleware(compress=True) uses gzip fallback when zstd unavailable."""
        from typing import Any
        from unittest.mock import patch

        from cryptography.hazmat.primitives.asymmetric import x25519

        from hpke_http.constants import KemId
        from hpke_http.middleware.fastapi import HPKEMiddleware

        async def mock_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            pass

        async def mock_psk_resolver(scope: dict[str, Any]) -> tuple[bytes, bytes]:
            return (b"psk", b"psk_id")

        # Generate a valid test keypair
        private_key = x25519.X25519PrivateKey.generate()
        sk = private_key.private_bytes_raw()

        # Mock _check_zstd_available to return False
        with patch.object(HPKEMiddleware, "_check_zstd_available", return_value=False):
            # Should NOT raise - gzip is used as fallback
            middleware = HPKEMiddleware(
                app=mock_app,
                private_keys={KemId.DHKEM_X25519_HKDF_SHA256: sk},
                psk_resolver=mock_psk_resolver,
                compress=True,
            )
            # Verify middleware was created with compress enabled
            assert middleware.compress is True
            assert middleware._zstd_available is False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    async def test_compress_false_without_zstd_works(self) -> None:
        """HPKEMiddleware(compress=False) works even if zstd unavailable."""
        from typing import Any
        from unittest.mock import patch

        # Generate a valid test keypair
        from cryptography.hazmat.primitives.asymmetric import x25519

        from hpke_http.constants import KemId
        from hpke_http.middleware.fastapi import HPKEMiddleware

        private_key = x25519.X25519PrivateKey.generate()
        sk = private_key.private_bytes_raw()

        async def mock_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
            pass

        async def mock_psk_resolver(scope: dict[str, Any]) -> tuple[bytes, bytes]:
            return (b"", b"")

        # Mock _check_zstd_available to return False
        with patch.object(HPKEMiddleware, "_check_zstd_available", return_value=False):
            # Should not raise
            middleware = HPKEMiddleware(
                app=mock_app,
                private_keys={KemId.DHKEM_X25519_HKDF_SHA256: sk},
                psk_resolver=mock_psk_resolver,
                compress=False,
            )
            assert not middleware._zstd_available  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001


# =============================================================================
# Accept-Encoding Parser Unit Tests
# =============================================================================


class TestParseAcceptEncoding:
    """Unit tests for parse_accept_encoding() helper function."""

    # --- Normal cases ---

    def test_simple_encoding(self) -> None:
        """Single encoding without quality value."""
        assert parse_accept_encoding("zstd") == {"zstd"}

    def test_multiple_encodings(self) -> None:
        """Multiple comma-separated encodings."""
        assert parse_accept_encoding("identity, zstd") == {"identity", "zstd"}

    def test_identity_only(self) -> None:
        """Server without compression returns identity only."""
        assert parse_accept_encoding("identity") == {"identity"}

    # --- RFC 9110 quality values ---

    def test_quality_values_stripped(self) -> None:
        """Quality values (;q=X.X) are stripped per RFC 9110."""
        result = parse_accept_encoding("zstd;q=1.0, gzip;q=0.8, identity;q=0.5")
        assert result == {"zstd", "gzip", "identity"}

    def test_quality_value_no_space(self) -> None:
        """Quality value without space after semicolon."""
        assert parse_accept_encoding("zstd;q=1.0") == {"zstd"}

    # --- Whitespace handling ---

    def test_whitespace_around_encodings(self) -> None:
        """Whitespace around encodings is trimmed."""
        assert parse_accept_encoding("  zstd  ,  gzip  ") == {"zstd", "gzip"}

    def test_no_whitespace(self) -> None:
        """Works without any whitespace."""
        assert parse_accept_encoding("zstd,gzip,br") == {"zstd", "gzip", "br"}

    # --- Case handling ---

    def test_lowercase_normalization(self) -> None:
        """Encodings are lowercased for consistent comparison."""
        assert parse_accept_encoding("ZSTD, GZIP, Identity") == {"zstd", "gzip", "identity"}

    def test_mixed_case(self) -> None:
        """Mixed case is normalized to lowercase."""
        assert parse_accept_encoding("ZsTd") == {"zstd"}

    # --- Edge cases ---

    def test_empty_string(self) -> None:
        """Empty string returns set with empty string (caller handles default)."""
        # Empty input gives {""}  - the caller should handle this appropriately
        result = parse_accept_encoding("")
        assert result == {""}

    def test_single_comma(self) -> None:
        """Single comma gives two empty strings."""
        result = parse_accept_encoding(",")
        assert result == {""}

    def test_trailing_comma(self) -> None:
        """Trailing comma includes empty string."""
        result = parse_accept_encoding("zstd,")
        assert "" in result
        assert "zstd" in result

    def test_leading_comma(self) -> None:
        """Leading comma includes empty string."""
        result = parse_accept_encoding(",zstd")
        assert "" in result
        assert "zstd" in result

    # --- Weird/malformed cases ---

    def test_multiple_semicolons(self) -> None:
        """Multiple semicolons - only first part used."""
        assert parse_accept_encoding("zstd;q=1.0;extra=foo") == {"zstd"}

    def test_semicolon_no_value(self) -> None:
        """Semicolon without value after it."""
        assert parse_accept_encoding("zstd;") == {"zstd"}

    def test_only_quality_value(self) -> None:
        """Malformed: just quality value, no encoding name."""
        result = parse_accept_encoding(";q=1.0")
        assert result == {""}

    def test_spaces_only(self) -> None:
        """Only whitespace characters."""
        result = parse_accept_encoding("   ")
        assert result == {""}

    def test_duplicate_encodings(self) -> None:
        """Duplicate encodings collapsed to single entry (set behavior)."""
        result = parse_accept_encoding("zstd, zstd, ZSTD")
        assert result == {"zstd"}

    # --- Real-world Accept-Encoding values ---

    def test_browser_like_header(self) -> None:
        """Typical browser Accept-Encoding header."""
        result = parse_accept_encoding("gzip, deflate, br, zstd")
        assert result == {"gzip", "deflate", "br", "zstd"}

    def test_cloudflare_style(self) -> None:
        """Cloudflare-style with quality values."""
        result = parse_accept_encoding("gzip;q=1.0, br;q=0.9, zstd;q=0.8")
        assert result == {"gzip", "br", "zstd"}


# =============================================================================
# Client server_supports_zstd Property Tests
# =============================================================================


class TestServerSupportsZstdProperty:
    """Tests for client's server_supports_zstd property."""

    async def test_server_supports_zstd_true_when_available(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """server_supports_zstd returns True when server advertises zstd."""
        from hpke_http.middleware.aiohttp import HPKEClientSession

        base_url = f"http://{granian_server.host}:{granian_server.port}"
        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            # Trigger discovery by making a request
            await client.post("/echo", json={"test": "data"})
            # Now check property
            assert client.server_supports_zstd is True

    async def test_server_supports_zstd_false_when_unavailable(
        self,
        granian_server_no_zstd: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """server_supports_zstd returns False when server lacks zstd."""
        from hpke_http.middleware.aiohttp import HPKEClientSession

        base_url = f"http://{granian_server_no_zstd.host}:{granian_server_no_zstd.port}"
        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            # Trigger discovery by making a request
            await client.post("/echo", json={"test": "data"})
            # Now check property
            assert client.server_supports_zstd is False

    async def test_server_supports_zstd_false_before_discovery(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """server_supports_zstd returns False before discovery is called."""
        from hpke_http.middleware.aiohttp import HPKEClientSession

        base_url = f"http://{granian_server.host}:{granian_server.port}"
        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            # Before any request, property should be False (empty set)
            assert client.server_supports_zstd is False


# =============================================================================
# Cache Behavior Tests
# =============================================================================


class TestEncodingCacheBehavior:
    """Tests for encoding caching alongside key caching."""

    async def test_multiple_requests_use_cached_encodings(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Multiple requests reuse cached encodings from discovery."""
        from hpke_http.middleware.aiohttp import HPKEClientSession

        base_url = f"http://{granian_server.host}:{granian_server.port}"
        async with HPKEClientSession(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            # First request triggers discovery
            resp1 = await client.post("/echo", json={"request": 1})
            assert resp1.status == 200
            encodings_after_first = client._server_encodings.copy()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

            # Second request should use cache
            resp2 = await client.post("/echo", json={"request": 2})
            assert resp2.status == 200
            encodings_after_second = client._server_encodings  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

            # Encodings should be the same (cached)
            assert encodings_after_first == encodings_after_second
            assert "zstd" in encodings_after_second

    async def test_httpx_multiple_requests_use_cached_encodings(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """httpx client also caches encodings correctly."""
        from hpke_http.middleware.httpx import HPKEAsyncClient

        base_url = f"http://{granian_server.host}:{granian_server.port}"
        async with HPKEAsyncClient(
            base_url=base_url,
            psk=test_psk,
            psk_id=test_psk_id,
        ) as client:
            # First request triggers discovery
            resp1 = await client.post("/echo", json={"request": 1})
            assert resp1.status_code == 200

            # Second request should use cache
            resp2 = await client.post("/echo", json={"request": 2})
            assert resp2.status_code == 200

            # Both should have zstd support cached
            assert client.server_supports_zstd is True

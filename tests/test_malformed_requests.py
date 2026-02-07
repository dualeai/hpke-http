"""E2E tests for malformed request handling with real granian server.

Tests that the server properly rejects invalid/malformed requests.
Uses raw aiohttp (not HPKEClientSession) to send intentionally broken requests.
"""

import aiohttp
import pytest

from hpke_http.constants import (
    CHACHA20_POLY1305_KEY_SIZE,
    HEADER_HPKE_ENC,
    HEADER_HPKE_ENCODING,
    HEADER_HPKE_ERROR,
    HEADER_HPKE_PSK_ID,
    HEADER_HPKE_STREAM,
    REQUEST_KEY_LABEL,
)
from hpke_http.headers import b64url_encode
from hpke_http.hpke import setup_sender_psk
from hpke_http.streaming import ChunkEncryptor, RawFormat, StreamingSession

from .conftest import E2EServer


class TestMalformedRequests:
    """Test server handling of malformed HPKE requests.

    Uses real granian server with raw aiohttp to send intentionally invalid requests.
    HPKEClientSession would prevent these malformed requests, so we bypass it.
    """

    @pytest.mark.parametrize(
        ("enc_header", "body", "expected_status", "description"),
        [
            ("not-valid-base64!!!", b"some body", 400, "invalid base64 in enc header"),
            ("dGVzdA==", b"short", 400, "truncated envelope body"),
        ],
    )
    async def test_malformed_hpke_request(
        self,
        granian_server: E2EServer,
        enc_header: str,
        body: bytes,
        expected_status: int,
        description: str,
    ) -> None:
        """Malformed HPKE request handling: {description}."""
        host, port = granian_server.host, granian_server.port

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                headers={
                    "X-HPKE-Enc": enc_header,
                    "Content-Type": "application/octet-stream",
                },
                data=body,
            ) as resp:
                assert resp.status == expected_status, f"Failed for {description}"

    async def test_plaintext_request_passes_through(
        self,
        granian_server: E2EServer,
        test_psk_id: bytes,
    ) -> None:
        """Plaintext request without X-HPKE-Enc header passes through (with PSK auth)."""
        host, port = granian_server.host, granian_server.port

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                json={"test": "plaintext"},
                headers={HEADER_HPKE_PSK_ID: b64url_encode(test_psk_id)},
            ) as resp:
                assert resp.status == 200

    async def test_health_endpoint_always_plaintext(
        self,
        granian_server: E2EServer,
    ) -> None:
        """Health endpoint works without encryption."""
        host, port = granian_server.host, granian_server.port

        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{host}:{port}/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"


def _encrypt_request(
    body: bytes,
    pk_r: bytes,
    psk: bytes,
    psk_id: bytes,
) -> tuple[bytes, str, str, str]:
    """Encrypt request body for testing using chunked streaming format.

    Returns:
        Tuple of (encrypted_body, enc_header_value, stream_header_value, psk_id_header_value)
    """
    ctx = setup_sender_psk(
        pk_r=pk_r,
        info=psk_id,
        psk=psk,
        psk_id=psk_id,
    )
    # Derive request key from HPKE context (matches HPKEClientSession)
    request_key = ctx.export(REQUEST_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
    session = StreamingSession.create(request_key)
    encryptor = ChunkEncryptor(session, format=RawFormat(), compress=False)

    # Encrypt body as single chunk
    encrypted_body = encryptor.encrypt(body) if body else encryptor.encrypt(b"")
    enc_header = b64url_encode(ctx.enc)
    stream_header = b64url_encode(session.session_salt)
    psk_id_header = b64url_encode(psk_id)
    return (encrypted_body, enc_header, stream_header, psk_id_header)


class TestMalformedCompressionHeaders:
    """Test server handling of invalid compression headers.

    Tests X-HPKE-Encoding header edge cases with properly encrypted requests.
    'zstd' and 'gzip' (lowercase) trigger decompression; unknown values are
    rejected with 415 Unsupported Media Type.
    """

    @pytest.mark.parametrize(
        ("encoding_value", "expected_status", "description"),
        [
            ("gzip", 400, "gzip with uncompressed body fails"),
            ("zstd", 400, "zstd with uncompressed body fails"),
            ("", 200, "empty header treated as identity (no compression)"),
            ("ZSTD", 415, "uppercase rejected (case-sensitive)"),
            ("GZIP", 415, "uppercase rejected (case-sensitive)"),
            ("deflate", 415, "deflate rejected as unsupported"),
            ("br", 415, "brotli rejected as unsupported"),
        ],
    )
    async def test_encoding_header_handling(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
        encoding_value: str,
        expected_status: int,
        description: str,
    ) -> None:
        """X-HPKE-Encoding header handling: {description}."""
        host, port, pk = granian_server.host, granian_server.port, granian_server.public_key
        body = b'{"test": "compression header test"}'

        encrypted_body, enc_header, stream_header, psk_id_header = _encrypt_request(body, pk, test_psk, test_psk_id)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                headers={
                    HEADER_HPKE_ENC: enc_header,
                    HEADER_HPKE_STREAM: stream_header,
                    HEADER_HPKE_ENCODING: encoding_value,
                    HEADER_HPKE_PSK_ID: psk_id_header,
                    "Content-Type": "application/octet-stream",
                },
                data=encrypted_body,
            ) as resp:
                assert resp.status == expected_status, f"Failed for {description}"


class TestEncodingValidation:
    """Test that unknown encodings are rejected with 415 (Fix #5)."""

    async def test_identity_encoding_accepted(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """X-HPKE-Encoding: identity -> 200."""
        host, port, pk = granian_server.host, granian_server.port, granian_server.public_key
        body = b'{"test": "identity"}'
        encrypted_body, enc_header, stream_header, psk_id_header = _encrypt_request(body, pk, test_psk, test_psk_id)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                headers={
                    HEADER_HPKE_ENC: enc_header,
                    HEADER_HPKE_STREAM: stream_header,
                    HEADER_HPKE_ENCODING: "identity",
                    HEADER_HPKE_PSK_ID: psk_id_header,
                    "Content-Type": "application/octet-stream",
                },
                data=encrypted_body,
            ) as resp:
                assert resp.status == 200

    async def test_brotli_encoding_rejected(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """X-HPKE-Encoding: brotli -> 415."""
        host, port, pk = granian_server.host, granian_server.port, granian_server.public_key
        body = b'{"test": "brotli"}'
        encrypted_body, enc_header, stream_header, psk_id_header = _encrypt_request(body, pk, test_psk, test_psk_id)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                headers={
                    HEADER_HPKE_ENC: enc_header,
                    HEADER_HPKE_STREAM: stream_header,
                    HEADER_HPKE_ENCODING: "brotli",
                    HEADER_HPKE_PSK_ID: psk_id_header,
                    "Content-Type": "application/octet-stream",
                },
                data=encrypted_body,
            ) as resp:
                assert resp.status == 415

    async def test_injection_attempt_rejected(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """X-HPKE-Encoding: ../../etc/passwd -> 415."""
        host, port, pk = granian_server.host, granian_server.port, granian_server.public_key
        body = b'{"test": "injection"}'
        encrypted_body, enc_header, stream_header, psk_id_header = _encrypt_request(body, pk, test_psk, test_psk_id)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                headers={
                    HEADER_HPKE_ENC: enc_header,
                    HEADER_HPKE_STREAM: stream_header,
                    HEADER_HPKE_ENCODING: "../../etc/passwd",
                    HEADER_HPKE_PSK_ID: psk_id_header,
                    "Content-Type": "application/octet-stream",
                },
                data=encrypted_body,
            ) as resp:
                assert resp.status == 415


class TestErrorResponseHeaders:
    """Test X-HPKE-Error header on error responses (Fix #10)."""

    async def test_error_response_has_hpke_error_header(
        self,
        granian_server: E2EServer,
    ) -> None:
        """Error responses include X-HPKE-Error: true header."""
        host, port = granian_server.host, granian_server.port

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                headers={"X-HPKE-Enc": "not-valid-base64!!!"},
                data=b"body",
            ) as resp:
                assert resp.status == 400
                assert resp.headers.get(HEADER_HPKE_ERROR) == "true"

    async def test_success_response_no_hpke_error_header(
        self,
        granian_server: E2EServer,
    ) -> None:
        """Successful health endpoint does NOT have X-HPKE-Error header."""
        host, port = granian_server.host, granian_server.port

        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{host}:{port}/health") as resp:
                assert resp.status == 200
                assert HEADER_HPKE_ERROR not in resp.headers

    async def test_missing_psk_error_has_header(
        self,
        granian_server: E2EServer,
    ) -> None:
        """401 (missing PSK) has X-HPKE-Error header."""
        host, port = granian_server.host, granian_server.port

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                json={"test": "no-psk"},
            ) as resp:
                assert resp.status == 401
                assert resp.headers.get(HEADER_HPKE_ERROR) == "true"


class TestPSKIDLengthBound:
    """Test PSK ID length validation (Fix #14)."""

    async def test_normal_psk_id_accepted(
        self,
        granian_server: E2EServer,
        test_psk_id: bytes,
    ) -> None:
        """Normal 10-byte PSK ID accepted."""
        host, port = granian_server.host, granian_server.port

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                json={"test": "psk"},
                headers={HEADER_HPKE_PSK_ID: b64url_encode(test_psk_id)},
            ) as resp:
                assert resp.status == 200

    async def test_large_psk_id_rejected(
        self,
        granian_server: E2EServer,
    ) -> None:
        """1KB PSK ID rejected with 400."""
        host, port = granian_server.host, granian_server.port
        large_psk_id = b"x" * 1024

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                json={"test": "large-psk"},
                headers={HEADER_HPKE_PSK_ID: b64url_encode(large_psk_id)},
            ) as resp:
                assert resp.status == 400

    async def test_128_byte_psk_id_accepted(
        self,
        granian_server: E2EServer,
    ) -> None:
        """128-byte PSK ID (at limit) accepted."""
        host, port = granian_server.host, granian_server.port
        psk_id_at_limit = b"x" * 128

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                json={"test": "limit-psk"},
                headers={HEADER_HPKE_PSK_ID: b64url_encode(psk_id_at_limit)},
            ) as resp:
                # May get 401 because this PSK ID doesn't match the server's,
                # but should NOT get 400 for size
                assert resp.status in (200, 401)

    async def test_129_byte_psk_id_rejected(
        self,
        granian_server: E2EServer,
    ) -> None:
        """129-byte PSK ID (over limit) rejected with 400."""
        host, port = granian_server.host, granian_server.port
        psk_id_over_limit = b"x" * 129

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                json={"test": "over-limit-psk"},
                headers={HEADER_HPKE_PSK_ID: b64url_encode(psk_id_over_limit)},
            ) as resp:
                assert resp.status == 400

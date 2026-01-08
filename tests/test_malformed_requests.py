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
    ) -> None:
        """Plaintext request without X-HPKE-Enc header passes through."""
        host, port = granian_server.host, granian_server.port

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                json={"test": "plaintext"},
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
    # Derive request key from HPKE context (matches HPKEClientSession)
    request_key = ctx.export(REQUEST_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
    session = StreamingSession.create(request_key)
    encryptor = ChunkEncryptor(session, format=RawFormat(), compress=False)

    # Encrypt body as single chunk
    encrypted_body = encryptor.encrypt(body) if body else encryptor.encrypt(b"")
    enc_header = b64url_encode(ctx.enc)
    stream_header = b64url_encode(session.session_salt)
    return (encrypted_body, enc_header, stream_header)


class TestMalformedCompressionHeaders:
    """Test server handling of invalid compression headers.

    Tests X-HPKE-Encoding header edge cases with properly encrypted requests.
    Only 'zstd' (lowercase) triggers decompression; other values are ignored.
    """

    @pytest.mark.parametrize(
        ("encoding_value", "expected_status", "description"),
        [
            ("gzip", 200, "invalid encoding ignored"),
            ("zstd", 400, "zstd with uncompressed body fails"),
            ("", 200, "empty header treated as identity"),
            ("ZSTD", 200, "uppercase ignored (case-sensitive)"),
            ("deflate", 200, "deflate ignored"),
            ("br", 200, "brotli ignored"),
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

        encrypted_body, enc_header, stream_header = _encrypt_request(body, pk, test_psk, test_psk_id)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                headers={
                    HEADER_HPKE_ENC: enc_header,
                    HEADER_HPKE_STREAM: stream_header,
                    HEADER_HPKE_ENCODING: encoding_value,
                    "Content-Type": "application/octet-stream",
                },
                data=encrypted_body,
            ) as resp:
                assert resp.status == expected_status, f"Failed for {description}"

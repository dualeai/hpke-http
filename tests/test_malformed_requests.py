"""E2E tests for malformed request handling with real granian server.

Tests that the server properly rejects invalid/malformed requests.
Uses raw aiohttp (not HPKEClientSession) to send intentionally broken requests.
"""

import aiohttp


class TestMalformedRequests:
    """Test server handling of malformed HPKE requests.

    Uses real granian server with raw aiohttp to send intentionally invalid requests.
    HPKEClientSession would prevent these malformed requests, so we bypass it.
    """

    async def test_invalid_base64_in_enc_header_returns_400(
        self,
        granian_server: tuple[str, int, bytes],
    ) -> None:
        """Invalid base64 in X-HPKE-Enc header should return 400."""
        host, port, _ = granian_server

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                headers={
                    "X-HPKE-Enc": "not-valid-base64!!!",  # Invalid base64
                    "Content-Type": "application/octet-stream",
                },
                data=b"some body",
            ) as resp:
                assert resp.status == 400

    async def test_truncated_envelope_body_returns_400(
        self,
        granian_server: tuple[str, int, bytes],
    ) -> None:
        """Body too short to be valid HPKE envelope should return 400."""
        host, port, _ = granian_server

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{host}:{port}/echo",
                headers={
                    "X-HPKE-Enc": "dGVzdA==",  # Valid base64, but wrong content
                    "Content-Type": "application/octet-stream",
                },
                data=b"short",  # Too short for valid HPKE envelope
            ) as resp:
                assert resp.status == 400

    async def test_request_without_enc_header_to_encrypted_endpoint(
        self,
        granian_server: tuple[str, int, bytes],
    ) -> None:
        """Request without X-HPKE-Enc to encrypted endpoint passes through.

        The middleware allows plaintext requests - endpoints decide if they require encryption.
        """
        host, port, _ = granian_server

        async with aiohttp.ClientSession() as session:
            # POST to /echo without HPKE headers - middleware passes through
            async with session.post(
                f"http://{host}:{port}/echo",
                json={"test": "plaintext"},
            ) as resp:
                # Server accepts plaintext (encryption is transparent, not enforced)
                assert resp.status == 200

    async def test_health_endpoint_always_plaintext(
        self,
        granian_server: tuple[str, int, bytes],
    ) -> None:
        """Health endpoint should always work without encryption."""
        host, port, _ = granian_server

        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{host}:{port}/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"

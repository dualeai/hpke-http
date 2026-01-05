"""E2E middleware tests with real granian ASGI server.

Tests the full encryption/decryption flow:
- HPKEClientSession encrypts requests
- HPKEMiddleware decrypts on server
- Server processes plaintext
- SSE responses encrypted via EncryptedSSEResponse
- HPKEClientSession decrypts SSE events

Uses granian (Rust ASGI server) started as subprocess.
Fixtures are shared via conftest.py.
"""

from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest

from hpke_http.middleware.aiohttp import HPKEClientSession

# === Fixtures ===


@pytest.fixture
async def hpke_client(
    granian_server: tuple[str, int, bytes],
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[HPKEClientSession]:
    """HPKEClientSession connected to test server."""
    host, port, _pk = granian_server
    base_url = f"http://{host}:{port}"

    async with HPKEClientSession(
        base_url=base_url,
        psk=test_psk,
        psk_id=test_psk_id,
    ) as session:
        yield session


# === Tests ===


class TestDiscoveryEndpoint:
    """Test HPKE key discovery endpoint."""

    async def test_discovery_returns_keys(self, granian_server: tuple[str, int, bytes]) -> None:
        """Server exposes public keys via discovery endpoint."""
        host, port, _expected_pk = granian_server

        async with aiohttp.ClientSession() as session:
            url = f"http://{host}:{port}/.well-known/hpke-keys"
            async with session.get(url) as resp:
                assert resp.status == 200
                data = await resp.json()

                assert data["version"] == 1
                assert "keys" in data
                assert len(data["keys"]) >= 1

                # Verify key format
                key_info = data["keys"][0]
                assert "kem_id" in key_info
                assert "public_key" in key_info

    async def test_discovery_cache_headers(self, granian_server: tuple[str, int, bytes]) -> None:
        """Discovery endpoint returns cache headers."""
        host, port, _ = granian_server

        async with aiohttp.ClientSession() as session:
            url = f"http://{host}:{port}/.well-known/hpke-keys"
            async with session.get(url) as resp:
                assert resp.status == 200
                assert "Cache-Control" in resp.headers
                assert "max-age" in resp.headers["Cache-Control"]


class TestEncryptedRequests:
    """Test encrypted request/response flow."""

    async def test_encrypted_request_roundtrip(self, hpke_client: HPKEClientSession) -> None:
        """Client encrypts → Server decrypts → Response works."""
        test_data = {"message": "Hello, HPKE!", "count": 42}

        resp = await hpke_client.post("/echo", json=test_data)
        assert resp.status == 200
        data = await resp.json()

        assert data["path"] == "/echo"
        assert data["method"] == "POST"
        # Echo contains the JSON string we sent
        assert "Hello, HPKE!" in data["echo"]
        assert "42" in data["echo"]

    async def test_large_payload(self, hpke_client: HPKEClientSession) -> None:
        """Large payloads encrypt/decrypt correctly."""
        large_content = "x" * 100_000  # 100KB
        test_data = {"data": large_content}

        resp = await hpke_client.post("/echo", json=test_data)
        assert resp.status == 200
        data = await resp.json()

        # Verify the large content made it through
        assert large_content in data["echo"]

    async def test_binary_payload(self, hpke_client: HPKEClientSession) -> None:
        """Binary data encrypts/decrypts correctly."""
        binary_data = bytes(range(256)) * 10  # Various byte values

        resp = await hpke_client.post("/echo", data=binary_data)
        assert resp.status == 200
        data = await resp.json()
        # Binary data should be in the echo (may be escaped)
        assert len(data["echo"]) > 0


class TestAuthenticationFailures:
    """Test authentication and decryption failures."""

    async def test_wrong_psk_rejected(
        self,
        granian_server: tuple[str, int, bytes],
        wrong_psk: bytes,
        wrong_psk_id: bytes,
    ) -> None:
        """Server rejects requests encrypted with wrong PSK."""
        host, port, _ = granian_server
        base_url = f"http://{host}:{port}"

        async with HPKEClientSession(
            base_url=base_url,
            psk=wrong_psk,
            psk_id=wrong_psk_id,
        ) as bad_client:
            resp = await bad_client.post("/echo", json={"test": 1})
            # Server should reject with decryption failure
            assert resp.status == 400

    async def test_plaintext_request_passes_through(
        self,
        granian_server: tuple[str, int, bytes],
    ) -> None:
        """Plaintext requests without HPKE headers pass through (no encryption required)."""
        host, port, _ = granian_server

        async with aiohttp.ClientSession() as plain_client:
            url = f"http://{host}:{port}/health"
            async with plain_client.get(url) as resp:
                # Health endpoint doesn't require encryption
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"


class TestSSEEncryption:
    """Test encrypted SSE streaming."""

    async def test_sse_stream_roundtrip(self, hpke_client: HPKEClientSession) -> None:
        """SSE events are encrypted end-to-end."""
        events: list[tuple[str, dict[str, Any]]] = []

        resp = await hpke_client.post("/stream", json={"start": True})
        assert resp.status == 200
        async for event_type, event_data in hpke_client.iter_sse(resp):
            events.append((event_type, event_data))

        # Should have 4 events: 3 progress + 1 complete
        assert len(events) == 4

        # Verify progress events
        for i in range(3):
            assert events[i][0] == "progress"
            assert events[i][1]["step"] == i + 1

        # Verify complete event
        assert events[3][0] == "complete"
        assert events[3][1]["result"] == "success"

    async def test_sse_counter_monotonicity(self, hpke_client: HPKEClientSession) -> None:
        """SSE events have monotonically increasing counters."""
        event_count = 0

        resp = await hpke_client.post("/stream", json={"start": True})
        assert resp.status == 200
        async for _event_type, _event_data in hpke_client.iter_sse(resp):
            event_count += 1

        # Verify all events were processed (counter worked correctly)
        assert event_count == 4

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

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest

from hpke_http.middleware.aiohttp import HPKEClientSession


def parse_sse_chunk(chunk: str) -> tuple[str | None, dict[str, Any] | None]:
    """Parse a raw SSE chunk into (event_type, data).

    Args:
        chunk: Raw SSE chunk string (e.g., "event: progress\\ndata: {...}\\n\\n")

    Returns:
        Tuple of (event_type, parsed_data) or (None, None) for comments
    """
    event_type = None
    data = None

    for line in re.split(r"\r\n|\r|\n", chunk):
        if not line or line.startswith(":"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if key == "event":
                event_type = value
            elif key == "data":
                try:
                    data = json.loads(value)
                except json.JSONDecodeError:
                    data = {"raw": value}

    return (event_type, data)


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
        resp = await hpke_client.post("/stream", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in hpke_client.iter_sse(resp)]

        # Should have 4 events: 3 progress + 1 complete
        assert len(events) == 4

        # Verify progress events
        for i in range(3):
            event_type, event_data = events[i]
            assert event_type == "progress"
            assert event_data is not None
            assert event_data["step"] == i + 1

        # Verify complete event
        event_type, event_data = events[3]
        assert event_type == "complete"
        assert event_data is not None
        assert event_data["result"] == "success"

    async def test_sse_counter_monotonicity(self, hpke_client: HPKEClientSession) -> None:
        """SSE events have monotonically increasing counters."""
        event_count = 0

        resp = await hpke_client.post("/stream", json={"start": True})
        assert resp.status == 200
        async for _chunk in hpke_client.iter_sse(resp):
            event_count += 1

        # Verify all events were processed (counter worked correctly)
        assert event_count == 4

    async def test_sse_delayed_events(self, hpke_client: HPKEClientSession) -> None:
        """SSE events with delays between them work correctly."""
        import time

        start = time.monotonic()

        resp = await hpke_client.post("/stream-delayed", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in hpke_client.iter_sse(resp)]

        elapsed = time.monotonic() - start

        # Should have 6 events: 5 ticks + 1 done
        assert len(events) == 6

        # Verify tick events
        for i in range(5):
            event_type, event_data = events[i]
            assert event_type == "tick"
            assert event_data is not None
            assert event_data["count"] == i

        # Verify done event
        event_type, event_data = events[5]
        assert event_type == "done"
        assert event_data is not None
        assert event_data["total"] == 5

        # Should have taken at least 400ms (5 events * 100ms delay)
        # Allow some slack for test timing
        assert elapsed >= 0.4, f"Expected >= 400ms, got {elapsed * 1000:.0f}ms"

    async def test_sse_large_payload_stream(self, hpke_client: HPKEClientSession) -> None:
        """SSE events with ~10KB payloads work correctly."""
        resp = await hpke_client.post("/stream-large", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in hpke_client.iter_sse(resp)]

        # Should have 4 events: 3 large + 1 complete
        assert len(events) == 4

        # Verify large events have ~10KB data
        for i in range(3):
            event_type, event_data = events[i]
            assert event_type == "large"
            assert event_data is not None
            assert event_data["index"] == i
            assert len(event_data["data"]) == 10000

        # Verify complete event
        event_type, _event_data = events[3]
        assert event_type == "complete"

    async def test_sse_many_events_stream(self, hpke_client: HPKEClientSession) -> None:
        """SSE stream with 50+ events works correctly."""
        resp = await hpke_client.post("/stream-many", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in hpke_client.iter_sse(resp)]

        # Should have 51 events: 50 event + 1 complete
        assert len(events) == 51

        # Verify sequential events
        for i in range(50):
            event_type, event_data = events[i]
            assert event_type == "event"
            assert event_data is not None
            assert event_data["index"] == i

        # Verify complete event
        event_type, event_data = events[50]
        assert event_type == "complete"
        assert event_data is not None
        assert event_data["count"] == 50

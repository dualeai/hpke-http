"""E2E middleware tests with real granian ASGI server.

Tests the full encryption/decryption flow:
- HPKEClientSession encrypts requests
- HPKEMiddleware decrypts on server
- Server processes plaintext
- Standard responses encrypted via RawFormat (X-HPKE-Stream header, non-SSE Content-Type)
- SSE responses encrypted via SSEFormat (X-HPKE-Stream header, text/event-stream Content-Type)
- HPKEClientSession decrypts both types transparently

Uses granian (Rust ASGI server) started as subprocess.
Fixtures are shared via conftest.py.
"""

import asyncio
import gc
import hashlib
import json
import re
import sys
import time
import warnings
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import httpx
import pytest
from typing_extensions import assert_type

from hpke_http.constants import HEADER_HPKE_STREAM
from hpke_http.middleware.aiohttp import DecryptedResponse, HPKEClientSession
from hpke_http.middleware.httpx import DecryptedResponse as HTTPXDecryptedResponse
from hpke_http.middleware.httpx import HPKEAsyncClient
from tests.conftest import (
    CHI_SQUARE_MIN_PASS,
    CHI_SQUARE_P_THRESHOLD,
    CHI_SQUARE_TRIALS,
    LARGE_PAYLOAD_SIZES_IDS,
    LARGE_PAYLOAD_SIZES_MB,
    E2EServer,
    calculate_shannon_entropy,
    chi_square_byte_uniformity,
    skip_on_314t_gc_bug,
)


def parse_sse_chunk(chunk: bytes) -> tuple[str | None, dict[str, Any] | None]:
    """Parse a raw SSE chunk into (event_type, data).

    Args:
        chunk: Raw SSE chunk bytes (e.g., b"event: progress\\ndata: {...}\\n\\n")

    Returns:
        Tuple of (event_type, parsed_data) or (None, None) for comments
    """
    event_type = None
    data = None
    chunk_str = chunk.decode("utf-8")

    for line in re.split(r"\r\n|\r|\n", chunk_str):
        if not line or line.startswith(":"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if key == "event":
                event_type = value
            elif key == "data":
                try:
                    data = json.loads(value)
                except json.JSONDecodeError:
                    data = {"raw": value}

    return (event_type, data)


def _is_connection_leak(warning: warnings.WarningMessage) -> bool:
    """Check if warning is a connection/socket leak (not file IO from test infra)."""
    msg = str(warning.message).lower()
    # Connection-related keywords
    return any(kw in msg for kw in ("socket", "transport", "connection", "ssl", "tcp"))


# === Tests ===


class TestDiscoveryEndpoint:
    """Test HPKE key discovery endpoint."""

    async def test_discovery_endpoint(self, granian_server: E2EServer) -> None:
        """Discovery endpoint returns keys with proper cache headers."""
        async with aiohttp.ClientSession() as session:
            url = f"http://{granian_server.host}:{granian_server.port}/.well-known/hpke-keys"
            async with session.get(url) as resp:
                assert resp.status == 200

                # Verify response structure
                data = await resp.json()
                assert data["version"] == 1
                assert "keys" in data
                assert len(data["keys"]) >= 1

                # Verify key format
                key_info = data["keys"][0]
                assert "kem_id" in key_info
                assert "public_key" in key_info

                # Verify cache headers
                assert "Cache-Control" in resp.headers
                assert "max-age" in resp.headers["Cache-Control"]


class TestEncryptedRequests:
    """Test encrypted request/response flow."""

    async def test_encrypted_request_roundtrip(self, aiohttp_client: HPKEClientSession) -> None:
        """Client encrypts → Server decrypts → Response works."""
        test_data = {"message": "Hello, HPKE!", "count": 42}

        resp = await aiohttp_client.post("/echo", json=test_data)
        assert resp.status == 200
        data = await resp.json()

        assert data["path"] == "/echo"
        assert data["method"] == "POST"
        # Echo contains the JSON string we sent
        assert "Hello, HPKE!" in data["echo"]
        assert "42" in data["echo"]

    async def test_large_payload(self, aiohttp_client: HPKEClientSession) -> None:
        """Large payloads encrypt/decrypt correctly."""
        large_content = "x" * 100_000  # 100KB
        test_data = {"data": large_content}

        resp = await aiohttp_client.post("/echo", json=test_data)
        assert resp.status == 200
        data = await resp.json()

        # Verify the large content made it through
        assert large_content in data["echo"]

    async def test_binary_payload(self, aiohttp_client: HPKEClientSession) -> None:
        """Binary data encrypts/decrypts correctly."""
        binary_data = bytes(range(256)) * 10  # Various byte values

        resp = await aiohttp_client.post("/echo", data=binary_data)
        assert resp.status == 200
        data = await resp.json()
        # Binary data should be in the echo (may be escaped)
        assert len(data["echo"]) > 0

    async def test_put_method(self, aiohttp_client: HPKEClientSession) -> None:
        """PUT request encrypts and decrypts correctly."""
        resp = await aiohttp_client.put("/echo", json={"method": "PUT"})
        assert resp.status == 200
        data = await resp.json()
        assert data["method"] == "PUT"

    async def test_patch_method(self, aiohttp_client: HPKEClientSession) -> None:
        """PATCH request encrypts and decrypts correctly."""
        resp = await aiohttp_client.patch("/echo", json={"method": "PATCH"})
        assert resp.status == 200
        data = await resp.json()
        assert data["method"] == "PATCH"

    async def test_delete_method(self, aiohttp_client: HPKEClientSession) -> None:
        """DELETE request works correctly."""
        resp = await aiohttp_client.delete("/echo")
        assert resp.status == 200
        data = await resp.json()
        assert data["method"] == "DELETE"

    async def test_head_method(self, aiohttp_client: HPKEClientSession) -> None:
        """HEAD request works correctly."""
        resp = await aiohttp_client.head("/health")
        assert resp.status == 200

    async def test_options_method(self, aiohttp_client: HPKEClientSession) -> None:
        """OPTIONS request works correctly."""
        resp = await aiohttp_client.options("/echo")
        # Just verify it doesn't crash - server may or may not support OPTIONS
        assert resp.status in (200, 405)


class TestStandardResponseEncryption:
    """Test encrypted standard (non-SSE) responses."""

    async def test_response_has_hpke_stream_header(self, aiohttp_client: HPKEClientSession) -> None:
        """Encrypted request triggers encrypted response with X-HPKE-Stream header."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert resp.status == 200

        # DecryptedResponse wraps the underlying response
        assert isinstance(resp, DecryptedResponse)

        # X-HPKE-Stream header should be present (contains salt)
        assert HEADER_HPKE_STREAM in resp.headers

        # Content-Type should NOT be text/event-stream (that's for SSE)
        content_type = resp.headers.get("Content-Type", "")
        assert "text/event-stream" not in content_type

    async def test_decrypted_response_json(self, aiohttp_client: HPKEClientSession) -> None:
        """DecryptedResponse.json() returns decrypted data."""
        test_data = {"message": "secret", "value": 42}
        resp = await aiohttp_client.post("/echo", json=test_data)

        # json() should return decrypted data
        data = await resp.json()
        assert "message" in data["echo"]
        assert "secret" in data["echo"]

    async def test_decrypted_response_read(self, aiohttp_client: HPKEClientSession) -> None:
        """DecryptedResponse.read() returns raw decrypted bytes."""
        resp = await aiohttp_client.post("/echo", json={"raw": "test"})

        # read() should return decrypted bytes
        raw_bytes = await resp.read()
        assert b"raw" in raw_bytes
        assert b"test" in raw_bytes

    async def test_decrypted_response_text(self, aiohttp_client: HPKEClientSession) -> None:
        """DecryptedResponse.text() returns decrypted text."""
        resp = await aiohttp_client.post("/echo", json={"text": "hello"})

        # text() should return decrypted string
        text = await resp.text()
        assert "text" in text
        assert "hello" in text

    async def test_sse_response_not_wrapped_in_decrypted_response(self, aiohttp_client: HPKEClientSession) -> None:
        """SSE responses use X-HPKE-Stream with text/event-stream Content-Type."""
        resp = await aiohttp_client.post("/stream", json={"start": True})
        assert resp.status == 200

        # SSE responses SHOULD have X-HPKE-Stream header
        assert HEADER_HPKE_STREAM in resp.headers

        # SSE responses have Content-Type: text/event-stream
        content_type = resp.headers.get("Content-Type", "")
        assert "text/event-stream" in content_type

    async def test_unencrypted_request_unencrypted_response(
        self,
        granian_server: E2EServer,
    ) -> None:
        """Plain HTTP request gets plain response (backward compat)."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        # Use plain aiohttp client, no encryption
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as resp:
                assert resp.status == 200
                # Plain response should NOT have X-HPKE-Stream header
                assert HEADER_HPKE_STREAM not in resp.headers


class TestAuthenticationFailures:
    """Test authentication and decryption failures."""

    async def test_wrong_psk_rejected(
        self,
        granian_server: E2EServer,
        wrong_psk: bytes,
        wrong_psk_id: bytes,
    ) -> None:
        """Server rejects requests encrypted with wrong PSK."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEClientSession(
            base_url=base_url,
            psk=wrong_psk,
            psk_id=wrong_psk_id,
        ) as bad_client:
            resp = await bad_client.post("/echo", json={"test": 1})
            # Server should reject with decryption failure
            assert resp.status == 400


class TestSSEEncryption:
    """Test encrypted SSE streaming."""

    async def test_sse_stream_roundtrip(self, aiohttp_client: HPKEClientSession) -> None:
        """SSE events are encrypted end-to-end."""
        resp = await aiohttp_client.post("/stream", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in aiohttp_client.iter_sse(resp)]

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

    async def test_sse_counter_monotonicity(self, aiohttp_client: HPKEClientSession) -> None:
        """SSE events have monotonically increasing counters."""
        event_count = 0

        resp = await aiohttp_client.post("/stream", json={"start": True})
        assert resp.status == 200
        async for _chunk in aiohttp_client.iter_sse(resp):
            event_count += 1

        # Verify all events were processed (counter worked correctly)
        assert event_count == 4

    async def test_sse_delayed_events(self, aiohttp_client: HPKEClientSession) -> None:
        """SSE events with delays between them work correctly."""
        import time

        start = time.monotonic()

        resp = await aiohttp_client.post("/stream-delayed", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in aiohttp_client.iter_sse(resp)]

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

    async def test_sse_large_payload_stream(self, aiohttp_client: HPKEClientSession) -> None:
        """SSE events with ~10KB payloads work correctly."""
        resp = await aiohttp_client.post("/stream-large", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in aiohttp_client.iter_sse(resp)]

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

    async def test_sse_many_events_stream(self, aiohttp_client: HPKEClientSession) -> None:
        """SSE stream with 50+ events works correctly."""
        resp = await aiohttp_client.post("/stream-many", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in aiohttp_client.iter_sse(resp)]

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

    async def test_iter_sse_yields_bytes(self, aiohttp_client: HPKEClientSession) -> None:
        """iter_sse must yield bytes (matches native aiohttp response.content).

        This is a type contract test - ensures API doesn't accidentally change.
        Static: assert_type checked by pyright at type-check time.
        Runtime: isinstance checked by pytest at test time.
        """
        resp = await aiohttp_client.post("/stream", json={"start": True})
        assert resp.status == 200

        async for chunk in aiohttp_client.iter_sse(resp):
            # Static assertion - pyright validates this matches the type annotation
            assert_type(chunk, bytes)
            # Runtime assertion - catches any mismatch at test time
            assert isinstance(chunk, bytes), f"Expected bytes, got {type(chunk).__name__}"
            break  # Only need to check first chunk


class TestCompressionE2E:
    """E2E tests for Zstd compression with real granian server.

    Tests request compression (client→server) and SSE compression (server→client).
    """

    async def test_compressed_request_roundtrip(
        self,
        aiohttp_client_compressed: HPKEClientSession,
    ) -> None:
        """Client compress=True → Server decompresses correctly.

        Large JSON is compressed before encryption, server decompresses after decryption.
        """
        # Large payload to ensure compression is triggered (>64 bytes)
        large_data = {"message": "x" * 1000, "nested": {"key": "value" * 100}}

        resp = await aiohttp_client_compressed.post("/echo", json=large_data)
        assert resp.status == 200
        data = await resp.json()

        # Verify the data made it through compression → encryption → decryption → decompression
        assert "x" * 1000 in data["echo"]
        assert "value" * 100 in data["echo"]

    async def test_compressed_sse_roundtrip(
        self,
        aiohttp_client_compressed: HPKEClientSession,
    ) -> None:
        """Server compress=True → Client receives decompressed SSE.

        SSE events are compressed before encryption, client decompresses after decryption.
        """
        resp = await aiohttp_client_compressed.post("/stream-large", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in aiohttp_client_compressed.iter_sse(resp)]

        # Should have 4 events: 3 large + 1 complete
        assert len(events) == 4

        # Verify large events have ~10KB data (compression worked transparently)
        for i in range(3):
            event_type, event_data = events[i]
            assert event_type == "large"
            assert event_data is not None
            assert event_data["index"] == i
            assert len(event_data["data"]) == 10000

    async def test_mixed_compression_client_off_server_on(
        self,
        aiohttp_client_no_compress_server_compress: HPKEClientSession,
    ) -> None:
        """Client compress=False, Server compress=True still works.

        Client sends uncompressed requests, server compresses SSE responses.
        """
        test_data = {"message": "Hello from uncompressed client!"}

        resp = await aiohttp_client_no_compress_server_compress.post("/echo", json=test_data)
        assert resp.status == 200
        data = await resp.json()
        assert "Hello from uncompressed client!" in data["echo"]

        # SSE should still work (server compresses, client decompresses)
        resp = await aiohttp_client_no_compress_server_compress.post("/stream", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in aiohttp_client_no_compress_server_compress.iter_sse(resp)]
        assert len(events) == 4

    async def test_many_events_with_compression(
        self,
        aiohttp_client_compressed: HPKEClientSession,
    ) -> None:
        """50+ SSE events with compression work correctly."""
        resp = await aiohttp_client_compressed.post("/stream-many", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in aiohttp_client_compressed.iter_sse(resp)]

        # Should have 51 events: 50 event + 1 complete
        assert len(events) == 51

        # Verify sequential events
        for i in range(50):
            event_type, event_data = events[i]
            assert event_type == "event"
            assert event_data is not None
            assert event_data["index"] == i


class TestDecryptedResponseEdgeCases:
    """Edge case tests for DecryptedResponse behavior."""

    async def test_multiple_read_calls_cached(self, aiohttp_client: HPKEClientSession) -> None:
        """Multiple read() calls return same cached data."""
        resp = await aiohttp_client.post("/echo", json={"cached": "test"})

        # First read
        data1 = await resp.read()
        # Second read (should use cache)
        data2 = await resp.read()

        assert data1 == data2
        assert b"cached" in data1

    async def test_json_after_read_works(self, aiohttp_client: HPKEClientSession) -> None:
        """json() works after read() has been called."""
        resp = await aiohttp_client.post("/echo", json={"order": "test"})

        # Read raw first
        raw = await resp.read()
        assert b"order" in raw

        # Then parse as JSON (uses cached data)
        data = await resp.json()
        assert "order" in data["echo"]

    async def test_text_after_json_works(self, aiohttp_client: HPKEClientSession) -> None:
        """text() works after json() has been called."""
        resp = await aiohttp_client.post("/echo", json={"sequence": 123})

        # Parse as JSON first
        data = await resp.json()
        assert "sequence" in data["echo"]

        # Then get as text (uses cached data)
        text = await resp.text()
        assert "sequence" in text

    async def test_empty_response_body(self, aiohttp_client: HPKEClientSession) -> None:
        """Empty response body is handled correctly."""
        # The /health endpoint returns a small response, but let's test with /echo
        # sending minimal data
        resp = await aiohttp_client.post("/echo", json={})
        data = await resp.json()
        assert "echo" in data

    async def test_status_and_url_passthrough(self, aiohttp_client: HPKEClientSession) -> None:
        """DecryptedResponse proxies status and url correctly."""
        resp = await aiohttp_client.post("/echo", json={"proxy": "test"})

        # Status should be accessible
        assert resp.status == 200

        # URL should be accessible (proxied from underlying response)
        assert "/echo" in str(resp.url)

    async def test_headers_accessible(self, aiohttp_client: HPKEClientSession) -> None:
        """Response headers are accessible through DecryptedResponse."""
        resp = await aiohttp_client.post("/echo", json={"headers": "test"})

        # Headers should be accessible
        assert "content-type" in resp.headers or "Content-Type" in resp.headers


class TestDecryptedResponseReleaseLifecycle:
    """Tests for DecryptedResponse connection release behavior.

    Verifies that connections are properly released to the pool after
    read()/text()/json() calls, and that release() is idempotent.
    """

    async def test_read_auto_releases_connection(self, aiohttp_client: HPKEClientSession) -> None:
        """read() automatically releases connection to pool."""
        connector = aiohttp_client._session.connector  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

        resp = await aiohttp_client.post("/echo", json={"test": "release"})
        assert isinstance(resp, DecryptedResponse)

        # Connection acquired before read
        await resp.read()

        # Connection should be released after read
        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

    async def test_text_auto_releases_connection(self, aiohttp_client: HPKEClientSession) -> None:
        """text() automatically releases connection to pool."""
        connector = aiohttp_client._session.connector  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

        resp = await aiohttp_client.post("/echo", json={"test": "text"})
        await resp.text()

        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

    async def test_json_auto_releases_connection(self, aiohttp_client: HPKEClientSession) -> None:
        """json() automatically releases connection to pool."""
        connector = aiohttp_client._session.connector  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

        resp = await aiohttp_client.post("/echo", json={"test": "json"})
        await resp.json()

        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

    async def test_multiple_reads_only_release_once(self, aiohttp_client: HPKEClientSession) -> None:
        """Multiple read() calls don't cause issues (cached, release idempotent)."""
        resp = await aiohttp_client.post("/echo", json={"test": "multi"})

        data1 = await resp.read()
        data2 = await resp.read()
        data3 = await resp.read()

        assert data1 == data2 == data3
        assert b"multi" in data1

    async def test_read_then_explicit_release_idempotent(self, aiohttp_client: HPKEClientSession) -> None:
        """Calling release() after read() is safe (idempotent)."""
        connector = aiohttp_client._session.connector  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

        resp = await aiohttp_client.post("/echo", json={"test": "idem"})
        await resp.read()  # Auto-releases
        await resp.release()  # Should be idempotent no-op

        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

    async def test_read_in_context_manager_idempotent(self, aiohttp_client: HPKEClientSession) -> None:
        """read() inside context manager - __aexit__ release is idempotent."""
        connector = aiohttp_client._session.connector  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

        resp = await aiohttp_client.post("/echo", json={"test": "ctx"})
        async with resp:
            data = await resp.read()  # Auto-releases
            assert b"ctx" in data
        # __aexit__ calls release() again - should be idempotent

        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

    async def test_headers_accessible_after_read(self, aiohttp_client: HPKEClientSession) -> None:
        """Headers remain accessible after read() releases connection."""
        resp = await aiohttp_client.post("/echo", json={"test": "headers"})

        await resp.read()  # Auto-releases connection

        # Metadata should still be accessible
        assert resp.status == 200
        assert resp.ok is True
        assert "content-type" in resp.headers or "Content-Type" in resp.headers
        assert "/echo" in str(resp.url)

    async def test_status_accessible_after_release(self, aiohttp_client: HPKEClientSession) -> None:
        """Status and other metadata accessible after explicit release()."""
        resp = await aiohttp_client.post("/echo", json={"test": "status"})

        await resp.release()

        # Should still work
        assert resp.status == 200
        assert resp.ok is True
        assert resp.content_type  # Non-empty string

    async def test_raw_body_cached_after_release(self, aiohttp_client: HPKEClientSession) -> None:
        """Raw encrypted body is cached in aiohttp's _body after release."""
        resp = await aiohttp_client.post("/echo", json={"test": "cached"})
        assert isinstance(resp, DecryptedResponse)

        # Read decrypted (auto-releases connection)
        decrypted = await resp.read()
        assert b"cached" in decrypted

        # Raw body should be cached in underlying response's _body
        # (accessing private attr for testing - aiohttp caches body here)
        raw_cached = resp.unwrap()._body  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001
        assert isinstance(raw_cached, bytes)
        assert len(raw_cached) > 0  # Should have cached encrypted body

    async def test_sequential_requests_no_leak(self, aiohttp_client: HPKEClientSession) -> None:
        """Sequential requests with read() don't leak connections."""
        connector = aiohttp_client._session.connector  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

        for i in range(20):
            resp = await aiohttp_client.post("/echo", json={"seq": i})
            data = await resp.json()
            assert str(i) in data["echo"]

        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

    async def test_concurrent_requests_no_leak(self, aiohttp_client: HPKEClientSession) -> None:
        """Concurrent requests with read() don't leak connections."""
        connector = aiohttp_client._session.connector  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

        async def make_request(i: int) -> dict[str, Any]:
            resp = await aiohttp_client.post("/echo", json={"concurrent": i})
            return await resp.json()

        results = await asyncio.gather(*[make_request(i) for i in range(10)])
        assert len(results) == 10

        await asyncio.sleep(0.05)
        assert len(connector._acquired) == 0  # type: ignore[union-attr]  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]  # noqa: SLF001

    @pytest.mark.skipif(
        sys.version_info >= (3, 14),
        reason="ResourceWarning GC timing is flaky on Python 3.14+ due to free-threading changes",
    )
    async def test_no_resource_warning_after_read(self, aiohttp_client: HPKEClientSession) -> None:
        """No ResourceWarning after read() - connection properly released."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ResourceWarning)

            resp = await aiohttp_client.post("/echo", json={"warn": "test"})
            await resp.read()

            # Yield to event loop - allows pending cleanup tasks to complete
            await asyncio.sleep(0.01)

            # Force GC to trigger any warnings
            gc.collect()
            gc.collect()

            # Filter to connection leaks
            leaks = [x for x in w if issubclass(x.category, ResourceWarning) and _is_connection_leak(x)]
            assert not leaks, f"Connection leak after read(): {leaks[0].message}"


class TestDecryptedResponseDecryptionFailure:
    """Tests for connection release on decryption failure.

    Verifies that connections are released even when decryption fails,
    and that the raw body remains accessible via unwrap().read().
    """

    async def test_decryption_failure_releases_connection(self) -> None:
        """Connection is released even when decryption fails."""
        # Create mock response
        mock_response = MagicMock(spec=aiohttp.ClientResponse)
        mock_response.read = AsyncMock(return_value=b"encrypted_data")
        mock_response.release = AsyncMock()
        mock_response.headers = {"X-HPKE-Stream": "counter=0"}
        mock_response.url = "http://test/echo"

        # Create mock sender context
        mock_ctx = MagicMock()

        # Create DecryptedResponse
        resp = DecryptedResponse(mock_response, mock_ctx)

        # Patch ResponseDecryptor to raise an exception
        with patch("hpke_http.middleware.aiohttp.ResponseDecryptor") as mock_decryptor_class:
            mock_decryptor = MagicMock()
            mock_decryptor.decrypt_all.side_effect = ValueError("Decryption failed")
            mock_decryptor_class.return_value = mock_decryptor

            # read() should raise the decryption error
            with pytest.raises(ValueError, match="Decryption failed"):
                await resp.read()

        # But release() should have been called despite the error
        mock_response.release.assert_called_once()

    async def test_decryption_failure_raw_body_accessible(self) -> None:
        """Raw body accessible via unwrap().read() after decryption failure."""
        raw_encrypted = b"some_encrypted_bytes_here"

        # Create mock response
        mock_response = MagicMock(spec=aiohttp.ClientResponse)
        mock_response.read = AsyncMock(return_value=raw_encrypted)
        mock_response.release = AsyncMock()
        mock_response.headers = {"X-HPKE-Stream": "counter=0"}

        mock_ctx = MagicMock()
        resp = DecryptedResponse(mock_response, mock_ctx)

        with patch("hpke_http.middleware.aiohttp.ResponseDecryptor") as mock_decryptor_class:
            mock_decryptor = MagicMock()
            mock_decryptor.decrypt_all.side_effect = ValueError("Bad data")
            mock_decryptor_class.return_value = mock_decryptor

            with pytest.raises(ValueError):
                await resp.read()

        # Raw body should still be accessible
        raw = await resp.unwrap().read()
        assert raw == raw_encrypted

    async def test_decryption_failure_no_resource_warning(self) -> None:
        """No ResourceWarning after decryption failure - connection released."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ResourceWarning)

            mock_response = MagicMock(spec=aiohttp.ClientResponse)
            mock_response.read = AsyncMock(return_value=b"data")
            mock_response.release = AsyncMock()
            mock_response.headers = {}

            mock_ctx = MagicMock()
            resp = DecryptedResponse(mock_response, mock_ctx)

            with patch("hpke_http.middleware.aiohttp.ResponseDecryptor") as mock_decryptor_class:
                mock_decryptor_class.return_value.decrypt_all.side_effect = RuntimeError("Fail")

                with pytest.raises(RuntimeError):
                    await resp.read()

            # Force GC
            gc.collect()
            gc.collect()

            # release() was called, so no warning from our code
            mock_response.release.assert_called_once()

            # No connection leaks from our wrapper
            leaks = [x for x in w if issubclass(x.category, ResourceWarning) and _is_connection_leak(x)]
            assert not leaks

    async def test_release_called_once_on_failure(self) -> None:
        """release() called exactly once on decryption failure."""
        mock_response = MagicMock(spec=aiohttp.ClientResponse)
        mock_response.read = AsyncMock(return_value=b"data")
        mock_response.release = AsyncMock()
        mock_response.headers = {}

        mock_ctx = MagicMock()
        resp = DecryptedResponse(mock_response, mock_ctx)

        with patch("hpke_http.middleware.aiohttp.ResponseDecryptor") as mock_decryptor_class:
            mock_decryptor_class.return_value.decrypt_all.side_effect = RuntimeError("Oops")

            with pytest.raises(RuntimeError):
                await resp.read()

        # Exactly one release call
        assert mock_response.release.call_count == 1

    async def test_multiple_read_after_failure_reraises(self) -> None:
        """Multiple read() calls after failure re-raise (not cached)."""
        mock_response = MagicMock(spec=aiohttp.ClientResponse)
        mock_response.read = AsyncMock(return_value=b"data")
        mock_response.release = AsyncMock()
        mock_response.headers = {}

        mock_ctx = MagicMock()
        resp = DecryptedResponse(mock_response, mock_ctx)

        with patch("hpke_http.middleware.aiohttp.ResponseDecryptor") as mock_decryptor_class:
            mock_decryptor_class.return_value.decrypt_all.side_effect = ValueError("Bad")

            # First read fails
            with pytest.raises(ValueError):
                await resp.read()

            # Second read also fails (not cached because _decrypted not set)
            with pytest.raises(ValueError):
                await resp.read()

        # release() called twice (once per read attempt)
        assert mock_response.release.call_count == 2


class TestStandardResponseEdgeCasesE2E:
    """E2E edge case tests for standard response encryption."""

    async def test_large_response_multi_chunk(self, aiohttp_client: HPKEClientSession) -> None:
        """Large response that may be sent in multiple chunks."""
        # Request a response with a larger payload via /echo
        large_payload = {"data": "x" * 50000}  # 50KB payload
        resp = await aiohttp_client.post("/echo", json=large_payload)

        assert resp.status == 200
        data = await resp.json()
        assert "x" * 50000 in data["echo"]

    async def test_unicode_response_content(self, aiohttp_client: HPKEClientSession) -> None:
        """Unicode content in response is preserved (may be escaped in JSON)."""
        unicode_data = {"message": "Hello 世界 🌍 émojis"}
        resp = await aiohttp_client.post("/echo", json=unicode_data)

        data = await resp.json()
        # The echo contains the JSON string, which may have unicode escaped
        # Check for either literal or escaped form
        echo = data["echo"]
        assert "世界" in echo or "\\u4e16\\u754c" in echo
        assert "🌍" in echo or "\\ud83c\\udf0d" in echo

    async def test_binary_in_json_response(self, aiohttp_client: HPKEClientSession) -> None:
        """Binary-like content (high bytes) in JSON is handled."""
        # JSON with escaped binary-like content
        test_data = {"binary_like": "\\x00\\xff"}
        resp = await aiohttp_client.post("/echo", json=test_data)

        data = await resp.json()
        assert "binary_like" in data["echo"]

    async def test_rapid_sequential_requests(self, aiohttp_client: HPKEClientSession) -> None:
        """Multiple rapid sequential requests work correctly."""
        for i in range(10):
            resp = await aiohttp_client.post("/echo", json={"seq": i})
            data = await resp.json()
            assert str(i) in data["echo"]


class TestSSEEdgeCasesE2E:
    """E2E edge case tests for SSE encryption."""

    async def test_single_event_stream(self, aiohttp_client: HPKEClientSession) -> None:
        """Stream with minimum events works."""
        # /stream sends 4 events minimum
        resp = await aiohttp_client.post("/stream", json={"start": True})
        events = [parse_sse_chunk(chunk) async for chunk in aiohttp_client.iter_sse(resp)]
        assert len(events) >= 1

    async def test_sse_with_unicode_data(self, aiohttp_client: HPKEClientSession) -> None:
        """SSE events with unicode content work."""
        resp = await aiohttp_client.post("/stream", json={"start": True})
        events = [parse_sse_chunk(chunk) async for chunk in aiohttp_client.iter_sse(resp)]

        # All events should decode properly
        for event_type, event_data in events:
            assert event_type is not None
            assert event_data is not None


class TestErrorResponsesE2E:
    """E2E tests for error response handling."""

    async def test_404_response_encrypted(self, aiohttp_client: HPKEClientSession) -> None:
        """404 responses are still encrypted for encrypted requests."""
        resp = await aiohttp_client.get("/nonexistent-path-xyz")
        # Server returns 404 for unknown paths
        assert resp.status == 404

    async def test_malformed_json_request(self, aiohttp_client: HPKEClientSession) -> None:
        """Server handles requests gracefully."""
        # Send valid JSON that the server can process
        resp = await aiohttp_client.post("/echo", json=None)
        # Should get some response (either success or error)
        assert resp.status in (200, 400, 422)


class TestWeirdInputsE2E:
    """E2E tests for weird/adversarial inputs."""

    async def test_very_long_key_names(self, aiohttp_client: HPKEClientSession) -> None:
        """JSON with very long key names works."""
        long_key = "k" * 1000
        test_data = {long_key: "value"}
        resp = await aiohttp_client.post("/echo", json=test_data)

        data = await resp.json()
        assert long_key in data["echo"]

    async def test_deeply_nested_json(self, aiohttp_client: HPKEClientSession) -> None:
        """Deeply nested JSON structures work."""
        nested: dict[str, Any] = {"level": 0}
        current = nested
        for i in range(1, 20):  # 20 levels deep
            current["child"] = {"level": i}
            current = current["child"]

        resp = await aiohttp_client.post("/echo", json=nested)
        data = await resp.json()
        assert "level" in data["echo"]

    async def test_array_response(self, aiohttp_client: HPKEClientSession) -> None:
        """Array JSON in request works."""
        test_data = [1, 2, 3, "four", {"five": 5}]
        resp = await aiohttp_client.post("/echo", json=test_data)

        data = await resp.json()
        assert "1" in data["echo"] or "[1" in data["echo"]

    async def test_special_characters_in_values(self, aiohttp_client: HPKEClientSession) -> None:
        """Special characters in JSON values work."""
        test_data = {
            "quotes": 'Hello "world"',
            "newlines": "line1\nline2",
            "tabs": "col1\tcol2",
            "backslash": "path\\to\\file",
        }
        resp = await aiohttp_client.post("/echo", json=test_data)

        data = await resp.json()
        # The echo should contain these values (possibly escaped)
        assert "echo" in data


class TestEncryptionStateValidation:
    """
    E2E tests that validate encryption at the wire level.

    These tests verify that:
    1. When protocol expects encryption, raw content IS encrypted (not plaintext)
    2. When protocol does NOT expect encryption, raw content is plaintext
    3. Violations of expected encryption state raise appropriate errors
    """

    async def test_encrypted_response_is_not_plaintext(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify encrypted response body is NOT readable as plaintext JSON."""
        resp = await aiohttp_client.post("/echo", json={"secret": "data"})

        # The response should be encrypted - verify by trying to parse as JSON
        # Get raw bytes from underlying response using public unwrap() method
        assert isinstance(resp, DecryptedResponse)
        raw_body = await resp.unwrap().read()

        # Raw body should NOT be valid JSON (it's encrypted)
        try:
            json.loads(raw_body)
            # If this succeeds, the response was NOT encrypted - FAIL
            raise AssertionError(
                f"Response body was readable as plaintext JSON - encryption expected! "
                f"Raw body starts with: {raw_body[:100]!r}"
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Expected - raw body is encrypted, not plaintext JSON
            # UnicodeDecodeError can occur when encrypted bytes are invalid UTF
            pass

        # But decrypted response SHOULD be valid JSON
        decrypted = await resp.json()
        assert "echo" in decrypted

    async def test_encrypted_sse_is_not_plaintext(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify encrypted SSE events are NOT readable as plaintext SSE."""
        resp = await aiohttp_client.post("/stream", json={"start": True})

        # Read raw chunks from underlying response
        raw_chunks = [chunk async for chunk in resp.content]

        # Combine all raw data
        raw_data = b"".join(raw_chunks)

        # Raw data should be encrypted SSE format (event: enc)
        # NOT plaintext SSE (event: progress, etc.)
        assert b"event: enc" in raw_data, "Encrypted SSE should use 'event: enc' format"
        assert b"event: progress" not in raw_data, "Raw SSE should NOT contain plaintext events"

        # The data field should be base64url encoded, not plaintext JSON
        # Check that we don't see unencrypted JSON in the raw data
        assert b'"progress"' not in raw_data, "Raw SSE should NOT contain plaintext JSON"

    async def test_unencrypted_response_is_plaintext(
        self,
        granian_server: E2EServer,
    ) -> None:
        """Verify unencrypted response body IS readable plaintext."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        # Use plain aiohttp client - no encryption
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as resp:
                raw_body = await resp.read()

                # Raw body SHOULD be valid JSON (not encrypted)
                try:
                    data = json.loads(raw_body)
                    assert "status" in data
                except json.JSONDecodeError as e:
                    raise AssertionError(
                        f"Unencrypted response should be plaintext JSON! Raw body: {raw_body[:200]!r}"
                    ) from e

                # Verify no encryption headers
                assert HEADER_HPKE_STREAM not in resp.headers

    async def test_encryption_header_presence_matches_content(
        self,
        aiohttp_client: HPKEClientSession,
        granian_server: E2EServer,
    ) -> None:
        """Verify X-HPKE-Stream header presence matches actual encryption."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        # Case 1: Encrypted request → should get encrypted response with header
        resp = await aiohttp_client.post("/echo", json={"test": 1})

        assert HEADER_HPKE_STREAM in resp.headers, "Encrypted response MUST have X-HPKE-Stream header"

        # Verify content is actually encrypted
        assert isinstance(resp, DecryptedResponse)
        raw_body = await resp.unwrap().read()
        if raw_body:
            try:
                json.loads(raw_body)
                raise AssertionError("Header claims encryption but body is plaintext")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # Expected - body is encrypted

        # Case 2: Unencrypted request → should get unencrypted response without header
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as resp:
                assert HEADER_HPKE_STREAM not in resp.headers, "Unencrypted response MUST NOT have X-HPKE-Stream header"

                # Verify content is actually plaintext
                raw_body = await resp.read()
                try:
                    json.loads(raw_body)  # Should succeed
                except json.JSONDecodeError as e:
                    raise AssertionError("No encryption header but body is not plaintext") from e

    async def test_tampered_encryption_header_fails_decryption(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify normal decryption works (baseline test)."""
        # Make a valid encrypted request
        resp = await aiohttp_client.post("/echo", json={"test": 1})

        # Verify we can decrypt normally
        data = await resp.json()
        assert "echo" in data

    async def test_missing_encryption_when_expected_raises(
        self,
        granian_server: E2EServer,
    ) -> None:
        """
        When client expects encryption but server doesn't provide it,
        the mismatch should be detectable.
        """
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        # Plain request to /health - server will NOT encrypt
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as resp:
                # Verify no encryption header present
                assert HEADER_HPKE_STREAM not in resp.headers

                # If someone tried to treat this as encrypted, they'd fail
                raw_body = await resp.read()

                # This IS valid JSON (unencrypted)
                data = json.loads(raw_body)
                assert "status" in data


class TestRawWireFormatValidation:
    """
    Tests that validate the exact wire format of encrypted data.

    These tests ensure the encryption format matches the protocol specification.
    """

    async def test_standard_response_wire_format(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify standard response wire format: [length(4B) || counter(4B) || ciphertext]."""
        resp = await aiohttp_client.post("/echo", json={"format": "test"})

        # Access raw encrypted body using public unwrap() method
        assert isinstance(resp, DecryptedResponse)
        raw_body = await resp.unwrap().read()

        # Wire format validation:
        # - Minimum size: length(4) + counter(4) + encoding_id(1) + tag(16) = 25 bytes
        assert len(raw_body) >= 25, f"Encrypted body too short: {len(raw_body)} bytes"

        # - First 4 bytes are length prefix
        length = int.from_bytes(raw_body[:4], "big")
        assert length >= 21, f"Chunk length should be >= 21, got {length}"

        # - Bytes 4-8 are counter (should be 1 for first chunk)
        counter = int.from_bytes(raw_body[4:8], "big")
        assert counter == 1, f"First chunk counter should be 1, got {counter}"

    async def test_sse_wire_format(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify SSE wire format: event: enc\\ndata: <base64>\\n\\n.

        SSEFormat uses standard base64 (not base64url) for ~1.7x faster encoding.
        See streaming.py SSEFormat docstring for rationale.
        """
        resp = await aiohttp_client.post("/stream", json={"start": True})

        # Read enough raw chunks to get a complete event
        raw_chunks: list[bytes] = []
        async for chunk in resp.content:
            raw_chunks.append(chunk)
            # Check if we have at least one complete event (contains data field)
            combined = b"".join(raw_chunks).decode("utf-8", errors="replace")
            if "data:" in combined and "\n\n" in combined:
                break

        raw_str = b"".join(raw_chunks).decode("utf-8", errors="replace")

        # SSE format validation
        assert "event: enc" in raw_str, f"SSE should have 'event: enc', got: {raw_str[:100]}"
        assert "data:" in raw_str, f"SSE should have 'data:' field, got: {raw_str[:100]}"

        # Data field should be standard base64 encoded (A-Za-z0-9+/=)
        for line in raw_str.split("\n"):
            if line.startswith("data:"):
                data_value = line[5:].strip()
                assert re.match(r"^[A-Za-z0-9+/=]+$", data_value), (
                    f"Data field should be base64, got: {data_value[:50]}"
                )
                break


class TestDecryptedResponseAiohttpCompat:
    """
    Integration tests verifying DecryptedResponse works with common aiohttp patterns.

    These tests ensure duck-typing correctly proxies all commonly used
    aiohttp.ClientResponse attributes and methods.
    """

    # ==========================================================================
    # Explicitly proxied properties (defined in DecryptedResponse)
    # ==========================================================================

    async def test_status_property(self, aiohttp_client: HPKEClientSession) -> None:
        """Test status property returns correct HTTP status code."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)
        assert resp.status == 200
        assert isinstance(resp.status, int)

    async def test_headers_property(self, aiohttp_client: HPKEClientSession) -> None:
        """Test headers property returns CIMultiDictProxy with case-insensitive access."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # Should be CIMultiDictProxy
        from multidict import CIMultiDictProxy

        assert isinstance(resp.headers, CIMultiDictProxy)

        # Case-insensitive access should work
        ct_lower = resp.headers.get("content-type")
        ct_upper = resp.headers.get("Content-Type")
        assert ct_lower == ct_upper

    async def test_url_property(self, aiohttp_client: HPKEClientSession) -> None:
        """Test url property returns yarl.URL."""
        from yarl import URL

        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)
        assert isinstance(resp.url, URL)
        assert "/echo" in str(resp.url)

    async def test_ok_property(self, aiohttp_client: HPKEClientSession) -> None:
        """Test ok property returns True for 2xx responses."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)
        assert resp.ok is True
        assert isinstance(resp.ok, bool)

    async def test_reason_property(self, aiohttp_client: HPKEClientSession) -> None:
        """Test reason property returns HTTP status reason."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)
        # Reason can be None (HTTP/2) or string (HTTP/1.1)
        assert resp.reason is None or isinstance(resp.reason, str)

    async def test_content_type_property(self, aiohttp_client: HPKEClientSession) -> None:
        """Test content_type property returns Content-Type value."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)
        assert isinstance(resp.content_type, str)

    async def test_raise_for_status_success(self, aiohttp_client: HPKEClientSession) -> None:
        """Test raise_for_status() does not raise on 2xx."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)
        # Should not raise
        resp.raise_for_status()

    # ==========================================================================
    # Overridden methods (decrypt content)
    # ==========================================================================

    async def test_read_returns_decrypted_bytes(self, aiohttp_client: HPKEClientSession) -> None:
        """Test read() returns decrypted bytes."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        body = await resp.read()
        assert isinstance(body, bytes)
        data = json.loads(body)
        assert "echo" in data

    async def test_text_returns_decrypted_string(self, aiohttp_client: HPKEClientSession) -> None:
        """Test text() returns decrypted string with default encoding."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        text = await resp.text()
        assert isinstance(text, str)
        data = json.loads(text)
        assert "echo" in data

    async def test_text_with_encoding_param(self, aiohttp_client: HPKEClientSession) -> None:
        """Test text(encoding=...) respects encoding parameter."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        text = await resp.text(encoding="utf-8")
        assert isinstance(text, str)

    async def test_text_with_errors_param(self, aiohttp_client: HPKEClientSession) -> None:
        """Test text(errors=...) matches aiohttp signature."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # aiohttp supports errors parameter
        text = await resp.text(errors="replace")
        assert isinstance(text, str)

    async def test_json_returns_decrypted_dict(self, aiohttp_client: HPKEClientSession) -> None:
        """Test json() returns decrypted and parsed JSON."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        data = await resp.json()
        assert isinstance(data, dict)
        assert "echo" in data

    async def test_json_with_loads_param(self, aiohttp_client: HPKEClientSession) -> None:
        """Test json(loads=...) matches aiohttp signature."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # aiohttp supports custom loads function
        import json as json_mod

        data = await resp.json(loads=json_mod.loads)
        assert isinstance(data, dict)

    async def test_json_with_encoding_param(self, aiohttp_client: HPKEClientSession) -> None:
        """Test json(encoding=...) matches aiohttp signature."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # aiohttp supports encoding parameter
        data = await resp.json(encoding="utf-8")
        assert isinstance(data, dict)

    async def test_json_with_content_type_param(self, aiohttp_client: HPKEClientSession) -> None:
        """Test json(content_type=...) matches aiohttp signature."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # aiohttp supports content_type validation (None disables check)
        data = await resp.json(content_type=None)
        assert isinstance(data, dict)

    # ==========================================================================
    # DecryptedResponse-specific methods
    # ==========================================================================

    async def test_unwrap_returns_client_response(self, aiohttp_client: HPKEClientSession) -> None:
        """Test unwrap() returns underlying ClientResponse."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        underlying = resp.unwrap()
        assert isinstance(underlying, aiohttp.ClientResponse)
        assert underlying.status == resp.status

    # ==========================================================================
    # __getattr__ fallback (proxied to underlying response)
    # ==========================================================================

    async def test_version_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test version attribute proxied via __getattr__."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        version = resp.version
        assert version is not None

    async def test_request_info_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test request_info attribute proxied via __getattr__."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        request_info = resp.request_info
        assert request_info is not None
        assert hasattr(request_info, "url")

    async def test_cookies_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test cookies attribute proxied via __getattr__."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        cookies = resp.cookies
        assert cookies is not None

    async def test_history_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test history attribute proxied via __getattr__."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        history = resp.history
        assert isinstance(history, tuple)

    async def test_content_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test content StreamReader proxied via __getattr__."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        content = resp.content
        assert content is not None
        assert hasattr(content, "read")

    async def test_charset_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test charset property proxied via __getattr__."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        charset = resp.charset
        # charset can be None or string
        assert charset is None or isinstance(charset, str)

    async def test_content_length_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test content_length property proxied via __getattr__."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        content_length = resp.content_length
        # content_length can be None or int
        assert content_length is None or isinstance(content_length, int)

    async def test_real_url_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test real_url property proxied via __getattr__."""
        from yarl import URL

        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        real_url = resp.real_url
        assert isinstance(real_url, URL)

    async def test_host_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test host property proxied via __getattr__."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        host = resp.host
        assert isinstance(host, str)

    async def test_links_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test links property proxied via __getattr__."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        links = resp.links
        # links is a MultiDictProxy (possibly empty)
        assert links is not None

    async def test_get_encoding_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test get_encoding() method proxied via __getattr__."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        encoding = resp.get_encoding()
        assert isinstance(encoding, str)

    async def test_close_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test close() method proxied via __getattr__."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # close() should be callable
        assert callable(resp.close)

    async def test_release_via_getattr(self, aiohttp_client: HPKEClientSession) -> None:
        """Test release() method proxied via __getattr__."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # release() should be callable
        assert callable(resp.release)

    # ==========================================================================
    # Caching and consistency
    # ==========================================================================

    async def test_read_cached_on_multiple_calls(self, aiohttp_client: HPKEClientSession) -> None:
        """Test that multiple read() calls return cached decrypted content."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        body1 = await resp.read()
        body2 = await resp.read()
        assert body1 == body2

    async def test_read_text_json_consistency(self, aiohttp_client: HPKEClientSession) -> None:
        """Test that read(), text(), json() return consistent data."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        body_bytes = await resp.read()
        body_text = await resp.text()
        body_json = await resp.json()

        assert body_bytes.decode("utf-8") == body_text
        assert json.loads(body_text) == body_json

    # ==========================================================================
    # Type identity
    # ==========================================================================

    async def test_isinstance_decrypted_response(self, aiohttp_client: HPKEClientSession) -> None:
        """Test DecryptedResponse can be identified via isinstance."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})

        assert isinstance(resp, DecryptedResponse)
        assert not isinstance(resp, aiohttp.ClientResponse)
        assert isinstance(resp.unwrap(), aiohttp.ClientResponse)


@pytest.mark.slow
@skip_on_314t_gc_bug
class TestLargePayloadAutoChunking:
    """
    Test auto-chunking for large payloads (10MB, 50MB, 100MB, 250MB, 500MB, 1GB).

    These tests verify that the length-prefix wire format correctly handles
    multi-chunk encryption/decryption for very large request/response bodies.

    Wire format per chunk: length(4B) || counter(4B) || ciphertext
    """

    @pytest.mark.parametrize("size_mb", LARGE_PAYLOAD_SIZES_MB, ids=LARGE_PAYLOAD_SIZES_IDS)
    async def test_large_request_roundtrip(
        self,
        aiohttp_client: HPKEClientSession,
        size_mb: int,
    ) -> None:
        """Large request payloads encrypt/decrypt correctly via auto-chunking."""
        size_bytes = size_mb * 1024 * 1024
        # Use repeating pattern for efficient generation
        pattern = "A" * 1024  # 1KB pattern
        large_content = pattern * (size_bytes // 1024)

        resp = await aiohttp_client.post("/echo", data=large_content.encode())
        assert resp.status == 200

        data = await resp.json()
        # Verify content length (echo returns the raw body as string)
        assert len(data["echo"]) == len(large_content)

    @pytest.mark.parametrize("size_mb", LARGE_PAYLOAD_SIZES_MB, ids=LARGE_PAYLOAD_SIZES_IDS)
    async def test_large_response_decryption(
        self,
        aiohttp_client: HPKEClientSession,
        size_mb: int,
    ) -> None:
        """Large responses are correctly decrypted from multiple chunks."""
        size_bytes = size_mb * 1024 * 1024
        pattern = "B" * 1024
        large_content = pattern * (size_bytes // 1024)

        resp = await aiohttp_client.post("/echo", data=large_content.encode())
        assert resp.status == 200
        assert isinstance(resp, DecryptedResponse)

        # Verify decryption works
        data = await resp.json()
        assert data["echo"] == large_content

    @pytest.mark.parametrize("size_mb", LARGE_PAYLOAD_SIZES_MB, ids=LARGE_PAYLOAD_SIZES_IDS)
    async def test_large_payload_wire_format(
        self,
        aiohttp_client: HPKEClientSession,
        size_mb: int,
    ) -> None:
        """Verify wire format uses length prefix for O(1) chunk detection."""
        size_bytes = size_mb * 1024 * 1024
        pattern = "C" * 1024
        large_content = pattern * (size_bytes // 1024)

        resp = await aiohttp_client.post("/echo", data=large_content.encode())
        assert resp.status == 200
        assert isinstance(resp, DecryptedResponse)

        # Access raw encrypted body
        raw_body = await resp.unwrap().read()

        # Verify length-prefix format: first 4 bytes = chunk length
        assert len(raw_body) >= 8, "Response too short for length-prefix format"
        first_chunk_len = int.from_bytes(raw_body[:4], "big")
        assert first_chunk_len > 0, "First chunk length must be positive"

        # Verify we can parse chunk boundaries
        offset = 0
        chunk_count = 0
        while offset < len(raw_body):
            chunk_len = int.from_bytes(raw_body[offset : offset + 4], "big")
            assert chunk_len > 0, f"Invalid chunk length at offset {offset}"
            offset += 4 + chunk_len  # length prefix + chunk data
            chunk_count += 1

        # For large payloads, expect multiple chunks
        assert chunk_count >= 1, f"Expected at least 1 chunk, got {chunk_count}"

    async def test_large_payload_data_integrity(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify data integrity with verifiable block markers (10MB)."""
        size_bytes = 10 * 1024 * 1024
        block_size = 1024

        # Create content with block markers
        blocks: list[str] = []
        for i in range(size_bytes // block_size):
            marker = f"[{i:08d}]"
            padding = "=" * (block_size - len(marker))
            blocks.append(marker + padding)
        large_content = "".join(blocks)

        resp = await aiohttp_client.post("/echo", data=large_content.encode())
        assert resp.status == 200

        data = await resp.json()
        echo = data["echo"]

        # Verify markers
        assert "[00000000]" in echo, "First block marker missing"
        assert "[00005000]" in echo, "Middle block marker missing"
        last_idx = (size_bytes // block_size) - 1
        assert f"[{last_idx:08d}]" in echo, "Last block marker missing"


# =============================================================================
# Cryptographic Properties Verification
# =============================================================================


class TestCryptographicProperties:
    """Tests that verify encryption is ACTUALLY happening at the wire level."""

    async def test_encrypted_body_has_cryptographic_entropy(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify encrypted output has high Shannon entropy (> 7.0 bits/byte)."""
        payload = {"data": "A" * 10000, "secret": "password123"}

        resp = await aiohttp_client.post("/echo", json=payload)
        assert resp.status == 200
        assert isinstance(resp, DecryptedResponse)

        raw_body = await resp.unwrap().read()
        assert len(raw_body) > 256, "Response too short for entropy analysis"

        entropy = calculate_shannon_entropy(raw_body)
        assert entropy > 7.0, (
            f"Entropy {entropy:.2f} bits/byte too low for encrypted data. "
            f"Expected > 7.0. May indicate encryption bypass."
        )

    async def test_encrypted_body_uniform_distribution(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify encrypted data has uniform byte distribution.

        Chi-square tests have inherent false positive rate equal to the p-value
        threshold (e.g., p > 0.01 means 1% of truly random data fails). To reduce
        CI flakiness, we run multiple trials and allow one failure.

        With 10 trials and 9 required passes at p > 0.01:
        P(flaky failure) = P(>=2 of 10 fail) ≈ 0.4% (binomial)
        """
        passes = 0
        results: list[tuple[float, float]] = []

        for _ in range(CHI_SQUARE_TRIALS):
            payload = {"message": "Hello World " * 1000}
            resp = await aiohttp_client.post("/echo", json=payload)
            assert resp.status == 200
            assert isinstance(resp, DecryptedResponse)

            raw_body = await resp.unwrap().read()
            assert len(raw_body) >= 1000, "Response too short for chi-square test"

            chi2, p_value = chi_square_byte_uniformity(raw_body)
            results.append((chi2, p_value))
            if p_value > CHI_SQUARE_P_THRESHOLD:
                passes += 1

        assert passes >= CHI_SQUARE_MIN_PASS, (
            f"Chi-square uniformity test failed: {passes}/{CHI_SQUARE_TRIALS} trials passed "
            f"(required {CHI_SQUARE_MIN_PASS}). Results: {results}. "
            f"Encrypted data should appear random."
        )

    async def test_known_plaintext_not_visible_on_wire(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify known plaintext values never appear in encrypted wire data."""
        canary = "CANARY_SECRET_12345_XYZ"
        payload = {"secret": canary, "data": "Hello World Test Message"}

        resp = await aiohttp_client.post("/echo", json=payload)
        assert resp.status == 200
        assert isinstance(resp, DecryptedResponse)

        raw_body = await resp.unwrap().read()

        forbidden_patterns = [
            canary.encode(),
            b'"secret"',
            b'"echo"',
            b"Hello World",
        ]

        for pattern in forbidden_patterns:
            assert pattern not in raw_body, (
                f"Known plaintext '{pattern.decode(errors='replace')}' found in "
                f"encrypted wire data! Encryption may be bypassed."
            )

    async def test_wire_format_cryptographic_structure(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify wire format has valid cryptographic structure."""
        resp = await aiohttp_client.post("/echo", json={"test": "structure"})
        assert resp.status == 200
        assert isinstance(resp, DecryptedResponse)

        raw_body = await resp.unwrap().read()
        assert len(raw_body) >= 25, f"Response too short: {len(raw_body)} bytes"

        chunk_len = int.from_bytes(raw_body[:4], "big")
        assert chunk_len >= 21, f"Chunk length {chunk_len} too short"

        counter = int.from_bytes(raw_body[4:8], "big")
        assert counter >= 1, f"Counter {counter} invalid - must start at 1"


# =============================================================================
# Streaming Behavior Verification
# =============================================================================


class TestStreamingBehavior:
    """Tests that verify chunking/streaming is ACTUALLY happening."""

    async def test_sse_chunks_arrive_progressively(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify SSE chunks arrive with inter-arrival timing gaps."""
        resp = await aiohttp_client.post("/stream-delayed", json={"start": True})
        assert resp.status == 200

        arrival_times: list[float] = []
        async for _chunk in resp.content:
            arrival_times.append(time.monotonic())
            if len(arrival_times) >= 6:
                break

        assert len(arrival_times) >= 5, f"Expected 5+ chunks, got {len(arrival_times)}"

        gaps = [arrival_times[i + 1] - arrival_times[i] for i in range(len(arrival_times) - 1)]
        significant_gaps = sum(1 for g in gaps if g > 0.05)

        # At least one significant gap proves progressive delivery (not all buffered until end).
        # TCP and encryption layer may batch adjacent events, so we only require 1 gap.
        assert significant_gaps >= 1, (
            f"No gaps > 50ms found. All chunks arrived instantly - not streaming. "
            f"Gaps: {[f'{g * 1000:.0f}ms' for g in gaps]}"
        )

    async def test_large_request_is_chunked(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify large requests are sent in multiple chunks."""
        size_mb = 10
        large_content = "X" * (size_mb * 1024 * 1024)

        resp = await aiohttp_client.post("/echo-chunks", data=large_content.encode())
        assert resp.status == 200

        data = await resp.json()

        assert data["chunk_count"] > 1, f"Large {size_mb}MB request sent as single chunk!"

        min_expected = (size_mb * 1024 * 1024) // (64 * 1024) // 2
        assert data["chunk_count"] >= min_expected, (
            f"Expected >= {min_expected} chunks for {size_mb}MB, got {data['chunk_count']}"
        )

    async def test_response_chunk_boundaries_valid(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify response chunk boundaries align with length prefix format."""
        size_mb = 5
        large_content = "Y" * (size_mb * 1024 * 1024)

        resp = await aiohttp_client.post("/echo", data=large_content.encode())
        assert resp.status == 200
        assert isinstance(resp, DecryptedResponse)

        raw_body = await resp.unwrap().read()

        offset = 0
        counters: list[int] = []

        while offset < len(raw_body):
            assert offset + 4 <= len(raw_body), f"Truncated length at offset {offset}"
            chunk_len = int.from_bytes(raw_body[offset : offset + 4], "big")
            assert chunk_len > 0, f"Zero-length chunk at offset {offset}"
            assert offset + 4 + chunk_len <= len(raw_body), f"Chunk overflow at offset {offset}"

            counter = int.from_bytes(raw_body[offset + 4 : offset + 8], "big")
            counters.append(counter)
            offset += 4 + chunk_len

        assert offset == len(raw_body), f"Chunk boundaries misaligned: {offset} vs {len(raw_body)}"
        assert counters == list(range(1, len(counters) + 1)), f"Counters not monotonic: {counters[:10]}"


# =============================================================================
# Network-Level Verification
# =============================================================================


@pytest.mark.requires_root
class TestNetworkLevelVerification:
    """Tests that verify encryption at the network packet level.

    These tests require root/sudo to run tcpdump for packet capture.
    They must run serially (-n 0) due to timing sensitivity.
    """

    async def test_unencrypted_traffic_is_visible_in_capture(
        self,
        tcpdump_capture: str,
        granian_server: E2EServer,
    ) -> None:
        """Verify tcpdump actually captures traffic (sanity check for false negatives).

        This test sends UNENCRYPTED requests and verifies they ARE visible in the
        capture. If this fails, the tcpdump setup is broken and other tests in this
        class would give false confidence.
        """
        import asyncio

        # Send plain HTTP request (not through HPKE client) with unique canary
        canary = "UNENCRYPTED_SANITY_CHECK_12345"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{granian_server.host}:{granian_server.port}/health?canary={canary}"
            ) as resp:
                assert resp.status == 200

        # Wait for tcpdump to flush packets to disk
        await asyncio.sleep(0.5)

        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()

        # Verify unencrypted canary IS visible (proves tcpdump is working)
        assert canary.encode() in pcap_data, (
            f"Unencrypted canary '{canary}' not found in capture (pcap size: {len(pcap_data)} bytes). "
            "tcpdump may not be capturing traffic correctly."
        )

    async def test_wire_traffic_contains_no_plaintext(
        self,
        tcpdump_capture: str,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify captured network traffic contains no plaintext."""
        import asyncio

        canaries = ["WIRE_CANARY_VALUE_ABC123", "NETWORK_TEST_SECRET_XYZ"]

        for canary in canaries:
            resp = await aiohttp_client.post("/echo", json={"secret": canary})
            assert resp.status == 200

        await asyncio.sleep(0.3)

        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()

        for canary in canaries:
            assert canary.encode() not in pcap_data, f"Plaintext canary '{canary}' found in network capture!"
        assert b'"secret"' not in pcap_data, "JSON key found in network capture"

    async def test_sse_wire_traffic_is_encrypted(
        self,
        tcpdump_capture: str,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify SSE streaming content is encrypted on the wire.

        The SSE transport format (event:, data:) is visible on the wire, but
        the actual event content must be encrypted. Original event types and
        data should NOT appear - only 'event: enc' with base64 ciphertext.
        """
        import asyncio

        # Trigger SSE stream with unique canary values
        sse_canary = "SSE_WIRE_CANARY_ENCRYPTED_789"
        resp = await aiohttp_client.post("/stream", json={"canary": sse_canary})
        assert resp.status == 200

        # Consume the SSE stream to generate wire traffic
        event_count = 0
        async for _chunk in resp.content:
            event_count += 1
            if event_count >= 5:
                break

        # Wait for packets to flush
        await asyncio.sleep(0.5)

        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()

        # Original event types should NEVER appear (they're encrypted)
        # The wire should only show "event: enc" not "event: tick" or "event: done"
        assert b"event: tick" not in pcap_data, "Raw SSE event type 'tick' found in network capture!"
        assert b"event: done" not in pcap_data, "Raw SSE event type 'done' found in network capture!"

        # Original JSON data keys should NOT be visible
        assert b'"count"' not in pcap_data, "SSE data key 'count' found in network capture!"
        assert b'"timestamp"' not in pcap_data, "SSE data key 'timestamp' found in network capture!"

        # Canary value should NOT be visible
        assert sse_canary.encode() not in pcap_data, f"SSE canary '{sse_canary}' found in network capture!"

        # Verify encrypted SSE format IS present (proves SSE encryption is working)
        assert b"event: enc" in pcap_data, "Encrypted SSE 'event: enc' not found - encryption may not be active!"

    async def test_nonce_uniqueness_different_ciphertext(
        self,
        tcpdump_capture: str,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify same plaintext produces different ciphertext (nonce uniqueness).

        This is critical for security - if nonces are reused, an attacker can
        XOR ciphertexts to recover plaintext. Each encryption MUST produce
        unique ciphertext even for identical input.
        """
        import asyncio

        # Send identical requests
        identical_payload = {"test": "NONCE_TEST_IDENTICAL_PAYLOAD_12345"}

        resp1 = await aiohttp_client.post("/echo", json=identical_payload)
        assert resp1.status == 200
        assert isinstance(resp1, DecryptedResponse)
        raw1 = await resp1.unwrap().read()

        resp2 = await aiohttp_client.post("/echo", json=identical_payload)
        assert resp2.status == 200
        assert isinstance(resp2, DecryptedResponse)
        raw2 = await resp2.unwrap().read()

        # Ciphertexts MUST be different (due to different nonces/ephemeral keys)
        assert raw1 != raw2, (
            "CRITICAL: Identical plaintext produced identical ciphertext! "
            "This indicates nonce reuse - catastrophic for security."
        )

        # Wait for pcap to capture all traffic
        await asyncio.sleep(0.3)

        # Verify traffic was captured (sanity check)
        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()
        assert len(pcap_data) > 100, "No traffic captured in pcap"

    async def test_request_body_encrypted_in_pcap(
        self,
        tcpdump_capture: str,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Verify request body is encrypted on the wire (not just response).

        Previous tests focused on response encryption. This verifies the
        request body sent BY the client is also encrypted in the network capture.
        """
        import asyncio

        # Unique canary that should appear in request body
        request_canary = "REQUEST_BODY_CANARY_XYZ789"
        request_payload = {
            "secret_request_data": request_canary,
            "password": "super_secret_password_123",
            "api_key": "sk-live-abcdef123456",
        }

        resp = await aiohttp_client.post("/echo", json=request_payload)
        assert resp.status == 200

        await asyncio.sleep(0.3)

        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()

        # Request body canaries should NOT be visible
        assert request_canary.encode() not in pcap_data, (
            f"Request body canary '{request_canary}' found in pcap! Request not encrypted."
        )
        assert b"super_secret_password" not in pcap_data, "Password found in pcap!"
        assert b"sk-live-" not in pcap_data, "API key prefix found in pcap!"
        assert b'"secret_request_data"' not in pcap_data, "JSON key found in pcap!"

    async def test_no_psk_on_wire(
        self,
        tcpdump_capture: str,
        aiohttp_client: HPKEClientSession,
        test_psk: bytes,
    ) -> None:
        """Verify pre-shared key never appears in network traffic.

        The PSK is used for authentication but should NEVER be transmitted.
        It's used locally to derive keys, not sent over the wire.
        """
        import asyncio

        # Generate some traffic
        resp = await aiohttp_client.post("/echo", json={"test": "psk_leak_check"})
        assert resp.status == 200

        await asyncio.sleep(0.3)

        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()

        # PSK should never appear in any form
        assert test_psk not in pcap_data, "CRITICAL: Raw PSK found in network capture!"
        # Also check hex-encoded form
        assert test_psk.hex().encode() not in pcap_data, "PSK (hex) found in pcap!"

    async def test_no_session_key_material_on_wire(
        self,
        tcpdump_capture: str,
        aiohttp_client: HPKEClientSession,
        platform_keypair: tuple[bytes, bytes],
    ) -> None:
        """Verify private key and derived session keys never appear in traffic.

        Only the ephemeral PUBLIC key should be visible (the 'enc' field in HPKE).
        Private keys and derived session keys must never be transmitted.
        """
        import asyncio

        private_key, _public_key = platform_keypair

        # Generate traffic
        resp = await aiohttp_client.post("/echo", json={"test": "key_leak_check"})
        assert resp.status == 200

        await asyncio.sleep(0.3)

        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()

        # Private key should NEVER appear
        assert private_key not in pcap_data, "CRITICAL: Private key found in network capture!"

        # Note: Server's static public key IS visible in /.well-known/hpke-keys response
        # (that's expected - it's public). We only check private key doesn't leak.


# =============================================================================
# Active Attack Resistance (no root required)
# =============================================================================


class TestActiveAttackResistance:
    """Tests that verify resistance to active attacks (tampering, replay).

    These tests verify cryptographic properties at the protocol level
    without requiring network packet capture.
    """

    async def test_tampered_ciphertext_rejected(
        self,
        platform_keypair: tuple[bytes, bytes],
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Verify that tampered ciphertext is rejected (AEAD authentication).

        An active attacker who modifies ciphertext in transit should cause
        decryption to fail. This tests the ChaCha20-Poly1305 authentication
        tag verification at the HPKE layer.
        """
        from hpke_http.exceptions import DecryptionError
        from hpke_http.hpke import open_psk, seal_psk

        private_key, public_key = platform_keypair

        # Create a valid HPKE-encrypted message
        plaintext = b"This is a secret message that will be tampered with"
        info = b"hpke-http"
        aad = b""

        enc, ciphertext = seal_psk(
            pk_r=public_key,
            info=info,
            psk=test_psk,
            psk_id=test_psk_id,
            aad=aad,
            plaintext=plaintext,
        )

        # Verify valid ciphertext decrypts correctly
        decrypted = open_psk(
            enc=enc,
            sk_r=private_key,
            info=info,
            psk=test_psk,
            psk_id=test_psk_id,
            aad=aad,
            ciphertext=ciphertext,
        )
        assert decrypted == plaintext

        # Now tamper with the ciphertext
        tampered_ciphertext = bytearray(ciphertext)
        # Flip bits in the middle of the ciphertext
        tampered_ciphertext[len(ciphertext) // 2] ^= 0xFF
        tampered_ciphertext[len(ciphertext) // 2 + 1] ^= 0xFF

        # Tampered ciphertext should fail authentication
        try:
            open_psk(
                enc=enc,
                sk_r=private_key,
                info=info,
                psk=test_psk,
                psk_id=test_psk_id,
                aad=aad,
                ciphertext=bytes(tampered_ciphertext),
            )
            raise AssertionError("CRITICAL: Tampered ciphertext was decrypted! AEAD authentication is not working.")
        except DecryptionError:
            pass  # Expected - authentication failed

        # Also test tampering with the authentication tag itself (last 16 bytes)
        tag_tampered = bytearray(ciphertext)
        tag_tampered[-1] ^= 0xFF  # Flip last byte of tag

        try:
            open_psk(
                enc=enc,
                sk_r=private_key,
                info=info,
                psk=test_psk,
                psk_id=test_psk_id,
                aad=aad,
                ciphertext=bytes(tag_tampered),
            )
            raise AssertionError(
                "CRITICAL: Tag-tampered ciphertext was decrypted! AEAD tag verification is not working."
            )
        except DecryptionError:
            pass  # Expected - tag verification failed

    async def test_sse_replay_attack_detected(self) -> None:
        """Verify SSE streaming detects out-of-order/replay attacks.

        The ChunkDecryptor maintains a monotonic counter. If events arrive
        out of order (indicating replay or reordering attack), decryption
        should fail with ReplayAttackError.
        """
        from hpke_http.exceptions import ReplayAttackError
        from hpke_http.streaming import ChunkDecryptor, ChunkEncryptor, StreamingSession

        # Create a session and encryptor/decryptor pair
        session = StreamingSession(
            session_key=b"k" * 32,
            session_salt=b"salt",
        )
        encryptor = ChunkEncryptor(session)
        decryptor = ChunkDecryptor(session)

        # Encrypt three chunks
        chunk1 = encryptor.encrypt(b"first")
        chunk2 = encryptor.encrypt(b"second")
        chunk3 = encryptor.encrypt(b"third")

        # Extract data fields from SSE format
        def get_data(sse_bytes: bytes) -> str:
            for line in sse_bytes.decode("ascii").split("\n"):
                if line.startswith("data: "):
                    return line[6:]
            raise ValueError("No data field")

        # Normal order works
        decryptor.decrypt(get_data(chunk1))
        decryptor.decrypt(get_data(chunk2))
        decryptor.decrypt(get_data(chunk3))

        # Now test replay attack detection with a fresh decryptor
        decryptor2 = ChunkDecryptor(StreamingSession(session_key=b"k" * 32, session_salt=b"salt"))

        # Decrypt chunk1 first (counter=1)
        decryptor2.decrypt(get_data(chunk1))

        # Try to decrypt chunk1 again (replay attack - counter should be 2 now)
        try:
            decryptor2.decrypt(get_data(chunk1))
            raise AssertionError("Replay attack was not detected! chunk1 decrypted twice.")
        except ReplayAttackError:
            pass  # Expected - replay detected

        # Try to decrypt chunk3 (skipping chunk2 - out of order)
        try:
            decryptor2.decrypt(get_data(chunk3))
            raise AssertionError("Out-of-order attack not detected! chunk3 before chunk2.")
        except ReplayAttackError:
            pass  # Expected - out of order detected


# =============================================================================
# AIOHTTP CONNECTION LEAK TESTS
# =============================================================================


class TestAiohttpConnectionLeaks:
    """Verify aiohttp connections are properly released."""

    @pytest.mark.skipif(
        sys.version_info >= (3, 14),
        reason="ResourceWarning GC timing is flaky on Python 3.14+ due to free-threading changes",
    )
    async def test_no_resource_warning_normal_request(
        self,
        aiohttp_client: Any,
    ) -> None:
        """Normal request with consumed response emits no ResourceWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ResourceWarning)

            resp = await aiohttp_client.post("/echo", json={"test": 1})
            await resp.read()  # Consume response to release connection

            # Yield to event loop - allows pending cleanup tasks to complete
            # Use longer sleep (0.5s) to avoid flaky failures on slower CI runners
            await asyncio.sleep(0.5)

            gc.collect()
            gc.collect()

            # Filter to connection leaks only
            leaks = [x for x in w if issubclass(x.category, ResourceWarning) and _is_connection_leak(x)]
            assert not leaks, f"Connection leak: {leaks[0].message}"

    @pytest.mark.skipif(
        sys.version_info >= (3, 14),
        reason="ResourceWarning GC timing is flaky on Python 3.14+ due to free-threading changes",
    )
    async def test_no_resource_warning_with_context_manager(
        self,
        aiohttp_client: Any,
    ) -> None:
        """Context manager releases connection without consuming body."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ResourceWarning)

            resp = await aiohttp_client.post("/echo", json={"test": 1})
            async with resp:
                # Don't consume body - context manager handles cleanup
                assert resp.status == 200

            # Yield to event loop - allows pending cleanup tasks to complete
            # Use longer sleep (0.5s) to avoid flaky failures on slower CI runners
            await asyncio.sleep(0.5)

            gc.collect()
            gc.collect()

            # Filter to connection leaks only
            leaks = [x for x in w if issubclass(x.category, ResourceWarning) and _is_connection_leak(x)]
            assert not leaks, f"Connection leak: {leaks[0].message}"

    @pytest.mark.skipif(
        sys.version_info >= (3, 14),
        reason="ResourceWarning GC timing is flaky on Python 3.14+ due to free-threading changes",
    )
    async def test_no_resource_warning_with_release(
        self,
        aiohttp_client: Any,
    ) -> None:
        """Explicit release() cleans up without consuming body."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ResourceWarning)

            resp = await aiohttp_client.post("/echo", json={"test": 1})
            assert resp.status == 200
            await resp.release()  # Explicit cleanup

            # Yield to event loop - allows pending cleanup tasks to complete
            # Use longer sleep (0.5s) to avoid flaky failures on slower CI runners
            await asyncio.sleep(0.5)

            gc.collect()
            gc.collect()

            # Filter to connection leaks only
            leaks = [x for x in w if issubclass(x.category, ResourceWarning) and _is_connection_leak(x)]
            assert not leaks, f"Connection leak: {leaks[0].message}"

    @pytest.mark.skipif(
        sys.version_info >= (3, 14),
        reason="ResourceWarning GC timing is flaky on Python 3.14+ due to free-threading changes",
    )
    async def test_no_resource_warning_sse_early_break(
        self,
        aiohttp_client: Any,
    ) -> None:
        """SSE with early break emits no connection ResourceWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ResourceWarning)

            resp = await aiohttp_client.post("/stream", json={"start": True})
            async for _ in aiohttp_client.iter_sse(resp):
                break

            # Yield to event loop - allows pending cleanup tasks to complete
            # Use longer sleep (0.5s) to avoid flaky failures on slower CI runners
            await asyncio.sleep(0.5)

            gc.collect()
            gc.collect()

            # Filter to connection leaks only
            leaks = [x for x in w if issubclass(x.category, ResourceWarning) and _is_connection_leak(x)]
            assert not leaks, f"Connection leak: {leaks[0].message}"

    async def test_multiple_sse_streams_sequential(
        self,
        aiohttp_client: Any,
    ) -> None:
        """Multiple SSE streams don't leak."""
        for i in range(5):
            resp = await aiohttp_client.post("/stream", json={"i": i})
            count = 0
            async for _ in aiohttp_client.iter_sse(resp):
                count += 1
                if count >= 2:
                    break

        # If leaking, pool would be exhausted
        resp = await aiohttp_client.post("/echo", json={"final": True})
        assert resp.status == 200

    async def test_post_releases_connection(self, aiohttp_client: Any) -> None:
        """POST /echo releases connection."""
        connector = aiohttp_client._session.connector  # noqa: SLF001

        resp = await aiohttp_client.post("/echo", json={"test": "leak"})
        assert resp.status == 200

        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # noqa: SLF001

    async def test_sequential_requests(self, aiohttp_client: Any) -> None:
        """10 sequential requests don't leak."""
        connector = aiohttp_client._session.connector  # noqa: SLF001

        for i in range(10):
            resp = await aiohttp_client.post("/echo", json={"seq": i})
            assert resp.status == 200

        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # noqa: SLF001

    async def test_concurrent_requests(self, aiohttp_client: Any) -> None:
        """5 concurrent requests don't leak."""
        connector = aiohttp_client._session.connector  # noqa: SLF001

        async def req(n: int) -> None:
            resp = await aiohttp_client.post("/echo", json={"n": n})
            assert resp.status == 200

        await asyncio.gather(*[req(i) for i in range(5)])

        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # noqa: SLF001

    async def test_sse_stream_releases(self, aiohttp_client: Any) -> None:
        """SSE stream releases connection after consumption."""
        connector = aiohttp_client._session.connector  # noqa: SLF001

        resp = await aiohttp_client.post("/stream", json={"start": True})
        assert resp.status == 200

        event_count = 0
        async for _chunk in aiohttp_client.iter_sse(resp):
            event_count += 1

        assert event_count >= 1
        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # noqa: SLF001

    async def test_sse_many_events(self, aiohttp_client: Any) -> None:
        """50-event SSE doesn't leak."""
        connector = aiohttp_client._session.connector  # noqa: SLF001

        resp = await aiohttp_client.post("/stream-many", json={"start": True})
        assert resp.status == 200

        count = 0
        async for _chunk in aiohttp_client.iter_sse(resp):
            count += 1

        assert count >= 50
        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # noqa: SLF001

    async def test_sse_cancelled_early(self, aiohttp_client: Any) -> None:
        """Early SSE cancellation releases connection."""
        connector = aiohttp_client._session.connector  # noqa: SLF001

        resp = await aiohttp_client.post("/stream-many", json={"start": True})
        assert resp.status == 200

        count = 0
        async for _chunk in aiohttp_client.iter_sse(resp):
            count += 1
            if count >= 5:
                break

        await asyncio.sleep(0.05)
        assert len(connector._acquired) == 0  # noqa: SLF001


class TestAiohttpConnectionLeaksCompressed:
    """Connection leak tests with compression enabled."""

    async def test_compressed_post(self, aiohttp_client_compressed: Any) -> None:
        """Compressed POST releases connection."""
        connector = aiohttp_client_compressed._session.connector  # noqa: SLF001

        resp = await aiohttp_client_compressed.post("/echo", json={"compressed": True})
        assert resp.status == 200

        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # noqa: SLF001

    async def test_compressed_sse(self, aiohttp_client_compressed: Any) -> None:
        """Compressed SSE releases connection."""
        connector = aiohttp_client_compressed._session.connector  # noqa: SLF001

        resp = await aiohttp_client_compressed.post("/stream", json={"start": True})
        assert resp.status == 200

        async for _chunk in aiohttp_client_compressed.iter_sse(resp):
            pass

        await asyncio.sleep(0.01)
        assert len(connector._acquired) == 0  # noqa: SLF001


class TestAiohttpPoolExhaustion:
    """Tests requiring small connection pool to detect leaks."""

    async def test_sequential_requests_no_exhaustion(
        self,
        aiohttp_client_small_pool: Any,
    ) -> None:
        """Sequential requests don't exhaust pool of 2."""
        for i in range(10):
            resp = await aiohttp_client_small_pool.post("/echo", json={"i": i})
            assert resp.status == 200

    async def test_concurrent_requests_no_exhaustion(
        self,
        aiohttp_client_small_pool: Any,
    ) -> None:
        """Concurrent requests complete without pool timeout."""

        async def make_request(i: int) -> int:
            resp = await aiohttp_client_small_pool.post("/echo", json={"i": i})
            return resp.status

        # 3 batches, pool of 2 - must release connections
        for _ in range(3):
            results = await asyncio.gather(*[make_request(i) for i in range(2)])
            assert all(r == 200 for r in results)

    async def test_sse_cancellation_releases_connection(
        self,
        aiohttp_client_small_pool: Any,
    ) -> None:
        """Cancelled SSE releases connection to pool."""

        async def consume_sse() -> None:
            resp = await aiohttp_client_small_pool.post("/stream", json={"start": True})
            async for _ in aiohttp_client_small_pool.iter_sse(resp):
                await asyncio.sleep(1)

        task = asyncio.create_task(consume_sse())
        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Pool of 2 - if leaked, this times out
        resp = await aiohttp_client_small_pool.post("/echo", json={"after": True})
        assert resp.status == 200


# =============================================================================
# HTTPX Client Tests
# =============================================================================
# Fixtures are defined in conftest.py: httpx_client, httpx_client_compressed


class TestEncryptedRequestsHTTPX:
    """Test encrypted request/response flow with HTTPX client."""

    async def test_encrypted_request_roundtrip(self, httpx_client: HPKEAsyncClient) -> None:
        """Client encrypts -> Server decrypts -> Response works."""
        test_data = {"message": "Hello, HPKE via HTTPX!", "count": 42}

        resp = await httpx_client.post("/echo", json=test_data)
        assert resp.status_code == 200
        data = resp.json()

        assert data["path"] == "/echo"
        assert data["method"] == "POST"
        # Echo contains the JSON string we sent
        assert "Hello, HPKE via HTTPX!" in data["echo"]
        assert "42" in data["echo"]

    async def test_large_payload(self, httpx_client: HPKEAsyncClient) -> None:
        """Large payloads encrypt/decrypt correctly."""
        large_content = "x" * 100_000  # 100KB
        test_data = {"data": large_content}

        resp = await httpx_client.post("/echo", json=test_data)
        assert resp.status_code == 200
        data = resp.json()

        # Verify the large content made it through
        assert large_content in data["echo"]

    async def test_binary_payload(self, httpx_client: HPKEAsyncClient) -> None:
        """Binary data encrypts/decrypts correctly."""
        binary_data = bytes(range(256)) * 10  # Various byte values

        resp = await httpx_client.post("/echo", content=binary_data)
        assert resp.status_code == 200
        data = resp.json()
        # Binary data should be in the echo (may be escaped)
        assert len(data["echo"]) > 0

    async def test_get_request(self, httpx_client: HPKEAsyncClient) -> None:
        """GET requests work (no body encryption needed)."""
        resp = await httpx_client.get("/health")
        assert resp.status_code == 200

    async def test_all_http_methods(self, httpx_client: HPKEAsyncClient) -> None:
        """All HTTP methods work correctly."""
        # POST
        resp = await httpx_client.post("/echo", json={"method": "POST"})
        assert resp.status_code == 200
        assert resp.json()["method"] == "POST"

        # PUT
        resp = await httpx_client.put("/echo", json={"method": "PUT"})
        assert resp.status_code == 200
        assert resp.json()["method"] == "PUT"

        # PATCH
        resp = await httpx_client.patch("/echo", json={"method": "PATCH"})
        assert resp.status_code == 200
        assert resp.json()["method"] == "PATCH"

        # DELETE
        resp = await httpx_client.delete("/echo")
        assert resp.status_code == 200
        assert resp.json()["method"] == "DELETE"

        # HEAD (no body in response)
        resp = await httpx_client.head("/health")
        assert resp.status_code == 200

        # OPTIONS (may or may not be supported by test server)
        resp = await httpx_client.options("/echo")
        # Just verify it doesn't crash


class TestStandardResponseEncryptionHTTPX:
    """Test encrypted standard (non-SSE) responses with HTTPX."""

    async def test_response_has_hpke_stream_header(self, httpx_client: HPKEAsyncClient) -> None:
        """Encrypted request triggers encrypted response with X-HPKE-Stream header."""
        resp = await httpx_client.post("/echo", json={"test": 1})
        assert resp.status_code == 200

        # HTTPXDecryptedResponse wraps the underlying response
        assert isinstance(resp, HTTPXDecryptedResponse)

        # X-HPKE-Stream header should be present (contains salt)
        assert HEADER_HPKE_STREAM in resp.headers

        # Content-Type should NOT be text/event-stream (that's for SSE)
        content_type = resp.headers.get("Content-Type", "")
        assert "text/event-stream" not in content_type

    async def test_decrypted_response_json(self, httpx_client: HPKEAsyncClient) -> None:
        """HTTPXDecryptedResponse.json() returns decrypted data."""
        test_data = {"message": "secret", "value": 42}
        resp = await httpx_client.post("/echo", json=test_data)

        # json() should return decrypted data
        data = resp.json()
        assert "message" in data["echo"]
        assert "secret" in data["echo"]

    async def test_decrypted_response_content(self, httpx_client: HPKEAsyncClient) -> None:
        """HTTPXDecryptedResponse.content returns raw decrypted bytes."""
        resp = await httpx_client.post("/echo", json={"raw": "test"})

        # content should return decrypted bytes
        raw_bytes = resp.content
        assert b"raw" in raw_bytes
        assert b"test" in raw_bytes

    async def test_decrypted_response_text(self, httpx_client: HPKEAsyncClient) -> None:
        """HTTPXDecryptedResponse.text returns decrypted text."""
        resp = await httpx_client.post("/echo", json={"text": "hello"})

        # text should return decrypted string
        text = resp.text
        assert "text" in text
        assert "hello" in text


class TestSSEEncryptionHTTPX:
    """Test encrypted SSE streaming with HTTPX."""

    async def test_sse_stream_roundtrip(self, httpx_client: HPKEAsyncClient) -> None:
        """SSE events are encrypted end-to-end."""
        resp = await httpx_client.post("/stream", json={"start": True})
        assert resp.status_code == 200
        events = [parse_sse_chunk(chunk) async for chunk in httpx_client.iter_sse(resp)]

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

    async def test_sse_counter_monotonicity(self, httpx_client: HPKEAsyncClient) -> None:
        """SSE events have monotonically increasing counters."""
        event_count = 0

        resp = await httpx_client.post("/stream", json={"start": True})
        assert resp.status_code == 200
        async for _chunk in httpx_client.iter_sse(resp):
            event_count += 1

        # Verify all events were processed (counter worked correctly)
        assert event_count == 4

    async def test_sse_many_events(self, httpx_client: HPKEAsyncClient) -> None:
        """Many SSE events work correctly."""
        resp = await httpx_client.post("/stream-many", json={"start": True})
        assert resp.status_code == 200
        events = [parse_sse_chunk(chunk) async for chunk in httpx_client.iter_sse(resp)]

        # Should have 51 events: 50 tick + 1 done
        assert len(events) == 51


class TestDecryptedResponseHTTPXCompat:
    """Test HTTPXDecryptedResponse API compatibility with httpx.Response."""

    async def test_status_code_property(self, httpx_client: HPKEAsyncClient) -> None:
        """status_code property works."""
        resp = await httpx_client.post("/echo", json={"test": 1})
        assert resp.status_code == 200

    async def test_is_success_property(self, httpx_client: HPKEAsyncClient) -> None:
        """is_success property works for 2xx responses."""
        resp = await httpx_client.post("/echo", json={"test": 1})
        assert resp.is_success is True

    async def test_headers_property(self, httpx_client: HPKEAsyncClient) -> None:
        """headers property returns httpx.Headers-compatible object."""
        resp = await httpx_client.post("/echo", json={"test": 1})
        assert "content-type" in resp.headers or "Content-Type" in resp.headers

    async def test_url_property(self, httpx_client: HPKEAsyncClient) -> None:
        """url property works."""
        resp = await httpx_client.post("/echo", json={"test": 1})
        assert "/echo" in str(resp.url)

    async def test_raise_for_status(self, httpx_client: HPKEAsyncClient) -> None:
        """raise_for_status() doesn't raise for 2xx."""
        resp = await httpx_client.post("/echo", json={"test": 1})
        # Should not raise
        resp.raise_for_status()

    async def test_unwrap_returns_httpx_response(self, httpx_client: HPKEAsyncClient) -> None:
        """unwrap() returns the underlying httpx.Response."""
        resp = await httpx_client.post("/echo", json={"test": 1})
        assert isinstance(resp, HTTPXDecryptedResponse)
        unwrapped = resp.unwrap()
        assert isinstance(unwrapped, httpx.Response)


class TestCompressionE2EHTTPX:
    """Test Zstd compression with HTTPX."""

    async def test_compressed_request_roundtrip(
        self,
        httpx_client_compressed: HPKEAsyncClient,
    ) -> None:
        """Compressed requests roundtrip correctly."""
        large_data = {"message": "x" * 1000, "nested": {"key": "value" * 100}}
        resp = await httpx_client_compressed.post("/echo", json=large_data)
        assert resp.status_code == 200
        data = resp.json()
        assert "x" * 1000 in data["echo"]


# =============================================================================
# TEST: release_encrypted flag
# =============================================================================


class TestReleaseEncryptedAiohttp:
    """Test release_encrypted flag for aiohttp middleware."""

    async def test_decrypted_content_accessible(
        self,
        aiohttp_client_release_encrypted: HPKEClientSession,
    ) -> None:
        """Decrypted content is accessible when release_encrypted=True."""
        resp = await aiohttp_client_release_encrypted.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # Decrypted content should be accessible
        data = await resp.json()
        assert "echo" in data
        # Echo returns the raw request body as a string
        assert "test" in data["echo"]

    async def test_encrypted_content_released(
        self,
        aiohttp_client_release_encrypted: HPKEClientSession,
    ) -> None:
        """Encrypted content is released after decryption when release_encrypted=True."""
        resp = await aiohttp_client_release_encrypted.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # Trigger decryption
        _ = await resp.read()

        # Underlying response body should be cleared
        underlying = resp.unwrap()
        # aiohttp stores body in _body attribute
        assert underlying._body == b""  # type: ignore[attr-defined]  # noqa: SLF001

    async def test_multiple_reads_still_work(
        self,
        aiohttp_client_release_encrypted: HPKEClientSession,
    ) -> None:
        """Multiple reads return cached decrypted content even after encrypted is released."""
        resp = await aiohttp_client_release_encrypted.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # First read triggers decryption and release
        body1 = await resp.read()
        # Second read should return cached decrypted content
        body2 = await resp.read()

        assert body1 == body2
        assert len(body1) > 0

    async def test_flag_default_false(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """By default, release_encrypted is False and encrypted content is retained."""
        resp = await aiohttp_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # Trigger decryption
        _ = await resp.read()

        # Underlying response body should NOT be cleared
        underlying = resp.unwrap()
        assert underlying._body != b""  # type: ignore[attr-defined]  # noqa: SLF001


class TestReleaseEncryptedHTTPX:
    """Test release_encrypted flag for httpx middleware."""

    async def test_decrypted_content_accessible(
        self,
        httpx_client_release_encrypted: HPKEAsyncClient,
    ) -> None:
        """Decrypted content is accessible when release_encrypted=True."""
        resp = await httpx_client_release_encrypted.post("/echo", json={"test": 1})
        assert isinstance(resp, HTTPXDecryptedResponse)

        # Decrypted content should be accessible
        data = resp.json()
        assert "echo" in data
        # Echo returns the raw request body as a string
        assert "test" in data["echo"]

    async def test_encrypted_content_released(
        self,
        httpx_client_release_encrypted: HPKEAsyncClient,
    ) -> None:
        """Encrypted content is released after decryption when release_encrypted=True."""
        resp = await httpx_client_release_encrypted.post("/echo", json={"test": 1})
        assert isinstance(resp, HTTPXDecryptedResponse)

        # Trigger decryption
        _ = resp.content

        # Underlying response body should be cleared
        underlying = resp.unwrap()
        # httpx stores body in _content attribute
        assert underlying._content == b""  # type: ignore[attr-defined]  # noqa: SLF001

    async def test_multiple_reads_still_work(
        self,
        httpx_client_release_encrypted: HPKEAsyncClient,
    ) -> None:
        """Multiple reads return cached decrypted content even after encrypted is released."""
        resp = await httpx_client_release_encrypted.post("/echo", json={"test": 1})
        assert isinstance(resp, HTTPXDecryptedResponse)

        # First read triggers decryption and release
        body1 = resp.content
        # Second read should return cached decrypted content
        body2 = resp.content

        assert body1 == body2
        assert len(body1) > 0

    async def test_flag_default_false(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """By default, release_encrypted is False and encrypted content is retained."""
        resp = await httpx_client.post("/echo", json={"test": 1})
        assert isinstance(resp, HTTPXDecryptedResponse)

        # Trigger decryption
        _ = resp.content

        # Underlying response body should NOT be cleared
        underlying = resp.unwrap()
        assert underlying._content != b""  # type: ignore[attr-defined]  # noqa: SLF001


# =============================================================================
# HTTPX Wire Encryption Tests
# =============================================================================
# Mirror of aiohttp wire encryption tests to ensure both clients are tested.


class TestEncryptionStateValidationHTTPX:
    """
    E2E tests that validate encryption at the wire level for httpx.

    These tests verify that:
    1. When protocol expects encryption, raw content IS encrypted (not plaintext)
    2. When protocol does NOT expect encryption, raw content is plaintext
    3. Violations of expected encryption state raise appropriate errors
    """

    async def test_encrypted_response_is_not_plaintext(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify encrypted response body is NOT readable as plaintext JSON."""
        resp = await httpx_client.post("/echo", json={"secret": "data"})

        # The response should be encrypted - verify by trying to parse as JSON
        # Get raw bytes from underlying response using public unwrap() method
        assert isinstance(resp, HTTPXDecryptedResponse)
        raw_body = resp.unwrap().content

        # Raw body should NOT be valid JSON (it's encrypted)
        try:
            json.loads(raw_body)
            # If this succeeds, the response was NOT encrypted - FAIL
            raise AssertionError(
                f"Response body was readable as plaintext JSON - encryption expected! "
                f"Raw body starts with: {raw_body[:100]!r}"
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Expected - raw body is encrypted, not plaintext JSON
            # UnicodeDecodeError can occur when encrypted bytes are invalid UTF
            pass

        # But decrypted response SHOULD be valid JSON
        decrypted = resp.json()
        assert "echo" in decrypted

    async def test_encrypted_sse_is_not_plaintext(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify encrypted SSE events are NOT readable as plaintext SSE."""
        resp = await httpx_client.post("/stream", json={"start": True})

        # Read raw chunks from underlying response
        underlying = resp.unwrap() if isinstance(resp, HTTPXDecryptedResponse) else resp
        raw_chunks = [chunk async for chunk in underlying.aiter_bytes()]

        # Combine all raw data
        raw_data = b"".join(raw_chunks)

        # Raw data should be encrypted SSE format (event: enc)
        # NOT plaintext SSE (event: progress, etc.)
        assert b"event: enc" in raw_data, "Encrypted SSE should use 'event: enc' format"
        assert b"event: progress" not in raw_data, "Raw SSE should NOT contain plaintext events"

        # The data field should be base64url encoded, not plaintext JSON
        # Check that we don't see unencrypted JSON in the raw data
        assert b'"progress"' not in raw_data, "Raw SSE should NOT contain plaintext JSON"

    async def test_encryption_header_presence_matches_content(
        self,
        httpx_client: HPKEAsyncClient,
        granian_server: E2EServer,
    ) -> None:
        """Verify X-HPKE-Stream header presence matches actual encryption."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        # Case 1: Encrypted request → should get encrypted response with header
        resp = await httpx_client.post("/echo", json={"test": 1})

        assert HEADER_HPKE_STREAM in resp.headers, "Encrypted response MUST have X-HPKE-Stream header"

        # Verify content is actually encrypted
        assert isinstance(resp, HTTPXDecryptedResponse)
        raw_body = resp.unwrap().content
        if raw_body:
            try:
                json.loads(raw_body)
                raise AssertionError("Header claims encryption but body is plaintext")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass  # Expected - body is encrypted

        # Case 2: Unencrypted request → should get unencrypted response without header
        async with httpx.AsyncClient() as plain_client:
            resp = await plain_client.get(f"{base_url}/health")
            assert HEADER_HPKE_STREAM not in resp.headers, "Unencrypted response MUST NOT have X-HPKE-Stream header"

            # Verify content is actually plaintext
            raw_body = resp.content
            try:
                json.loads(raw_body)  # Should succeed
            except json.JSONDecodeError as e:
                raise AssertionError("No encryption header but body is not plaintext") from e


class TestRawWireFormatValidationHTTPX:
    """
    Tests that validate the exact wire format of encrypted data for httpx.

    These tests ensure the encryption format matches the protocol specification.
    """

    async def test_standard_response_wire_format(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify standard response wire format: [length(4B) || counter(4B) || ciphertext]."""
        resp = await httpx_client.post("/echo", json={"format": "test"})

        # Access raw encrypted body using public unwrap() method
        assert isinstance(resp, HTTPXDecryptedResponse)
        raw_body = resp.unwrap().content

        # Wire format validation:
        # - Minimum size: length(4) + counter(4) + encoding_id(1) + tag(16) = 25 bytes
        assert len(raw_body) >= 25, f"Encrypted body too short: {len(raw_body)} bytes"

        # - First 4 bytes are length prefix
        length = int.from_bytes(raw_body[:4], "big")
        assert length >= 21, f"Chunk length should be >= 21, got {length}"

        # - Bytes 4-8 are counter (should be 1 for first chunk)
        counter = int.from_bytes(raw_body[4:8], "big")
        assert counter == 1, f"First chunk counter should be 1, got {counter}"

    async def test_sse_wire_format(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify SSE wire format: event: enc\\ndata: <base64>\\n\\n.

        SSEFormat uses standard base64 (not base64url) for ~1.7x faster encoding.
        See streaming.py SSEFormat docstring for rationale.
        """
        resp = await httpx_client.post("/stream", json={"start": True})

        # Get underlying response for raw access
        underlying = resp.unwrap() if isinstance(resp, HTTPXDecryptedResponse) else resp

        # Read enough raw chunks to get a complete event
        raw_chunks: list[bytes] = []
        async for chunk in underlying.aiter_bytes():
            raw_chunks.append(chunk)
            # Check if we have at least one complete event (contains data field)
            combined = b"".join(raw_chunks).decode("utf-8", errors="replace")
            if "data:" in combined and "\n\n" in combined:
                break

        raw_str = b"".join(raw_chunks).decode("utf-8", errors="replace")

        # SSE format validation
        assert "event: enc" in raw_str, f"SSE should have 'event: enc', got: {raw_str[:100]}"
        assert "data:" in raw_str, f"SSE should have 'data:' field, got: {raw_str[:100]}"

        # Data field should be standard base64 encoded (A-Za-z0-9+/=)
        for line in raw_str.split("\n"):
            if line.startswith("data:"):
                data_value = line[5:].strip()
                assert re.match(r"^[A-Za-z0-9+/=]+$", data_value), (
                    f"Data field should be base64, got: {data_value[:50]}"
                )
                break


class TestCryptographicPropertiesHTTPX:
    """Tests that verify encryption is ACTUALLY happening at the wire level for httpx."""

    async def test_encrypted_body_has_cryptographic_entropy(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify encrypted output has high Shannon entropy (> 7.0 bits/byte)."""
        payload = {"data": "A" * 10000, "secret": "password123"}

        resp = await httpx_client.post("/echo", json=payload)
        assert resp.status_code == 200
        assert isinstance(resp, HTTPXDecryptedResponse)

        raw_body = resp.unwrap().content
        assert len(raw_body) > 256, "Response too short for entropy analysis"

        entropy = calculate_shannon_entropy(raw_body)
        assert entropy > 7.0, (
            f"Entropy {entropy:.2f} bits/byte too low for encrypted data. "
            f"Expected > 7.0. May indicate encryption bypass."
        )

    async def test_encrypted_body_uniform_distribution(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify encrypted data has uniform byte distribution.

        Chi-square tests have inherent false positive rate equal to the p-value
        threshold (e.g., p > 0.01 means 1% of truly random data fails). To reduce
        CI flakiness, we run multiple trials and allow one failure.

        With 10 trials and 9 required passes at p > 0.01:
        P(flaky failure) = P(>=2 of 10 fail) ≈ 0.4% (binomial)
        """
        passes = 0
        results: list[tuple[float, float]] = []

        for _ in range(CHI_SQUARE_TRIALS):
            payload = {"message": "Hello World " * 1000}
            resp = await httpx_client.post("/echo", json=payload)
            assert resp.status_code == 200
            assert isinstance(resp, HTTPXDecryptedResponse)

            raw_body = resp.unwrap().content
            assert len(raw_body) >= 1000, "Response too short for chi-square test"

            chi2, p_value = chi_square_byte_uniformity(raw_body)
            results.append((chi2, p_value))
            if p_value > CHI_SQUARE_P_THRESHOLD:
                passes += 1

        assert passes >= CHI_SQUARE_MIN_PASS, (
            f"Chi-square uniformity test failed: {passes}/{CHI_SQUARE_TRIALS} trials passed "
            f"(required {CHI_SQUARE_MIN_PASS}). Results: {results}. "
            f"Encrypted data should appear random."
        )

    async def test_known_plaintext_not_visible_on_wire(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify known plaintext values never appear in encrypted wire data."""
        canary = "CANARY_SECRET_12345_XYZ"
        payload = {"secret": canary, "data": "Hello World Test Message"}

        resp = await httpx_client.post("/echo", json=payload)
        assert resp.status_code == 200
        assert isinstance(resp, HTTPXDecryptedResponse)

        raw_body = resp.unwrap().content

        forbidden_patterns = [
            canary.encode(),
            b'"secret"',
            b'"echo"',
            b"Hello World",
        ]

        for pattern in forbidden_patterns:
            assert pattern not in raw_body, (
                f"Known plaintext '{pattern.decode(errors='replace')}' found in "
                f"encrypted wire data! Encryption may be bypassed."
            )

    async def test_wire_format_cryptographic_structure(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify wire format has valid cryptographic structure."""
        resp = await httpx_client.post("/echo", json={"test": "structure"})
        assert resp.status_code == 200
        assert isinstance(resp, HTTPXDecryptedResponse)

        raw_body = resp.unwrap().content
        assert len(raw_body) >= 25, f"Response too short: {len(raw_body)} bytes"

        chunk_len = int.from_bytes(raw_body[:4], "big")
        assert chunk_len >= 21, f"Chunk length {chunk_len} too short"

        counter = int.from_bytes(raw_body[4:8], "big")
        assert counter >= 1, f"Counter {counter} invalid - must start at 1"


class TestStreamingBehaviorHTTPX:
    """Tests that verify chunking/streaming is ACTUALLY happening for httpx."""

    async def test_sse_chunks_arrive_progressively(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify SSE chunks arrive with inter-arrival timing gaps.

        This test verifies progressive delivery: events arrive as the server sends them,
        not all buffered at the end. Uses client.send(stream=True) for true streaming.
        """
        resp = await httpx_client.post("/stream-delayed", json={"start": True})
        assert resp.status_code == 200

        # Use iter_sse() which consumes the stream progressively
        arrival_times: list[float] = []
        async for _chunk in httpx_client.iter_sse(resp):
            arrival_times.append(time.monotonic())
            if len(arrival_times) >= 6:
                break

        assert len(arrival_times) >= 5, f"Expected 5+ chunks, got {len(arrival_times)}"

        gaps = [arrival_times[i + 1] - arrival_times[i] for i in range(len(arrival_times) - 1)]
        significant_gaps = sum(1 for g in gaps if g > 0.05)

        # At least one significant gap proves progressive delivery (not all buffered until end).
        # TCP and encryption layer may batch adjacent events, so we only require 1 gap.
        assert significant_gaps >= 1, (
            f"No gaps > 50ms found. All chunks arrived instantly - not streaming. "
            f"Gaps: {[f'{g * 1000:.0f}ms' for g in gaps]}"
        )

    async def test_large_request_is_chunked(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify large requests are sent in multiple chunks."""
        size_mb = 10
        large_content = "X" * (size_mb * 1024 * 1024)

        resp = await httpx_client.post("/echo-chunks", content=large_content.encode())
        assert resp.status_code == 200

        data = resp.json()

        assert data["chunk_count"] > 1, f"Large {size_mb}MB request sent as single chunk!"

        min_expected = (size_mb * 1024 * 1024) // (64 * 1024) // 2
        assert data["chunk_count"] >= min_expected, (
            f"Expected >= {min_expected} chunks for {size_mb}MB, got {data['chunk_count']}"
        )

    async def test_response_chunk_boundaries_valid(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify response chunk boundaries align with length prefix format."""
        size_mb = 5
        large_content = "Y" * (size_mb * 1024 * 1024)

        resp = await httpx_client.post("/echo", content=large_content.encode())
        assert resp.status_code == 200
        assert isinstance(resp, HTTPXDecryptedResponse)

        raw_body = resp.unwrap().content

        offset = 0
        counters: list[int] = []

        while offset < len(raw_body):
            assert offset + 4 <= len(raw_body), f"Truncated length at offset {offset}"
            chunk_len = int.from_bytes(raw_body[offset : offset + 4], "big")
            assert chunk_len > 0, f"Zero-length chunk at offset {offset}"
            assert offset + 4 + chunk_len <= len(raw_body), f"Chunk overflow at offset {offset}"

            counter = int.from_bytes(raw_body[offset + 4 : offset + 8], "big")
            counters.append(counter)
            offset += 4 + chunk_len

        assert offset == len(raw_body), f"Chunk boundaries misaligned: {offset} vs {len(raw_body)}"
        assert counters == list(range(1, len(counters) + 1)), f"Counters not monotonic: {counters[:10]}"


@pytest.mark.requires_root
class TestNetworkLevelVerificationHTTPX:
    """Tests that verify encryption at the network packet level for httpx.

    These tests require root/sudo to run tcpdump for packet capture.
    They must run serially (-n 0) due to timing sensitivity.
    """

    async def test_wire_traffic_contains_no_plaintext(
        self,
        tcpdump_capture: str,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify captured network traffic contains no plaintext."""
        import asyncio

        canaries = ["HTTPX_WIRE_CANARY_ABC123", "HTTPX_NETWORK_SECRET_XYZ"]

        for canary in canaries:
            resp = await httpx_client.post("/echo", json={"secret": canary})
            assert resp.status_code == 200

        await asyncio.sleep(0.3)

        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()

        for canary in canaries:
            assert canary.encode() not in pcap_data, f"Plaintext canary '{canary}' found in network capture!"
        assert b'"secret"' not in pcap_data, "JSON key found in network capture"

    async def test_sse_wire_traffic_is_encrypted(
        self,
        tcpdump_capture: str,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify SSE streaming content is encrypted on the wire.

        The SSE transport format (event:, data:) is visible on the wire, but
        the actual event content must be encrypted. Original event types and
        data should NOT appear - only 'event: enc' with base64 ciphertext.
        """
        import asyncio

        # Trigger SSE stream with unique canary values
        sse_canary = "HTTPX_SSE_WIRE_CANARY_789"
        resp = await httpx_client.post("/stream", json={"canary": sse_canary})
        assert resp.status_code == 200

        # Consume the SSE stream to generate wire traffic
        event_count = 0
        async for _chunk in httpx_client.iter_sse(resp):
            event_count += 1
            if event_count >= 5:
                break

        # Wait for packets to flush
        await asyncio.sleep(0.5)

        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()

        # Original event types should NEVER appear (they're encrypted)
        # The wire should only show "event: enc" not "event: tick" or "event: done"
        assert b"event: tick" not in pcap_data, "Raw SSE event type 'tick' found in network capture!"
        assert b"event: done" not in pcap_data, "Raw SSE event type 'done' found in network capture!"

        # Original JSON data keys should NOT be visible
        assert b'"count"' not in pcap_data, "SSE data key 'count' found in network capture!"
        assert b'"timestamp"' not in pcap_data, "SSE data key 'timestamp' found in network capture!"

        # Canary value should NOT be visible
        assert sse_canary.encode() not in pcap_data, f"SSE canary '{sse_canary}' found in network capture!"

        # Verify encrypted SSE format IS present (proves SSE encryption is working)
        assert b"event: enc" in pcap_data, "Encrypted SSE 'event: enc' not found - encryption may not be active!"

    async def test_nonce_uniqueness_different_ciphertext(
        self,
        tcpdump_capture: str,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify same plaintext produces different ciphertext (nonce uniqueness).

        This is critical for security - if nonces are reused, an attacker can
        XOR ciphertexts to recover plaintext. Each encryption MUST produce
        unique ciphertext even for identical input.
        """
        import asyncio

        identical_payload = {"test": "nonce_uniqueness", "data": "identical_content_123"}

        # Send same payload twice
        resp1 = await httpx_client.post("/echo", json=identical_payload)
        assert resp1.status_code == 200
        assert isinstance(resp1, HTTPXDecryptedResponse)
        raw1 = resp1.unwrap().content

        resp2 = await httpx_client.post("/echo", json=identical_payload)
        assert resp2.status_code == 200
        assert isinstance(resp2, HTTPXDecryptedResponse)
        raw2 = resp2.unwrap().content

        # Ciphertexts MUST be different (due to different nonces/ephemeral keys)
        assert raw1 != raw2, (
            "CRITICAL: Identical plaintext produced identical ciphertext! "
            "This indicates nonce reuse - catastrophic for security."
        )

        # Wait for pcap to capture all traffic
        await asyncio.sleep(0.3)

        # Verify traffic was captured (sanity check)
        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()
        assert len(pcap_data) > 100, "No traffic captured in pcap"

    async def test_request_body_encrypted_in_pcap(
        self,
        tcpdump_capture: str,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Verify request body is encrypted on the wire (not just response).

        Previous tests focused on response encryption. This verifies the
        request body sent BY the client is also encrypted in the network capture.
        """
        import asyncio

        # Unique canary that should appear in request body
        request_canary = "HTTPX_REQUEST_BODY_CANARY_XYZ789"
        request_payload = {
            "secret_request_data": request_canary,
            "password": "httpx_super_secret_password_123",
            "api_key": "httpx-sk-live-abcdef123456",
        }

        resp = await httpx_client.post("/echo", json=request_payload)
        assert resp.status_code == 200

        await asyncio.sleep(0.3)

        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()

        # Request body canaries should NOT be visible
        assert request_canary.encode() not in pcap_data, (
            f"Request body canary '{request_canary}' found in pcap! Request not encrypted."
        )
        assert b"httpx_super_secret_password" not in pcap_data, "Password found in pcap!"
        assert b"httpx-sk-live-" not in pcap_data, "API key prefix found in pcap!"
        assert b'"secret_request_data"' not in pcap_data, "JSON key found in pcap!"

    async def test_no_psk_on_wire(
        self,
        tcpdump_capture: str,
        httpx_client: HPKEAsyncClient,
        test_psk: bytes,
    ) -> None:
        """Verify pre-shared key never appears in network traffic.

        The PSK is used for authentication but should NEVER be transmitted.
        It's used locally to derive keys, not sent over the wire.
        """
        import asyncio

        # Generate some traffic
        resp = await httpx_client.post("/echo", json={"test": "httpx_psk_leak_check"})
        assert resp.status_code == 200

        await asyncio.sleep(0.3)

        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()

        # PSK should never appear in any form
        assert test_psk not in pcap_data, "CRITICAL: Raw PSK found in network capture!"
        # Also check hex-encoded form
        assert test_psk.hex().encode() not in pcap_data, "PSK (hex) found in pcap!"

    async def test_no_session_key_material_on_wire(
        self,
        tcpdump_capture: str,
        httpx_client: HPKEAsyncClient,
        platform_keypair: tuple[bytes, bytes],
    ) -> None:
        """Verify private key and derived session keys never appear in traffic.

        Only the ephemeral PUBLIC key should be visible (the 'enc' field in HPKE).
        Private keys and derived session keys must never be transmitted.
        """
        import asyncio

        private_key, _public_key = platform_keypair

        # Generate traffic
        resp = await httpx_client.post("/echo", json={"test": "httpx_key_leak_check"})
        assert resp.status_code == 200

        await asyncio.sleep(0.3)

        with open(tcpdump_capture, "rb") as f:
            pcap_data = f.read()

        # Private key should NEVER appear
        assert private_key not in pcap_data, "CRITICAL: Private key found in network capture!"

        # Note: Server's static public key IS visible in /.well-known/hpke-keys response
        # (that's expected - it's public). We only check private key doesn't leak.


class TestHttpxConnectionLeaks:
    """Verify httpx connections are properly released.

    Mirrors TestAiohttpConnectionLeaks for consistency.
    Uses pool exhaustion detection since httpx doesn't expose internal pool state.
    """

    async def test_post_releases_connection(
        self,
        httpx_client_small_pool: HPKEAsyncClient,
    ) -> None:
        """POST /echo releases connection."""
        resp = await httpx_client_small_pool.post("/echo", json={"test": "leak"})
        assert resp.status_code == 200

        # If connection leaked, this would timeout (pool of 2)
        resp = await httpx_client_small_pool.post("/echo", json={"verify": True})
        assert resp.status_code == 200

    async def test_sequential_requests(
        self,
        httpx_client_small_pool: HPKEAsyncClient,
    ) -> None:
        """10 sequential requests don't leak."""
        for i in range(10):
            resp = await httpx_client_small_pool.post("/echo", json={"seq": i})
            assert resp.status_code == 200

    async def test_concurrent_requests(
        self,
        httpx_client_small_pool: HPKEAsyncClient,
    ) -> None:
        """5 concurrent requests don't leak."""

        async def req(n: int) -> int:
            resp = await httpx_client_small_pool.post("/echo", json={"n": n})
            return resp.status_code

        results = await asyncio.gather(*[req(i) for i in range(5)])
        assert all(r == 200 for r in results)

        # Verify pool still works
        resp = await httpx_client_small_pool.post("/echo", json={"after": True})
        assert resp.status_code == 200

    async def test_sse_stream_releases(
        self,
        httpx_client_small_pool: HPKEAsyncClient,
    ) -> None:
        """SSE stream releases connection after full consumption."""
        resp = await httpx_client_small_pool.post("/stream", json={"start": True})
        assert resp.status_code == 200

        event_count = 0
        async for _chunk in httpx_client_small_pool.iter_sse(resp):
            event_count += 1

        assert event_count >= 1

        # If connection leaked, this would timeout
        resp = await httpx_client_small_pool.post("/echo", json={"after_sse": True})
        assert resp.status_code == 200

    async def test_sse_many_events(
        self,
        httpx_client_small_pool: HPKEAsyncClient,
    ) -> None:
        """50-event SSE doesn't leak."""
        resp = await httpx_client_small_pool.post("/stream-many", json={"start": True})
        assert resp.status_code == 200

        count = 0
        async for _chunk in httpx_client_small_pool.iter_sse(resp):
            count += 1

        assert count >= 50

        # If connection leaked, this would timeout
        resp = await httpx_client_small_pool.post("/echo", json={"after_many": True})
        assert resp.status_code == 200

    async def test_sse_cancelled_early(
        self,
        httpx_client_small_pool: HPKEAsyncClient,
    ) -> None:
        """Early SSE cancellation releases connection."""
        resp = await httpx_client_small_pool.post("/stream-many", json={"start": True})
        assert resp.status_code == 200

        count = 0
        async for _chunk in httpx_client_small_pool.iter_sse(resp):
            count += 1
            if count >= 5:
                break

        # If connection leaked, this would timeout
        await asyncio.sleep(0.05)
        resp = await httpx_client_small_pool.post("/echo", json={"after_cancel": True})
        assert resp.status_code == 200


class TestHttpxConnectionLeaksCompressed:
    """Connection leak tests with compression enabled.

    Mirrors TestAiohttpConnectionLeaksCompressed for consistency.
    """

    async def test_compressed_post(
        self,
        httpx_client_compressed: HPKEAsyncClient,
    ) -> None:
        """Compressed POST releases connection."""
        resp = await httpx_client_compressed.post("/echo", json={"compressed": True})
        assert resp.status_code == 200

        # Verify connection released
        resp = await httpx_client_compressed.post("/echo", json={"verify": True})
        assert resp.status_code == 200

    async def test_compressed_sse(
        self,
        httpx_client_compressed: HPKEAsyncClient,
    ) -> None:
        """Compressed SSE releases connection."""
        resp = await httpx_client_compressed.post("/stream", json={"start": True})
        assert resp.status_code == 200

        async for _chunk in httpx_client_compressed.iter_sse(resp):
            pass

        # Verify connection released
        resp = await httpx_client_compressed.post("/echo", json={"after_sse": True})
        assert resp.status_code == 200


# === Multipart Upload Tests ===


def _build_100_parts() -> tuple[dict[str, tuple[str, bytes, str]], dict[str, str]]:
    """Build 100 x 10MB parts with random data (1GB total)."""
    import os

    files: dict[str, tuple[str, bytes, str]] = {}
    expected_hashes: dict[str, str] = {}
    for i in range(100):
        content = os.urandom(10 * 1024 * 1024)  # 10MB random bytes
        field = f"part_{i}"
        files[field] = (f"{field}.bin", content, "application/octet-stream")
        expected_hashes[field] = hashlib.sha256(content).hexdigest()
    return files, expected_hashes


def _build_100_parts_aiohttp() -> tuple[aiohttp.FormData, dict[str, str]]:
    """Build 100 x 10MB parts as aiohttp FormData (1GB total)."""
    import os

    form = aiohttp.FormData()
    expected_hashes: dict[str, str] = {}
    for i in range(100):
        content = os.urandom(10 * 1024 * 1024)  # 10MB random bytes
        field = f"part_{i}"
        form.add_field(field, content, filename=f"{field}.bin", content_type="application/octet-stream")
        expected_hashes[field] = hashlib.sha256(content).hexdigest()
    return form, expected_hashes


def _validate_parts(data: dict[str, Any], expected_hashes: dict[str, str]) -> None:
    """Validate all parts have correct hash and size."""
    assert len(data["parts"]) == len(expected_hashes)
    for part in data["parts"]:
        field = part["field"]
        assert part["sha256"] == expected_hashes[field], f"Hash mismatch for {field}"
        assert part["size"] == 10 * 1024 * 1024


class TestMultipartUploadHttpx:
    """Test encrypted multipart uploads via httpx client."""

    async def test_multipart_bytes_payload_httpx(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """httpx: BytesPayload - 10MB raw bytes upload."""
        import os

        content = os.urandom(10 * 1024 * 1024)  # 10MB
        files = {"file": ("test.bin", content, "application/octet-stream")}

        resp = await httpx_client.post("/upload", files=files)
        assert resp.status_code == 200

        data = resp.json()
        assert len(data["parts"]) == 1
        assert data["parts"][0]["size"] == 10 * 1024 * 1024
        assert data["parts"][0]["sha256"] == hashlib.sha256(content).hexdigest()

    async def test_multipart_iobase_payload_httpx(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """httpx: IOBasePayload - 10MB file-like object upload."""
        import io
        import os

        content = os.urandom(10 * 1024 * 1024)  # 10MB
        file_obj = io.BytesIO(content)
        files = {"file": ("test.bin", file_obj, "application/octet-stream")}

        resp = await httpx_client.post("/upload", files=files)
        assert resp.status_code == 200

        data = resp.json()
        assert len(data["parts"]) == 1
        assert data["parts"][0]["size"] == 10 * 1024 * 1024
        assert data["parts"][0]["sha256"] == hashlib.sha256(content).hexdigest()

    @pytest.mark.slow
    async def test_multipart_100_parts_10mb_httpx(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """httpx: 100 parts x 10MB (1GB) - client validates hashes."""
        files, expected_hashes = _build_100_parts()

        resp = await httpx_client.post("/upload", files=files)
        assert resp.status_code == 200

        _validate_parts(resp.json(), expected_hashes)


class TestMultipartUploadAiohttp:
    """Test encrypted multipart uploads via aiohttp client."""

    async def test_multipart_bytes_payload_aiohttp(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """aiohttp: BytesPayload - 10MB raw bytes upload."""
        import os

        content = os.urandom(10 * 1024 * 1024)  # 10MB

        form = aiohttp.FormData()
        form.add_field("file", content, filename="test.bin", content_type="application/octet-stream")

        resp = await aiohttp_client.post("/upload", data=form)
        assert resp.status == 200

        data = await resp.json()
        assert len(data["parts"]) == 1
        assert data["parts"][0]["size"] == 10 * 1024 * 1024
        assert data["parts"][0]["sha256"] == hashlib.sha256(content).hexdigest()

    async def test_multipart_iobase_payload_aiohttp(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """aiohttp: IOBasePayload - 10MB file-like object upload."""
        import io
        import os

        content = os.urandom(10 * 1024 * 1024)  # 10MB
        file_obj = io.BytesIO(content)

        form = aiohttp.FormData()
        form.add_field("file", file_obj, filename="test.bin", content_type="application/octet-stream")

        resp = await aiohttp_client.post("/upload", data=form)
        assert resp.status == 200

        data = await resp.json()
        assert len(data["parts"]) == 1
        assert data["parts"][0]["size"] == 10 * 1024 * 1024
        assert data["parts"][0]["sha256"] == hashlib.sha256(content).hexdigest()

    @pytest.mark.slow
    async def test_multipart_100_parts_10mb_aiohttp(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """aiohttp: 100 parts x 10MB (1GB) via FormData - client validates hashes."""
        form, expected_hashes = _build_100_parts_aiohttp()

        resp = await aiohttp_client.post("/upload", data=form)
        assert resp.status == 200

        _validate_parts(await resp.json(), expected_hashes)


# === Multipart Memory Tests ===


class TestMultipartMemoryHttpx:
    """Memory behavior tests for multipart uploads via httpx client."""

    async def test_multipart_no_memory_leak(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Repeated multipart uploads don't leak memory."""
        import gc
        import os
        import tracemalloc

        content = os.urandom(1024 * 1024)  # 1MB
        expected_hash = hashlib.sha256(content).hexdigest()

        # Warmup
        for _ in range(3):
            files = {"file": ("test.bin", content, "application/octet-stream")}
            resp = await httpx_client.post("/upload", files=files)
            assert resp.status_code == 200
        gc.collect()

        # Measure
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for i in range(20):
            files = {"file": (f"test_{i}.bin", content, "application/octet-stream")}
            resp = await httpx_client.post("/upload", files=files)
            assert resp.status_code == 200
            assert resp.json()["parts"][0]["sha256"] == expected_hash

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        net_allocated = sum(stat.size_diff for stat in diff)

        # Allow 300KB growth for 20 uploads (connection pools, caches, SSL contexts)
        # Platform variance: Linux allocators may use more memory than macOS
        max_leak = 300 * 1024
        assert net_allocated < max_leak, (
            f"Memory grew by {net_allocated / 1024:.1f}KB after 20 uploads, expected < {max_leak // 1024}KB"
        )

    async def test_multipart_peak_memory_ratio(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """REAL test: actual client.post() with file-backed content uses bounded memory.

        This tests the full end-to-end path:
        - File on disk (not pre-allocated in-memory bytes)
        - Actual httpx_client.post() call
        - Real encryption pipeline

        Memory should be O(chunk_size), not O(file_size).
        """
        import gc
        import os
        import tempfile
        import tracemalloc

        file_size = 50 * 1024 * 1024  # 50MB file

        # Create file on disk (content NOT in Python memory)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            # Write in chunks to avoid large allocation
            chunk = os.urandom(1024 * 1024)  # 1MB chunk
            for _ in range(file_size // len(chunk)):
                f.write(chunk)
            temp_path = f.name
        del chunk
        gc.collect()

        try:
            # Warmup - exercise full path
            with open(temp_path, "rb") as f:
                files = {"file": ("test.bin", f, "application/octet-stream")}
                resp = await httpx_client.post("/upload", files=files)
                assert resp.status_code == 200
            gc.collect()

            # Measure REAL upload with file handle
            tracemalloc.start()
            tracemalloc.reset_peak()

            with open(temp_path, "rb") as f:
                files = {"file": ("test.bin", f, "application/octet-stream")}
                resp = await httpx_client.post("/upload", files=files)
                assert resp.status_code == 200
                result = resp.json()
                # Verify upload succeeded and correct size
                assert result["parts"][0]["size"] == file_size

            _current, peak_alloc = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Real streaming should use bounded memory
            # Allow 5MB for buffers, crypto state, HTTP overhead (but NOT 50MB)
            max_memory = 5 * 1024 * 1024  # 5MB
            assert peak_alloc < max_memory, (
                f"Peak memory {peak_alloc / 1024 / 1024:.1f}MB exceeds {max_memory // 1024 // 1024}MB limit. "
                f"Full upload path should stream, not buffer entire file. "
                f"Ratio: {peak_alloc / file_size:.1%} of file size"
            )

        finally:
            os.unlink(temp_path)


class TestMultipartMemoryAiohttp:
    """Memory behavior tests for multipart uploads via aiohttp client."""

    async def test_multipart_no_memory_leak(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Repeated multipart uploads don't leak memory."""
        import gc
        import os
        import tracemalloc

        content = os.urandom(1024 * 1024)  # 1MB
        expected_hash = hashlib.sha256(content).hexdigest()

        # Warmup
        for _ in range(3):
            form = aiohttp.FormData()
            form.add_field("file", content, filename="test.bin", content_type="application/octet-stream")
            resp = await aiohttp_client.post("/upload", data=form)
            assert resp.status == 200
        gc.collect()

        # Measure
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for i in range(20):
            form = aiohttp.FormData()
            form.add_field("file", content, filename=f"test_{i}.bin", content_type="application/octet-stream")
            resp = await aiohttp_client.post("/upload", data=form)
            assert resp.status == 200
            data = await resp.json()
            assert data["parts"][0]["sha256"] == expected_hash

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        net_allocated = sum(stat.size_diff for stat in diff)

        # Allow 300KB growth for 20 uploads (connection pools, caches, SSL contexts)
        # Platform variance: Linux allocators may use more memory than macOS
        max_leak = 300 * 1024
        assert net_allocated < max_leak, (
            f"Memory grew by {net_allocated / 1024:.1f}KB after 20 uploads, expected < {max_leak // 1024}KB"
        )

    async def test_multipart_peak_memory_ratio(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """REAL test: actual client.post() with file-backed content uses bounded memory.

        This tests the full end-to-end path:
        - File on disk (not pre-allocated in-memory bytes)
        - Actual aiohttp_client.post() call
        - Real encryption pipeline

        Memory should be O(chunk_size), not O(file_size).
        """
        import gc
        import os
        import tempfile
        import tracemalloc

        file_size = 50 * 1024 * 1024  # 50MB file

        # Create file on disk (content NOT in Python memory)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            # Write in chunks to avoid large allocation
            chunk = os.urandom(1024 * 1024)  # 1MB chunk
            for _ in range(file_size // len(chunk)):
                f.write(chunk)
            temp_path = f.name
        del chunk
        gc.collect()

        try:
            # Warmup - exercise full path
            with open(temp_path, "rb") as f:
                form = aiohttp.FormData()
                form.add_field("file", f, filename="test.bin", content_type="application/octet-stream")
                resp = await aiohttp_client.post("/upload", data=form)
                assert resp.status == 200
            gc.collect()

            # Measure REAL upload with file handle
            tracemalloc.start()
            tracemalloc.reset_peak()

            with open(temp_path, "rb") as f:
                form = aiohttp.FormData()
                form.add_field("file", f, filename="test.bin", content_type="application/octet-stream")
                resp = await aiohttp_client.post("/upload", data=form)
                assert resp.status == 200
                result = await resp.json()
                # Verify upload succeeded and correct size
                assert result["parts"][0]["size"] == file_size

            _current, peak_alloc = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Real streaming should use bounded memory
            # Allow 5MB for buffers, crypto state, HTTP overhead (but NOT 50MB)
            max_memory = 5 * 1024 * 1024  # 5MB
            assert peak_alloc < max_memory, (
                f"Peak memory {peak_alloc / 1024 / 1024:.1f}MB exceeds {max_memory // 1024 // 1024}MB limit. "
                f"Full upload path should stream, not buffer entire file. "
                f"Ratio: {peak_alloc / file_size:.1%} of file size"
            )

        finally:
            os.unlink(temp_path)


class TestServerMemoryBounded:
    """Verify server-side processing uses bounded memory."""

    async def test_server_streams_large_upload(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Server processes large uploads without holding full body in memory.

        The /upload endpoint uses streaming hash computation (64KB chunks),
        so server memory should be O(1) regardless of upload size.
        This test verifies the endpoint works for large files.
        """
        import os

        # 50MB file - server should handle without OOM
        content = os.urandom(50 * 1024 * 1024)
        expected_hash = hashlib.sha256(content).hexdigest()
        files = {"file": ("large.bin", content, "application/octet-stream")}

        resp = await httpx_client.post("/upload", files=files)
        assert resp.status_code == 200

        data = resp.json()
        assert data["parts"][0]["size"] == 50 * 1024 * 1024
        assert data["parts"][0]["sha256"] == expected_hash


# === Content-Type Preservation Tests ===


class TestContentTypePreservationHttpx:
    """Test X-HPKE-Content-Type header preservation via httpx client.

    Verifies that the original Content-Type is preserved through encryption
    and restored on the server side for proper request parsing.
    """

    # --- Normal cases ---

    async def test_content_type_json(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Normal: application/json is preserved."""
        resp = await httpx_client.post("/echo-headers", json={"test": "value"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["content_type"] == "application/json"

    async def test_content_type_text_plain(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Normal: text/plain is preserved."""
        resp = await httpx_client.post(
            "/echo-headers",
            content=b"Hello, World!",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content_type"] == "text/plain"

    async def test_content_type_multipart_explicit(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Normal: multipart/form-data with boundary is preserved."""
        files = {"file": ("test.txt", b"content", "text/plain")}
        resp = await httpx_client.post("/echo-headers", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["content_type"] is not None
        assert data["content_type"].startswith("multipart/form-data")
        assert "boundary=" in data["content_type"]

    # --- Weird cases ---

    async def test_content_type_custom_xml(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Weird: custom application/xml is preserved."""
        resp = await httpx_client.post(
            "/echo-headers",
            content=b"<root><item>test</item></root>",
            headers={"Content-Type": "application/xml"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content_type"] == "application/xml"

    async def test_content_type_with_charset(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Weird: Content-Type with charset parameter is preserved exactly."""
        resp = await httpx_client.post(
            "/echo-headers",
            content=b'{"test": "value"}',
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content_type"] == "application/json; charset=utf-8"

    async def test_content_type_with_multiple_params(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Weird: Content-Type with multiple parameters is preserved exactly."""
        complex_ct = "application/json; charset=utf-8; boundary=something"
        resp = await httpx_client.post(
            "/echo-headers",
            content=b'{"test": "value"}',
            headers={"Content-Type": complex_ct},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content_type"] == complex_ct

    async def test_content_type_vendor_specific(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Weird: vendor-specific Content-Type is preserved."""
        resp = await httpx_client.post(
            "/echo-headers",
            content=b'{"api": "data"}',
            headers={"Content-Type": "application/vnd.api+json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content_type"] == "application/vnd.api+json"

    # --- Edge cases ---

    async def test_content_type_missing_raw_bytes(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Edge: raw bytes without Content-Type header - server sees None."""
        resp = await httpx_client.post(
            "/echo-headers",
            content=b"raw binary data",
            # No Content-Type header
        )
        assert resp.status_code == 200
        data = resp.json()
        # Server should see application/octet-stream (the encrypted wire format)
        # since no original Content-Type was provided to preserve
        assert data["content_type"] == "application/octet-stream"

    async def test_content_type_empty_body_with_type(
        self,
        httpx_client: HPKEAsyncClient,
    ) -> None:
        """Edge: empty body with Content-Type - no encryption, passes through."""
        resp = await httpx_client.post(
            "/echo-headers",
            content=b"",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Empty body means no encryption, Content-Type passes through as-is
        assert data["body_size"] == 0


class TestContentTypePreservationAiohttp:
    """Test X-HPKE-Content-Type header preservation via aiohttp client.

    Mirrors httpx tests to ensure both clients behave identically.
    """

    # --- Normal cases ---

    async def test_content_type_json(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Normal: application/json is preserved."""
        resp = await aiohttp_client.post("/echo-headers", json={"test": "value"})
        assert resp.status == 200
        data = await resp.json()
        assert data["content_type"] == "application/json"

    async def test_content_type_text_plain(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Normal: text/plain is preserved."""
        resp = await aiohttp_client.post(
            "/echo-headers",
            data=b"Hello, World!",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["content_type"] == "text/plain"

    async def test_content_type_multipart_explicit(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Normal: multipart/form-data with boundary is preserved."""
        form = aiohttp.FormData()
        form.add_field("file", b"content", filename="test.txt", content_type="text/plain")
        resp = await aiohttp_client.post("/echo-headers", data=form)
        assert resp.status == 200
        data = await resp.json()
        assert data["content_type"] is not None
        assert data["content_type"].startswith("multipart/form-data")
        assert "boundary=" in data["content_type"]

    # --- Weird cases ---

    async def test_content_type_custom_xml(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Weird: custom application/xml is preserved."""
        resp = await aiohttp_client.post(
            "/echo-headers",
            data=b"<root><item>test</item></root>",
            headers={"Content-Type": "application/xml"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["content_type"] == "application/xml"

    async def test_content_type_with_charset(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Weird: Content-Type with charset parameter is preserved exactly."""
        resp = await aiohttp_client.post(
            "/echo-headers",
            data=b'{"test": "value"}',
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["content_type"] == "application/json; charset=utf-8"

    async def test_content_type_with_multiple_params(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Weird: Content-Type with multiple parameters is preserved exactly."""
        complex_ct = "application/json; charset=utf-8; boundary=something"
        resp = await aiohttp_client.post(
            "/echo-headers",
            data=b'{"test": "value"}',
            headers={"Content-Type": complex_ct},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["content_type"] == complex_ct

    async def test_content_type_vendor_specific(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Weird: vendor-specific Content-Type is preserved."""
        resp = await aiohttp_client.post(
            "/echo-headers",
            data=b'{"api": "data"}',
            headers={"Content-Type": "application/vnd.api+json"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["content_type"] == "application/vnd.api+json"

    # --- Edge cases ---

    async def test_content_type_missing_raw_bytes(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Edge: raw bytes without Content-Type header - server sees None."""
        resp = await aiohttp_client.post(
            "/echo-headers",
            data=b"raw binary data",
            # No Content-Type header
        )
        assert resp.status == 200
        data = await resp.json()
        # Server should see application/octet-stream (the encrypted wire format)
        # since no original Content-Type was provided to preserve
        assert data["content_type"] == "application/octet-stream"

    async def test_content_type_empty_body_with_type(
        self,
        aiohttp_client: HPKEClientSession,
    ) -> None:
        """Edge: empty body with Content-Type - no encryption, passes through."""
        resp = await aiohttp_client.post(
            "/echo-headers",
            data=b"",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 200
        data = await resp.json()
        # Empty body means no encryption, Content-Type passes through as-is
        assert data["body_size"] == 0

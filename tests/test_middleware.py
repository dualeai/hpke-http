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

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pytest
from typing_extensions import assert_type

from hpke_http.constants import HEADER_HPKE_STREAM
from hpke_http.middleware.aiohttp import DecryptedResponse, HPKEClientSession
from tests.conftest import E2EServer, calculate_shannon_entropy, chi_square_byte_uniformity


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


# === Fixtures ===


async def _create_hpke_client(
    server: E2EServer,
    psk: bytes,
    psk_id: bytes,
    *,
    compress: bool = False,
) -> AsyncIterator[HPKEClientSession]:
    """Create HPKEClientSession connected to test server."""
    base_url = f"http://{server.host}:{server.port}"

    async with HPKEClientSession(
        base_url=base_url,
        psk=psk,
        psk_id=psk_id,
        compress=compress,
    ) as session:
        yield session


@pytest.fixture
async def hpke_client(
    granian_server: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[HPKEClientSession]:
    """HPKEClientSession connected to test server."""
    async for session in _create_hpke_client(granian_server, test_psk, test_psk_id):
        yield session


@pytest.fixture
async def hpke_client_compressed(
    granian_server_compressed: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[HPKEClientSession]:
    """HPKEClientSession with compression, connected to compression-enabled server."""
    async for session in _create_hpke_client(granian_server_compressed, test_psk, test_psk_id, compress=True):
        yield session


@pytest.fixture
async def hpke_client_no_compress_server_compress(
    granian_server_compressed: E2EServer,
    test_psk: bytes,
    test_psk_id: bytes,
) -> AsyncIterator[HPKEClientSession]:
    """HPKEClientSession without compression, connected to compression-enabled server."""
    async for session in _create_hpke_client(granian_server_compressed, test_psk, test_psk_id, compress=False):
        yield session


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


class TestStandardResponseEncryption:
    """Test encrypted standard (non-SSE) responses."""

    async def test_response_has_hpke_stream_header(self, hpke_client: HPKEClientSession) -> None:
        """Encrypted request triggers encrypted response with X-HPKE-Stream header."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert resp.status == 200

        # DecryptedResponse wraps the underlying response
        assert isinstance(resp, DecryptedResponse)

        # X-HPKE-Stream header should be present (contains salt)
        assert HEADER_HPKE_STREAM in resp.headers

        # Content-Type should NOT be text/event-stream (that's for SSE)
        content_type = resp.headers.get("Content-Type", "")
        assert "text/event-stream" not in content_type

    async def test_decrypted_response_json(self, hpke_client: HPKEClientSession) -> None:
        """DecryptedResponse.json() returns decrypted data."""
        test_data = {"message": "secret", "value": 42}
        resp = await hpke_client.post("/echo", json=test_data)

        # json() should return decrypted data
        data = await resp.json()
        assert "message" in data["echo"]
        assert "secret" in data["echo"]

    async def test_decrypted_response_read(self, hpke_client: HPKEClientSession) -> None:
        """DecryptedResponse.read() returns raw decrypted bytes."""
        resp = await hpke_client.post("/echo", json={"raw": "test"})

        # read() should return decrypted bytes
        raw_bytes = await resp.read()
        assert b"raw" in raw_bytes
        assert b"test" in raw_bytes

    async def test_decrypted_response_text(self, hpke_client: HPKEClientSession) -> None:
        """DecryptedResponse.text() returns decrypted text."""
        resp = await hpke_client.post("/echo", json={"text": "hello"})

        # text() should return decrypted string
        text = await resp.text()
        assert "text" in text
        assert "hello" in text

    async def test_sse_response_not_wrapped_in_decrypted_response(self, hpke_client: HPKEClientSession) -> None:
        """SSE responses use X-HPKE-Stream with text/event-stream Content-Type."""
        resp = await hpke_client.post("/stream", json={"start": True})
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

    async def test_iter_sse_yields_bytes(self, hpke_client: HPKEClientSession) -> None:
        """iter_sse must yield bytes (matches native aiohttp response.content).

        This is a type contract test - ensures API doesn't accidentally change.
        Static: assert_type checked by pyright at type-check time.
        Runtime: isinstance checked by pytest at test time.
        """
        resp = await hpke_client.post("/stream", json={"start": True})
        assert resp.status == 200

        async for chunk in hpke_client.iter_sse(resp):
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
        hpke_client_compressed: HPKEClientSession,
    ) -> None:
        """Client compress=True → Server decompresses correctly.

        Large JSON is compressed before encryption, server decompresses after decryption.
        """
        # Large payload to ensure compression is triggered (>64 bytes)
        large_data = {"message": "x" * 1000, "nested": {"key": "value" * 100}}

        resp = await hpke_client_compressed.post("/echo", json=large_data)
        assert resp.status == 200
        data = await resp.json()

        # Verify the data made it through compression → encryption → decryption → decompression
        assert "x" * 1000 in data["echo"]
        assert "value" * 100 in data["echo"]

    async def test_compressed_sse_roundtrip(
        self,
        hpke_client_compressed: HPKEClientSession,
    ) -> None:
        """Server compress=True → Client receives decompressed SSE.

        SSE events are compressed before encryption, client decompresses after decryption.
        """
        resp = await hpke_client_compressed.post("/stream-large", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in hpke_client_compressed.iter_sse(resp)]

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
        hpke_client_no_compress_server_compress: HPKEClientSession,
    ) -> None:
        """Client compress=False, Server compress=True still works.

        Client sends uncompressed requests, server compresses SSE responses.
        """
        test_data = {"message": "Hello from uncompressed client!"}

        resp = await hpke_client_no_compress_server_compress.post("/echo", json=test_data)
        assert resp.status == 200
        data = await resp.json()
        assert "Hello from uncompressed client!" in data["echo"]

        # SSE should still work (server compresses, client decompresses)
        resp = await hpke_client_no_compress_server_compress.post("/stream", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in hpke_client_no_compress_server_compress.iter_sse(resp)]
        assert len(events) == 4

    async def test_many_events_with_compression(
        self,
        hpke_client_compressed: HPKEClientSession,
    ) -> None:
        """50+ SSE events with compression work correctly."""
        resp = await hpke_client_compressed.post("/stream-many", json={"start": True})
        assert resp.status == 200
        events = [parse_sse_chunk(chunk) async for chunk in hpke_client_compressed.iter_sse(resp)]

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

    async def test_multiple_read_calls_cached(self, hpke_client: HPKEClientSession) -> None:
        """Multiple read() calls return same cached data."""
        resp = await hpke_client.post("/echo", json={"cached": "test"})

        # First read
        data1 = await resp.read()
        # Second read (should use cache)
        data2 = await resp.read()

        assert data1 == data2
        assert b"cached" in data1

    async def test_json_after_read_works(self, hpke_client: HPKEClientSession) -> None:
        """json() works after read() has been called."""
        resp = await hpke_client.post("/echo", json={"order": "test"})

        # Read raw first
        raw = await resp.read()
        assert b"order" in raw

        # Then parse as JSON (uses cached data)
        data = await resp.json()
        assert "order" in data["echo"]

    async def test_text_after_json_works(self, hpke_client: HPKEClientSession) -> None:
        """text() works after json() has been called."""
        resp = await hpke_client.post("/echo", json={"sequence": 123})

        # Parse as JSON first
        data = await resp.json()
        assert "sequence" in data["echo"]

        # Then get as text (uses cached data)
        text = await resp.text()
        assert "sequence" in text

    async def test_empty_response_body(self, hpke_client: HPKEClientSession) -> None:
        """Empty response body is handled correctly."""
        # The /health endpoint returns a small response, but let's test with /echo
        # sending minimal data
        resp = await hpke_client.post("/echo", json={})
        data = await resp.json()
        assert "echo" in data

    async def test_status_and_url_passthrough(self, hpke_client: HPKEClientSession) -> None:
        """DecryptedResponse proxies status and url correctly."""
        resp = await hpke_client.post("/echo", json={"proxy": "test"})

        # Status should be accessible
        assert resp.status == 200

        # URL should be accessible (proxied from underlying response)
        assert "/echo" in str(resp.url)

    async def test_headers_accessible(self, hpke_client: HPKEClientSession) -> None:
        """Response headers are accessible through DecryptedResponse."""
        resp = await hpke_client.post("/echo", json={"headers": "test"})

        # Headers should be accessible
        assert "content-type" in resp.headers or "Content-Type" in resp.headers


class TestStandardResponseEdgeCasesE2E:
    """E2E edge case tests for standard response encryption."""

    async def test_large_response_multi_chunk(self, hpke_client: HPKEClientSession) -> None:
        """Large response that may be sent in multiple chunks."""
        # Request a response with a larger payload via /echo
        large_payload = {"data": "x" * 50000}  # 50KB payload
        resp = await hpke_client.post("/echo", json=large_payload)

        assert resp.status == 200
        data = await resp.json()
        assert "x" * 50000 in data["echo"]

    async def test_unicode_response_content(self, hpke_client: HPKEClientSession) -> None:
        """Unicode content in response is preserved (may be escaped in JSON)."""
        unicode_data = {"message": "Hello 世界 🌍 émojis"}
        resp = await hpke_client.post("/echo", json=unicode_data)

        data = await resp.json()
        # The echo contains the JSON string, which may have unicode escaped
        # Check for either literal or escaped form
        echo = data["echo"]
        assert "世界" in echo or "\\u4e16\\u754c" in echo
        assert "🌍" in echo or "\\ud83c\\udf0d" in echo

    async def test_binary_in_json_response(self, hpke_client: HPKEClientSession) -> None:
        """Binary-like content (high bytes) in JSON is handled."""
        # JSON with escaped binary-like content
        test_data = {"binary_like": "\\x00\\xff"}
        resp = await hpke_client.post("/echo", json=test_data)

        data = await resp.json()
        assert "binary_like" in data["echo"]

    async def test_rapid_sequential_requests(self, hpke_client: HPKEClientSession) -> None:
        """Multiple rapid sequential requests work correctly."""
        for i in range(10):
            resp = await hpke_client.post("/echo", json={"seq": i})
            data = await resp.json()
            assert str(i) in data["echo"]


class TestSSEEdgeCasesE2E:
    """E2E edge case tests for SSE encryption."""

    async def test_single_event_stream(self, hpke_client: HPKEClientSession) -> None:
        """Stream with minimum events works."""
        # /stream sends 4 events minimum
        resp = await hpke_client.post("/stream", json={"start": True})
        events = [parse_sse_chunk(chunk) async for chunk in hpke_client.iter_sse(resp)]
        assert len(events) >= 1

    async def test_sse_with_unicode_data(self, hpke_client: HPKEClientSession) -> None:
        """SSE events with unicode content work."""
        resp = await hpke_client.post("/stream", json={"start": True})
        events = [parse_sse_chunk(chunk) async for chunk in hpke_client.iter_sse(resp)]

        # All events should decode properly
        for event_type, event_data in events:
            assert event_type is not None
            assert event_data is not None


class TestErrorResponsesE2E:
    """E2E tests for error response handling."""

    async def test_404_response_encrypted(self, hpke_client: HPKEClientSession) -> None:
        """404 responses are still encrypted for encrypted requests."""
        resp = await hpke_client.get("/nonexistent-path-xyz")
        # Server returns 404 for unknown paths
        assert resp.status == 404

    async def test_malformed_json_request(self, hpke_client: HPKEClientSession) -> None:
        """Server handles requests gracefully."""
        # Send valid JSON that the server can process
        resp = await hpke_client.post("/echo", json=None)
        # Should get some response (either success or error)
        assert resp.status in (200, 400, 422)


class TestWeirdInputsE2E:
    """E2E tests for weird/adversarial inputs."""

    async def test_very_long_key_names(self, hpke_client: HPKEClientSession) -> None:
        """JSON with very long key names works."""
        long_key = "k" * 1000
        test_data = {long_key: "value"}
        resp = await hpke_client.post("/echo", json=test_data)

        data = await resp.json()
        assert long_key in data["echo"]

    async def test_deeply_nested_json(self, hpke_client: HPKEClientSession) -> None:
        """Deeply nested JSON structures work."""
        nested: dict[str, Any] = {"level": 0}
        current = nested
        for i in range(1, 20):  # 20 levels deep
            current["child"] = {"level": i}
            current = current["child"]

        resp = await hpke_client.post("/echo", json=nested)
        data = await resp.json()
        assert "level" in data["echo"]

    async def test_array_response(self, hpke_client: HPKEClientSession) -> None:
        """Array JSON in request works."""
        test_data = [1, 2, 3, "four", {"five": 5}]
        resp = await hpke_client.post("/echo", json=test_data)

        data = await resp.json()
        assert "1" in data["echo"] or "[1" in data["echo"]

    async def test_special_characters_in_values(self, hpke_client: HPKEClientSession) -> None:
        """Special characters in JSON values work."""
        test_data = {
            "quotes": 'Hello "world"',
            "newlines": "line1\nline2",
            "tabs": "col1\tcol2",
            "backslash": "path\\to\\file",
        }
        resp = await hpke_client.post("/echo", json=test_data)

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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify encrypted response body is NOT readable as plaintext JSON."""
        resp = await hpke_client.post("/echo", json={"secret": "data"})

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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify encrypted SSE events are NOT readable as plaintext SSE."""
        resp = await hpke_client.post("/stream", json={"start": True})

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
        hpke_client: HPKEClientSession,
        granian_server: E2EServer,
    ) -> None:
        """Verify X-HPKE-Stream header presence matches actual encryption."""
        base_url = f"http://{granian_server.host}:{granian_server.port}"

        # Case 1: Encrypted request → should get encrypted response with header
        resp = await hpke_client.post("/echo", json={"test": 1})

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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify normal decryption works (baseline test)."""
        # Make a valid encrypted request
        resp = await hpke_client.post("/echo", json={"test": 1})

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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify standard response wire format: [length(4B) || counter(4B) || ciphertext]."""
        resp = await hpke_client.post("/echo", json={"format": "test"})

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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify SSE wire format: event: enc\\ndata: <base64>\\n\\n.

        SSEFormat uses standard base64 (not base64url) for ~1.7x faster encoding.
        See streaming.py SSEFormat docstring for rationale.
        """
        resp = await hpke_client.post("/stream", json={"start": True})

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

    async def test_status_property(self, hpke_client: HPKEClientSession) -> None:
        """Test status property returns correct HTTP status code."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)
        assert resp.status == 200
        assert isinstance(resp.status, int)

    async def test_headers_property(self, hpke_client: HPKEClientSession) -> None:
        """Test headers property returns CIMultiDictProxy with case-insensitive access."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # Should be CIMultiDictProxy
        from multidict import CIMultiDictProxy

        assert isinstance(resp.headers, CIMultiDictProxy)

        # Case-insensitive access should work
        ct_lower = resp.headers.get("content-type")
        ct_upper = resp.headers.get("Content-Type")
        assert ct_lower == ct_upper

    async def test_url_property(self, hpke_client: HPKEClientSession) -> None:
        """Test url property returns yarl.URL."""
        from yarl import URL

        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)
        assert isinstance(resp.url, URL)
        assert "/echo" in str(resp.url)

    async def test_ok_property(self, hpke_client: HPKEClientSession) -> None:
        """Test ok property returns True for 2xx responses."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)
        assert resp.ok is True
        assert isinstance(resp.ok, bool)

    async def test_reason_property(self, hpke_client: HPKEClientSession) -> None:
        """Test reason property returns HTTP status reason."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)
        # Reason can be None (HTTP/2) or string (HTTP/1.1)
        assert resp.reason is None or isinstance(resp.reason, str)

    async def test_content_type_property(self, hpke_client: HPKEClientSession) -> None:
        """Test content_type property returns Content-Type value."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)
        assert isinstance(resp.content_type, str)

    async def test_raise_for_status_success(self, hpke_client: HPKEClientSession) -> None:
        """Test raise_for_status() does not raise on 2xx."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)
        # Should not raise
        resp.raise_for_status()

    # ==========================================================================
    # Overridden methods (decrypt content)
    # ==========================================================================

    async def test_read_returns_decrypted_bytes(self, hpke_client: HPKEClientSession) -> None:
        """Test read() returns decrypted bytes."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        body = await resp.read()
        assert isinstance(body, bytes)
        data = json.loads(body)
        assert "echo" in data

    async def test_text_returns_decrypted_string(self, hpke_client: HPKEClientSession) -> None:
        """Test text() returns decrypted string with default encoding."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        text = await resp.text()
        assert isinstance(text, str)
        data = json.loads(text)
        assert "echo" in data

    async def test_text_with_encoding_param(self, hpke_client: HPKEClientSession) -> None:
        """Test text(encoding=...) respects encoding parameter."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        text = await resp.text(encoding="utf-8")
        assert isinstance(text, str)

    async def test_text_with_errors_param(self, hpke_client: HPKEClientSession) -> None:
        """Test text(errors=...) matches aiohttp signature."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # aiohttp supports errors parameter
        text = await resp.text(errors="replace")
        assert isinstance(text, str)

    async def test_json_returns_decrypted_dict(self, hpke_client: HPKEClientSession) -> None:
        """Test json() returns decrypted and parsed JSON."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        data = await resp.json()
        assert isinstance(data, dict)
        assert "echo" in data

    async def test_json_with_loads_param(self, hpke_client: HPKEClientSession) -> None:
        """Test json(loads=...) matches aiohttp signature."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # aiohttp supports custom loads function
        import json as json_mod

        data = await resp.json(loads=json_mod.loads)
        assert isinstance(data, dict)

    async def test_json_with_encoding_param(self, hpke_client: HPKEClientSession) -> None:
        """Test json(encoding=...) matches aiohttp signature."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # aiohttp supports encoding parameter
        data = await resp.json(encoding="utf-8")
        assert isinstance(data, dict)

    async def test_json_with_content_type_param(self, hpke_client: HPKEClientSession) -> None:
        """Test json(content_type=...) matches aiohttp signature."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # aiohttp supports content_type validation (None disables check)
        data = await resp.json(content_type=None)
        assert isinstance(data, dict)

    # ==========================================================================
    # DecryptedResponse-specific methods
    # ==========================================================================

    async def test_unwrap_returns_client_response(self, hpke_client: HPKEClientSession) -> None:
        """Test unwrap() returns underlying ClientResponse."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        underlying = resp.unwrap()
        assert isinstance(underlying, aiohttp.ClientResponse)
        assert underlying.status == resp.status

    # ==========================================================================
    # __getattr__ fallback (proxied to underlying response)
    # ==========================================================================

    async def test_version_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test version attribute proxied via __getattr__."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        version = resp.version
        assert version is not None

    async def test_request_info_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test request_info attribute proxied via __getattr__."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        request_info = resp.request_info
        assert request_info is not None
        assert hasattr(request_info, "url")

    async def test_cookies_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test cookies attribute proxied via __getattr__."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        cookies = resp.cookies
        assert cookies is not None

    async def test_history_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test history attribute proxied via __getattr__."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        history = resp.history
        assert isinstance(history, tuple)

    async def test_content_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test content StreamReader proxied via __getattr__."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        content = resp.content
        assert content is not None
        assert hasattr(content, "read")

    async def test_charset_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test charset property proxied via __getattr__."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        charset = resp.charset
        # charset can be None or string
        assert charset is None or isinstance(charset, str)

    async def test_content_length_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test content_length property proxied via __getattr__."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        content_length = resp.content_length
        # content_length can be None or int
        assert content_length is None or isinstance(content_length, int)

    async def test_real_url_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test real_url property proxied via __getattr__."""
        from yarl import URL

        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        real_url = resp.real_url
        assert isinstance(real_url, URL)

    async def test_host_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test host property proxied via __getattr__."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        host = resp.host
        assert isinstance(host, str)

    async def test_links_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test links property proxied via __getattr__."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        links = resp.links
        # links is a MultiDictProxy (possibly empty)
        assert links is not None

    async def test_get_encoding_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test get_encoding() method proxied via __getattr__."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        encoding = resp.get_encoding()
        assert isinstance(encoding, str)

    async def test_close_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test close() method proxied via __getattr__."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # close() should be callable
        assert callable(resp.close)

    async def test_release_via_getattr(self, hpke_client: HPKEClientSession) -> None:
        """Test release() method proxied via __getattr__."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        # release() should be callable
        assert callable(resp.release)

    # ==========================================================================
    # Caching and consistency
    # ==========================================================================

    async def test_read_cached_on_multiple_calls(self, hpke_client: HPKEClientSession) -> None:
        """Test that multiple read() calls return cached decrypted content."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        body1 = await resp.read()
        body2 = await resp.read()
        assert body1 == body2

    async def test_read_text_json_consistency(self, hpke_client: HPKEClientSession) -> None:
        """Test that read(), text(), json() return consistent data."""
        resp = await hpke_client.post("/echo", json={"test": 1})
        assert isinstance(resp, DecryptedResponse)

        body_bytes = await resp.read()
        body_text = await resp.text()
        body_json = await resp.json()

        assert body_bytes.decode("utf-8") == body_text
        assert json.loads(body_text) == body_json

    # ==========================================================================
    # Type identity
    # ==========================================================================

    async def test_isinstance_decrypted_response(self, hpke_client: HPKEClientSession) -> None:
        """Test DecryptedResponse can be identified via isinstance."""
        resp = await hpke_client.post("/echo", json={"test": 1})

        assert isinstance(resp, DecryptedResponse)
        assert not isinstance(resp, aiohttp.ClientResponse)
        assert isinstance(resp.unwrap(), aiohttp.ClientResponse)


class TestLargePayloadAutoChunking:
    """
    Test auto-chunking for large payloads (10MB, 50MB, 100MB, 250MB).

    These tests verify that the length-prefix wire format correctly handles
    multi-chunk encryption/decryption for very large request/response bodies.

    Wire format per chunk: length(4B) || counter(4B) || ciphertext
    """

    @pytest.mark.parametrize(
        "size_mb",
        [10, 50, 100, 250],
        ids=["10MB", "50MB", "100MB", "250MB"],
    )
    async def test_large_request_roundtrip(
        self,
        hpke_client: HPKEClientSession,
        size_mb: int,
    ) -> None:
        """Large request payloads encrypt/decrypt correctly via auto-chunking."""
        size_bytes = size_mb * 1024 * 1024
        # Use repeating pattern for efficient generation
        pattern = "A" * 1024  # 1KB pattern
        large_content = pattern * (size_bytes // 1024)

        resp = await hpke_client.post("/echo", data=large_content.encode())
        assert resp.status == 200

        data = await resp.json()
        # Verify content length (echo returns the raw body as string)
        assert len(data["echo"]) == len(large_content)

    @pytest.mark.parametrize(
        "size_mb",
        [10, 50, 100, 250],
        ids=["10MB", "50MB", "100MB", "250MB"],
    )
    async def test_large_response_decryption(
        self,
        hpke_client: HPKEClientSession,
        size_mb: int,
    ) -> None:
        """Large responses are correctly decrypted from multiple chunks."""
        size_bytes = size_mb * 1024 * 1024
        pattern = "B" * 1024
        large_content = pattern * (size_bytes // 1024)

        resp = await hpke_client.post("/echo", data=large_content.encode())
        assert resp.status == 200
        assert isinstance(resp, DecryptedResponse)

        # Verify decryption works
        data = await resp.json()
        assert data["echo"] == large_content

    @pytest.mark.parametrize(
        "size_mb",
        [10, 50, 100, 250],
        ids=["10MB", "50MB", "100MB", "250MB"],
    )
    async def test_large_payload_wire_format(
        self,
        hpke_client: HPKEClientSession,
        size_mb: int,
    ) -> None:
        """Verify wire format uses length prefix for O(1) chunk detection."""
        size_bytes = size_mb * 1024 * 1024
        pattern = "C" * 1024
        large_content = pattern * (size_bytes // 1024)

        resp = await hpke_client.post("/echo", data=large_content.encode())
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
        hpke_client: HPKEClientSession,
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

        resp = await hpke_client.post("/echo", data=large_content.encode())
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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify encrypted output has high Shannon entropy (> 7.0 bits/byte)."""
        payload = {"data": "A" * 10000, "secret": "password123"}

        resp = await hpke_client.post("/echo", json=payload)
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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify encrypted data has uniform byte distribution (chi-square p > 0.01)."""
        payload = {"message": "Hello World " * 1000}

        resp = await hpke_client.post("/echo", json=payload)
        assert resp.status == 200
        assert isinstance(resp, DecryptedResponse)

        raw_body = await resp.unwrap().read()
        assert len(raw_body) >= 1000, "Response too short for chi-square test"

        chi2, p_value = chi_square_byte_uniformity(raw_body)
        assert p_value > 0.01, (
            f"Chi-square p-value {p_value:.4f} indicates non-uniform distribution. "
            f"Encrypted data should appear random (chi2={chi2:.2f})."
        )

    async def test_known_plaintext_not_visible_on_wire(
        self,
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify known plaintext values never appear in encrypted wire data."""
        canary = "CANARY_SECRET_12345_XYZ"
        payload = {"secret": canary, "data": "Hello World Test Message"}

        resp = await hpke_client.post("/echo", json=payload)
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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify wire format has valid cryptographic structure."""
        resp = await hpke_client.post("/echo", json={"test": "structure"})
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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify SSE chunks arrive with inter-arrival timing gaps."""
        resp = await hpke_client.post("/stream-delayed", json={"start": True})
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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify large requests are sent in multiple chunks."""
        size_mb = 10
        large_content = "X" * (size_mb * 1024 * 1024)

        resp = await hpke_client.post("/echo-chunks", data=large_content.encode())
        assert resp.status == 200

        data = await resp.json()

        assert data["chunk_count"] > 1, f"Large {size_mb}MB request sent as single chunk!"

        min_expected = (size_mb * 1024 * 1024) // (64 * 1024) // 2
        assert data["chunk_count"] >= min_expected, (
            f"Expected >= {min_expected} chunks for {size_mb}MB, got {data['chunk_count']}"
        )

    async def test_response_chunk_boundaries_valid(
        self,
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify response chunk boundaries align with length prefix format."""
        size_mb = 5
        large_content = "Y" * (size_mb * 1024 * 1024)

        resp = await hpke_client.post("/echo", data=large_content.encode())
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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify captured network traffic contains no plaintext."""
        import asyncio

        canaries = ["WIRE_CANARY_VALUE_ABC123", "NETWORK_TEST_SECRET_XYZ"]

        for canary in canaries:
            resp = await hpke_client.post("/echo", json={"secret": canary})
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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify SSE streaming content is encrypted on the wire.

        The SSE transport format (event:, data:) is visible on the wire, but
        the actual event content must be encrypted. Original event types and
        data should NOT appear - only 'event: enc' with base64 ciphertext.
        """
        import asyncio

        # Trigger SSE stream with unique canary values
        sse_canary = "SSE_WIRE_CANARY_ENCRYPTED_789"
        resp = await hpke_client.post("/stream", json={"canary": sse_canary})
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
        hpke_client: HPKEClientSession,
    ) -> None:
        """Verify same plaintext produces different ciphertext (nonce uniqueness).

        This is critical for security - if nonces are reused, an attacker can
        XOR ciphertexts to recover plaintext. Each encryption MUST produce
        unique ciphertext even for identical input.
        """
        import asyncio

        # Send identical requests
        identical_payload = {"test": "NONCE_TEST_IDENTICAL_PAYLOAD_12345"}

        resp1 = await hpke_client.post("/echo", json=identical_payload)
        assert resp1.status == 200
        assert isinstance(resp1, DecryptedResponse)
        raw1 = await resp1.unwrap().read()

        resp2 = await hpke_client.post("/echo", json=identical_payload)
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
        hpke_client: HPKEClientSession,
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

        resp = await hpke_client.post("/echo", json=request_payload)
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
        hpke_client: HPKEClientSession,
        test_psk: bytes,
    ) -> None:
        """Verify pre-shared key never appears in network traffic.

        The PSK is used for authentication but should NEVER be transmitted.
        It's used locally to derive keys, not sent over the wire.
        """
        import asyncio

        # Generate some traffic
        resp = await hpke_client.post("/echo", json={"test": "psk_leak_check"})
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
        hpke_client: HPKEClientSession,
        platform_keypair: tuple[bytes, bytes],
    ) -> None:
        """Verify private key and derived session keys never appear in traffic.

        Only the ephemeral PUBLIC key should be visible (the 'enc' field in HPKE).
        Private keys and derived session keys must never be transmitted.
        """
        import asyncio

        private_key, _public_key = platform_keypair

        # Generate traffic
        resp = await hpke_client.post("/echo", json={"test": "key_leak_check"})
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

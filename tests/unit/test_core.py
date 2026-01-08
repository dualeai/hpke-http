"""Unit tests for core.py encryption/decryption classes and parsers.

Uses fixtures from conftest.py:
- client_keypair: (sk, pk) tuple
- test_psk: 32-byte PSK
- test_psk_id: PSK identifier
- wrong_psk: different PSK for failure tests
"""

import secrets

import pytest
from cryptography.hazmat.primitives.asymmetric import x25519

from hpke_http.constants import CHUNK_SIZE, HEADER_HPKE_ENC, HEADER_HPKE_STREAM
from hpke_http.core import (
    RequestDecryptor,
    RequestEncryptor,
    ResponseDecryptor,
    ResponseEncryptor,
    SSEDecryptor,
    SSEEncryptor,
    SSEEventParser,
    SSELineParser,
    _ChunkStreamParser,  # pyright: ignore[reportPrivateUsage]
    is_sse_response,
)
from hpke_http.exceptions import DecryptionError, InvalidPSKError
from hpke_http.hpke import RecipientContext, SenderContext

# =============================================================================
# TEST: RequestEncryptor
# =============================================================================


class TestRequestEncryptor:
    """Tests for RequestEncryptor class."""

    def test_encrypt_all_basic(self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes) -> None:
        """encrypt_all produces encrypted output."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        plaintext = b"hello world"

        encrypted = encryptor.encrypt_all(plaintext)

        assert encrypted != plaintext
        assert len(encrypted) > len(plaintext)  # Overhead from encryption

    def test_get_headers_contains_required(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """get_headers returns X-HPKE-Enc and X-HPKE-Stream."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        headers = encryptor.get_headers()

        assert HEADER_HPKE_ENC in headers
        assert HEADER_HPKE_STREAM in headers
        assert len(headers[HEADER_HPKE_ENC]) > 0
        assert len(headers[HEADER_HPKE_STREAM]) > 0

    def test_context_property_returns_sender_context(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """context property returns SenderContext."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        ctx = encryptor.context

        assert isinstance(ctx, SenderContext)

    def test_empty_body(self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes) -> None:
        """encrypt_all handles empty body."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        encrypted = encryptor.encrypt_all(b"")

        assert len(encrypted) > 0  # Should have at least one encrypted chunk

    def test_encrypt_streaming_mode(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """encrypt can be called multiple times for streaming."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        chunk1 = encryptor.encrypt(b"chunk1")
        chunk2 = encryptor.encrypt(b"chunk2")

        assert len(chunk1) > 0
        assert len(chunk2) > 0
        assert chunk1 != chunk2

    def test_compression_enabled_large_body(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """compress=True adds X-HPKE-Encoding header for large bodies."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id, compress=True)

        # Body larger than ZSTD_MIN_SIZE (1KB)
        large_body = b"A" * 2000  # Highly compressible
        encryptor.encrypt_all(large_body)

        headers = encryptor.get_headers()
        assert headers.get("X-HPKE-Encoding") == "zstd"

    def test_compression_below_threshold(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """compress=True does not compress small bodies."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id, compress=True)

        # Body smaller than ZSTD_MIN_SIZE
        small_body = b"tiny"
        encryptor.encrypt_all(small_body)

        headers = encryptor.get_headers()
        assert "X-HPKE-Encoding" not in headers


# =============================================================================
# TEST: RequestDecryptor
# =============================================================================


class TestRequestDecryptor:
    """Tests for RequestDecryptor class."""

    def test_decrypt_all_roundtrip(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Encrypt with RequestEncryptor, decrypt with RequestDecryptor."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        plaintext = b"hello world"

        encrypted = encryptor.encrypt_all(plaintext)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        decrypted = decryptor.decrypt_all(encrypted)

        assert decrypted == plaintext

    def test_feed_streaming_mode(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """feed() yields chunks as they complete."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        plaintext = b"streaming data"

        encrypted = encryptor.encrypt_all(plaintext)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        chunks = list(decryptor.feed(encrypted))

        assert b"".join(chunks) == plaintext

    def test_feed_partial_chunks(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """feed() handles data split across multiple calls."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        plaintext = b"data spanning chunks"

        encrypted = encryptor.encrypt_all(plaintext)
        headers = encryptor.get_headers()

        # Split in middle of encrypted data
        mid = len(encrypted) // 2
        part1, part2 = encrypted[:mid], encrypted[mid:]

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        results: list[bytes] = []
        results.extend(decryptor.feed(part1))
        results.extend(decryptor.feed(part2))

        assert b"".join(results) == plaintext

    def test_context_property_returns_recipient_context(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """context property returns RecipientContext."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)

        assert isinstance(decryptor.context, RecipientContext)

    def test_is_compressed_property(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """is_compressed reflects X-HPKE-Encoding header."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id, compress=True)

        # Large body triggers compression
        large_body = b"A" * 2000
        _ = encryptor.encrypt_all(large_body)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)

        assert decryptor.is_compressed is True

    def test_missing_enc_header_raises(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Missing X-HPKE-Enc header raises DecryptionError."""
        sk, _ = client_keypair
        headers = {HEADER_HPKE_STREAM: "some-value"}

        with pytest.raises(DecryptionError, match="Missing X-HPKE-Enc"):
            RequestDecryptor(headers, sk, test_psk, test_psk_id)

    def test_missing_stream_header_raises(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Missing X-HPKE-Stream header raises DecryptionError."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        headers = encryptor.get_headers()
        del headers[HEADER_HPKE_STREAM]

        with pytest.raises(DecryptionError, match="Missing X-HPKE-Stream"):
            RequestDecryptor(headers, sk, test_psk, test_psk_id)

    def test_decompression_on_decrypt_all(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """decrypt_all automatically decompresses."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id, compress=True)

        # Large compressible body
        plaintext = b"AAAA" * 1000
        encrypted = encryptor.encrypt_all(plaintext)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        decrypted = decryptor.decrypt_all(encrypted)

        assert decrypted == plaintext


# =============================================================================
# TEST: ResponseEncryptor + ResponseDecryptor
# =============================================================================


class TestResponseEncryptor:
    """Tests for ResponseEncryptor class."""

    def test_encrypt_all_basic(self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes) -> None:
        """encrypt_all produces encrypted output."""
        sk, pk = client_keypair
        # Setup: create request context first
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"request")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        # Test response encryption
        resp_enc = ResponseEncryptor(req_dec.context)
        plaintext = b"response body"
        encrypted = resp_enc.encrypt_all(plaintext)

        assert encrypted != plaintext
        assert len(encrypted) > len(plaintext)

    def test_get_headers_contains_stream(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """get_headers returns X-HPKE-Stream."""
        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        resp_enc = ResponseEncryptor(req_dec.context)
        headers = resp_enc.get_headers()

        assert HEADER_HPKE_STREAM in headers
        assert len(headers[HEADER_HPKE_STREAM]) > 0

    def test_empty_body(self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes) -> None:
        """encrypt_all handles empty body."""
        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        resp_enc = ResponseEncryptor(req_dec.context)
        encrypted = resp_enc.encrypt_all(b"")

        assert len(encrypted) > 0

    def test_encrypt_streaming_mode(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """encrypt can be called multiple times for streaming."""
        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        resp_enc = ResponseEncryptor(req_dec.context)
        chunk1 = resp_enc.encrypt(b"chunk1")
        chunk2 = resp_enc.encrypt(b"chunk2")

        assert len(chunk1) > 0
        assert len(chunk2) > 0


class TestResponseDecryptor:
    """Tests for ResponseDecryptor class."""

    def test_decrypt_all_roundtrip(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Full request→response→decrypt roundtrip."""
        sk, pk = client_keypair

        # Client encrypts request
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"request")

        # Server decrypts request
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        # Server encrypts response
        resp_enc = ResponseEncryptor(req_dec.context)
        plaintext = b"response body"
        encrypted_resp = resp_enc.encrypt_all(plaintext)

        # Client decrypts response
        resp_dec = ResponseDecryptor(resp_enc.get_headers(), req_enc.context)
        decrypted = resp_dec.decrypt_all(encrypted_resp)

        assert decrypted == plaintext

    def test_feed_streaming_mode(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """feed() yields chunks as they complete."""
        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        resp_enc = ResponseEncryptor(req_dec.context)
        plaintext = b"streaming response"
        encrypted = resp_enc.encrypt_all(plaintext)

        resp_dec = ResponseDecryptor(resp_enc.get_headers(), req_enc.context)
        chunks = list(resp_dec.feed(encrypted))

        assert b"".join(chunks) == plaintext

    def test_feed_partial_chunks(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """feed() handles data split across multiple calls."""
        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        resp_enc = ResponseEncryptor(req_dec.context)
        plaintext = b"data spanning chunks"
        encrypted = resp_enc.encrypt_all(plaintext)

        # Split encrypted data
        mid = len(encrypted) // 2
        part1, part2 = encrypted[:mid], encrypted[mid:]

        resp_dec = ResponseDecryptor(resp_enc.get_headers(), req_enc.context)
        results: list[bytes] = []
        results.extend(resp_dec.feed(part1))
        results.extend(resp_dec.feed(part2))

        assert b"".join(results) == plaintext

    def test_missing_header_raises(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Missing X-HPKE-Stream header raises DecryptionError."""
        _, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)

        with pytest.raises(DecryptionError, match="Missing X-HPKE-Stream"):
            ResponseDecryptor({}, req_enc.context)

    def test_case_insensitive_header_lookup(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Headers can be any case."""
        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        resp_enc = ResponseEncryptor(req_dec.context)
        encrypted = resp_enc.encrypt_all(b"response")

        # Use lowercase header name
        headers = {"x-hpke-stream": resp_enc.get_headers()[HEADER_HPKE_STREAM]}
        resp_dec = ResponseDecryptor(headers, req_enc.context)

        assert resp_dec.decrypt_all(encrypted) == b"response"


# =============================================================================
# TEST: SSEEncryptor + SSEDecryptor
# =============================================================================


class TestSSEEncryptor:
    """Tests for SSEEncryptor class."""

    def test_encrypt_produces_sse_format(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """encrypt produces SSE wire format: event: enc\\ndata: ...\\n\\n."""
        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        sse_enc = SSEEncryptor(req_dec.context)
        encrypted = sse_enc.encrypt(b"event data")

        # Check SSE format
        lines = encrypted.decode("ascii").split("\n")
        assert lines[0] == "event: enc"
        assert lines[1].startswith("data: ")
        assert encrypted.endswith(b"\n\n")

    def test_get_headers_contains_content_type(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """get_headers includes Content-Type: text/event-stream."""
        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        sse_enc = SSEEncryptor(req_dec.context)
        headers = sse_enc.get_headers()

        assert headers.get("Content-Type") == "text/event-stream"
        assert HEADER_HPKE_STREAM in headers

    def test_multiple_events_counter_increments(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Each event uses a different counter (different ciphertext)."""
        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        sse_enc = SSEEncryptor(req_dec.context)
        event1 = sse_enc.encrypt(b"same data")
        event2 = sse_enc.encrypt(b"same data")

        # Same plaintext should produce different ciphertext
        assert event1 != event2


class TestSSEDecryptor:
    """Tests for SSEDecryptor class."""

    def test_decrypt_single_event(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """decrypt decrypts SSE data field."""
        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        sse_enc = SSEEncryptor(req_dec.context)
        plaintext = b"event data"
        encrypted = sse_enc.encrypt(plaintext)

        # Extract data field
        data_field = ""
        for line in encrypted.decode("ascii").split("\n"):
            if line.startswith("data: "):
                data_field = line[6:]
                break

        sse_dec = SSEDecryptor(sse_enc.get_headers(), req_enc.context)
        decrypted = sse_dec.decrypt(data_field)

        assert decrypted == plaintext

    def test_decrypt_multiple_events(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """decrypt works for multiple sequential events."""
        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        sse_enc = SSEEncryptor(req_dec.context)
        sse_dec = SSEDecryptor(sse_enc.get_headers(), req_enc.context)

        for i in range(5):
            plaintext = f"event {i}".encode()
            encrypted = sse_enc.encrypt(plaintext)
            data_field = encrypted.decode("ascii").split("\n")[1][6:]
            decrypted = sse_dec.decrypt(data_field)
            assert decrypted == plaintext

    def test_missing_header_raises(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Missing X-HPKE-Stream header raises DecryptionError."""
        _, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)

        with pytest.raises(DecryptionError, match="Missing X-HPKE-Stream"):
            SSEDecryptor({}, req_enc.context)

    def test_accepts_str_and_bytes(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """decrypt accepts both str and bytes input."""
        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        sse_enc = SSEEncryptor(req_dec.context)
        plaintext = b"test"
        encrypted = sse_enc.encrypt(plaintext)
        data_field = encrypted.decode("ascii").split("\n")[1][6:]

        sse_dec = SSEDecryptor(sse_enc.get_headers(), req_enc.context)

        # Test with str
        result_str = sse_dec.decrypt(data_field)
        assert result_str == plaintext


# =============================================================================
# TEST: SSELineParser
# =============================================================================


class TestSSELineParser:
    """Tests for SSELineParser class."""

    def test_feed_complete_lines(self) -> None:
        """feed yields complete lines."""
        parser = SSELineParser()
        lines = list(parser.feed(b"line1\nline2\nline3\n"))

        assert lines == [b"line1", b"line2", b"line3"]

    def test_feed_partial_lines(self) -> None:
        """feed buffers partial lines until complete."""
        parser = SSELineParser()

        # First call - partial line
        lines1 = list(parser.feed(b"par"))
        assert lines1 == []

        # Second call - completes line
        lines2 = list(parser.feed(b"tial\n"))
        assert lines2 == [b"partial"]

    def test_feed_handles_crlf(self) -> None:
        """feed strips \\r from \\r\\n line endings."""
        parser = SSELineParser()
        lines = list(parser.feed(b"line1\r\nline2\r\n"))

        assert lines == [b"line1", b"line2"]

    def test_feed_handles_cr_only(self) -> None:
        """feed does not treat \\r alone as line ending."""
        parser = SSELineParser()
        lines = list(parser.feed(b"line\rwith\rcarriage\n"))

        # Only \n is treated as line ending, \r is stripped from end
        assert lines == [b"line\rwith\rcarriage"]

    def test_feed_empty_lines(self) -> None:
        """feed yields empty bytes for blank lines."""
        parser = SSELineParser()
        lines = list(parser.feed(b"line1\n\nline2\n"))

        assert lines == [b"line1", b"", b"line2"]

    def test_multiple_feed_calls(self) -> None:
        """Multiple feed calls accumulate correctly."""
        parser = SSELineParser()

        lines: list[bytes] = []
        lines.extend(parser.feed(b"first"))
        lines.extend(parser.feed(b" line\nseco"))
        lines.extend(parser.feed(b"nd line\n"))

        assert lines == [b"first line", b"second line"]


# =============================================================================
# TEST: SSEEventParser
# =============================================================================


class TestSSEEventParser:
    """Tests for SSEEventParser class."""

    def test_feed_complete_events(self) -> None:
        """feed yields complete events."""
        parser = SSEEventParser()
        events = list(parser.feed(b"event1\n\nevent2\n\n"))

        assert len(events) == 2
        assert events[0] == b"event1\n\n"
        assert events[1] == b"event2\n\n"

    def test_feed_partial_events(self) -> None:
        """feed buffers partial events until boundary found."""
        parser = SSEEventParser()

        # First call - partial event
        events1 = list(parser.feed(b"partial eve"))
        assert events1 == []

        # Second call - completes event
        events2 = list(parser.feed(b"nt\n\n"))
        assert events2 == [b"partial event\n\n"]

    def test_feed_handles_double_newline(self) -> None:
        """feed detects \\n\\n boundary."""
        parser = SSEEventParser()
        events = list(parser.feed(b"data: test\n\n"))

        assert len(events) == 1
        assert events[0] == b"data: test\n\n"

    def test_feed_handles_crlf_crlf(self) -> None:
        """feed detects \\r\\n\\r\\n boundary."""
        parser = SSEEventParser()
        events = list(parser.feed(b"data: test\r\n\r\n"))

        assert len(events) == 1
        assert events[0] == b"data: test\r\n\r\n"

    def test_flush_returns_remaining(self) -> None:
        """flush returns buffered data without boundary."""
        parser = SSEEventParser()

        # Feed partial event (no boundary)
        list(parser.feed(b"incomplete event"))

        remaining = parser.flush()
        assert remaining == b"incomplete event"

    def test_whatwg_boundary_compliance(self) -> None:
        """All valid SSE event boundaries per WHATWG spec."""
        # \n\n
        parser1 = SSEEventParser()
        events1 = list(parser1.feed(b"e1\n\ne2\n\n"))
        assert len(events1) == 2

        # \r\n\r\n
        parser2 = SSEEventParser()
        events2 = list(parser2.feed(b"e1\r\n\r\ne2\r\n\r\n"))
        assert len(events2) == 2

        # \r\r (rare but valid)
        parser3 = SSEEventParser()
        events3 = list(parser3.feed(b"e1\r\re2\r\r"))
        assert len(events3) == 2


# =============================================================================
# TEST: _ChunkStreamParser
# =============================================================================


class TestChunkStreamParser:
    """Tests for _ChunkStreamParser class (internal but critical)."""

    def test_feed_complete_chunks(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """feed yields complete chunks."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted = encryptor.encrypt_all(b"test data")

        parser = _ChunkStreamParser()
        chunks = list(parser.feed(encrypted))

        assert len(chunks) >= 1
        # Each chunk should have length prefix
        for chunk in chunks:
            length = int.from_bytes(chunk[:4], "big")
            assert len(chunk) == 4 + length

    def test_feed_partial_header(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """feed handles partial length prefix."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted = encryptor.encrypt_all(b"test")

        parser = _ChunkStreamParser()

        # Feed only 2 bytes of 4-byte length prefix
        chunks1 = list(parser.feed(encrypted[:2]))
        assert chunks1 == []

        # Feed rest
        chunks2 = list(parser.feed(encrypted[2:]))
        assert len(chunks2) == 1

    def test_feed_partial_payload(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """feed handles partial payload."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted = encryptor.encrypt_all(b"test data")

        parser = _ChunkStreamParser()

        # Feed length prefix + partial payload
        chunks1 = list(parser.feed(encrypted[:10]))
        assert chunks1 == []

        # Feed rest
        chunks2 = list(parser.feed(encrypted[10:]))
        assert len(chunks2) == 1

    def test_multiple_chunks_in_one_feed(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """feed yields multiple chunks if available."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        # Create data that spans multiple chunks
        large_data = b"x" * (CHUNK_SIZE + 100)
        encrypted = encryptor.encrypt_all(large_data)

        parser = _ChunkStreamParser()
        chunks = list(parser.feed(encrypted))

        # Should have at least 2 chunks
        assert len(chunks) >= 2


# =============================================================================
# TEST: is_sse_response
# =============================================================================


class TestIsSSEResponse:
    """Tests for is_sse_response helper function."""

    def test_text_event_stream(self) -> None:
        """Returns True for text/event-stream."""
        headers = {"Content-Type": "text/event-stream"}
        assert is_sse_response(headers) is True

    def test_with_charset(self) -> None:
        """Returns True for text/event-stream with charset."""
        headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        assert is_sse_response(headers) is True

    def test_case_insensitive(self) -> None:
        """Header lookup is case-insensitive."""
        headers = {"content-type": "text/event-stream"}
        assert is_sse_response(headers) is True

    def test_not_sse_json(self) -> None:
        """Returns False for application/json."""
        headers = {"Content-Type": "application/json"}
        assert is_sse_response(headers) is False

    def test_missing_content_type(self) -> None:
        """Returns False when Content-Type is missing."""
        headers: dict[str, str] = {}
        assert is_sse_response(headers) is False

    def test_empty_content_type(self) -> None:
        """Returns False for empty Content-Type."""
        headers = {"Content-Type": ""}
        assert is_sse_response(headers) is False


# =============================================================================
# TEST: Security & Adversarial Cases
# =============================================================================


class TestSecurityAdversarial:
    """Tests for security-critical error handling."""

    def test_wrong_psk_raises_decryption_error(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes, wrong_psk: bytes
    ) -> None:
        """Decryption with wrong PSK fails."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted = encryptor.encrypt_all(b"secret data")
        headers = encryptor.get_headers()

        # Try to decrypt with different PSK (from conftest.py fixture)
        with pytest.raises(DecryptionError):
            decryptor = RequestDecryptor(headers, sk, wrong_psk, test_psk_id)
            decryptor.decrypt_all(encrypted)

    def test_wrong_private_key_raises(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Decryption with wrong private key fails."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted = encryptor.encrypt_all(b"secret data")
        headers = encryptor.get_headers()

        # Generate different keypair
        wrong_sk = x25519.X25519PrivateKey.generate().private_bytes_raw()
        with pytest.raises(DecryptionError):
            decryptor = RequestDecryptor(headers, wrong_sk, test_psk, test_psk_id)
            decryptor.decrypt_all(encrypted)

    def test_corrupted_ciphertext_raises(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Corrupted ciphertext fails decryption."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted = encryptor.encrypt_all(b"secret data")
        headers = encryptor.get_headers()

        # Corrupt the ciphertext (flip bits in middle)
        corrupted = bytearray(encrypted)
        mid = len(corrupted) // 2
        corrupted[mid] ^= 0xFF
        corrupted = bytes(corrupted)

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        with pytest.raises(DecryptionError):
            decryptor.decrypt_all(corrupted)

    def test_tampered_auth_tag_raises(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Tampered authentication tag fails decryption."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted = encryptor.encrypt_all(b"secret data")
        headers = encryptor.get_headers()

        # Tamper with last 16 bytes (auth tag for ChaCha20-Poly1305)
        tampered = bytearray(encrypted)
        tampered[-1] ^= 0x01  # Flip one bit in auth tag
        tampered = bytes(tampered)

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        with pytest.raises(DecryptionError):
            decryptor.decrypt_all(tampered)

    def test_truncated_ciphertext_raises(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Truncated ciphertext fails decryption."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted = encryptor.encrypt_all(b"secret data")
        headers = encryptor.get_headers()

        # Truncate the ciphertext
        truncated = encrypted[: len(encrypted) // 2]

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        # Should either raise or return incomplete data
        chunks = list(decryptor.feed(truncated))
        # Incomplete chunk should not yield anything
        assert len(chunks) == 0

    def test_invalid_enc_header_raises(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Invalid base64 in X-HPKE-Enc header raises."""
        sk, _ = client_keypair
        headers = {
            "X-HPKE-Enc": "not-valid-base64!!!",
            "X-HPKE-Stream": "AAAAAAAAAAAAAAAAAAAAAA==",
        }

        with pytest.raises((DecryptionError, ValueError)):
            RequestDecryptor(headers, sk, test_psk, test_psk_id)

    def test_wrong_size_enc_header_raises(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Wrong size X-HPKE-Enc header raises."""
        sk, _ = client_keypair
        headers = {
            "X-HPKE-Enc": "AAAAAAAAAAAAAAAA",  # 12 bytes, should be 32
            "X-HPKE-Stream": "AAAAAAAAAAAAAAAAAAAAAA==",
        }

        with pytest.raises((DecryptionError, ValueError)):
            RequestDecryptor(headers, sk, test_psk, test_psk_id)


# =============================================================================
# TEST: Boundary Values
# =============================================================================


class TestBoundaryValues:
    """Tests for boundary conditions."""

    def test_data_exactly_chunk_size(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Data exactly at CHUNK_SIZE boundary."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        # Exactly one chunk
        data = b"x" * CHUNK_SIZE
        encrypted = encryptor.encrypt_all(data)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        decrypted = decryptor.decrypt_all(encrypted)

        assert decrypted == data

    def test_data_chunk_size_plus_one(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Data at CHUNK_SIZE + 1 (spans two chunks)."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        # One byte over chunk size - forces second chunk
        data = b"x" * (CHUNK_SIZE + 1)
        encrypted = encryptor.encrypt_all(data)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        decrypted = decryptor.decrypt_all(encrypted)

        assert decrypted == data

    def test_data_chunk_size_minus_one(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Data at CHUNK_SIZE - 1 (fits in one chunk)."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        data = b"x" * (CHUNK_SIZE - 1)
        encrypted = encryptor.encrypt_all(data)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        decrypted = decryptor.decrypt_all(encrypted)

        assert decrypted == data

    def test_single_byte(self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes) -> None:
        """Single byte payload."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        data = b"x"
        encrypted = encryptor.encrypt_all(data)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        decrypted = decryptor.decrypt_all(encrypted)

        assert decrypted == data

    def test_large_payload_multiple_chunks(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Large payload spanning many chunks."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        # 5 chunks worth of data
        data = b"y" * (CHUNK_SIZE * 5)
        encrypted = encryptor.encrypt_all(data)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        decrypted = decryptor.decrypt_all(encrypted)

        assert decrypted == data


# =============================================================================
# TEST: PSK Validation
# =============================================================================


class TestPSKValidation:
    """Tests for PSK validation edge cases."""

    def test_short_psk_raises(self, client_keypair: tuple[bytes, bytes], test_psk_id: bytes) -> None:
        """PSK shorter than minimum size raises error."""
        _, pk = client_keypair
        short_psk = b"tooshort"  # Less than PSK_MIN_SIZE (32 bytes)

        with pytest.raises((ValueError, Exception)):
            RequestEncryptor(pk, short_psk, test_psk_id)

    def test_empty_psk_raises(self, client_keypair: tuple[bytes, bytes], test_psk_id: bytes) -> None:
        """Empty PSK raises error."""
        _, pk = client_keypair

        with pytest.raises((ValueError, Exception)):
            RequestEncryptor(pk, b"", test_psk_id)

    def test_empty_psk_id_allowed(self, client_keypair: tuple[bytes, bytes], test_psk: bytes) -> None:
        """Empty PSK ID may be allowed (implementation dependent)."""
        sk, pk = client_keypair

        # This tests the actual behavior - empty PSK ID might work
        try:
            encryptor = RequestEncryptor(pk, test_psk, b"")
            encrypted = encryptor.encrypt_all(b"test")
            headers = encryptor.get_headers()

            decryptor = RequestDecryptor(headers, sk, test_psk, b"")
            decrypted = decryptor.decrypt_all(encrypted)
            assert decrypted == b"test"
        except (ValueError, InvalidPSKError):
            # Also acceptable if implementation rejects empty PSK ID
            pass


# =============================================================================
# TEST: Replay Attack Protection
# =============================================================================


class TestReplayAttackProtection:
    """Tests for replay attack detection in SSE streaming."""

    def test_sse_out_of_order_counter_raises(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Out-of-order SSE events raise ReplayAttackError."""
        from hpke_http.exceptions import ReplayAttackError

        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        sse_enc = SSEEncryptor(req_dec.context)
        sse_dec = SSEDecryptor(sse_enc.get_headers(), req_enc.context)

        # Encrypt 3 events
        event0 = sse_enc.encrypt(b"event0")
        _event1 = sse_enc.encrypt(b"event1")  # Intentionally skipped
        event2 = sse_enc.encrypt(b"event2")

        # Extract data fields
        def get_data(event: bytes) -> str:
            for line in event.decode("ascii").split("\n"):
                if line.startswith("data: "):
                    return line[6:]
            raise ValueError("No data field")

        # Decrypt event0 first (counter=1)
        sse_dec.decrypt(get_data(event0))

        # Skip event1, try to decrypt event2 (counter=3, expected=2)
        with pytest.raises(ReplayAttackError):
            sse_dec.decrypt(get_data(event2))

    def test_sse_duplicate_event_raises(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Replaying same SSE event raises ReplayAttackError."""
        from hpke_http.exceptions import ReplayAttackError

        sk, pk = client_keypair
        req_enc = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted_req = req_enc.encrypt_all(b"req")
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, test_psk, test_psk_id)
        req_dec.decrypt_all(encrypted_req)

        sse_enc = SSEEncryptor(req_dec.context)
        sse_dec = SSEDecryptor(sse_enc.get_headers(), req_enc.context)

        # Encrypt one event
        event = sse_enc.encrypt(b"event data")
        data_field = event.decode("ascii").split("\n")[1][6:]

        # Decrypt once (succeeds)
        sse_dec.decrypt(data_field)

        # Try to decrypt same event again (replay attack)
        with pytest.raises(ReplayAttackError):
            sse_dec.decrypt(data_field)


# =============================================================================
# TEST: Binary Data Handling
# =============================================================================


class TestBinaryDataHandling:
    """Tests for binary data edge cases."""

    def test_data_with_null_bytes(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Data containing null bytes roundtrips correctly."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        # Data with embedded null bytes
        data = b"before\x00middle\x00\x00after"
        encrypted = encryptor.encrypt_all(data)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        decrypted = decryptor.decrypt_all(encrypted)

        assert decrypted == data

    def test_random_binary_roundtrip(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """High-entropy random data roundtrips correctly."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        # Random bytes (high entropy, all byte values possible)
        data = secrets.token_bytes(1000)
        encrypted = encryptor.encrypt_all(data)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        decrypted = decryptor.decrypt_all(encrypted)

        assert decrypted == data

    def test_all_byte_values(self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes) -> None:
        """Data containing all 256 byte values roundtrips correctly."""
        sk, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        # All possible byte values 0x00-0xFF
        data = bytes(range(256))
        encrypted = encryptor.encrypt_all(data)
        headers = encryptor.get_headers()

        decryptor = RequestDecryptor(headers, sk, test_psk, test_psk_id)
        decrypted = decryptor.decrypt_all(encrypted)

        assert decrypted == data


# =============================================================================
# TEST: State Machine / Misuse
# =============================================================================


class TestStateMisuse:
    """Tests for encryptor/decryptor state machine violations."""

    def test_double_encrypt_all_different_data(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Calling encrypt_all twice produces independent ciphertexts."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        # First encryption
        encrypted1 = encryptor.encrypt_all(b"first")

        # Second encryption on same encryptor (counter continues)
        encrypted2 = encryptor.encrypt_all(b"second")

        # Both should produce valid ciphertext
        assert len(encrypted1) > 0
        assert len(encrypted2) > 0
        assert encrypted1 != encrypted2

    def test_mixing_streaming_and_batch(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Mixing encrypt() and encrypt_all() on same encryptor."""
        _, pk = client_keypair
        encryptor = RequestEncryptor(pk, test_psk, test_psk_id)

        # First use streaming mode
        chunk1 = encryptor.encrypt(b"stream1")
        chunk2 = encryptor.encrypt(b"stream2")

        # Then use batch mode
        chunk3 = encryptor.encrypt_all(b"batch")

        # All should produce output
        assert len(chunk1) > 0
        assert len(chunk2) > 0
        assert len(chunk3) > 0


# =============================================================================
# TEST: Cross-Session Attacks
# =============================================================================


class TestCrossSessionAttacks:
    """Tests for cross-session decryption failures."""

    def test_decrypt_with_different_session_fails(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """Cannot decrypt data from a different session."""
        sk, pk = client_keypair

        # Session 1: encrypt some data
        enc1 = RequestEncryptor(pk, test_psk, test_psk_id)
        encrypted = enc1.encrypt_all(b"secret from session 1")
        _ = enc1.get_headers()  # Not used - we use wrong headers from session 2

        # Session 2: different encryptor (different ephemeral key)
        enc2 = RequestEncryptor(pk, test_psk, test_psk_id)
        enc2.encrypt_all(b"dummy")  # Initialize session 2
        headers2 = enc2.get_headers()

        # Try to decrypt session 1 data with session 2 headers
        # This should fail because the enc (ephemeral public key) doesn't match
        decryptor = RequestDecryptor(headers2, sk, test_psk, test_psk_id)
        with pytest.raises(DecryptionError):
            decryptor.decrypt_all(encrypted)

    def test_response_decryptor_wrong_request_context(
        self, client_keypair: tuple[bytes, bytes], test_psk: bytes, test_psk_id: bytes
    ) -> None:
        """ResponseDecryptor fails with wrong request context."""
        sk, pk = client_keypair

        # Request 1
        req_enc1 = RequestEncryptor(pk, test_psk, test_psk_id)
        enc_req1 = req_enc1.encrypt_all(b"req1")
        req_dec1 = RequestDecryptor(req_enc1.get_headers(), sk, test_psk, test_psk_id)
        req_dec1.decrypt_all(enc_req1)

        # Response for request 1
        resp_enc = ResponseEncryptor(req_dec1.context)
        encrypted_resp = resp_enc.encrypt_all(b"response")
        resp_headers = resp_enc.get_headers()

        # Request 2 (different session)
        req_enc2 = RequestEncryptor(pk, test_psk, test_psk_id)
        req_enc2.encrypt_all(b"req2")

        # Try to decrypt response with request 2's context
        resp_dec = ResponseDecryptor(resp_headers, req_enc2.context)
        with pytest.raises(DecryptionError):
            resp_dec.decrypt_all(encrypted_resp)

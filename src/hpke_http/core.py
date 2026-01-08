"""
High-level encryption/decryption classes for HTTP transport.

This module provides stateful classes that encapsulate the full HPKE encryption
lifecycle for HTTP communication. These classes handle:
- HPKE context setup and key derivation
- Header parsing and building (base64url)
- Chunk format (length prefix parsing/building)
- Compression (Zstd)
- Counter state management

Usage (Client - httpx example):
    from hpke_http.core import RequestEncryptor, ResponseDecryptor

    encryptor = RequestEncryptor(server_pk, api_key, tenant_id)
    response = httpx.post(
        url,
        content=encryptor.encrypt_all(json.dumps(data).encode()),
        headers=encryptor.get_headers(),
    )
    decryptor = ResponseDecryptor(response.headers, encryptor.context)
    plaintext = decryptor.decrypt_all(response.content)

Usage (Server - Flask/Django example):
    from hpke_http.core import RequestDecryptor, ResponseEncryptor

    decryptor = RequestDecryptor(request.headers, private_key, psk, psk_id)
    plaintext = decryptor.decrypt_all(request.data)
    # Process request...
    encryptor = ResponseEncryptor(decryptor.context)
    encrypted = encryptor.encrypt_all(json.dumps(response_data).encode())
    return Response(encrypted, headers=encryptor.get_headers())
"""

from __future__ import annotations

import base64
import re
from collections.abc import Iterator, Mapping
from typing import Any

from hpke_http.constants import (
    CHACHA20_POLY1305_KEY_SIZE,
    CHUNK_SIZE,
    HEADER_HPKE_ENC,
    HEADER_HPKE_ENCODING,
    HEADER_HPKE_STREAM,
    RAW_LENGTH_PREFIX_SIZE,
    REQUEST_KEY_LABEL,
    RESPONSE_KEY_LABEL,
    SSE_SESSION_KEY_LABEL,
    ZSTD_MIN_SIZE,
)
from hpke_http.exceptions import DecryptionError
from hpke_http.hpke import (
    RecipientContext,
    SenderContext,
    setup_recipient_psk,
    setup_sender_psk,
)
from hpke_http.streaming import (
    ChunkDecryptor,
    ChunkEncryptor,
    RawFormat,
    SSEFormat,
    StreamingSession,
    create_session_from_context,
    zstd_compress,
    zstd_decompress,
)

__all__ = [
    # Server-side
    "RequestDecryptor",
    # Client-side
    "RequestEncryptor",
    "ResponseDecryptor",
    "ResponseEncryptor",
    "SSEDecryptor",
    "SSEEncryptor",
    # Helpers
    "SSEEventParser",
    "SSELineParser",
    "is_sse_response",
]


def _b64url_encode(data: bytes) -> str:
    """Base64url encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


_B64_PAD_SIZE = 4  # Base64 padding block size


def _b64url_decode(data: str) -> bytes:
    """Base64url decode with padding restoration."""
    # Add padding if needed
    padding = _B64_PAD_SIZE - (len(data) % _B64_PAD_SIZE)
    if padding != _B64_PAD_SIZE:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


def _get_header(headers: Mapping[str, Any], name: str) -> str | None:
    """Get header value, handling case-insensitive lookups."""
    # Try exact match first (faster)
    if name in headers:
        return str(headers[name])
    # Fall back to case-insensitive search
    name_lower = name.lower()
    for key in headers:
        if str(key).lower() == name_lower:
            return str(headers[key])
    return None


# =============================================================================
# STREAMING PARSERS
# =============================================================================


class _ChunkStreamParser:
    """Zero-copy streaming chunk boundary detection.

    Parses length-prefixed chunks from a byte stream, handling partial
    chunks that span multiple feed() calls.

    Wire format per chunk: [length(4B BE)] [payload(N bytes)]
    """

    __slots__ = ("_buffer",)

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> Iterator[bytes]:
        """Feed data, yield complete chunks as they're found.

        Args:
            data: Raw bytes from network/stream

        Yields:
            Complete chunks (including length prefix) ready for decryption
        """
        self._buffer.extend(data)
        while True:
            chunk = self._try_extract_chunk()
            if chunk is None:
                break
            yield chunk

    def _try_extract_chunk(self) -> bytes | None:
        """Extract one complete chunk if available."""
        if len(self._buffer) < RAW_LENGTH_PREFIX_SIZE:
            return None
        chunk_len = int.from_bytes(self._buffer[:RAW_LENGTH_PREFIX_SIZE], "big")
        total = RAW_LENGTH_PREFIX_SIZE + chunk_len
        if len(self._buffer) < total:
            return None
        chunk = bytes(self._buffer[:total])
        del self._buffer[:total]
        return chunk


class SSELineParser:
    """Zero-copy streaming SSE line parsing.

    Parses lines from a byte stream, handling partial lines that span
    multiple feed() calls. Lines are split on \\n and \\r is stripped
    (WHATWG SSE spec compliance).

    Used by client-side SSE decryption to parse encrypted event streams.
    """

    __slots__ = ("_buffer",)

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes) -> Iterator[bytes]:
        """Feed data, yield complete lines as they're found.

        Args:
            data: Raw bytes from network/stream

        Yields:
            Complete lines (without line ending, \\r stripped)
        """
        self._buffer.extend(data)
        consumed = 0
        while True:
            try:
                newline_pos = self._buffer.index(b"\n", consumed)
            except ValueError:
                break
            # Extract line, strip \\r (handles both \\n and \\r\\n)
            line = bytes(self._buffer[consumed:newline_pos]).rstrip(b"\r")
            consumed = newline_pos + 1
            yield line
        # Single compaction after extracting all lines
        if consumed:
            del self._buffer[:consumed]


class SSEEventParser:
    """Zero-copy streaming SSE event boundary detection.

    Parses complete SSE events from a byte stream. An event is delimited
    by two consecutive line endings (\\n\\n, \\r\\n\\r\\n, etc.) per WHATWG spec.

    Used by server-side SSE encryption to detect event boundaries in
    plaintext streams before encryption.
    """

    __slots__ = ("_buffer", "_pattern")

    def __init__(self) -> None:
        self._buffer = bytearray()
        # WHATWG-compliant event boundary: two consecutive line endings
        # Handles \n\n, \r\r, \r\n\r\n, and mixed combinations
        self._pattern = re.compile(rb"(?:\r\n|\r(?!\n)|\n)(?:\r\n|\r(?!\n)|\n)")

    def feed(self, data: bytes) -> Iterator[bytes]:
        """Feed data, yield complete events as boundaries are found.

        Args:
            data: Raw bytes from application

        Yields:
            Complete events (including the boundary delimiter)
        """
        self._buffer.extend(data)
        consumed = 0
        while True:
            match = self._pattern.search(self._buffer, pos=consumed)
            if not match:
                break
            # Extract complete event including boundary
            event_end = match.end()
            event = bytes(self._buffer[consumed:event_end])
            consumed = event_end
            yield event
        # Single compaction after extracting all events
        if consumed:
            del self._buffer[:consumed]

    def flush(self) -> bytes:
        """Flush remaining buffer content (for end of stream).

        Returns:
            Any remaining data that didn't form a complete event
        """
        remaining = bytes(self._buffer)
        self._buffer.clear()
        return remaining


# =============================================================================
# SSE PROTOCOL HELPERS
# =============================================================================


def is_sse_response(headers: Mapping[str, Any]) -> bool:
    """
    Check if response is SSE based on Content-Type header.

    Args:
        headers: Response headers (dict-like, case-sensitive or case-insensitive)

    Returns:
        True if Content-Type indicates text/event-stream

    Example:
        if is_sse_response(response.headers):
            decryptor = SSEDecryptor(response.headers, ctx)
        else:
            decryptor = ResponseDecryptor(response.headers, ctx)
    """
    content_type = _get_header(headers, "Content-Type")
    if not content_type:
        return False
    return "text/event-stream" in content_type.lower()


# =============================================================================
# CLIENT SIDE
# =============================================================================


class RequestEncryptor:
    """
    Encrypt request body for transmission.

    Handles HPKE context setup, header generation, chunking, and optional
    compression. Supports both streaming (chunk-by-chunk) and all-at-once modes.

    Example:
        encryptor = RequestEncryptor(server_pk, api_key, tenant_id)
        response = httpx.post(
            url,
            content=encryptor.encrypt_all(body),
            headers=encryptor.get_headers(),
        )
        # Use encryptor.context for response decryption
    """

    def __init__(
        self,
        public_key: bytes,
        psk: bytes,
        psk_id: bytes,
        *,
        compress: bool = False,
    ) -> None:
        """
        Initialize request encryptor.

        Args:
            public_key: Server's X25519 public key (32 bytes)
            psk: Pre-shared key / API key (>= 32 bytes)
            psk_id: PSK identifier (e.g., tenant ID)
            compress: Enable Zstd compression for request body
        """
        self._compress = compress
        self._was_compressed = False

        # Set up HPKE context
        self._ctx = setup_sender_psk(pk_r=public_key, info=psk_id, psk=psk, psk_id=psk_id)

        # Derive request key and create session
        request_key = self._ctx.export(REQUEST_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
        self._session = StreamingSession.create(request_key)

        # Create chunk encryptor (never per-chunk compression for requests)
        self._encryptor = ChunkEncryptor(self._session, format=RawFormat(), compress=False)

    def get_headers(self) -> dict[str, str]:
        """
        Get headers to send with the request.

        Returns:
            Dict with X-HPKE-Enc, X-HPKE-Stream, and optionally X-HPKE-Encoding
        """
        headers = {
            HEADER_HPKE_ENC: _b64url_encode(self._ctx.enc),
            HEADER_HPKE_STREAM: _b64url_encode(self._session.session_salt),
        }
        if self._was_compressed:
            headers[HEADER_HPKE_ENCODING] = "zstd"
        return headers

    def encrypt(self, chunk: bytes) -> bytes:
        """
        Encrypt a single chunk.

        For streaming mode, call this repeatedly for each chunk.
        Note: When using compress=True, you should use encrypt_all() instead,
        as whole-body compression provides better ratios.

        Args:
            chunk: Raw chunk bytes

        Returns:
            Encrypted chunk in wire format (length || counter || ciphertext)
        """
        return self._encryptor.encrypt(chunk)

    def encrypt_all(self, body: bytes) -> bytes:
        """
        Encrypt entire body at once.

        Handles compression (if enabled) and chunking automatically.
        This is the recommended method for most use cases.

        Uses memoryview for zero-copy slicing (~10-15% faster for large payloads).

        Args:
            body: Complete request body

        Returns:
            Encrypted body ready for transmission
        """
        # Compress whole body first (better ratio than per-chunk)
        if self._compress and len(body) >= ZSTD_MIN_SIZE:
            body = zstd_compress(body)
            self._was_compressed = True

        # Use memoryview for zero-copy slicing (avoids bytes copy on each slice)
        body_view = memoryview(body)
        body_len = len(body)

        # Chunk and encrypt - pass memoryview slices directly (no copy until needed)
        chunks = [
            self._encryptor.encrypt(body_view[offset : offset + CHUNK_SIZE])
            for offset in range(0, body_len, CHUNK_SIZE)
        ]

        # Handle empty body
        if not chunks:
            chunks.append(self._encryptor.encrypt(b""))

        return b"".join(chunks)

    @property
    def context(self) -> SenderContext:
        """Get HPKE context for response decryption."""
        return self._ctx


class ResponseDecryptor:
    """
    Decrypt standard (non-SSE) response.

    Handles header parsing, chunk boundary detection, and decryption.
    Supports streaming (feed), single-chunk, and all-at-once modes.

    Example (all-at-once):
        decryptor = ResponseDecryptor(response.headers, encryptor.context)
        plaintext = decryptor.decrypt_all(response.content)

    Example (streaming):
        decryptor = ResponseDecryptor(response.headers, encryptor.context)
        async for chunk in response.content:
            for plaintext in decryptor.feed(chunk):
                process(plaintext)
    """

    def __init__(
        self,
        headers: Mapping[str, Any],
        context: SenderContext,
    ) -> None:
        """
        Initialize response decryptor.

        Args:
            headers: Response headers (parses X-HPKE-Stream)
            context: SenderContext from RequestEncryptor.context

        Raises:
            DecryptionError: If X-HPKE-Stream header is missing
        """
        # Parse session salt from header
        stream_header = _get_header(headers, HEADER_HPKE_STREAM)
        if not stream_header:
            raise DecryptionError(f"Missing {HEADER_HPKE_STREAM} header")

        session_salt = _b64url_decode(stream_header)

        # Derive response key and create session
        response_key = context.export(RESPONSE_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
        session = StreamingSession(session_key=response_key, session_salt=session_salt)

        # Create chunk decryptor and stream parser
        self._decryptor = ChunkDecryptor(session, format=RawFormat())
        self._parser = _ChunkStreamParser()

    def decrypt(self, chunk: bytes) -> bytes:
        """
        Decrypt a single pre-parsed chunk.

        Use this when you've already extracted a complete chunk (with length prefix).
        For streaming where chunk boundaries are unknown, use feed() instead.

        Args:
            chunk: Complete encrypted chunk in wire format

        Returns:
            Decrypted plaintext
        """
        return self._decryptor.decrypt(chunk)

    def feed(self, data: bytes) -> Iterator[bytes]:
        """
        Feed raw data, yield decrypted chunks as boundaries are found.

        Handles partial chunks that span multiple feed() calls.
        Use this for streaming decryption where data arrives in arbitrary sizes.

        Args:
            data: Raw bytes from network/stream

        Yields:
            Decrypted plaintext chunks as they complete

        Example:
            async for raw_chunk in response.content:
                for plaintext in decryptor.feed(raw_chunk):
                    process(plaintext)
        """
        for chunk in self._parser.feed(data):
            yield self._decryptor.decrypt(chunk)

    def decrypt_all(self, body: bytes) -> bytes:
        """
        Decrypt entire response body at once.

        Convenience method that handles chunk boundary detection automatically.

        Args:
            body: Complete encrypted response body

        Returns:
            Decrypted plaintext
        """
        return b"".join(self.feed(body))


class SSEDecryptor:
    """
    Decrypt SSE stream event-by-event.

    Handles SSE-specific wire format (base64-encoded payloads).

    Example:
        decryptor = SSEDecryptor(response.headers, encryptor.context)
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                plaintext = decryptor.decrypt(line[6:])
    """

    def __init__(
        self,
        headers: Mapping[str, Any],
        context: SenderContext,
    ) -> None:
        """
        Initialize SSE decryptor.

        Args:
            headers: Response headers (parses X-HPKE-Stream)
            context: SenderContext from RequestEncryptor.context

        Raises:
            DecryptionError: If X-HPKE-Stream header is missing
        """
        # Parse session params from header
        stream_header = _get_header(headers, HEADER_HPKE_STREAM)
        if not stream_header:
            raise DecryptionError(f"Missing {HEADER_HPKE_STREAM} header")

        session_params = _b64url_decode(stream_header)

        # Derive SSE session key and create session
        session_key = context.export(SSE_SESSION_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
        session = StreamingSession.deserialize(session_params, session_key)

        # Create chunk decryptor with SSE format
        self._decryptor = ChunkDecryptor(session, format=SSEFormat())

    def decrypt(self, data: str | bytes) -> bytes:
        """
        Decrypt SSE data field.

        Args:
            data: Base64-encoded payload from SSE 'data: <payload>' field

        Returns:
            Decrypted plaintext (original SSE event content)
        """
        return self._decryptor.decrypt(data)


# =============================================================================
# SERVER SIDE
# =============================================================================


class RequestDecryptor:
    """
    Decrypt encrypted request body.

    Handles header parsing, HPKE context setup, chunking, and decompression.
    Supports streaming (feed), single-chunk, and all-at-once modes.

    Example (all-at-once):
        decryptor = RequestDecryptor(request.headers, private_key, psk, psk_id)
        plaintext = decryptor.decrypt_all(request.data)
        # Use decryptor.context for response encryption

    Example (streaming - ASGI):
        decryptor = RequestDecryptor(headers, private_key, psk, psk_id)
        async for message in receive():
            for plaintext in decryptor.feed(message["body"]):
                process(plaintext)
    """

    def __init__(
        self,
        headers: Mapping[str, Any],
        private_key: bytes,
        psk: bytes,
        psk_id: bytes,
    ) -> None:
        """
        Initialize request decryptor.

        Args:
            headers: Request headers (parses X-HPKE-Enc, X-HPKE-Stream)
            private_key: Server's X25519 private key (32 bytes)
            psk: Pre-shared key (must match client)
            psk_id: PSK identifier (must match client)

        Raises:
            DecryptionError: If required headers are missing
        """
        # Parse enc from header
        enc_header = _get_header(headers, HEADER_HPKE_ENC)
        if not enc_header:
            raise DecryptionError(f"Missing {HEADER_HPKE_ENC} header")
        enc = _b64url_decode(enc_header)

        # Parse session salt from header
        stream_header = _get_header(headers, HEADER_HPKE_STREAM)
        if not stream_header:
            raise DecryptionError(f"Missing {HEADER_HPKE_STREAM} header")
        session_salt = _b64url_decode(stream_header)

        # Check for compression
        encoding_header = _get_header(headers, HEADER_HPKE_ENCODING)
        self._is_compressed = encoding_header == "zstd"

        # Set up HPKE context
        self._ctx = setup_recipient_psk(enc=enc, sk_r=private_key, info=psk_id, psk=psk, psk_id=psk_id)

        # Derive request key and create session
        request_key = self._ctx.export(REQUEST_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
        session = StreamingSession(session_key=request_key, session_salt=session_salt)

        # Create chunk decryptor and stream parser
        self._decryptor = ChunkDecryptor(session, format=RawFormat())
        self._parser = _ChunkStreamParser()

    @property
    def is_compressed(self) -> bool:
        """Whether request body is compressed (X-HPKE-Encoding: zstd)."""
        return self._is_compressed

    def decrypt(self, chunk: bytes) -> bytes:
        """
        Decrypt a single pre-parsed chunk.

        Use this when you've already extracted a complete chunk (with length prefix).
        For streaming where chunk boundaries are unknown, use feed() instead.

        Note: Returns raw decrypted bytes. If is_compressed is True, the full
        reassembled body needs decompression (handled by decrypt_all).

        Args:
            chunk: Complete encrypted chunk in wire format

        Returns:
            Decrypted chunk (may still be compressed if X-HPKE-Encoding: zstd)
        """
        return self._decryptor.decrypt(chunk)

    def feed(self, data: bytes) -> Iterator[bytes]:
        """
        Feed raw data, yield decrypted chunks as boundaries are found.

        Handles partial chunks that span multiple feed() calls.
        Use this for streaming decryption where data arrives in arbitrary sizes.

        Note: Yields raw decrypted bytes. If is_compressed is True, you must
        collect all chunks and decompress the full body yourself, or use
        decrypt_all() instead.

        Args:
            data: Raw bytes from network/stream

        Yields:
            Decrypted chunks as they complete (may be compressed)

        Example (ASGI middleware):
            decryptor = RequestDecryptor(headers, sk, psk, psk_id)
            chunks = []
            async for message in receive():
                for chunk in decryptor.feed(message["body"]):
                    chunks.append(chunk)
            body = b"".join(chunks)
            if decryptor.is_compressed:
                body = zstd_decompress(body)
        """
        for chunk in self._parser.feed(data):
            yield self._decryptor.decrypt(chunk)

    def decrypt_all(self, body: bytes) -> bytes:
        """
        Decrypt entire request body at once.

        Handles chunk boundary detection and decompression automatically.

        Args:
            body: Complete encrypted request body

        Returns:
            Decrypted (and decompressed if needed) plaintext
        """
        plaintext = b"".join(self.feed(body))

        # Decompress if needed (whole-body compression)
        if self._is_compressed:
            plaintext = zstd_decompress(plaintext)

        return plaintext

    @property
    def context(self) -> RecipientContext:
        """Get HPKE context for response encryption."""
        return self._ctx


class ResponseEncryptor:
    """
    Encrypt standard (non-SSE) response.

    Handles header generation, chunking, and optional compression.
    Supports both streaming (chunk-by-chunk) and all-at-once modes.

    Example:
        encryptor = ResponseEncryptor(decryptor.context)
        encrypted = encryptor.encrypt_all(json.dumps(response_data).encode())
        return Response(encrypted, headers=encryptor.get_headers())
    """

    def __init__(
        self,
        context: RecipientContext,
        *,
        compress: bool = False,
    ) -> None:
        """
        Initialize response encryptor.

        Args:
            context: RecipientContext from RequestDecryptor.context
            compress: Enable per-chunk Zstd compression
        """
        # Derive response key and create session
        response_key = context.export(RESPONSE_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
        self._session = StreamingSession.create(response_key)

        # Create chunk encryptor
        self._encryptor = ChunkEncryptor(self._session, format=RawFormat(), compress=compress)

    def get_headers(self) -> dict[str, str]:
        """
        Get headers to send with the response.

        Returns:
            Dict with X-HPKE-Stream header
        """
        return {
            HEADER_HPKE_STREAM: _b64url_encode(self._session.session_salt),
        }

    def encrypt(self, chunk: bytes) -> bytes:
        """
        Encrypt a single chunk.

        For streaming mode, call this repeatedly for each chunk.

        Args:
            chunk: Raw chunk bytes

        Returns:
            Encrypted chunk in wire format
        """
        return self._encryptor.encrypt(chunk)

    def encrypt_all(self, body: bytes) -> bytes:
        """
        Encrypt entire response body at once.

        Handles chunking automatically.
        Uses memoryview for zero-copy slicing (~10-15% faster for large payloads).

        Args:
            body: Complete response body

        Returns:
            Encrypted body ready for transmission
        """
        # Use memoryview for zero-copy slicing (avoids bytes copy on each slice)
        body_view = memoryview(body)
        body_len = len(body)

        # Chunk and encrypt - pass memoryview slices directly (no copy until needed)
        chunks = [
            self._encryptor.encrypt(body_view[offset : offset + CHUNK_SIZE])
            for offset in range(0, body_len, CHUNK_SIZE)
        ]

        # Handle empty body
        if not chunks:
            chunks.append(self._encryptor.encrypt(b""))

        return b"".join(chunks)


class SSEEncryptor:
    """
    Encrypt SSE stream event-by-event.

    Produces SSE wire format: event: enc\\ndata: <base64>\\n\\n

    Example:
        encryptor = SSEEncryptor(decryptor.context)
        headers = encryptor.get_headers()
        for event in events:
            yield encryptor.encrypt(event)
    """

    def __init__(
        self,
        context: RecipientContext,
        *,
        compress: bool = False,
    ) -> None:
        """
        Initialize SSE encryptor.

        Args:
            context: RecipientContext from RequestDecryptor.context
            compress: Enable per-event Zstd compression
        """
        # Create SSE session from context
        self._session = create_session_from_context(context)

        # Create chunk encryptor with SSE format
        self._encryptor = ChunkEncryptor(self._session, format=SSEFormat(), compress=compress)

    def get_headers(self) -> dict[str, str]:
        """
        Get headers to send with the SSE response.

        Returns:
            Dict with X-HPKE-Stream and Content-Type headers
        """
        return {
            HEADER_HPKE_STREAM: _b64url_encode(self._session.serialize()),
            "Content-Type": "text/event-stream",
        }

    def encrypt(self, event: bytes) -> bytes:
        """
        Encrypt SSE event.

        Args:
            event: Raw SSE event content

        Returns:
            Encrypted SSE event: 'event: enc\\ndata: <base64>\\n\\n'
        """
        return self._encryptor.encrypt(event)

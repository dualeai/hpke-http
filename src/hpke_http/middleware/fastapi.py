"""
FastAPI/Starlette ASGI middleware for transparent HPKE encryption.

Provides:
- Automatic request body decryption
- Automatic SSE response encryption (transparent - no code changes needed)
- Built-in key discovery endpoint (/.well-known/hpke-keys)

Usage:
    from hpke_http.middleware.fastapi import HPKEMiddleware
    from starlette.responses import StreamingResponse

    app = FastAPI()
    app.add_middleware(
        HPKEMiddleware,
        private_keys={KemId.DHKEM_X25519_HKDF_SHA256: private_key_bytes},
        psk_resolver=get_api_key_from_request,
    )

    @app.post("/chat")
    async def chat(request: Request):
        data = await request.json()  # Decrypted by middleware

        async def generate():
            yield b"event: progress\\ndata: {}\\n\\n"
            yield b"event: done\\ndata: {}\\n\\n"

        # Just use StreamingResponse - encryption is automatic!
        return StreamingResponse(generate(), media_type="text/event-stream")

Reference: RFC-065 §4.3, §5.3
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives.asymmetric import x25519

from hpke_http._logging import get_logger
from hpke_http.constants import (
    AEAD_ID,
    CHUNK_SIZE,
    DISCOVERY_CACHE_MAX_AGE,
    DISCOVERY_PATH,
    GZIP_STREAMING_THRESHOLD,
    HEADER_HPKE_CONTENT_TYPE,
    HEADER_HPKE_ENC,
    HEADER_HPKE_ENCODING,
    HEADER_HPKE_STREAM,
    KDF_ID,
    SCOPE_HPKE_CONTEXT,
    SSE_MAX_EVENT_SIZE,
    ZSTD_DECOMPRESS_STREAMING_THRESHOLD,
    EncodingName,
    KemId,
    build_accept_encoding,
)
from hpke_http.core import (
    RequestDecryptor,
    ResponseEncryptor,
    SSEEncryptor,
    SSEEventParser,
    is_sse_response,
)
from hpke_http.exceptions import CryptoError, DecryptionError
from hpke_http.headers import b64url_encode
from hpke_http.streaming import gzip_decompress, zstd_decompress

__all__ = [
    "HPKEMiddleware",
]

_logger = get_logger(__name__)


@dataclass
class ResponseEncryptionState:
    """Per-request state for response encryption."""

    is_sse: bool = False
    """Whether the response is an SSE stream requiring encryption."""

    encrypt_response: bool = False
    """Whether standard (non-SSE) response should be encrypted."""

    sse_encryptor: SSEEncryptor | None = None
    """SSE encryptor instance."""

    response_encryptor: ResponseEncryptor | None = None
    """Standard response encryptor instance."""

    event_parser: SSEEventParser | None = None
    """SSE event parser for boundary detection (centralized from core.py)."""

    sse_buffer_size: int = 0
    """Track SSE buffer size for DoS protection."""

    body_buffer: bytearray = field(default_factory=bytearray)
    """Buffer for standard response body to enforce consistent chunk sizes."""

    headers_sent: bool = False
    """Whether response headers have been sent."""


@dataclass
class _DecryptionState:
    """Shared state for streaming request decryption closures."""

    decryptor: RequestDecryptor
    """Request decryptor for this request (has internal chunk buffer)."""

    http_done: bool = False
    """Whether all HTTP body data has been received."""

    body_returned: bool = False
    """Whether final body chunk (more_body=False) has been returned."""

    first_chunk_returned: bool = False
    """Whether pre-validated first chunk has been returned."""


# Type alias for PSK resolver callback
PSKResolver = Callable[[dict[str, Any]], Awaitable[tuple[bytes, bytes]]]
"""
Callback to resolve PSK and PSK ID from request scope.

Args:
    scope: ASGI scope dict

Returns:
    Tuple of (psk, psk_id) - typically (api_key, tenant_id)
"""


class HPKEMiddleware:
    """
    Pure ASGI middleware for transparent HPKE encryption.

    Features:
    - Decrypts request bodies encrypted with HPKE PSK mode
    - Auto-encrypts ALL responses when request was encrypted:
      - SSE responses: Uses SSEFormat (base64url in SSE events)
      - Standard responses: Uses RawFormat (binary length || counter || ciphertext)
    - Auto-registers /.well-known/hpke-keys discovery endpoint

    Response encryption is fully transparent - just use normal responses
    and encryption happens automatically when the request was encrypted.
    """

    def __init__(
        self,
        app: Any,
        private_keys: dict[KemId, bytes],
        psk_resolver: PSKResolver,
        discovery_path: str = DISCOVERY_PATH,
        max_sse_event_size: int = SSE_MAX_EVENT_SIZE,
        *,
        compress: bool = False,
        require_encryption: bool = False,
    ) -> None:
        """
        Initialize HPKE middleware.

        Args:
            app: ASGI application
            private_keys: Private keys by KEM ID (e.g., {KemId.DHKEM_X25519_HKDF_SHA256: sk})
            psk_resolver: Async callback to get (psk, psk_id) from request scope
            discovery_path: Path for key discovery endpoint
            max_sse_event_size: Maximum SSE event buffer size in bytes (default 64MB).
                This is a DoS protection for malformed events without proper \\n\\n boundaries.
                SSE is text-only; binary data must be base64-encoded (+33% overhead).
            compress: Enable Zstd compression for SSE responses (RFC 8878).
                When enabled, SSE chunks >= 64 bytes are compressed before encryption.
                Client must have backports.zstd installed (Python < 3.14).
            require_encryption: If True, reject plaintext requests with HTTP 426.
                Discovery endpoint is always allowed without encryption.
        """
        self.app = app
        self.private_keys = private_keys
        self.psk_resolver = psk_resolver
        self.discovery_path = discovery_path
        self.max_sse_event_size = max_sse_event_size
        self.compress = compress
        self.require_encryption = require_encryption

        # Check zstd availability for compression support
        # If zstd unavailable, gzip (stdlib) is used as fallback
        self._zstd_available = self._check_zstd_available()

        # Derive public keys for discovery endpoint
        self._public_keys: dict[KemId, bytes] = {}
        for kem_id, sk in private_keys.items():
            if kem_id == KemId.DHKEM_X25519_HKDF_SHA256:
                private_key = x25519.X25519PrivateKey.from_private_bytes(sk)
                self._public_keys[kem_id] = private_key.public_key().public_bytes_raw()

    @staticmethod
    def _check_zstd_available() -> bool:
        """Check if zstd decompression is available."""
        try:
            from hpke_http.streaming import import_zstd

            import_zstd()
            return True
        except ImportError:
            return False

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """ASGI interface."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Handle discovery endpoint
        path = scope.get("path", "")
        method = scope.get("method", "")
        if path == self.discovery_path:
            _logger.debug("Discovery endpoint requested: path=%s", path)
            await self._handle_discovery(scope, receive, send)
            return

        # Check for HPKE encryption header
        headers = dict(scope.get("headers", []))
        enc_header = headers.get(HEADER_HPKE_ENC.lower().encode())

        if not enc_header:
            # Not encrypted - reject or pass through based on configuration
            if self.require_encryption:
                _logger.debug("Rejected plaintext request: method=%s path=%s", method, path)
                await self._send_error(send, 426, "Encryption required")
                return
            _logger.debug("Unencrypted request: method=%s path=%s", method, path)
            await self.app(scope, receive, send)
            return

        _logger.debug("Encrypted request received: method=%s path=%s", method, path)

        # Early rejection for unsupported encoding (RFC 9110 §12.5.3)
        # Check before attempting decryption to fail fast with clear error
        encoding_header = headers.get(HEADER_HPKE_ENCODING.lower().encode())
        if encoding_header == b"zstd" and not self._zstd_available:
            _logger.debug(
                "Rejected unsupported encoding: method=%s path=%s encoding=%s",
                method,
                path,
                encoding_header,
            )
            await self._send_error(
                send,
                415,
                "Unsupported encoding: zstd",
                extra_headers=[(b"accept-encoding", build_accept_encoding(zstd_available=False).encode())],
            )
            return

        # Decrypt request AND wrap send for response encryption
        # Track if response has started so we know if we can send error responses
        response_started = False

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            decrypted_receive = await self._create_decrypted_receive(scope, receive, enc_header)
            encrypting_send = self._create_encrypting_send(scope, tracked_send)
            await self.app(scope, decrypted_receive, encrypting_send)
        except CryptoError as e:
            # Don't expose internal error details to clients
            _logger.debug("Decryption failed: method=%s path=%s error_type=%s", method, path, type(e).__name__)
            if response_started:
                # Response already started, can't send error - re-raise to close connection
                raise
            await self._send_error(send, 400, "Request decryption failed")

    def _create_encrypting_send(  # noqa: PLR0915
        self,
        scope: dict[str, Any],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> Callable[[dict[str, Any]], Awaitable[None]]:
        """Create send wrapper that auto-encrypts responses."""
        # Per-request state (closure)
        state = ResponseEncryptionState()

        async def encrypting_send(message: dict[str, Any]) -> None:
            msg_type = message["type"]

            if msg_type == "http.response.start":
                await _handle_response_start(message)
            elif msg_type == "http.response.body":
                await _handle_response_body(message)
            else:
                await send(message)

        async def _handle_response_start(message: dict[str, Any]) -> None:
            """Handle response start - detect response type and set up encryption."""
            headers = message.get("headers", [])
            # Convert ASGI headers (list of byte tuples) to dict for is_sse_response()
            headers_dict = {n.decode("latin-1"): v.decode("latin-1") for n, v in headers}

            ctx = scope.get(SCOPE_HPKE_CONTEXT)
            if ctx and is_sse_response(headers_dict):
                # SSE response - use SSEEncryptor class and centralized event parser
                state.is_sse = True
                state.sse_encryptor = SSEEncryptor(ctx, compress=self.compress)
                state.event_parser = SSEEventParser()
                _logger.debug(
                    "SSE encryption enabled: path=%s compress=%s",
                    scope.get("path", ""),
                    self.compress,
                )

                # Add headers from encryptor (X-HPKE-Stream, Content-Type already set)
                crypto_headers = state.sse_encryptor.get_headers()
                new_headers = [
                    *headers,
                    (HEADER_HPKE_STREAM.encode(), crypto_headers[HEADER_HPKE_STREAM].encode()),
                ]
                message = {**message, "headers": new_headers}
                await send(message)

            elif ctx:
                # Standard response - use ResponseEncryptor class
                state.encrypt_response = True
                state.response_encryptor = ResponseEncryptor(ctx, compress=self.compress)
                _logger.debug(
                    "Response encryption enabled: path=%s compress=%s",
                    scope.get("path", ""),
                    self.compress,
                )

                # Modify headers: remove Content-Length, add X-HPKE-Stream
                # Client detects standard vs SSE via Content-Type (standard HTTP)
                crypto_headers = state.response_encryptor.get_headers()
                new_headers = [
                    (n, v)
                    for n, v in headers
                    if n.lower() not in (b"content-length",)  # Remove - size changes
                ]
                new_headers.append((HEADER_HPKE_STREAM.encode(), crypto_headers[HEADER_HPKE_STREAM].encode()))
                message = {**message, "headers": new_headers}
                state.headers_sent = True
                await send(message)

            else:
                # No encryption context, pass through
                await send(message)

        async def _handle_response_body(message: dict[str, Any]) -> None:
            """Handle response body - encrypt SSE events or standard response."""
            if state.is_sse:
                # SSE path - buffer events and encrypt
                await _handle_sse_body(message)
            elif state.encrypt_response:
                # Standard response - encrypt chunks directly
                await _handle_standard_body(message)
            else:
                # No encryption, pass through
                await send(message)

        async def _handle_sse_body(message: dict[str, Any]) -> None:
            """Handle SSE response body - parse events and encrypt."""
            body: bytes = message.get("body", b"")
            more_body = message.get("more_body", False)
            encryptor = state.sse_encryptor
            parser = state.event_parser
            if encryptor is None or parser is None:
                raise CryptoError("SSE encryption state corrupted")

            sent_any = False

            if body:
                # DoS protection: track buffer size
                state.sse_buffer_size += len(body)
                if state.sse_buffer_size > self.max_sse_event_size:
                    # Reset size tracking after overflow handling
                    state.sse_buffer_size = 0

                # Feed to parser, encrypt complete events
                for event in parser.feed(body):
                    encrypted = encryptor.encrypt(event)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": encrypted,
                            "more_body": True,
                        }
                    )
                    sent_any = True
                    # Reset size tracking after each event
                    state.sse_buffer_size = 0

            # Handle end of stream
            if not more_body:
                # Flush any remaining partial event
                remaining = parser.flush()
                if remaining:
                    encrypted = encryptor.encrypt(remaining)
                    await send(
                        {
                            "type": "http.response.body",
                            "body": encrypted,
                            "more_body": False,
                        }
                    )
                elif not sent_any:
                    # Send empty final body if nothing was sent
                    await send(
                        {
                            "type": "http.response.body",
                            "body": b"",
                            "more_body": False,
                        }
                    )

        async def _handle_standard_body(message: dict[str, Any]) -> None:
            """Handle standard response body - buffer and emit fixed-size chunks."""
            body: bytes = message.get("body", b"")
            more_body = message.get("more_body", False)
            encryptor = state.response_encryptor
            if encryptor is None:
                raise CryptoError("Response encryption state corrupted: response_encryptor is None")

            # Buffer incoming body
            state.body_buffer.extend(body)

            # Emit full chunks (CHUNK_SIZE bytes each) using offset tracking
            # O(1) per chunk, single O(n) compaction at end instead of O(n) per chunk
            consumed = 0
            while len(state.body_buffer) - consumed >= CHUNK_SIZE:
                chunk = bytes(state.body_buffer[consumed : consumed + CHUNK_SIZE])
                consumed += CHUNK_SIZE
                encrypted = encryptor.encrypt(chunk)
                await send(
                    {
                        "type": "http.response.body",
                        "body": encrypted,
                        "more_body": True,
                    }
                )
            # Single compaction after emitting all full chunks
            if consumed:
                del state.body_buffer[:consumed]

            # Final chunk (when no more body coming)
            if not more_body:
                # Emit remaining buffer (may be smaller than CHUNK_SIZE)
                encrypted = encryptor.encrypt(bytes(state.body_buffer))
                state.body_buffer.clear()
                await send(
                    {
                        "type": "http.response.body",
                        "body": encrypted,
                        "more_body": False,
                    }
                )

        return encrypting_send

    async def _handle_discovery(
        self,
        _scope: dict[str, Any],
        _receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Handle /.well-known/hpke-keys endpoint."""
        # Build response
        keys = [
            {
                "kem_id": f"0x{kem_id:04x}",
                "kdf_id": f"0x{KDF_ID:04x}",
                "aead_id": f"0x{AEAD_ID:04x}",
                "public_key": b64url_encode(pk),
            }
            for kem_id, pk in self._public_keys.items()
        ]

        response = {
            "version": 1,
            "keys": keys,
            "default_suite": {
                "kem_id": f"0x{KemId.DHKEM_X25519_HKDF_SHA256:04x}",
                "kdf_id": f"0x{KDF_ID:04x}",
                "aead_id": f"0x{AEAD_ID:04x}",
            },
        }

        # NOTE: json.dumps().encode() creates 2 allocations (str → bytes).
        # For higher throughput, consider orjson.dumps() which returns bytes
        # directly (1 allocation, ~10x faster). Not included as dependency
        # to keep the library lightweight.
        body = json.dumps(response).encode()

        # Advertise supported request encodings (RFC 9110 §12.5.3)
        accept_encoding = build_accept_encoding(zstd_available=self._zstd_available).encode()

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"cache-control", f"public, max-age={DISCOVERY_CACHE_MAX_AGE}".encode()),
                    (b"accept-encoding", accept_encoding),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )

    async def _setup_decryption(
        self,
        scope: dict[str, Any],
        enc_header: bytes,
        stream_header: bytes,
        encoding_header: bytes | None,
    ) -> RequestDecryptor:
        """
        Set up HPKE decryption context and return request decryptor.

        Resolves PSK asynchronously, creates RequestDecryptor with headers,
        and stores context in scope for response encryption.
        """
        # Get PSK from resolver (async)
        try:
            psk, psk_id = await self.psk_resolver(scope)
        except Exception as e:
            raise DecryptionError(f"PSK resolution failed: {e}") from e

        # Get private key for the KEM (default X25519)
        kem_id = KemId.DHKEM_X25519_HKDF_SHA256
        if kem_id not in self.private_keys:
            raise DecryptionError(f"Unsupported KEM: 0x{kem_id:04x}")
        sk_r = self.private_keys[kem_id]

        # Build headers dict for RequestDecryptor
        request_headers: dict[str, str] = {
            HEADER_HPKE_ENC: enc_header.decode("ascii"),
            HEADER_HPKE_STREAM: stream_header.decode("ascii"),
        }
        if encoding_header:
            request_headers[HEADER_HPKE_ENCODING] = encoding_header.decode("ascii")

        # Create request decryptor (handles HPKE setup, key derivation, chunk parsing)
        try:
            decryptor = RequestDecryptor(request_headers, sk_r, psk, psk_id)
        except Exception as e:
            raise DecryptionError(f"Decryption setup failed: {e}") from e

        # Store context in scope for response encryption
        scope[SCOPE_HPKE_CONTEXT] = decryptor.context

        _logger.debug(
            "Request decryption context created: path=%s kem_id=0x%04x compressed=%s",
            scope.get("path", ""),
            kem_id,
            decryptor.is_compressed,
        )

        return decryptor

    async def _read_first_chunk(
        self,
        state: _DecryptionState,
        receive: Callable[[], Awaitable[dict[str, Any]]],
    ) -> bytes:
        """
        Read and decrypt first chunk to validate PSK/key before app starts.

        This ensures decryption errors return 400 (Bad Request) instead of
        500 (Server Error). Returns the decrypted first chunk.
        """
        while True:
            # Need more data from HTTP layer
            message = await receive()
            if message["type"] == "http.disconnect":
                raise DecryptionError("Client disconnected during request validation")

            body = message.get("body", b"")
            more_body = message.get("more_body", False)

            if not more_body:
                state.http_done = True

            # Feed data to decryptor, return first chunk when available
            for plaintext in state.decryptor.feed(body):
                return plaintext

            # No complete chunk yet
            if state.http_done and not body:
                return b""  # Empty body

    async def _decrypt_all_compressed(
        self,
        state: _DecryptionState,
        receive: Callable[[], Awaitable[dict[str, Any]]],
        first_plaintext: bytes,
    ) -> bytes:
        """
        Read and decrypt all chunks for compressed request, then decompress.

        Returns the full decompressed body. Must buffer all data because
        client compresses full body before chunking.
        """
        parts: list[bytes] = [first_plaintext] if first_plaintext else []

        # Read and decrypt all remaining chunks using decryptor.feed()
        while not state.http_done:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise DecryptionError("Client disconnected during request")

            body = message.get("body", b"")
            if not message.get("more_body", False):
                state.http_done = True

            # Feed data to decryptor, collect all decrypted chunks
            parts.extend(state.decryptor.feed(body))

        # Decompress full body using streaming for memory efficiency
        compressed_body = b"".join(parts)
        encoding = state.decryptor.encoding
        try:
            match encoding:
                case EncodingName.ZSTD:
                    decompressed = zstd_decompress(
                        compressed_body,
                        streaming_threshold=ZSTD_DECOMPRESS_STREAMING_THRESHOLD,
                    )
                case EncodingName.GZIP:
                    decompressed = gzip_decompress(
                        compressed_body,
                        streaming_threshold=GZIP_STREAMING_THRESHOLD,
                    )
                case _:
                    raise DecryptionError(f"Unknown encoding: {encoding}")
            _logger.debug(
                "Request decompressed: encoding=%s compressed=%d decompressed=%d",
                encoding,
                len(compressed_body),
                len(decompressed),
            )
            return decompressed
        except DecryptionError:
            raise
        except Exception as e:
            raise DecryptionError(f"{encoding} decompression failed: {e}") from e

    async def _create_decrypted_receive(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        enc_header: bytes,
    ) -> Callable[[], Awaitable[dict[str, Any]]]:
        """
        Create a receive wrapper that decrypts chunked request body.

        Uses streaming decryption - reads chunks from HTTP layer and decrypts
        on-demand. Memory usage is O(chunk_size) regardless of body size.

        Wire format: [length(4B BE)] [counter(4B BE)] [ciphertext(N + 16B tag)]
        """
        headers = dict(scope.get("headers", []))

        # Get session salt from X-HPKE-Stream header (required for chunked format)
        stream_header = headers.get(HEADER_HPKE_STREAM.lower().encode())
        if not stream_header:
            raise DecryptionError(f"Missing {HEADER_HPKE_STREAM} header")

        # Get encoding header for compression detection
        encoding_header = headers.get(HEADER_HPKE_ENCODING.lower().encode())

        # Set up decryption context (resolves PSK, creates RequestDecryptor)
        decryptor = await self._setup_decryption(scope, enc_header, stream_header, encoding_header)

        # Restore original Content-Type for multipart parsing
        # Client sends original Content-Type (with boundary) via X-HPKE-Content-Type header
        original_ct = headers.get(HEADER_HPKE_CONTENT_TYPE.lower().encode())
        if original_ct:
            # Rebuild scope headers with restored Content-Type
            # ASGI headers are list of (name, value) byte tuples, lowercase names
            new_headers = [
                (b"content-type", original_ct) if name == b"content-type" else (name, value)
                for name, value in scope.get("headers", [])
            ]
            scope["headers"] = new_headers
            _logger.debug("Content-Type restored: %s", original_ct.decode())

        # Initialize shared state for streaming decryption
        state = _DecryptionState(decryptor=decryptor)

        _logger.debug(
            "Request decryption started: path=%s compress=%s",
            scope.get("path", ""),
            decryptor.is_compressed,
        )

        # Early validation: read and decrypt first chunk to validate PSK/key
        first_plaintext = await self._read_first_chunk(state, receive)

        # Handle compressed requests: buffer all chunks and decompress
        if decryptor.is_compressed:
            decompressed_body = await self._decrypt_all_compressed(state, receive, first_plaintext)
            body_returned_compressed = False

            async def decrypted_receive_compressed() -> dict[str, Any]:
                nonlocal body_returned_compressed
                if not body_returned_compressed:
                    body_returned_compressed = True
                    return {"type": "http.request", "body": decompressed_body, "more_body": False}
                return await receive()

            return decrypted_receive_compressed

        # Non-compressed: stream chunks directly using shared state
        return self._create_streaming_receive(state, receive, first_plaintext)

    def _create_streaming_receive(
        self,
        state: _DecryptionState,
        receive: Callable[[], Awaitable[dict[str, Any]]],
        first_plaintext: bytes,
    ) -> Callable[[], Awaitable[dict[str, Any]]]:
        """Create receive function for non-compressed streaming decryption."""
        # Pending chunks from previous feed() calls
        pending_chunks: list[bytes] = []

        async def decrypted_receive() -> dict[str, Any]:
            # After returning more_body=False, wait for disconnect
            if state.body_returned:
                return await receive()

            # Return pre-validated first chunk on first call
            if not state.first_chunk_returned:
                state.first_chunk_returned = True
                # Check if there's more data expected
                has_pending = len(pending_chunks) > 0
                more_body = not state.http_done or has_pending
                if not more_body:
                    state.body_returned = True
                return {"type": "http.request", "body": first_plaintext, "more_body": more_body}

            while True:
                # Return pending chunks first
                if pending_chunks:
                    plaintext = pending_chunks.pop(0)
                    has_more_pending = len(pending_chunks) > 0
                    more_body = not state.http_done or has_more_pending
                    if not more_body:
                        state.body_returned = True
                    return {"type": "http.request", "body": plaintext, "more_body": more_body}

                # Need more data from HTTP layer
                if state.http_done:
                    state.body_returned = True
                    return {"type": "http.request", "body": b"", "more_body": False}

                # Fetch more from underlying receive
                message = await receive()
                if message["type"] == "http.disconnect":
                    raise DecryptionError("Client disconnected during chunked request")

                body = message.get("body", b"")
                if not message.get("more_body", False):
                    state.http_done = True

                # Feed data to decryptor, collect all decrypted chunks
                pending_chunks.extend(state.decryptor.feed(body))

        return decrypted_receive

    async def _send_error(
        self,
        send: Callable[[dict[str, Any]], Awaitable[None]],
        status: int,
        message: str,
        extra_headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        """Send an error response.

        Args:
            send: ASGI send callable
            status: HTTP status code
            message: Error message for JSON body
            extra_headers: Additional headers to include (e.g., Accept-Encoding for 415)
        """
        body = json.dumps({"error": message}).encode()
        headers: list[tuple[bytes, bytes]] = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        if extra_headers:
            headers.extend(extra_headers)
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": headers,
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )

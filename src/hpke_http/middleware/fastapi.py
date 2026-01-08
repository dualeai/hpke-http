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
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives.asymmetric import x25519

from hpke_http._logging import get_logger
from hpke_http.constants import (
    AEAD_ID,
    CHACHA20_POLY1305_KEY_SIZE,
    CHUNK_SIZE,
    DISCOVERY_CACHE_MAX_AGE,
    DISCOVERY_PATH,
    HEADER_HPKE_ENC,
    HEADER_HPKE_ENCODING,
    HEADER_HPKE_STREAM,
    KDF_ID,
    REQUEST_KEY_LABEL,
    RESPONSE_KEY_LABEL,
    SCOPE_HPKE_CONTEXT,
    SSE_MAX_EVENT_SIZE,
    KemId,
)
from hpke_http.exceptions import CryptoError, DecryptionError
from hpke_http.headers import b64url_decode, b64url_encode
from hpke_http.hpke import setup_recipient_psk
from hpke_http.streaming import (
    ChunkDecryptor,
    ChunkEncryptor,
    RawFormat,
    StreamingSession,
    create_session_from_context,
    zstd_decompress,
)

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

    encryptor: ChunkEncryptor | None = None
    """Chunk encryptor instance (for both SSE and standard responses)."""

    buffer: bytearray = field(default_factory=bytearray)
    """Buffer for incomplete SSE events awaiting boundary detection (zero-copy)."""

    body_buffer: bytearray = field(default_factory=bytearray)
    """Buffer for standard response body to enforce consistent chunk sizes."""

    headers_sent: bool = False
    """Whether response headers have been sent."""


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
        """
        self.app = app
        self.private_keys = private_keys
        self.psk_resolver = psk_resolver
        self.discovery_path = discovery_path
        self.max_sse_event_size = max_sse_event_size
        self.compress = compress

        # Derive public keys for discovery endpoint
        self._public_keys: dict[KemId, bytes] = {}
        for kem_id, sk in private_keys.items():
            if kem_id == KemId.DHKEM_X25519_HKDF_SHA256:
                private_key = x25519.X25519PrivateKey.from_private_bytes(sk)
                self._public_keys[kem_id] = private_key.public_key().public_bytes_raw()

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
            # Not encrypted, pass through
            _logger.debug("Unencrypted request: method=%s path=%s", method, path)
            await self.app(scope, receive, send)
            return

        _logger.debug("Encrypted request received: method=%s path=%s", method, path)

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

        # WHATWG-compliant event boundary: two consecutive line endings (bytes pattern)
        # Handles \n\n, \r\r, \r\n\r\n, and mixed combinations
        event_boundary = re.compile(rb"(?:\r\n|\r(?!\n)|\n)(?:\r\n|\r(?!\n)|\n)")

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
            content_type = next(
                (v for n, v in headers if n.lower() == b"content-type"),
                None,
            )

            ctx = scope.get(SCOPE_HPKE_CONTEXT)
            if ctx and content_type and b"text/event-stream" in content_type:
                # SSE response - use SSEFormat (default)
                state.is_sse = True
                session = create_session_from_context(ctx)
                state.encryptor = ChunkEncryptor(session, compress=self.compress)
                _logger.debug(
                    "SSE encryption enabled: path=%s compress=%s",
                    scope.get("path", ""),
                    self.compress,
                )

                # Add X-HPKE-Stream header
                session_params = session.serialize()
                new_headers = [
                    *headers,
                    (HEADER_HPKE_STREAM.encode(), b64url_encode(session_params).encode()),
                ]
                message = {**message, "headers": new_headers}
                await send(message)

            elif ctx:
                # Standard response - use RawFormat
                state.encrypt_response = True
                # Derive key with response-specific label, create session with random salt
                response_key = ctx.export(RESPONSE_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
                session = StreamingSession.create(response_key)
                state.encryptor = ChunkEncryptor(session, format=RawFormat(), compress=self.compress)
                _logger.debug(
                    "Response encryption enabled: path=%s compress=%s",
                    scope.get("path", ""),
                    self.compress,
                )

                # Modify headers: remove Content-Length, add X-HPKE-Stream
                # Client detects standard vs SSE via Content-Type (standard HTTP)
                new_headers = [
                    (n, v)
                    for n, v in headers
                    if n.lower() not in (b"content-length",)  # Remove - size changes
                ]
                new_headers.append((HEADER_HPKE_STREAM.encode(), b64url_encode(session.session_salt).encode()))
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
            """Handle SSE response body - buffer and encrypt events."""
            body: bytes = message.get("body", b"")
            more_body = message.get("more_body", False)
            encryptor = state.encryptor
            if encryptor is None:
                raise CryptoError("SSE encryption state corrupted: encryptor is None")

            # Add to buffer (zero-copy extend)
            if body:
                # Enforce buffer size limit to prevent DoS
                if len(state.buffer) + len(body) > self.max_sse_event_size:
                    # Force flush oversized buffer as partial event
                    if state.buffer:
                        encrypted = encryptor.encrypt(bytes(state.buffer))
                        await send(
                            {
                                "type": "http.response.body",
                                "body": encrypted,
                                "more_body": True,
                            }
                        )
                        state.buffer.clear()
                    # Keep tail that fits
                    state.buffer.extend(body[-self.max_sse_event_size :])
                else:
                    state.buffer.extend(body)

            # Extract and encrypt complete events
            sent_any = await _extract_and_send_events(encryptor, more_body=more_body)

            # Handle end of stream
            if not more_body:
                await _handle_end_of_stream(encryptor, sent_any=sent_any)

        async def _handle_standard_body(message: dict[str, Any]) -> None:
            """Handle standard response body - buffer and emit fixed-size chunks."""
            body: bytes = message.get("body", b"")
            more_body = message.get("more_body", False)
            encryptor = state.encryptor
            if encryptor is None:
                raise CryptoError("Response encryption state corrupted: encryptor is None")

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

        async def _extract_and_send_events(encryptor: ChunkEncryptor, *, more_body: bool) -> bool:
            """Extract complete events from buffer and send encrypted.

            Uses offset tracking with single compaction for O(1) per event
            instead of O(n) per event from del buffer[:n].
            """
            sent_any = False
            consumed = 0
            while True:
                # Search starting from consumed position (Pattern.search supports pos arg)
                match = event_boundary.search(state.buffer, pos=consumed)
                if not match:
                    break

                # Extract complete event (including boundary)
                event_end = match.end()
                chunk = bytes(state.buffer[consumed:event_end])
                consumed = event_end

                # Send with more_body=False only if final message AND buffer empty after compaction
                remaining_after = len(state.buffer) - consumed
                is_final = not more_body and remaining_after == 0
                encrypted = encryptor.encrypt(chunk)
                await send(
                    {
                        "type": "http.response.body",
                        "body": encrypted,
                        "more_body": not is_final,
                    }
                )
                sent_any = True

            # Single compaction after extracting all events
            if consumed:
                del state.buffer[:consumed]
            return sent_any

        async def _handle_end_of_stream(encryptor: ChunkEncryptor, *, sent_any: bool) -> None:
            """Handle end of stream - flush buffer or send empty body."""
            if state.buffer:
                # Flush remaining buffer (partial event)
                encrypted = encryptor.encrypt(bytes(state.buffer))
                await send(
                    {
                        "type": "http.response.body",
                        "body": encrypted,
                        "more_body": False,
                    }
                )
            elif not sent_any:
                # Only send final empty body if we didn't send anything this round
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"",
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

        body = json.dumps(response).encode()

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"cache-control", f"public, max-age={DISCOVERY_CACHE_MAX_AGE}".encode()),
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

        # Decode encapsulated key and session salt
        try:
            enc = bytes(b64url_decode(enc_header.decode("ascii")))
            session_salt = bytes(b64url_decode(stream_header.decode("ascii")))
        except Exception as e:
            raise DecryptionError("Invalid header encoding") from e

        # Get PSK from resolver
        try:
            psk, psk_id = await self.psk_resolver(scope)
        except Exception as e:
            raise DecryptionError(f"PSK resolution failed: {e}") from e

        # Get private key for the KEM (default X25519)
        kem_id = KemId.DHKEM_X25519_HKDF_SHA256
        if kem_id not in self.private_keys:
            raise DecryptionError(f"Unsupported KEM: 0x{kem_id:04x}")
        sk_r = self.private_keys[kem_id]

        # Set up HPKE context
        ctx = setup_recipient_psk(
            enc=enc,
            sk_r=sk_r,
            info=psk_id,
            psk=psk,
            psk_id=psk_id,
        )

        # Derive request key from HPKE context
        request_key = ctx.export(REQUEST_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
        session = StreamingSession(session_key=request_key, session_salt=session_salt)
        decryptor = ChunkDecryptor(session, format=RawFormat())

        # Store context in scope for response encryption
        scope[SCOPE_HPKE_CONTEXT] = ctx

        # Check for compression (body-level, not chunk-level)
        encoding_header = headers.get(HEADER_HPKE_ENCODING.lower().encode())
        is_compressed = encoding_header == b"zstd"

        # State for streaming decryption
        buffer = bytearray()
        http_done = False

        _logger.debug(
            "Request decryption started: path=%s kem_id=0x%04x compress=%s",
            scope.get("path", ""),
            kem_id,
            is_compressed,
        )

        # Early validation: read and decrypt first chunk to validate PSK/key before app starts.
        # This ensures decryption errors return 400 (Bad Request) instead of 500 (Server Error).
        # Buffer data until we have a complete first encrypted chunk.
        first_plaintext: bytes | None = None
        while True:
            if len(buffer) >= 4:
                chunk_len = int.from_bytes(buffer[:4], "big")
                total_size = 4 + chunk_len
                if len(buffer) >= total_size:
                    # Have complete first chunk - decrypt to validate key
                    chunk_data = bytes(buffer[:total_size])
                    del buffer[:total_size]
                    first_plaintext = decryptor.decrypt(chunk_data)  # Raises if PSK wrong
                    break

            # Need more data
            message = await receive()
            if message["type"] == "http.disconnect":
                raise DecryptionError("Client disconnected during request validation")
            buffer.extend(message.get("body", b""))
            if not message.get("more_body", False):
                http_done = True
                if len(buffer) == 0:
                    # Empty body - nothing to validate
                    first_plaintext = b""
                    break
                # Still need to decrypt whatever we have
                continue

        # For compressed requests, we must buffer all chunks and decompress at end
        # (client compresses full body before chunking)
        # Do all decryption and decompression during setup to catch errors early (return 400)
        if is_compressed:
            # Start with pre-validated first chunk (if any)
            parts: list[bytes] = [first_plaintext] if first_plaintext else []

            # Read and decrypt all remaining chunks
            while True:
                # Try to extract complete chunks from buffer using offset tracking
                # O(1) per chunk, single O(n) compaction instead of O(n) per chunk
                consumed = 0
                while consumed + 4 <= len(buffer):
                    chunk_len = int.from_bytes(buffer[consumed : consumed + 4], "big")
                    total_size = 4 + chunk_len
                    if consumed + total_size > len(buffer):
                        break
                    chunk_data = bytes(buffer[consumed : consumed + total_size])
                    consumed += total_size
                    parts.append(decryptor.decrypt(chunk_data))

                # Single compaction after extracting all available chunks
                if consumed:
                    del buffer[:consumed]

                # Need more data?
                if http_done:
                    break

                message = await receive()
                if message["type"] == "http.disconnect":
                    raise DecryptionError("Client disconnected during request")
                buffer.extend(message.get("body", b""))
                if not message.get("more_body", False):
                    http_done = True

            # Decompress full body (raises DecryptionError if invalid, caught by __call__)
            compressed_body = b"".join(parts)
            try:
                decompressed_body = zstd_decompress(compressed_body)
                _logger.debug(
                    "Request decompressed: compressed=%d decompressed=%d",
                    len(compressed_body),
                    len(decompressed_body),
                )
            except Exception as e:
                raise DecryptionError(f"Zstd decompression failed: {e}") from e

            # Return decompressed body and wait for disconnect
            body_returned_compressed = False

            async def decrypted_receive_compressed() -> dict[str, Any]:
                nonlocal body_returned_compressed

                if not body_returned_compressed:
                    body_returned_compressed = True
                    return {"type": "http.request", "body": decompressed_body, "more_body": False}
                # Subsequent calls: wait for disconnect
                return await receive()

            return decrypted_receive_compressed

        # Non-compressed: stream chunks directly
        body_returned = False  # Track if we've returned the final body
        first_chunk_returned = False  # Track if we've returned the pre-validated first chunk

        async def decrypted_receive() -> dict[str, Any]:
            nonlocal http_done, body_returned, first_chunk_returned

            # After returning more_body=False, wait for disconnect
            if body_returned:
                return await receive()

            # Return pre-validated first chunk on first call
            # (first_plaintext is always assigned before the validation loop exits)
            if not first_chunk_returned:
                first_chunk_returned = True
                more_body = not (http_done and len(buffer) == 0)
                if not more_body:
                    body_returned = True
                return {"type": "http.request", "body": first_plaintext, "more_body": more_body}

            while True:
                # Try to extract complete chunk from buffer
                if len(buffer) >= 4:
                    chunk_len = int.from_bytes(buffer[:4], "big")
                    total_size = 4 + chunk_len

                    if len(buffer) >= total_size:
                        # Extract and decrypt chunk
                        chunk_data = bytes(buffer[:total_size])
                        del buffer[:total_size]
                        plaintext = decryptor.decrypt(chunk_data)

                        more_body = not (http_done and len(buffer) == 0)
                        if not more_body:
                            body_returned = True
                        return {"type": "http.request", "body": plaintext, "more_body": more_body}

                # Need more data from HTTP layer
                if http_done:
                    # No more HTTP data available
                    body_returned = True
                    return {"type": "http.request", "body": b"", "more_body": False}

                # Fetch more from underlying receive
                message = await receive()
                if message["type"] == "http.disconnect":
                    raise DecryptionError("Client disconnected during chunked request")

                buffer.extend(message.get("body", b""))
                if not message.get("more_body", False):
                    http_done = True

        return decrypted_receive

    async def _send_error(
        self,
        send: Callable[[dict[str, Any]], Awaitable[None]],
        status: int,
        message: str,
    ) -> None:
        """Send an error response."""
        body = json.dumps({"error": message}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
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

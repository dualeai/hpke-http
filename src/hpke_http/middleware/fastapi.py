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

from hpke_http.constants import (
    AEAD_ID,
    DISCOVERY_CACHE_MAX_AGE,
    DISCOVERY_PATH,
    HEADER_HPKE_ENC,
    HEADER_HPKE_STREAM,
    KDF_ID,
    SCOPE_HPKE_CONTEXT,
    SSE_MAX_EVENT_SIZE,
    KemId,
)
from hpke_http.envelope import decode_envelope
from hpke_http.exceptions import CryptoError, DecryptionError, EnvelopeError
from hpke_http.headers import b64url_decode, b64url_encode
from hpke_http.hpke import setup_recipient_psk
from hpke_http.streaming import SSEEncryptor, create_session_from_context

__all__ = [
    "HPKEMiddleware",
]


@dataclass
class SSEEncryptionState:
    """Per-request state for SSE response encryption."""

    is_sse: bool = False
    """Whether the response is an SSE stream requiring encryption."""

    encryptor: SSEEncryptor | None = None
    """SSE encryptor instance, set when SSE response detected."""

    buffer: str = field(default="")
    """Buffer for incomplete SSE events awaiting boundary detection."""


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
    - Auto-encrypts SSE responses when request was encrypted
    - Auto-registers /.well-known/hpke-keys discovery endpoint

    SSE encryption is fully transparent - just use normal StreamingResponse
    with media_type="text/event-stream" and encryption happens automatically.
    """

    def __init__(
        self,
        app: Any,
        private_keys: dict[KemId, bytes],
        psk_resolver: PSKResolver,
        discovery_path: str = DISCOVERY_PATH,
        max_sse_event_size: int = SSE_MAX_EVENT_SIZE,
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
        """
        self.app = app
        self.private_keys = private_keys
        self.psk_resolver = psk_resolver
        self.discovery_path = discovery_path
        self.max_sse_event_size = max_sse_event_size

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
        if path == self.discovery_path:
            await self._handle_discovery(scope, receive, send)
            return

        # Check for HPKE encryption header
        headers = dict(scope.get("headers", []))
        enc_header = headers.get(HEADER_HPKE_ENC.lower().encode())

        if not enc_header:
            # Not encrypted, pass through
            await self.app(scope, receive, send)
            return

        # Decrypt request AND wrap send for response encryption
        try:
            decrypted_receive = await self._create_decrypted_receive(scope, receive, enc_header)
            encrypting_send = self._create_encrypting_send(scope, send)
            await self.app(scope, decrypted_receive, encrypting_send)
        except CryptoError:
            # Don't expose internal error details to clients
            await self._send_error(send, 400, "Request decryption failed")

    def _create_encrypting_send(  # noqa: PLR0915
        self,
        scope: dict[str, Any],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> Callable[[dict[str, Any]], Awaitable[None]]:
        """Create send wrapper that auto-encrypts SSE responses."""
        # Per-request state (closure)
        state = SSEEncryptionState()

        # WHATWG-compliant event boundary: two consecutive line endings
        # Handles \n\n, \r\r, \r\n\r\n, and mixed combinations
        event_boundary = re.compile(r"(?:\r\n|\r(?!\n)|\n)(?:\r\n|\r(?!\n)|\n)")

        async def encrypting_send(message: dict[str, Any]) -> None:
            msg_type = message["type"]

            if msg_type == "http.response.start":
                await _handle_response_start(message)
            elif msg_type == "http.response.body":
                await _handle_response_body(message)
            else:
                await send(message)

        async def _handle_response_start(message: dict[str, Any]) -> None:
            """Handle response start - detect SSE and set up encryption."""
            headers = message.get("headers", [])
            content_type = next(
                (v for n, v in headers if n.lower() == b"content-type"),
                None,
            )

            # Enable encryption if SSE + HPKE context exists
            ctx = scope.get(SCOPE_HPKE_CONTEXT)
            if ctx and content_type and b"text/event-stream" in content_type:
                state.is_sse = True
                session = create_session_from_context(ctx)
                state.encryptor = SSEEncryptor(session)

                # Add X-HPKE-Stream header
                session_params = session.serialize()
                new_headers = [
                    *headers,
                    (HEADER_HPKE_STREAM.encode(), b64url_encode(session_params).encode()),
                ]
                message = {**message, "headers": new_headers}

            await send(message)

        async def _handle_response_body(message: dict[str, Any]) -> None:
            """Handle response body - buffer and encrypt SSE events."""
            if not state.is_sse:
                await send(message)
                return

            body = message.get("body", b"")
            more_body = message.get("more_body", False)
            encryptor = state.encryptor
            if encryptor is None:  # Should never happen when is_sse=True
                raise CryptoError("SSE encryption state corrupted: encryptor is None")

            # Add to buffer (decode UTF-8, replace invalid chars)
            if body:
                decoded = body.decode("utf-8", errors="replace")
                # Enforce buffer size limit to prevent DoS
                if len(state.buffer) + len(decoded) > self.max_sse_event_size:
                    # Force flush oversized buffer as partial event
                    if state.buffer:
                        encrypted = encryptor.encrypt(state.buffer)
                        await send(
                            {
                                "type": "http.response.body",
                                "body": encrypted.encode(),
                                "more_body": True,
                            }
                        )
                    state.buffer = decoded[-self.max_sse_event_size :]  # Keep tail
                else:
                    state.buffer += decoded

            # Extract and encrypt complete events
            sent_any = await _extract_and_send_events(encryptor, more_body=more_body)

            # Handle end of stream
            if not more_body:
                await _handle_end_of_stream(encryptor, sent_any=sent_any)

        async def _extract_and_send_events(encryptor: SSEEncryptor, *, more_body: bool) -> bool:
            """Extract complete events from buffer and send encrypted."""
            sent_any = False
            while True:
                match = event_boundary.search(state.buffer)
                if not match:
                    break

                # Extract complete event (including boundary)
                event_end = match.end()
                chunk = state.buffer[:event_end]
                state.buffer = state.buffer[event_end:]

                # Send with more_body=False only if final message AND buffer empty
                is_final = not more_body and not state.buffer
                encrypted = encryptor.encrypt(chunk)
                await send(
                    {
                        "type": "http.response.body",
                        "body": encrypted.encode(),
                        "more_body": not is_final,
                    }
                )
                sent_any = True
            return sent_any

        async def _handle_end_of_stream(encryptor: SSEEncryptor, *, sent_any: bool) -> None:
            """Handle end of stream - flush buffer or send empty body."""
            if state.buffer:
                # Flush remaining buffer (partial event)
                encrypted = encryptor.encrypt(state.buffer)
                await send(
                    {
                        "type": "http.response.body",
                        "body": encrypted.encode(),
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
        Create a receive wrapper that decrypts the request body.

        Returns a new receive callable that yields the decrypted body.
        """
        # Decode encapsulated key from header
        try:
            enc = b64url_decode(enc_header.decode("ascii"))
        except Exception as e:
            raise DecryptionError(f"Invalid {HEADER_HPKE_ENC} header encoding") from e

        # Collect encrypted body
        body_chunks: list[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise DecryptionError("Client disconnected before sending body")
            body_chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break

        encrypted_body = b"".join(body_chunks)

        # Parse envelope
        try:
            header, ciphertext = decode_envelope(encrypted_body)
        except EnvelopeError as e:
            raise DecryptionError(f"Invalid envelope: {e}") from e

        # Get PSK from resolver
        try:
            psk, psk_id = await self.psk_resolver(scope)
        except Exception as e:
            raise DecryptionError(f"PSK resolution failed: {e}") from e

        # Get private key for the KEM
        kem_id = KemId(header.kem_id)
        if kem_id not in self.private_keys:
            raise DecryptionError(f"Unsupported KEM: 0x{kem_id:04x}")
        sk_r = self.private_keys[kem_id]

        # Decrypt
        ctx = setup_recipient_psk(
            enc=enc,
            sk_r=sk_r,
            info=psk_id,  # Use psk_id as info for domain separation
            psk=psk,
            psk_id=psk_id,
        )

        plaintext = ctx.open(aad=b"", ciphertext=ciphertext)

        # Store context in scope for SSE response encryption
        scope[SCOPE_HPKE_CONTEXT] = ctx

        # Create receive wrapper that returns decrypted body
        # After body is sent, we need to wait for actual disconnect from client
        # NOT return http.disconnect immediately (which aborts streaming responses)
        body_sent = False

        async def decrypted_receive() -> dict[str, Any]:
            nonlocal body_sent
            if body_sent:
                # After body is delivered, wait for actual client disconnect
                # This is called by StreamingResponse to check for client abort
                return await receive()
            body_sent = True
            return {
                "type": "http.request",
                "body": plaintext,
                "more_body": False,
            }

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

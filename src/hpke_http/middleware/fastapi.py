"""
FastAPI/Starlette ASGI middleware for transparent HPKE encryption.

Provides:
- Automatic request body decryption
- Built-in key discovery endpoint (/.well-known/hpke-keys)
- SSE response encryption wrapper

Usage:
    from hpke_http.middleware.fastapi import HPKEMiddleware

    app = FastAPI()
    app.add_middleware(
        HPKEMiddleware,
        private_keys={KemId.DHKEM_X25519_HKDF_SHA256: private_key_bytes},
        psk_resolver=get_api_key_from_request,
    )

Reference: RFC-065 §4.3, §5.3
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any

from cryptography.hazmat.primitives.asymmetric import x25519

from hpke_http.constants import (
    AEAD_ID,
    DISCOVERY_CACHE_MAX_AGE,
    DISCOVERY_PATH,
    HEADER_HPKE_ENC,
    HEADER_HPKE_STREAM,
    KDF_ID,
    KemId,
)
from hpke_http.envelope import decode_envelope
from hpke_http.exceptions import CryptoError, DecryptionError, EnvelopeError
from hpke_http.headers import b64url_decode, b64url_encode
from hpke_http.hpke import setup_recipient_psk
from hpke_http.streaming import SSEEncryptor, create_session_from_context

__all__ = [
    "EncryptedSSEResponse",
    "HPKEMiddleware",
]

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
    Pure ASGI middleware for HPKE request decryption.

    Features:
    - Decrypts request bodies encrypted with HPKE PSK mode
    - Auto-registers /.well-known/hpke-keys discovery endpoint
    - Stores recipient context in scope for SSE response encryption

    Note: SSE response encryption requires using EncryptedSSEResponse.
    """

    def __init__(
        self,
        app: Any,
        private_keys: dict[KemId, bytes],
        psk_resolver: PSKResolver,
        discovery_path: str = DISCOVERY_PATH,
    ) -> None:
        """
        Initialize HPKE middleware.

        Args:
            app: ASGI application
            private_keys: Private keys by KEM ID (e.g., {KemId.DHKEM_X25519_HKDF_SHA256: sk})
            psk_resolver: Async callback to get (psk, psk_id) from request scope
            discovery_path: Path for key discovery endpoint
        """
        self.app = app
        self.private_keys = private_keys
        self.psk_resolver = psk_resolver
        self.discovery_path = discovery_path

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

        # Decrypt request
        try:
            decrypted_receive = await self._create_decrypted_receive(scope, receive, enc_header)
            await self.app(scope, decrypted_receive, send)
        except CryptoError as e:
            await self._send_error(send, 400, str(e))

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
        psk, psk_id = await self.psk_resolver(scope)

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
        scope["hpke_context"] = ctx

        # Create receive wrapper that returns decrypted body
        body_sent = False

        async def decrypted_receive() -> dict[str, Any]:
            nonlocal body_sent
            if body_sent:
                return {"type": "http.disconnect"}
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
            }
        )


class EncryptedSSEResponse:
    """
    Wrapper for creating encrypted SSE responses.

    Usage in handler:
        @app.post("/tasks")
        async def create_task(request: Request):
            ctx = request.scope.get("hpke_context")
            if ctx:
                return EncryptedSSEResponse(ctx, event_generator())
            return StreamingResponse(event_generator(), media_type="text/event-stream")
    """

    def __init__(
        self,
        ctx: Any,  # RecipientContext
        event_generator: Any,  # AsyncGenerator yielding (event_type, data) tuples
    ) -> None:
        """
        Initialize encrypted SSE response.

        Args:
            ctx: HPKE RecipientContext from request decryption
            event_generator: Async generator yielding (event_type, data) tuples
        """
        self.ctx = ctx
        self.event_generator = event_generator

    async def __call__(
        self,
        _scope: dict[str, Any],
        _receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """ASGI interface for SSE streaming."""
        # Create streaming session from HPKE context
        session = create_session_from_context(self.ctx)
        encryptor = SSEEncryptor(session)

        # Send headers with session parameters
        session_params = session.serialize()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream"),
                    (b"cache-control", b"no-cache"),
                    (b"connection", b"keep-alive"),
                    (HEADER_HPKE_STREAM.encode(), b64url_encode(session_params).encode()),
                ],
            }
        )

        # Stream encrypted events
        event_id = 0
        async for event_type, data in self.event_generator:
            event_id += 1
            encrypted_event = encryptor.encrypt_event(event_type, data, event_id)
            await send(
                {
                    "type": "http.response.body",
                    "body": encrypted_event.encode(),
                    "more_body": True,
                }
            )

        # End stream
        await send(
            {
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            }
        )

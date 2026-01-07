"""
aiohttp client session with transparent HPKE encryption.

Provides a drop-in replacement for aiohttp.ClientSession that automatically:
- Fetches and caches platform public keys from discovery endpoint
- Encrypts request bodies
- Decrypts SSE event streams (transparent - yields exact server output)

Usage:
    async with HPKEClientSession(base_url="https://api.example.com", psk=api_key) as session:
        async with session.post("/tasks", json=data) as response:
            async for chunk in session.iter_sse(response):
                # chunk is exactly what server sent (event, comment, retry, etc.)
                print(chunk)

Reference: RFC-065 §4.4, §5.2
"""

import asyncio
import json as json_module
import re
import time
import types
import weakref
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any, ClassVar
from urllib.parse import urljoin, urlparse

import aiohttp
from typing_extensions import Self

from hpke_http._logging import get_logger
from hpke_http.constants import (
    DISCOVERY_CACHE_MAX_AGE,
    DISCOVERY_PATH,
    HEADER_HPKE_ENC,
    HEADER_HPKE_ENCODING,
    HEADER_HPKE_STREAM,
    ZSTD_MIN_SIZE,
    KemId,
)
from hpke_http.envelope import encode_envelope
from hpke_http.exceptions import DecryptionError, KeyDiscoveryError
from hpke_http.headers import b64url_decode, b64url_encode
from hpke_http.hpke import SenderContext, setup_sender_psk
from hpke_http.streaming import SSEDecryptor, StreamingSession, zstd_compress

__all__ = [
    "HPKEClientSession",
]

_logger = get_logger(__name__)


class HPKEClientSession:
    """
    aiohttp-compatible client session with transparent HPKE encryption.

    Features:
    - Automatic key discovery from /.well-known/hpke-keys
    - Request body encryption with HPKE PSK mode
    - SSE response stream decryption (transparent pass-through)
    - Class-level key caching with TTL
    """

    # Class-level key cache: host -> (keys_dict, expires_at)
    _key_cache: ClassVar[dict[str, tuple[dict[KemId, bytes], float]]] = {}
    _cache_lock: ClassVar[asyncio.Lock | None] = None

    def __init__(
        self,
        base_url: str,
        psk: bytes,
        psk_id: bytes | None = None,
        discovery_url: str | None = None,
        *,
        compress: bool = False,
        **aiohttp_kwargs: Any,
    ) -> None:
        """
        Initialize HPKE-enabled client session.

        Args:
            base_url: Base URL for API requests (e.g., "https://api.example.com")
            psk: Pre-shared key (API key as bytes)
            psk_id: Pre-shared key identifier (defaults to psk)
            discovery_url: Override discovery endpoint URL (for testing)
            compress: Enable Zstd compression for request bodies (RFC 8878).
                When enabled, requests >= 64 bytes are compressed before encryption.
                Server must have backports.zstd installed (Python < 3.14).
            **aiohttp_kwargs: Additional arguments passed to aiohttp.ClientSession
        """
        self.base_url = base_url.rstrip("/")
        self.psk = psk
        self.psk_id = psk_id or psk
        self.discovery_url = discovery_url or urljoin(self.base_url, DISCOVERY_PATH)
        self.compress = compress

        self._session: aiohttp.ClientSession | None = None
        self._aiohttp_kwargs = aiohttp_kwargs
        self._platform_keys: dict[KemId, bytes] | None = None

        # Maps responses to their sender contexts for SSE decryption (thread-safe for concurrent requests)
        self._response_contexts: weakref.WeakKeyDictionary[aiohttp.ClientResponse, SenderContext] = (
            weakref.WeakKeyDictionary()
        )

    @classmethod
    def _get_cache_lock(cls) -> asyncio.Lock:
        """Get or create cache lock (lazy initialization for event loop safety)."""
        if cls._cache_lock is None:
            cls._cache_lock = asyncio.Lock()
        return cls._cache_lock

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(**self._aiohttp_kwargs)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_keys(self) -> dict[KemId, bytes]:
        """
        Fetch and cache platform public keys.

        Returns:
            Dict mapping KemId to public key bytes
        """
        if self._platform_keys:
            return self._platform_keys

        host = urlparse(self.base_url).netloc
        lock = self._get_cache_lock()

        async with lock:
            # Check cache
            if host in self._key_cache:
                keys, expires_at = self._key_cache[host]
                if time.time() < expires_at:
                    _logger.debug("Key cache hit: host=%s", host)
                    self._platform_keys = keys
                    return self._platform_keys

            _logger.debug("Key cache miss: host=%s fetching from %s", host, self.discovery_url)
            # Fetch from discovery endpoint
            if not self._session:
                raise RuntimeError("Session not initialized. Use 'async with' context manager.")

            try:
                async with self._session.get(self.discovery_url) as resp:
                    if resp.status != HTTPStatus.OK:
                        raise KeyDiscoveryError(f"Discovery endpoint returned {resp.status}")

                    data = await resp.json()

                    # Parse Cache-Control for TTL
                    cache_control = resp.headers.get("Cache-Control", "")
                    max_age = self._parse_max_age(cache_control)
                    expires_at = time.time() + max_age

                    # Parse keys
                    keys = self._parse_keys(data)

                    # Cache
                    self._key_cache[host] = (keys, expires_at)
                    self._platform_keys = keys
                    _logger.debug(
                        "Keys fetched: host=%s kem_ids=%s ttl=%ds",
                        host,
                        [f"0x{k:04x}" for k in keys],
                        max_age,
                    )
                    return self._platform_keys

            except aiohttp.ClientError as e:
                _logger.debug("Key discovery failed: host=%s error=%s", host, e)
                raise KeyDiscoveryError(f"Failed to fetch keys: {e}") from e

    def _parse_max_age(self, cache_control: str) -> int:
        """Parse max-age from Cache-Control header."""
        for directive in cache_control.split(","):
            directive_stripped = directive.strip()
            if directive_stripped.startswith("max-age="):
                try:
                    return int(directive_stripped[8:])
                except ValueError:
                    pass
        return DISCOVERY_CACHE_MAX_AGE

    def _parse_keys(self, response: dict[str, Any]) -> dict[KemId, bytes]:
        """Parse keys from discovery response."""
        result: dict[KemId, bytes] = {}
        for key_info in response.get("keys", []):
            kem_id = KemId(int(key_info["kem_id"], 16))
            public_key = b64url_decode(key_info["public_key"])
            result[kem_id] = public_key
        return result

    async def _encrypt_request(
        self,
        body: bytes,
    ) -> tuple[bytes, str, SenderContext, bool]:
        """
        Encrypt request body with HPKE.

        Returns:
            Tuple of (encrypted_body, enc_header_value, sender_context, was_compressed)
        """
        keys = await self._ensure_keys()

        # Use X25519 (default suite)
        pk_r = keys.get(KemId.DHKEM_X25519_HKDF_SHA256)
        if not pk_r:
            raise KeyDiscoveryError("No X25519 key available from platform")

        # Compress if enabled and body is large enough
        was_compressed = False
        if self.compress and len(body) >= ZSTD_MIN_SIZE:
            original_size = len(body)
            body = zstd_compress(body)
            was_compressed = True
            _logger.debug(
                "Request compressed: original=%d compressed=%d ratio=%.1f%%",
                original_size,
                len(body),
                len(body) / original_size * 100,
            )

        # Set up sender context
        ctx = setup_sender_psk(
            pk_r=pk_r,
            info=self.psk_id,
            psk=self.psk,
            psk_id=self.psk_id,
        )

        # Encrypt body
        ciphertext = ctx.seal(aad=b"", plaintext=body)

        # Encode as envelope
        envelope = encode_envelope(ciphertext)

        # Encode enc for header
        enc_header = b64url_encode(ctx.enc)

        return (envelope, enc_header, ctx, was_compressed)

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        data: bytes | None = None,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse:
        """
        Make an encrypted HTTP request.

        Args:
            method: HTTP method
            url: URL (relative to base_url or absolute)
            json: JSON body (will be serialized and encrypted)
            data: Raw body bytes (will be encrypted)
            **kwargs: Additional arguments passed to aiohttp

        Returns:
            aiohttp.ClientResponse
        """
        if not self._session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")

        # Resolve URL
        if not url.startswith(("http://", "https://")):
            url = urljoin(self.base_url + "/", url.lstrip("/"))

        # Prepare body
        body: bytes | None = None
        if json is not None:
            body = json_module.dumps(json).encode()
        elif data is not None:
            body = data

        # Encrypt if we have a body
        headers = dict(kwargs.pop("headers", {}))
        sender_ctx: SenderContext | None = None
        if body:
            encrypted_body, enc_header, ctx, was_compressed = await self._encrypt_request(body)
            headers[HEADER_HPKE_ENC] = enc_header
            headers["Content-Type"] = "application/octet-stream"
            if was_compressed:
                headers[HEADER_HPKE_ENCODING] = "zstd"
            sender_ctx = ctx
            kwargs["data"] = encrypted_body
            _logger.debug(
                "Request encrypted: method=%s url=%s body_size=%d compressed=%s",
                method,
                url,
                len(body),
                was_compressed,
            )

        kwargs["headers"] = headers
        response = await self._session.request(method, url, **kwargs)

        # Store context per-response for concurrent request safety
        if sender_ctx:
            self._response_contexts[response] = sender_ctx

        return response

    # Convenience methods
    async def get(self, url: str, **kwargs: Any) -> aiohttp.ClientResponse:
        """GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(
        self,
        url: str,
        *,
        json: Any = None,
        data: bytes | None = None,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse:
        """POST request."""
        return await self.request("POST", url, json=json, data=data, **kwargs)

    async def put(
        self,
        url: str,
        *,
        json: Any = None,
        data: bytes | None = None,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse:
        """PUT request."""
        return await self.request("PUT", url, json=json, data=data, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> aiohttp.ClientResponse:
        """DELETE request."""
        return await self.request("DELETE", url, **kwargs)

    async def iter_sse(
        self,
        response: aiohttp.ClientResponse,
    ) -> AsyncIterator[bytes]:
        """
        Iterate over encrypted SSE stream, yielding decrypted chunks.

        Transparent pass-through: yields the exact SSE chunks the server sent.
        Events, comments, retry directives - everything is preserved exactly.

        Args:
            response: Response from an SSE endpoint

        Yields:
            Raw SSE chunks as bytes exactly as the server sent them
        """
        sender_ctx = self._response_contexts.get(response)
        if not sender_ctx:
            raise RuntimeError("No encryption context for this response. Was the request encrypted?")

        # Get session parameters from header
        stream_header = response.headers.get(HEADER_HPKE_STREAM)
        if not stream_header:
            raise DecryptionError(f"Missing {HEADER_HPKE_STREAM} header")

        # Derive session key from sender context
        session_key = sender_ctx.export(b"sse-session-key", 32)
        session_params = b64url_decode(stream_header)
        session = StreamingSession.deserialize(session_params, session_key)
        decryptor = SSEDecryptor(session)
        _logger.debug("SSE decryption started: url=%s", response.url)

        # WHATWG event boundary: blank line (any combination of line endings)
        event_boundary = re.compile(r"(?:\r\n|\r|\n)(?:\r\n|\r|\n)")

        # Parse encrypted SSE stream
        buffer = ""
        async for chunk in response.content:
            buffer += chunk.decode("utf-8")

            # Process complete events (separated by blank line)
            while True:
                match = event_boundary.search(buffer)
                if not match:
                    break
                event_text = buffer[: match.start()]
                buffer = buffer[match.end() :]

                # Extract data field from encrypted event
                data_value = None
                lines = re.split(r"\r\n|\r|\n", event_text)
                for line in lines:
                    if not line or line.startswith(":"):
                        continue
                    if ":" in line:
                        key, _, value = line.partition(":")
                        if value.startswith(" "):
                            value = value[1:]
                        if key == "data":
                            data_value = value

                if data_value and data_value.strip():
                    # Decrypt to get original SSE chunk
                    raw_chunk = decryptor.decrypt(data_value.strip())
                    yield raw_chunk

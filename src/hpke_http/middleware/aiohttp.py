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
import time
import types
import weakref
from collections.abc import AsyncIterator, Callable
from http import HTTPStatus
from typing import Any, ClassVar
from urllib.parse import urljoin, urlparse

import aiohttp
from multidict import CIMultiDictProxy
from typing_extensions import Self
from yarl import URL

from hpke_http._logging import get_logger
from hpke_http.constants import (
    DISCOVERY_CACHE_MAX_AGE,
    DISCOVERY_PATH,
    HEADER_HPKE_STREAM,
    KemId,
)
from hpke_http.core import (
    RequestEncryptor,
    ResponseDecryptor,
    SSEDecryptor,
    SSELineParser,
)
from hpke_http.exceptions import EncryptionRequiredError, KeyDiscoveryError
from hpke_http.headers import b64url_decode
from hpke_http.hpke import SenderContext

__all__ = [
    "DecryptedResponse",
    "HPKEClientSession",
]

_logger = get_logger(__name__)


class DecryptedResponse:
    """
    Transparent wrapper that decrypts response body on access.

    Wraps an aiohttp.ClientResponse and transparently decrypts the body
    when accessed via read(), text(), or json() methods. The underlying
    response uses counter-based chunk encryption (RawFormat).

    Duck-types common aiohttp.ClientResponse attributes (status, headers, url,
    ok, reason, content_type, raise_for_status) for seamless usage. Use unwrap()
    to access the underlying ClientResponse directly.

    This class is returned automatically by HPKEClientSession.request()
    when the server responds with an encrypted standard (non-SSE) response
    (detected via X-HPKE-Stream header and non-SSE Content-Type).

    Design note: This class intentionally does NOT inherit from a Protocol/ABC.
    aiohttp does not expose a public ClientResponse protocol. More importantly,
    ClientResponse uses `propcache.under_cached_property` for `headers` and `url`
    which has different type semantics than standard `@property` - pyright rejects
    Protocol matching due to this type mismatch. Creating a custom protocol that
    structurally matches ClientResponse is not feasible without matching these
    internal implementation details. Duck typing with __getattr__ fallback is the
    pragmatic approach here.
    """

    def __init__(
        self,
        response: aiohttp.ClientResponse,
        sender_ctx: "SenderContext",
    ) -> None:
        """
        Initialize decrypted response wrapper.

        Args:
            response: The underlying aiohttp response
            sender_ctx: HPKE sender context for key derivation
        """
        self._response = response
        self._sender_ctx = sender_ctx
        self._decrypted: bytes | None = None

    async def _ensure_decrypted(self) -> bytes:
        """Decrypt response body using ResponseDecryptor.

        Uses the centralized ResponseDecryptor class which handles
        header parsing, chunk boundary detection, and decryption.
        """
        if self._decrypted is not None:
            return self._decrypted

        # Read entire response body (cached by aiohttp, safe to call multiple times)
        raw_body = await self._response.read()

        # Use centralized ResponseDecryptor - handles headers, chunking, decryption
        decryptor = ResponseDecryptor(self._response.headers, self._sender_ctx)
        self._decrypted = decryptor.decrypt_all(raw_body)

        _logger.debug(
            "Response decrypted: url=%s size=%d",
            self._response.url,
            len(self._decrypted),
        )
        return self._decrypted

    async def read(self) -> bytes:
        """Read and decrypt the response body."""
        return await self._ensure_decrypted()

    async def text(self, encoding: str | None = None, errors: str = "strict") -> str:
        """Read and decrypt response body as text.

        Matches aiohttp.ClientResponse.text() signature.

        Args:
            encoding: Character encoding to use. If None, uses UTF-8.
            errors: Error handling scheme for decoding (default: 'strict').

        Returns:
            Decoded string content.
        """
        enc = encoding or "utf-8"
        return (await self._ensure_decrypted()).decode(enc, errors=errors)

    async def json(
        self,
        *,
        encoding: str | None = None,
        loads: Callable[[str], Any] = json_module.loads,
        content_type: str | None = "application/json",
    ) -> Any:
        """Read and decrypt response body as JSON.

        Matches aiohttp.ClientResponse.json() signature.

        Args:
            encoding: Character encoding for decoding bytes to string.
                If None, uses UTF-8.
            loads: Custom JSON decoder function (default: json.loads).
            content_type: Expected Content-Type (None disables validation).
                Default: 'application/json'.

        Returns:
            Parsed JSON data.

        Raises:
            aiohttp.ContentTypeError: If content_type validation fails.
        """
        # Validate content type if specified
        if content_type is not None:
            actual_ct = self._response.content_type or ""
            if content_type not in actual_ct:
                raise aiohttp.ContentTypeError(
                    self._response.request_info,
                    self._response.history,
                    message=f"Attempt to decode JSON with unexpected content type: {actual_ct}",
                )

        enc = encoding or "utf-8"
        text = (await self._ensure_decrypted()).decode(enc)
        return loads(text)

    # Proxy common aiohttp.ClientResponse attributes
    @property
    def status(self) -> int:
        """HTTP status code."""
        return self._response.status

    @property
    def headers(self) -> CIMultiDictProxy[str]:
        """Response headers as case-insensitive multidict."""
        return self._response.headers

    @property
    def url(self) -> URL:
        """Request URL."""
        return self._response.url

    @property
    def ok(self) -> bool:
        """True if status is less than 400."""
        return self._response.ok

    @property
    def reason(self) -> str | None:
        """HTTP status reason (e.g., 'OK')."""
        return self._response.reason

    @property
    def content_type(self) -> str:
        """Content-Type header value."""
        return self._response.content_type or ""

    def raise_for_status(self) -> None:
        """Raise an exception if status is 400 or higher."""
        self._response.raise_for_status()

    def unwrap(self) -> aiohttp.ClientResponse:
        """Return the underlying aiohttp.ClientResponse.

        Useful when you need direct access to the raw response object,
        for example when passing to iter_sse().
        """
        return self._response

    def __getattr__(self, name: str) -> Any:
        """Proxy unknown attributes to underlying response."""
        return getattr(self._response, name)


def _extract_sse_data(event_lines: list[bytes]) -> str | None:
    """Extract data field value from SSE event lines.

    Per WHATWG spec, the data field is prefixed with "data: ".
    Returns stripped value if non-empty, None otherwise.
    """
    for line in event_lines:
        if line.startswith(b"data: "):
            value = line[6:].decode("ascii").strip()
            return value if value else None
    return None


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
        require_encryption: bool = False,
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
            require_encryption: If True, raise EncryptionRequiredError when
                server responds with plaintext instead of encrypted response.
            **aiohttp_kwargs: Additional arguments passed to aiohttp.ClientSession
        """
        self.base_url = base_url.rstrip("/")
        self.psk = psk
        self.psk_id = psk_id or psk
        self.discovery_url = discovery_url or urljoin(self.base_url, DISCOVERY_PATH)
        self.compress = compress
        self.require_encryption = require_encryption

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
            public_key = bytes(b64url_decode(key_info["public_key"]))
            result[kem_id] = public_key
        return result

    async def _encrypt_request(
        self,
        body: bytes,
    ) -> tuple[AsyncIterator[bytes], dict[str, str], SenderContext]:
        """
        Encrypt request body using RequestEncryptor with streaming.

        Returns an async generator for memory-efficient chunked upload.
        Uses HTTP chunked transfer encoding (Transfer-Encoding: chunked).

        Returns:
            Tuple of (async_chunk_generator, headers_dict, sender_context)
        """
        keys = await self._ensure_keys()

        # Use X25519 (default suite)
        pk_r = keys.get(KemId.DHKEM_X25519_HKDF_SHA256)
        if not pk_r:
            raise KeyDiscoveryError("No X25519 key available from platform")

        # Use centralized RequestEncryptor - handles compression, chunking, encryption
        encryptor = RequestEncryptor(
            public_key=pk_r,
            psk=self.psk,
            psk_id=self.psk_id,
            compress=self.compress,
        )

        # Prime the generator to trigger compression BEFORE get_headers()
        # Generator code doesn't execute until iteration starts, so we must
        # consume at least the first chunk to set _was_compressed flag
        chunk_iter = encryptor.encrypt_iter(body)
        first_chunk = next(chunk_iter)

        # Now compression has happened and _was_compressed is set
        headers = encryptor.get_headers()

        # Create async generator for streaming upload
        # Memory: O(chunk_size) instead of O(body_size)
        async def stream_chunks() -> AsyncIterator[bytes]:
            yield first_chunk  # Return the primed first chunk
            for chunk in chunk_iter:  # Then yield remaining chunks
                yield chunk

        _logger.debug(
            "Request encryption prepared: body_size=%d streaming=True",
            len(body),
        )

        return (stream_chunks(), headers, encryptor.context)

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        data: bytes | None = None,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse | DecryptedResponse:
        """
        Make an encrypted HTTP request.

        Args:
            method: HTTP method
            url: URL (relative to base_url or absolute)
            json: JSON body (will be serialized and encrypted)
            data: Raw body bytes (will be encrypted)
            **kwargs: Additional arguments passed to aiohttp

        Returns:
            aiohttp.ClientResponse for plain responses or SSE streams,
            DecryptedResponse for encrypted standard responses
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
            encrypted_stream, crypto_headers, ctx = await self._encrypt_request(body)
            headers.update(crypto_headers)
            headers["Content-Type"] = "application/octet-stream"
            sender_ctx = ctx
            # Pass async generator for streaming upload (chunked transfer encoding)
            kwargs["data"] = encrypted_stream
            _logger.debug(
                "Request encrypted: method=%s url=%s body_size=%d streaming=True",
                method,
                url,
                len(body),
            )

        kwargs["headers"] = headers
        response = await self._session.request(method, url, **kwargs)

        # Store context per-response for concurrent request safety (needed for SSE)
        if sender_ctx:
            self._response_contexts[response] = sender_ctx

            # Return DecryptedResponse wrapper for encrypted standard responses
            # Detection: X-HPKE-Stream present AND Content-Type is NOT text/event-stream
            if HEADER_HPKE_STREAM in response.headers:
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" not in content_type:
                    _logger.debug("Encrypted response detected: url=%s", url)
                    return DecryptedResponse(response, sender_ctx)
            elif self.require_encryption:
                # We sent encrypted request but got plaintext response
                raise EncryptionRequiredError("Response was not encrypted")

        return response

    # Convenience methods
    async def get(self, url: str, **kwargs: Any) -> aiohttp.ClientResponse | DecryptedResponse:
        """GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(
        self,
        url: str,
        *,
        json: Any = None,
        data: bytes | None = None,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse | DecryptedResponse:
        """POST request."""
        return await self.request("POST", url, json=json, data=data, **kwargs)

    async def put(
        self,
        url: str,
        *,
        json: Any = None,
        data: bytes | None = None,
        **kwargs: Any,
    ) -> aiohttp.ClientResponse | DecryptedResponse:
        """PUT request."""
        return await self.request("PUT", url, json=json, data=data, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> aiohttp.ClientResponse | DecryptedResponse:
        """DELETE request."""
        return await self.request("DELETE", url, **kwargs)

    async def iter_sse(
        self,
        response: aiohttp.ClientResponse | DecryptedResponse,
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
        # Extract underlying response if wrapped
        if isinstance(response, DecryptedResponse):
            response = response.unwrap()

        sender_ctx = self._response_contexts.get(response)
        if not sender_ctx:
            raise RuntimeError("No encryption context for this response. Was the request encrypted?")

        # Use centralized SSEDecryptor - handles header parsing and key derivation
        decryptor = SSEDecryptor(response.headers, sender_ctx)
        _logger.debug("SSE decryption started: url=%s", response.url)

        # Use centralized line parser for O(n) streaming (vs O(n²) regex)
        line_parser = SSELineParser()
        current_event_lines: list[bytes] = []

        async for chunk in response.content:
            for line in line_parser.feed(chunk):
                if not line:
                    # Empty line = event boundary (WHATWG spec)
                    data_value = _extract_sse_data(current_event_lines)
                    if data_value:
                        raw_chunk = decryptor.decrypt(data_value)
                        yield raw_chunk
                    current_event_lines = []
                else:
                    current_event_lines.append(line)

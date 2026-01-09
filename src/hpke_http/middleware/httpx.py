"""
httpx client with transparent HPKE encryption.

Provides a drop-in replacement for httpx.AsyncClient that automatically:
- Fetches and caches platform public keys from discovery endpoint
- Encrypts request bodies
- Decrypts SSE event streams (transparent - yields exact server output)

Usage:
    async with HPKEAsyncClient(base_url="https://api.example.com", psk=api_key) as client:
        response = await client.post("/tasks", json=data)
        async for chunk in client.iter_sse(response):
            # chunk is exactly what server sent (event, comment, retry, etc.)
            print(chunk)

Architecture note:
    This module uses a wrapper pattern rather than httpx's native extensibility
    mechanisms. Here's what was evaluated:

    1. Custom Transport (AsyncBaseTransport): Can encrypt requests and attach context
       via response.extensions, BUT handle_async_request() must return httpx.Response -
       cannot return DecryptedResponse or add iter_sse() method.

    2. Event Hooks: Read-only by design. Cannot mutate request/response objects,
       only observe them. See github.com/encode/httpx/issues/1343.

    3. Auth (auth_flow): Can modify requests by yielding new Request with encrypted
       body, BUT cannot change response type or add methods to client.

    All mechanisms share one fundamental limitation: they cannot change the return
    type from httpx.Response to DecryptedResponse, nor add custom methods like
    iter_sse(). The wrapper pattern is required for:
    - Returning DecryptedResponse with transparent content/text/json() decryption
    - Providing iter_sse() for streaming SSE decryption
    - Async key discovery via HTTP (transport receives Request, not async context)

    See also: aiohttp.py uses the same pattern for identical reasons.

Reference: RFC-065 §4.4, §5.2
"""

import json as json_module
import types
import weakref
from collections.abc import AsyncIterator, Iterator
from datetime import timedelta
from http import HTTPStatus
from typing import Any

import httpx
from httpx._content import encode_multipart_data
from typing_extensions import Self

from hpke_http._logging import get_logger
from hpke_http.constants import CHUNK_SIZE, HEADER_HPKE_CONTENT_TYPE, HEADER_HPKE_STREAM, KemId
from hpke_http.core import (
    BaseHPKEClient,
    RequestEncryptor,
    ResponseDecryptor,
    SSEDecryptor,
    SSELineParser,
    extract_sse_data,
)
from hpke_http.exceptions import EncryptionRequiredError, KeyDiscoveryError
from hpke_http.hpke import SenderContext

__all__ = [
    "DecryptedResponse",
    "HPKEAsyncClient",
]

_logger = get_logger(__name__)


def _rechunk_iter(
    source: Iterator[bytes],
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[bytes]:
    """
    Re-chunk an iterator to yield bounded chunks.

    httpx's MultipartStream yields full file content for in-memory bytes.
    This function re-chunks large pieces into bounded chunks for memory-efficient
    streaming encryption (same pattern as aiohttp's _stream_multipart).

    Args:
        source: Iterator yielding bytes (may yield large chunks)
        chunk_size: Maximum chunk size to yield (default: 64KB)

    Yields:
        Chunks of at most chunk_size bytes
    """
    for chunk in source:
        if len(chunk) <= chunk_size:
            yield chunk
        else:
            # Re-chunk large pieces (e.g., full file content from bytes input)
            for offset in range(0, len(chunk), chunk_size):
                yield chunk[offset : offset + chunk_size]


class DecryptedResponse:
    """
    Transparent wrapper that decrypts response body on access.

    Wraps an httpx.Response and transparently decrypts the body
    when accessed via content, text, or json() methods. The underlying
    response uses counter-based chunk encryption (RawFormat).

    Matches httpx.Response API exactly for seamless drop-in usage.
    Use unwrap() to access the underlying Response directly.

    This class is returned automatically by HPKEAsyncClient.request()
    when the server responds with an encrypted standard (non-SSE) response
    (detected via X-HPKE-Stream header and non-SSE Content-Type).
    """

    def __init__(
        self,
        response: httpx.Response,
        sender_ctx: "SenderContext",
        *,
        release_encrypted: bool = False,
    ) -> None:
        """
        Initialize decrypted response wrapper.

        Args:
            response: The underlying httpx response
            sender_ctx: HPKE sender context for key derivation
            release_encrypted: If True, release encrypted content after decryption
                to reduce held memory from 2x to 1x payload. Trade-off: unwrap().content
                will return empty bytes after decryption.
        """
        self._response = response
        self._sender_ctx = sender_ctx
        self._decrypted: bytes | None = None
        self._release_encrypted = release_encrypted

    def _ensure_decrypted(self) -> bytes:
        """Decrypt response body using ResponseDecryptor.

        Uses the centralized ResponseDecryptor class which handles
        header parsing, chunk boundary detection, and decryption.
        """
        if self._decrypted is not None:
            return self._decrypted

        # Read entire response body (httpx auto-loads content)
        raw_body = self._response.content

        # Use centralized ResponseDecryptor - handles headers, chunking, decryption
        decryptor = ResponseDecryptor(self._response.headers, self._sender_ctx)
        self._decrypted = decryptor.decrypt_all(raw_body)

        # Release encrypted content to reduce held memory (2x → 1x)
        if self._release_encrypted:
            # Clear httpx's internal content cache
            # pyright: ignore[reportPrivateUsage] - intentional for memory optimization
            self._response._content = b""  # type: ignore[attr-defined]  # noqa: SLF001

        _logger.debug(
            "Response decrypted: url=%s size=%d release_encrypted=%s",
            self._response.url,
            len(self._decrypted),
            self._release_encrypted,
        )
        return self._decrypted

    # httpx.Response content access - decrypted
    @property
    def content(self) -> bytes:
        """Decrypted response body as bytes."""
        return self._ensure_decrypted()

    @property
    def text(self) -> str:
        """Decrypted response body as text."""
        encoding = self._response.encoding or "utf-8"
        return self._ensure_decrypted().decode(encoding)

    def json(self, **kwargs: Any) -> Any:
        """Decrypted response body as JSON.

        Args:
            **kwargs: Passed to json.loads()

        Returns:
            Parsed JSON data.
        """
        return json_module.loads(self.text, **kwargs)

    # httpx.Response properties - proxied directly
    @property
    def status_code(self) -> int:
        """HTTP status code."""
        return self._response.status_code

    @property
    def reason_phrase(self) -> str:
        """HTTP reason phrase (e.g., 'OK')."""
        return self._response.reason_phrase

    @property
    def http_version(self) -> str:
        """HTTP version (e.g., 'HTTP/1.1' or 'HTTP/2')."""
        return self._response.http_version

    @property
    def url(self) -> httpx.URL:
        """Request URL."""
        return self._response.url

    @property
    def headers(self) -> httpx.Headers:
        """Response headers."""
        return self._response.headers

    @property
    def cookies(self) -> httpx.Cookies:
        """Response cookies."""
        return self._response.cookies

    @property
    def encoding(self) -> str | None:
        """Response encoding."""
        return self._response.encoding

    # Status check properties - proxied
    @property
    def is_informational(self) -> bool:
        """True if status is 1xx."""
        return self._response.is_informational

    @property
    def is_success(self) -> bool:
        """True if status is 2xx."""
        return self._response.is_success

    @property
    def is_redirect(self) -> bool:
        """True if status is 3xx."""
        return self._response.is_redirect

    @property
    def is_client_error(self) -> bool:
        """True if status is 4xx."""
        return self._response.is_client_error

    @property
    def is_server_error(self) -> bool:
        """True if status is 5xx."""
        return self._response.is_server_error

    # Request/response metadata - proxied
    @property
    def request(self) -> httpx.Request:
        """The request that led to this response."""
        return self._response.request

    @property
    def next_request(self) -> httpx.Request | None:
        """The next request in a redirect chain, if any."""
        return self._response.next_request

    @property
    def history(self) -> list[httpx.Response]:
        """List of redirect responses."""
        return self._response.history

    @property
    def elapsed(self) -> timedelta:
        """Time elapsed between sending the request and receiving the response."""
        return self._response.elapsed

    # Methods - proxied
    def raise_for_status(self) -> Self:
        """Raise HTTPStatusError for non-2xx responses."""
        self._response.raise_for_status()
        return self

    async def aread(self) -> bytes:
        """Read response content (already loaded, returns decrypted)."""
        return self._ensure_decrypted()

    async def aclose(self) -> None:
        """Close the response."""
        await self._response.aclose()

    # HPKE-specific
    def unwrap(self) -> httpx.Response:
        """Return the underlying httpx.Response.

        Useful when you need direct access to the raw response object,
        for example when passing to iter_sse().
        """
        return self._response

    def __getattr__(self, name: str) -> Any:
        """Proxy unknown attributes to underlying response."""
        return getattr(self._response, name)


class HPKEAsyncClient(BaseHPKEClient):
    """
    httpx-compatible async client with transparent HPKE encryption.

    Drop-in replacement for httpx.AsyncClient with automatic:
    - Key discovery from /.well-known/hpke-keys
    - Request body encryption with HPKE PSK mode
    - SSE response stream decryption (transparent pass-through)
    - Class-level key caching with TTL

    All httpx.AsyncClient methods are supported with identical signatures.
    """

    def __init__(
        self,
        base_url: str,
        psk: bytes,
        psk_id: bytes | None = None,
        discovery_url: str | None = None,
        *,
        compress: bool = False,
        require_encryption: bool = False,
        release_encrypted: bool = False,
        **httpx_kwargs: Any,
    ) -> None:
        """
        Initialize HPKE-enabled async client.

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
            release_encrypted: If True, release encrypted response content after
                decryption to reduce held memory from 2x to 1x payload. Trade-off:
                response.unwrap().content will return empty bytes after decryption.
            **httpx_kwargs: Additional arguments passed to httpx.AsyncClient
        """
        super().__init__(
            base_url=base_url,
            psk=psk,
            psk_id=psk_id,
            discovery_url=discovery_url,
            compress=compress,
            require_encryption=require_encryption,
            logger=_logger,
        )

        self._client: httpx.AsyncClient | None = None
        self._httpx_kwargs = httpx_kwargs
        self._release_encrypted = release_encrypted

        # Maps responses to their sender contexts for SSE decryption
        self._response_contexts: weakref.WeakKeyDictionary[httpx.Response, SenderContext] = weakref.WeakKeyDictionary()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        self._client = httpx.AsyncClient(base_url=self.base_url, **self._httpx_kwargs)
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: types.TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def aclose(self) -> None:
        """Close the client (matches httpx.AsyncClient.aclose)."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _fetch_discovery(self) -> tuple[dict[str, Any], str, str]:
        """Fetch discovery endpoint using httpx.

        Returns:
            Tuple of (response JSON dict, Cache-Control header, Accept-Encoding header)
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        try:
            resp = await self._client.get(self.discovery_url)
            if resp.status_code != HTTPStatus.OK:
                raise KeyDiscoveryError(f"Discovery endpoint returned {resp.status_code}")

            data = resp.json()
            cache_control = resp.headers.get("Cache-Control", "")
            accept_encoding = resp.headers.get("Accept-Encoding", "identity")
            return (data, cache_control, accept_encoding)

        except httpx.HTTPError as e:
            _logger.debug("Key discovery failed: host=%s error=%s", self.base_url, e)
            raise KeyDiscoveryError(f"Failed to fetch keys: {e}") from e

    def _encrypt_request(
        self,
        body: bytes,
        keys: dict[KemId, bytes],
    ) -> tuple[AsyncIterator[bytes], dict[str, str], SenderContext]:
        """
        Encrypt request body using base class encryption with async wrapper.

        Returns an async iterator for memory-efficient chunked upload.
        Uses HTTP chunked transfer encoding (Transfer-Encoding: chunked).

        Returns:
            Tuple of (async_chunk_iterator, headers_dict, sender_context)
        """
        # Use base class sync encryption
        sync_iter, headers, ctx = self._encrypt_request_sync(body, keys)

        # Wrap sync iterator in async generator for httpx
        async def async_stream() -> AsyncIterator[bytes]:
            for chunk in sync_iter:
                yield chunk

        _logger.debug(
            "Request encryption prepared: body_size=%d streaming=True",
            len(body),
        )

        return (async_stream(), headers, ctx)

    def _encrypt_stream_sync(
        self,
        chunks: Iterator[bytes],
        keys: dict[KemId, bytes],
    ) -> tuple[Iterator[bytes], dict[str, str], SenderContext]:
        """
        Encrypt sync stream using RequestEncryptor.feed/finalize API.

        Mirrors aiohttp's _encrypt_stream_async but for sync iterators.
        Uses the same RequestEncryptor API that handles buffering internally.

        Note: Does NOT support whole-body compression (same as aiohttp).
        Each chunk is encrypted as-is with IDENTITY encoding.

        Args:
            chunks: Sync iterator of raw bytes (from MultipartStream)
            keys: Platform public keys from discovery

        Returns:
            Tuple of (sync_encrypted_iterator, headers_dict, sender_context)
        """
        pk_r = keys.get(KemId.DHKEM_X25519_HKDF_SHA256)
        if not pk_r:
            raise KeyDiscoveryError("No X25519 key available from platform")

        # Create encryptor without compression (streaming doesn't support whole-body)
        # Same pattern as aiohttp.py:515-520
        encryptor = RequestEncryptor(
            public_key=pk_r,
            psk=self.psk,
            psk_id=self.psk_id,
            compress=False,  # Streaming doesn't support whole-body compression
        )

        def encrypt_gen() -> Iterator[bytes]:
            for chunk in chunks:
                yield from encryptor.feed(chunk)
            yield from encryptor.finalize()

        return encrypt_gen(), encryptor.get_headers(), encryptor.context

    async def _encrypt_multipart_stream(
        self,
        multipart_iter: Iterator[bytes],
    ) -> tuple[AsyncIterator[bytes], dict[str, str], SenderContext]:
        """
        Encrypt streaming multipart with O(chunk_size) memory.

        Async wrapper around _encrypt_stream_sync for httpx compatibility.
        Mirrors aiohttp's _encrypt_stream_async pattern.

        Args:
            multipart_iter: Iterator from httpx encode_multipart_data()

        Returns:
            Tuple of (async_encrypted_iterator, headers_dict, sender_context)
        """
        keys = await self._ensure_keys()
        sync_iter, headers, ctx = self._encrypt_stream_sync(multipart_iter, keys)

        async def async_stream() -> AsyncIterator[bytes]:
            for chunk in sync_iter:
                yield chunk

        return async_stream(), headers, ctx

    def _prepare_body(
        self,
        *,
        content: Any,
        data: Any,
        files: Any,
        json: Any,
        headers: dict[str, Any],
    ) -> tuple[bytes | None, str | None, Any, Any]:
        """
        Prepare request body for encryption, preserving original Content-Type.

        Note: files/multipart is handled separately in request() with streaming.
        This method only handles json, content, and raw bytes.

        Returns:
            Tuple of (body_bytes, original_content_type, files, data)
            - body_bytes: Serialized body ready for encryption (or None)
            - original_content_type: Original Content-Type to preserve (or None)
            - files: Passed through unchanged (handled in request())
            - data: Cleared to None if serialized into body
        """
        body: bytes | None = None
        original_content_type: str | None = None

        # Note: multipart files are handled in request() with streaming encryption
        # to avoid O(payload_size) memory usage from b"".join()
        if json is not None:
            # NOTE: json.dumps().encode() creates 2 allocations (str → bytes).
            # For higher throughput, consider orjson.dumps() which returns bytes
            # directly (1 allocation, ~10x faster). Not included as dependency
            # to keep the library lightweight.
            body = json_module.dumps(json).encode()
            original_content_type = "application/json"
        elif content is not None and isinstance(content, bytes):
            body = content
            # Preserve Content-Type from headers if present
            original_content_type = headers.get("Content-Type")
        elif data is not None and isinstance(data, bytes):
            body = data
            original_content_type = headers.get("Content-Type")

        return body, original_content_type, files, data

    async def request(
        self,
        method: str,
        url: str,
        *,
        content: Any = None,
        data: Any = None,
        files: Any = None,
        json: Any = None,
        params: Any = None,
        headers: Any = None,
        cookies: Any = None,
        auth: Any = None,
        follow_redirects: bool | None = None,
        timeout: Any = None,
        extensions: Any = None,
    ) -> httpx.Response | DecryptedResponse:
        """
        Make an encrypted HTTP request.

        Matches httpx.AsyncClient.request() signature exactly.

        Args:
            method: HTTP method
            url: URL (relative to base_url or absolute)
            content: Raw body content
            data: Form data
            files: Files to upload
            json: JSON body (will be serialized and encrypted)
            params: URL query parameters
            headers: Request headers
            cookies: Request cookies
            auth: Authentication
            follow_redirects: Whether to follow redirects
            timeout: Request timeout
            extensions: Request extensions

        Returns:
            httpx.Response for plain responses or SSE streams,
            DecryptedResponse for encrypted standard responses
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        # Build headers dict early (needed for Content-Type preservation)
        request_headers = dict(headers or {})

        sender_ctx: SenderContext | None = None
        encrypted_content: Any = content

        # Handle multipart files with streaming (avoids full materialization)
        # Pattern from aiohttp.py:601-620 - handle BEFORE _prepare_body()
        if files is not None:
            multipart_headers, multipart_stream = encode_multipart_data(
                data=data or {},
                files=files,
                boundary=None,  # Let httpx generate random boundary
            )
            original_content_type = multipart_headers.get("Content-Type", "")

            # Use streaming encryption - memory O(chunk_size) instead of O(total_size)
            # _rechunk_iter breaks large chunks (e.g., full file content) into 64KB pieces
            encrypted_stream, crypto_headers, ctx = await self._encrypt_multipart_stream(
                _rechunk_iter(iter(multipart_stream))
            )
            request_headers.update(crypto_headers)
            request_headers[HEADER_HPKE_CONTENT_TYPE] = original_content_type
            request_headers["Content-Type"] = "application/octet-stream"
            sender_ctx = ctx
            encrypted_content = encrypted_stream

            _logger.debug("Multipart encrypted with streaming: method=%s url=%s", method, url)

            # Clear to skip _prepare_body processing (multipart fully handled above)
            files = data = json = None

        # Prepare body for encryption (handles json, raw bytes - NOT multipart)
        body, original_content_type, files, data = self._prepare_body(
            content=content, data=data, files=files, json=json, headers=request_headers
        )

        # Encrypt if we have a body to encrypt (non-multipart path)
        if body:
            keys = await self._ensure_keys()
            encrypted_stream, crypto_headers, ctx = self._encrypt_request(body, keys)
            request_headers.update(crypto_headers)
            # Preserve original Content-Type for downstream middleware
            if original_content_type:
                request_headers[HEADER_HPKE_CONTENT_TYPE] = original_content_type
            request_headers["Content-Type"] = "application/octet-stream"
            sender_ctx = ctx
            # Pass iterator for streaming upload (chunked transfer encoding)
            encrypted_content = encrypted_stream
            _logger.debug(
                "Request encrypted: method=%s url=%s body_size=%d streaming=True",
                method,
                url,
                len(body),
            )
            # Clear json/data since we're using encrypted content
            json = None
            data = None

        # Build kwargs for httpx request, filtering out None values
        optional_kwargs = {
            "content": encrypted_content,
            "data": data,
            "files": files,
            "json": json,
            "params": params,
            "cookies": cookies,
            "auth": auth,
            "follow_redirects": follow_redirects,
            "timeout": timeout,
            "extensions": extensions,
        }
        kwargs: dict[str, Any] = {
            "method": method,
            "url": url,
            "headers": request_headers,
            **{k: v for k, v in optional_kwargs.items() if v is not None},
        }

        # Use manual streaming mode for progressive SSE delivery
        # See: https://www.python-httpx.org/async/#streaming-responses
        # client.request() buffers entire response; stream=True enables true streaming
        request = self._client.build_request(**kwargs)
        response = await self._client.send(request, stream=True)

        # Store context per-response for concurrent request safety (needed for SSE)
        if sender_ctx:
            self._response_contexts[response] = sender_ctx

            # Detect response type from headers (available before body is read)
            # Detection: X-HPKE-Stream present AND Content-Type is NOT text/event-stream
            if HEADER_HPKE_STREAM in response.headers:
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" not in content_type:
                    # Non-SSE: read full body into memory for DecryptedResponse
                    await response.aread()
                    _logger.debug("Encrypted response detected: url=%s", url)
                    return DecryptedResponse(
                        response,
                        sender_ctx,
                        release_encrypted=self._release_encrypted,
                    )
                # SSE: keep stream open for iter_sse() to consume progressively
                _logger.debug("SSE streaming response: url=%s", url)
            elif self.require_encryption:
                # We sent encrypted request but got plaintext response
                await response.aclose()
                raise EncryptionRequiredError("Response was not encrypted")
            else:
                # Unencrypted response: read full body
                await response.aread()
        else:
            # No encryption context: read full body for non-encrypted request
            await response.aread()

        return response

    # HTTP method convenience methods - match httpx.AsyncClient signatures exactly

    async def get(
        self,
        url: str,
        *,
        params: Any = None,
        headers: Any = None,
        cookies: Any = None,
        auth: Any = None,
        follow_redirects: bool | None = None,
        timeout: Any = None,
        extensions: Any = None,
    ) -> httpx.Response | DecryptedResponse:
        """GET request."""
        return await self.request(
            "GET",
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def head(
        self,
        url: str,
        *,
        params: Any = None,
        headers: Any = None,
        cookies: Any = None,
        auth: Any = None,
        follow_redirects: bool | None = None,
        timeout: Any = None,
        extensions: Any = None,
    ) -> httpx.Response | DecryptedResponse:
        """HEAD request."""
        return await self.request(
            "HEAD",
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def options(
        self,
        url: str,
        *,
        params: Any = None,
        headers: Any = None,
        cookies: Any = None,
        auth: Any = None,
        follow_redirects: bool | None = None,
        timeout: Any = None,
        extensions: Any = None,
    ) -> httpx.Response | DecryptedResponse:
        """OPTIONS request."""
        return await self.request(
            "OPTIONS",
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def post(
        self,
        url: str,
        *,
        content: Any = None,
        data: Any = None,
        files: Any = None,
        json: Any = None,
        params: Any = None,
        headers: Any = None,
        cookies: Any = None,
        auth: Any = None,
        follow_redirects: bool | None = None,
        timeout: Any = None,
        extensions: Any = None,
    ) -> httpx.Response | DecryptedResponse:
        """POST request."""
        return await self.request(
            "POST",
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def put(
        self,
        url: str,
        *,
        content: Any = None,
        data: Any = None,
        files: Any = None,
        json: Any = None,
        params: Any = None,
        headers: Any = None,
        cookies: Any = None,
        auth: Any = None,
        follow_redirects: bool | None = None,
        timeout: Any = None,
        extensions: Any = None,
    ) -> httpx.Response | DecryptedResponse:
        """PUT request."""
        return await self.request(
            "PUT",
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def patch(
        self,
        url: str,
        *,
        content: Any = None,
        data: Any = None,
        files: Any = None,
        json: Any = None,
        params: Any = None,
        headers: Any = None,
        cookies: Any = None,
        auth: Any = None,
        follow_redirects: bool | None = None,
        timeout: Any = None,
        extensions: Any = None,
    ) -> httpx.Response | DecryptedResponse:
        """PATCH request."""
        return await self.request(
            "PATCH",
            url,
            content=content,
            data=data,
            files=files,
            json=json,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def delete(
        self,
        url: str,
        *,
        params: Any = None,
        headers: Any = None,
        cookies: Any = None,
        auth: Any = None,
        follow_redirects: bool | None = None,
        timeout: Any = None,
        extensions: Any = None,
    ) -> httpx.Response | DecryptedResponse:
        """DELETE request."""
        return await self.request(
            "DELETE",
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            auth=auth,
            follow_redirects=follow_redirects,
            timeout=timeout,
            extensions=extensions,
        )

    async def iter_sse(
        self,
        response: httpx.Response | DecryptedResponse,
    ) -> AsyncIterator[bytes]:
        """
        Iterate over encrypted SSE stream, yielding decrypted chunks.

        Transparent pass-through: yields the exact SSE chunks the server sent.
        Events, comments, retry directives - everything is preserved exactly.

        Args:
            response: Response from an SSE endpoint

        Yields:
            Raw SSE chunks as bytes exactly as the server sent them

        Note:
            Response is automatically closed when iteration completes or on error.
            With stream=True, we must call aclose() to avoid connection leaks.
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

        # httpx uses aiter_bytes() for streaming
        # Ensure response is closed when done (required for stream=True mode)
        try:
            async for chunk in response.aiter_bytes():
                for line in line_parser.feed(chunk):
                    if not line:
                        # Empty line = event boundary (WHATWG spec)
                        data_value = extract_sse_data(current_event_lines)
                        if data_value:
                            raw_chunk = decryptor.decrypt(data_value)
                            yield raw_chunk
                        current_event_lines = []
                    else:
                        current_event_lines.append(line)
        finally:
            # Clean up response context and close connection
            self._response_contexts.pop(response, None)
            await response.aclose()

"""Buffered aiohttp client adapter for the native hpke-http protocol."""

from __future__ import annotations

import json as json_module
from collections.abc import AsyncIterable, Iterable, Mapping
from contextlib import ExitStack
from http import HTTPStatus
from typing import Any, Literal, cast
from urllib.parse import urlencode

import aiohttp
from multidict import CIMultiDict, CIMultiDictProxy
from typing_extensions import Self
from yarl import URL

from hpke_http.middleware._native_async import run_native
from hpke_http.protocol import Client, Limits, Method, ProtocolError, Request
from hpke_http.transport import (
    REQUEST_MEDIA_TYPE,
    RESPONSE_MEDIA_TYPE,
    TransportError,
    filter_request_headers,
    filter_response_headers,
    max_body_len,
    max_envelope_len,
    media_type,
)

_DEFAULT_LIMITS = Limits()
_CLIENT_ERROR_STATUS = 400
_OUTER_OK_STATUS = 200


class HPKEResponse:
    """A complete authenticated response with a small aiohttp-like surface.

    This class is not ``aiohttp.ClientResponse``. It exposes only ``status``,
    ``headers``, ``url``, ``method``, ``reason``, the properties and buffered
    readers below, status checking, and no-op release/close methods. There is no
    live socket, streaming body, redirect history, or cookie-jar side effect.
    """

    def __init__(
        self,
        *,
        status: int,
        headers: Iterable[tuple[str, str]],
        body: bytes,
        url: URL,
        method: str,
    ) -> None:
        fields = CIMultiDict[str]()
        for name, value in headers:
            fields.add(name, value)
        self.status = status
        self.headers = CIMultiDictProxy(fields)
        self._body = body
        self.url = url
        self.method = method
        self.reason = _reason_phrase(status)

    @property
    def ok(self) -> bool:
        """Return whether the authenticated status is below 400."""
        return self.status < _CLIENT_ERROR_STATUS

    @property
    def closed(self) -> bool:
        """Return ``True`` because the outer response is already consumed."""
        return True

    @property
    def content_type(self) -> str:
        """Return the normalized authenticated media type without parameters."""
        return media_type(self.headers.get("content-type"))

    async def read(self) -> bytes:
        """Return the complete authenticated body."""
        return self._body

    async def text(self, encoding: str | None = None, errors: str = "strict") -> str:
        """Decode the buffered body with an explicit or declared character set."""
        selected = encoding or _charset(self.headers.get("content-type")) or "utf-8"
        return self._body.decode(selected, errors)

    async def json(
        self,
        *,
        encoding: str | None = None,
        loads: Any = json_module.loads,
        content_type: str | None = "application/json",
    ) -> Any:
        """Decode JSON after optional exact or structured-suffix media-type validation."""
        if content_type is not None and not _json_content_type_matches(self.content_type, content_type):
            raise aiohttp.ContentTypeError(
                self.request_info,
                (),
                status=self.status,
                message=f"unexpected content type {self.content_type!r}",
                headers=self.headers,
            )
        return loads(await self.text(encoding=encoding))

    @property
    def request_info(self) -> aiohttp.RequestInfo:
        """Return synthetic request metadata for aiohttp-compatible errors."""
        return aiohttp.RequestInfo(self.url, self.method, CIMultiDictProxy(CIMultiDict[str]()), self.url)

    def raise_for_status(self) -> None:
        """Raise ``aiohttp.ClientResponseError`` for status 400 or greater."""
        if self.status >= _CLIENT_ERROR_STATUS:
            raise aiohttp.ClientResponseError(
                self.request_info,
                (),
                status=self.status,
                message=self.reason,
                headers=self.headers,
            )

    def release(self) -> None:
        """Compatibility no-op; the outer response is already released."""

    def close(self) -> None:
        """Compatibility no-op; the outer response is already closed."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        self.release()


class _RequestContextManager:
    def __init__(self, coroutine: Any) -> None:
        self._coroutine = coroutine
        self._response: HPKEResponse | None = None

    def __await__(self) -> Any:
        return self._coroutine.__await__()

    async def __aenter__(self) -> HPKEResponse:
        response = await self._coroutine
        self._response = response
        return response

    async def __aexit__(self, *_exc_info: object) -> None:
        if self._response is not None:
            self._response.release()


class HPKEClientSession:
    """Compose a dedicated aiohttp transport with one recipient configuration.

    ``base_url`` resolves logical relative targets. ``transport_endpoint`` can
    select one fixed HTTPS envelope endpoint. The connector and accepted session
    options configure only the dedicated outer connection pool. Session default
    credentials, cookies, headers, and environment proxy state are rejected or
    disabled. ``compression`` selects optional Rust protocol body coding, not
    HTTP ``Content-Encoding``. This is a supported subset, not a drop-in
    ``ClientSession``.
    """

    def __init__(
        self,
        recipient_public_key: bytes,
        recipient_key_id: bytes,
        psk: bytes,
        psk_id: bytes,
        *,
        base_url: str | URL | None = None,
        transport_endpoint: str | URL | None = None,
        limits: Limits = _DEFAULT_LIMITS,
        compression: Literal["gzip", "zstd"] | None = None,
        connector: aiohttp.BaseConnector | None = None,
        **session_options: Any,
    ) -> None:
        sensitive_options = {"auth", "connector", "cookie_jar", "cookies", "headers"}.intersection(session_options)
        if sensitive_options:
            names = ", ".join(sorted(sensitive_options))
            msg = f"outer transport session cannot use credential-bearing defaults: {names}"
            raise ValueError(msg)
        if session_options.get("trust_env") is True:
            raise ValueError("outer transport session cannot use ambient proxy or netrc state: trust_env")

        with ExitStack() as cleanup:
            client = Client(recipient_public_key, recipient_key_id, psk, psk_id, limits=limits, compression=compression)
            cleanup.callback(client.close)
            http = aiohttp.ClientSession(
                connector=connector,
                cookie_jar=aiohttp.DummyCookieJar(),
                **session_options,
            )
            cleanup.pop_all()
        self._http = http
        self._base_url = URL(base_url) if base_url is not None else None
        self._transport_endpoint = transport_endpoint
        self._client = client
        self._max_request_len = max_body_len(limits)
        self._max_response_envelope_len = max_envelope_len(limits)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    @property
    def closed(self) -> bool:
        """Return whether the dedicated outer connection pool is closed."""
        return self._http.closed

    async def close(self) -> None:
        """Release credential copies and close the outer connection pool."""
        self._client.close()
        await self._http.close()

    def request(
        self,
        method: str,
        url: str | URL,
        *,
        params: Mapping[str, str | int | float] | None = None,
        headers: Mapping[str, str] | Iterable[tuple[str, str]] | None = None,
        data: object = None,
        json: object = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> _RequestContextManager:
        """Create an awaitable context manager for one protected request.

        The target must resolve to absolute HTTPS without embedded credentials.
        Bodies can be bytes, text, async byte iterables, or JSON-compatible data
        accepted by this adapter. The complete response authenticates before the
        context manager yields :class:`HPKEResponse`.
        """
        return _RequestContextManager(
            self._request(
                method,
                url,
                params=params,
                headers=headers,
                data=data,
                json=json,
                timeout=timeout,
            )
        )

    def get(self, url: str | URL, **options: Any) -> _RequestContextManager:
        """Create one protected ``GET`` request context."""
        return self.request("GET", url, **options)

    def options(self, url: str | URL, **options: Any) -> _RequestContextManager:
        """Create one protected ``OPTIONS`` request context."""
        return self.request("OPTIONS", url, **options)

    def head(self, url: str | URL, **options: Any) -> _RequestContextManager:
        """Create one protected ``HEAD`` request context."""
        return self.request("HEAD", url, **options)

    def post(self, url: str | URL, **options: Any) -> _RequestContextManager:
        """Create one protected ``POST`` request context."""
        return self.request("POST", url, **options)

    def put(self, url: str | URL, **options: Any) -> _RequestContextManager:
        """Create one protected ``PUT`` request context."""
        return self.request("PUT", url, **options)

    def patch(self, url: str | URL, **options: Any) -> _RequestContextManager:
        """Create one protected ``PATCH`` request context."""
        return self.request("PATCH", url, **options)

    def delete(self, url: str | URL, **options: Any) -> _RequestContextManager:
        """Create one protected ``DELETE`` request context."""
        return self.request("DELETE", url, **options)

    async def _request(
        self,
        method: str,
        url: str | URL,
        *,
        params: Mapping[str, str | int | float] | None,
        headers: Mapping[str, str] | Iterable[tuple[str, str]] | None,
        data: object,
        json: object,
        timeout: aiohttp.ClientTimeout | None,
    ) -> HPKEResponse:
        target = self._resolve(url)
        if params is not None:
            target = target.update_query(params)
        body, default_content_type = await _encode_body(data, json, self._max_request_len)
        fields = CIMultiDict[str](headers or ())
        if default_content_type is not None and "content-type" not in fields:
            fields["content-type"] = default_content_type
        try:
            protocol_method = Method(method.upper())
        except ValueError as error:
            raise ProtocolError("unsupported_method", "request method is not supported by hpke-http") from error

        protected = await run_native(
            self._client.protect,
            Request(
                method=protocol_method,
                authority=target.raw_authority,
                path=target.raw_path_qs,
                headers=filter_request_headers(fields.items()),
                body=body,
            ),
        )
        try:
            endpoint = target if self._transport_endpoint is None else self._resolve(self._transport_endpoint)
            envelope = await self._exchange(endpoint, protected.envelope, timeout)
            authenticated = await run_native(protected.open_response, envelope)
            authenticated_headers = filter_response_headers(
                (field.name, field.value) for field in authenticated.headers
            )
            return HPKEResponse(
                status=authenticated.status,
                headers=((field.name, field.value) for field in authenticated_headers),
                body=authenticated.body,
                url=target,
                method=method.upper(),
            )
        finally:
            protected.close()

    def _resolve(self, value: str | URL) -> URL:
        candidate = URL(value)
        if not candidate.is_absolute():
            if self._base_url is None:
                raise TransportError("invalid_target", "protected requests require an absolute HTTPS URL")
            candidate = self._base_url.join(candidate)
        if candidate.scheme != "https" or candidate.user is not None or candidate.password is not None:
            raise TransportError(
                "invalid_target",
                "protected requests require an HTTPS URL without embedded credentials",
            )
        return candidate.with_fragment(None)

    async def _exchange(
        self,
        endpoint: URL,
        envelope: bytes,
        timeout: aiohttp.ClientTimeout | None,
    ) -> bytes:
        try:
            headers = {
                "accept": RESPONSE_MEDIA_TYPE,
                "accept-encoding": "identity",
                "cache-control": "no-store",
                "content-type": REQUEST_MEDIA_TYPE,
            }
            context = (
                self._http.post(
                    endpoint,
                    data=envelope,
                    headers=headers,
                    allow_redirects=False,
                    auth=None,
                    auto_decompress=False,
                    timeout=timeout,
                )
                if timeout is not None
                else self._http.post(
                    endpoint,
                    data=envelope,
                    headers=headers,
                    allow_redirects=False,
                    auth=None,
                    auto_decompress=False,
                )
            )
            async with context as response:
                if response.status != _OUTER_OK_STATUS:
                    raise TransportError(
                        "outer_status",
                        f"protected endpoint returned outer status {response.status}",
                        status_code=response.status,
                    )
                content_types = response.headers.getall("content-type", ())
                if len(content_types) != 1 or media_type(content_types[0]) != RESPONSE_MEDIA_TYPE:
                    raise TransportError(
                        "outer_content_type",
                        f"protected endpoint must return {RESPONSE_MEDIA_TYPE}",
                    )
                content_encodings = response.headers.getall("content-encoding", ())
                if len(content_encodings) > 1 or (content_encodings and content_encodings[0].lower() != "identity"):
                    raise TransportError("outer_content_encoding", "protected envelope must not use content encoding")
                return await _read_outer_body(response, self._max_response_envelope_len)
        except TransportError:
            raise
        except (aiohttp.ClientError, TimeoutError) as error:
            raise TransportError("network_error", "protected HTTP request failed") from error


async def _encode_body(data: object, json: object, maximum: int) -> tuple[bytes, str | None]:
    if data is not None and json is not None:
        raise ValueError("data and json are mutually exclusive")
    content_type: str | None = None
    if json is not None:
        body = json_module.dumps(json, ensure_ascii=False, separators=(",", ":")).encode()
        content_type = "application/json"
    elif data is None:
        body = b""
    elif isinstance(data, (bytes, bytearray)):
        body = bytes(data)
    elif isinstance(data, str):
        body = data.encode()
        content_type = "text/plain; charset=utf-8"
    elif isinstance(data, Mapping):
        body = urlencode(cast(Mapping[str, Any], data), doseq=True).encode()
        content_type = "application/x-www-form-urlencoded"
    elif isinstance(data, AsyncIterable):
        body = await _collect_async(cast(AsyncIterable[bytes], data), maximum)
    else:
        raise TypeError("data must be buffered bytes, text, a form mapping, or an async bytes iterable")
    if len(body) > maximum:
        raise TransportError("request_too_large", "buffered request body exceeds the configured limit")
    return body, content_type


async def _collect_async(source: AsyncIterable[bytes], maximum: int) -> bytes:
    body = bytearray()
    length = 0
    async for chunk in source:
        length += len(chunk)
        if length > maximum:
            raise TransportError("request_too_large", "buffered request body exceeds the configured limit")
        body.extend(chunk)
    return bytes(body)


async def _read_outer_body(response: aiohttp.ClientResponse, maximum: int) -> bytes:
    declared = response.headers.get("content-length")
    if declared is not None and declared.isdecimal() and int(declared) > maximum:
        raise TransportError("response_too_large", "buffered response exceeds the configured limit")
    body = bytearray()
    length = 0
    async for chunk in response.content.iter_chunked(64 * 1024):
        length += len(chunk)
        if length > maximum:
            raise TransportError("response_too_large", "buffered response exceeds the configured limit")
        body.extend(chunk)
    return bytes(body)


def _charset(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    for parameter in content_type.split(";")[1:]:
        name, separator, value = parameter.partition("=")
        if separator and name.strip().lower() == "charset":
            return value.strip().strip('"')
    return None


def _json_content_type_matches(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    if expected != "application/json" or not actual.startswith("application/"):
        return False
    subtype = actual.removeprefix("application/")
    return len(subtype) > len("+json") and subtype.endswith("+json")


def _reason_phrase(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return ""


__all__ = ["HPKEClientSession", "HPKEResponse"]

"""aiohttp client adapter for checked finite and live SSE responses."""

from __future__ import annotations

import json as json_module
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import ExitStack
from http import HTTPStatus
from typing import Any, Literal, cast
from urllib.parse import urlencode

import aiohttp
from multidict import CIMultiDict, CIMultiDictProxy
from typing_extensions import Self
from yarl import URL

from hpke_http.middleware._discovery import (
    KEY_MEDIA_TYPE,
    Discover,
    PinnedKey,
    make_discovered_client,
    protect_discovered_client,
    read_key_record,
    same_origin,
    validate_client_configuration,
    validate_endpoint,
    validate_key_response,
    validate_target_origin,
)
from hpke_http.middleware._native_async import run_native
from hpke_http.middleware._records import CheckedStream
from hpke_http.protocol import Client, Limits, Method, ProtectedRequest, ProtocolError, Request, StateError
from hpke_http.transport import (
    REQUEST_MEDIA_TYPE,
    RESPONSE_MEDIA_TYPE,
    TransportError,
    filter_request_headers,
    filter_response_headers,
    max_body_len,
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
        """Do nothing; the finite outer response is already released."""

    def close(self) -> None:
        """Do nothing; the finite outer response is already closed."""

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
    """Compose a dedicated aiohttp transport with one fixed key endpoint.

    ``Discover()`` fetches one key per call. ``PinnedKey`` sends no key GET.
    The connector and accepted session
    options configure only the dedicated outer connection pool. Session default
    credentials, cookies, headers, and environment proxy state are rejected or
    disabled. ``compression`` selects optional Rust protocol body coding, not
    HTTP ``Content-Encoding``. This is a supported subset, not a drop-in
    ``ClientSession``.
    """

    def __init__(
        self,
        endpoint: str,
        key_source: Discover | PinnedKey,
        psk: bytes,
        psk_id: bytes,
        *,
        target_origin: str | None = None,
        limits: Limits = _DEFAULT_LIMITS,
        compression: Literal["gzip", "zstd"] | None = None,
        connector: aiohttp.BaseConnector | None = None,
        **session_options: Any,
    ) -> None:
        endpoint = validate_endpoint(endpoint)
        target_key = validate_target_origin(target_origin, endpoint)
        validate_client_configuration(psk, psk_id, limits, compression)
        if type(key_source) not in (Discover, PinnedKey):
            raise TypeError("key_source must be Discover() or PinnedKey")
        sensitive_options = {
            "auth",
            "connector",
            "cookie_jar",
            "cookies",
            "headers",
            "base_url",
            "transport_endpoint",
        }.intersection(session_options)
        if sensitive_options:
            names = ", ".join(sorted(sensitive_options))
            msg = f"outer transport session cannot use credential-bearing defaults: {names}"
            raise ValueError(msg)
        if session_options.pop("trust_env", False) is not False:
            raise ValueError("outer transport session cannot use ambient proxy or netrc state: trust_env")

        with ExitStack() as cleanup:
            client = (
                Client(key_source.public_key, key_source.key_id, psk, psk_id, limits=limits, compression=compression)
                if isinstance(key_source, PinnedKey)
                else None
            )
            if client is not None:
                cleanup.callback(client.close)
            http = aiohttp.ClientSession(
                connector=connector,
                cookie_jar=aiohttp.DummyCookieJar(),
                trust_env=False,
                **session_options,
            )
            cleanup.pop_all()
        self._http = http
        self._endpoint = URL(endpoint)
        self._target_origin = target_key
        self._client = client
        self._psk = bytes(psk) if isinstance(key_source, Discover) else b""
        self._psk_id = bytes(psk_id)
        self._limits = limits
        self._compression: Literal["gzip", "zstd"] | None = compression
        self._closed = False
        self._max_request_len = max_body_len(limits)
        self._streams: set[HPKEStreamResponse] = set()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()

    @property
    def closed(self) -> bool:
        """Return whether the dedicated outer connection pool is closed."""
        return self._closed

    async def close(self) -> None:
        """Release credential copies and close the outer connection pool."""
        self._closed = True
        for stream in tuple(self._streams):
            await stream.aclose()
        self._psk = b""
        if self._client is not None:
            self._client.close()
        await self._http.close()

    def stream(
        self,
        method: str,
        url: str | URL,
        *,
        params: Mapping[str, str | int | float] | None = None,
        headers: Mapping[str, str] | Iterable[tuple[str, str]] | None = None,
        data: object = None,
        json: object = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> _StreamContext:
        """Open a context-owned reply after its START record passes."""
        return _StreamContext(self, method, url, params, headers, data, json, timeout)

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
        stream = await self._open_stream(
            method, url, params=params, headers=headers, data=data, json=json, timeout=timeout, live=False
        )
        try:
            if stream.mode == "sse":
                raise StateError("use stream() for an SSE response")
            body = await stream.read()
            return HPKEResponse(
                status=stream.status,
                headers=stream.headers.items(),
                body=body,
                url=stream.url,
                method=stream.method,
            )
        finally:
            await stream.aclose()

    async def _open_stream(
        self,
        method: str,
        url: str | URL,
        *,
        params: Mapping[str, str | int | float] | None,
        headers: Mapping[str, str] | Iterable[tuple[str, str]] | None,
        data: object,
        json: object,
        timeout: aiohttp.ClientTimeout | None,
        live: bool,
    ) -> HPKEStreamResponse:
        if self.closed:
            raise StateError("client is closed")
        target = self._resolve(url)
        if params is not None:
            target = target.update_query(params)
        if not same_origin(str(target), self._target_origin):
            raise TransportError("invalid_target", "logical target has the wrong HTTPS origin")
        body, default_content_type = await _encode_body(data, json, self._max_request_len)
        fields = CIMultiDict[str](headers or ())
        if default_content_type is not None and "content-type" not in fields:
            fields["content-type"] = default_content_type
        try:
            protocol_method = Method(method.upper())
        except ValueError as error:
            raise ProtocolError("unsupported_method", "request method is not supported by hpke-http") from error

        protected = await self._protect(target, protocol_method, fields, body, timeout)
        try:
            if self.closed:
                raise StateError("client is closed")
            outer_headers = {
                "accept": RESPONSE_MEDIA_TYPE,
                "accept-encoding": "identity",
                "cache-control": "no-store",
                "content-type": REQUEST_MEDIA_TYPE,
            }
            chosen_timeout = timeout or self._http.timeout
            if live and timeout is None:
                session_timeout = self._http.timeout
                chosen_timeout = aiohttp.ClientTimeout(
                    total=None,
                    connect=session_timeout.connect,
                    sock_connect=session_timeout.sock_connect,
                    sock_read=None,
                )
            try:
                outer = await self._http.post(
                    self._endpoint,
                    data=protected.envelope,
                    headers=outer_headers,
                    allow_redirects=False,
                    auth=None,
                    auto_decompress=False,
                    raise_for_status=False,
                    timeout=chosen_timeout,
                )
            except (aiohttp.ClientError, TimeoutError) as error:
                raise TransportError("network_error", "protected HTTP request failed") from error
            driver: CheckedStream | None = None
            try:
                _check_outer(outer)
                driver = CheckedStream(
                    protected.into_opener(), outer.content.iter_chunked(64 * 1024), _close_outer(outer)
                )
                await driver.start()
                filtered = filter_response_headers((field.name, field.value) for field in driver.headers)
                response = HPKEStreamResponse(
                    driver,
                    driver.status,
                    ((field.name, field.value) for field in filtered),
                    target,
                    method.upper(),
                    self._streams,
                )
                self._streams.add(response)
                return response
            except BaseException:
                if driver is not None:
                    await driver.aclose()
                else:
                    outer.close()
                raise
        finally:
            protected.close()

    async def _protect(
        self,
        target: URL,
        method: Method,
        fields: CIMultiDict[str],
        body: bytes,
        timeout: aiohttp.ClientTimeout | None,
    ) -> ProtectedRequest:
        request = Request(
            method=method,
            authority=target.raw_authority,
            path=target.raw_path_qs,
            headers=filter_request_headers(fields.items()),
            body=body,
        )
        if self._client is not None:
            return await run_native(self._client.protect, request)
        client = await self._discover_client(timeout)
        return await protect_discovered_client(client, request)

    async def _discover_client(self, timeout: aiohttp.ClientTimeout | None) -> Client:
        try:
            response = await self._http.get(
                self._endpoint,
                headers={"accept": KEY_MEDIA_TYPE, "accept-encoding": "identity", "cache-control": "no-store"},
                allow_redirects=False,
                auth=None,
                auto_decompress=False,
                raise_for_status=False,
                timeout=timeout or self._http.timeout,
            )
        except (aiohttp.ClientError, TimeoutError) as error:
            raise TransportError("discovery_network", "key GET failed") from error
        try:
            validate_key_response(
                response.status,
                response.headers.getall("content-type", ()),
                response.headers.getall("content-encoding", ()),
            )
            key_id, public_key = await read_key_record(response.content.iter_chunked(64 * 1024))
            if self.closed:
                raise StateError("client is closed")
            return make_discovered_client(public_key, key_id, self._psk, self._psk_id, self._limits, self._compression)
        except (aiohttp.ClientError, TimeoutError) as error:
            raise TransportError("discovery_network", "key GET failed") from error
        finally:
            response.close()

    def _resolve(self, value: str | URL) -> URL:
        candidate = URL(value)
        if not candidate.is_absolute():
            raise TransportError("invalid_target", "protected requests require an absolute HTTPS URL")
        if candidate.scheme != "https" or candidate.user is not None or candidate.password is not None:
            raise TransportError(
                "invalid_target",
                "protected requests require an HTTPS URL without embedded credentials",
            )
        return candidate.with_fragment(None)


class HPKEStreamResponse:
    """Checked head and one context-owned aiohttp body reader."""

    def __init__(
        self,
        driver: CheckedStream,
        status: int,
        headers: Iterable[tuple[str, str]],
        url: URL,
        method: str,
        registry: set[HPKEStreamResponse],
    ) -> None:
        fields = CIMultiDict[str]()
        for name, value in headers:
            fields.add(name, value)
        self._driver = driver
        self._registry = registry
        self.status = status
        self.headers = CIMultiDictProxy(fields)
        self.url = url
        self.method = method
        self.mode = driver.mode

    async def read(self) -> bytes:
        """Return finite body bytes after END and outer EOF pass."""
        return await self._driver.read()

    async def iter_sse(self) -> AsyncIterator[bytes]:
        """Yield clear bytes for each complete checked SSE block."""
        async for block in self._driver.iter_sse():
            yield block

    async def aclose(self) -> None:
        """Close the live outer reply and native opener."""
        self._registry.discard(self)
        await self._driver.aclose()


class _StreamContext:
    def __init__(
        self,
        client: HPKEClientSession,
        method: str,
        url: str | URL,
        params: Mapping[str, str | int | float] | None,
        headers: Mapping[str, str] | Iterable[tuple[str, str]] | None,
        data: object,
        json: object,
        timeout: aiohttp.ClientTimeout | None,
    ) -> None:
        self._client = client
        self._method = method
        self._url = url
        self._params = params
        self._headers = headers
        self._data = data
        self._json = json
        self._timeout = timeout
        self._response: HPKEStreamResponse | None = None

    async def __aenter__(self) -> HPKEStreamResponse:
        self._response = await self._client._open_stream(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            self._method,
            self._url,
            params=self._params,
            headers=self._headers,
            data=self._data,
            json=self._json,
            timeout=self._timeout,
            live=True,
        )
        return self._response

    async def __aexit__(self, *_exc_info: object) -> None:
        if self._response is not None:
            await self._response.aclose()


def _check_outer(response: aiohttp.ClientResponse) -> None:
    if response.status != _OUTER_OK_STATUS:
        raise TransportError(
            "outer_status", f"protected endpoint returned outer status {response.status}", status_code=response.status
        )
    content_types = response.headers.getall("content-type", ())
    if len(content_types) != 1 or media_type(content_types[0]) != RESPONSE_MEDIA_TYPE:
        raise TransportError("outer_content_type", f"protected endpoint must return {RESPONSE_MEDIA_TYPE}")
    content_encodings = response.headers.getall("content-encoding", ())
    if len(content_encodings) > 1 or (content_encodings and content_encodings[0].lower() != "identity"):
        raise TransportError("outer_content_encoding", "protected envelope must not use content encoding")


def _close_outer(response: aiohttp.ClientResponse) -> Callable[[], Awaitable[None]]:
    async def close() -> None:
        response.close()

    return close


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


__all__ = ["HPKEClientSession", "HPKEResponse", "HPKEStreamResponse"]

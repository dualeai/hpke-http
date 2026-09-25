"""HTTPX client adapter for checked finite and live SSE responses."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import ExitStack
from http.cookiejar import Cookie, CookieJar, DefaultCookiePolicy
from typing import Any

import httpx
from typing_extensions import Self

from hpke_http.middleware._discovery import (
    KEY_MEDIA_TYPE,
    PinnedKey,
    make_discovered_client,
    read_key_record,
    same_origin,
    validate_client_configuration,
    validate_endpoint,
    validate_key_response,
    validate_target_origin,
)
from hpke_http.middleware._native_async import run_native
from hpke_http.middleware._records import CheckedStream, seal_request_chunk
from hpke_http.middleware._shared_key import KeyLease, SharedKey
from hpke_http.protocol import (
    Client,
    Limits,
    Method,
    ProtectedRequest,
    ProtocolError,
    RequestHead,
    StateError,
    StreamRequestSealer,
)
from hpke_http.transport import (
    REQUEST_MEDIA_TYPE,
    RESPONSE_MEDIA_TYPE,
    TransportError,
    filter_request_headers,
    filter_response_headers,
    validate_outer_response,
)

_DEFAULT_LIMITS = Limits()


class _RejectAllCookiePolicy(DefaultCookiePolicy):
    def set_ok(self, cookie: Cookie, request: Any) -> bool:
        del cookie, request
        return False

    def return_ok(self, cookie: Cookie, request: Any) -> bool:
        del cookie, request
        return False


class _RequestOnlyTransport(httpx.AsyncBaseTransport):
    """Let HTTPX build logical requests without opening a second pool."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        del request
        raise StateError("logical request client cannot send outer HTTP")


class DiscoveredEndpoint:
    """Share one checked key and one HTTPX pool for one HTTPS endpoint.

    Set TLS and pool options here. Use this source on one event loop. A
    cancelled caller does not stop a key GET needed by other callers.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        get_timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
        **transport_options: Any,
    ) -> None:
        endpoint = validate_endpoint(endpoint)
        forbidden = {
            "auth",
            "cookies",
            "event_hooks",
            "follow_redirects",
            "transport",
            "base_url",
            "headers",
            "params",
            "transport_endpoint",
            "proxy",
        }.intersection(transport_options)
        if forbidden:
            raise ValueError(f"outer transport cannot use request defaults: {', '.join(sorted(forbidden))}")
        if transport_options.pop("trust_env", False) is not False:
            raise ValueError("outer transport cannot use ambient proxy credentials: trust_env")
        self._endpoint = httpx.URL(endpoint)
        self._http = httpx.AsyncClient(
            transport=transport,
            cookies=CookieJar(policy=_RejectAllCookiePolicy()),
            follow_redirects=False,
            trust_env=False,
            **transport_options,
        )
        self._key = SharedKey(self._fetch_key, get_timeout_s=get_timeout_s)
        self._closed = False

    @property
    def endpoint(self) -> str:
        """Return the full HTTPS endpoint bound to this source."""
        return str(self._endpoint)

    async def _fetch_key(self) -> tuple[bytes, bytes, int]:
        request = httpx.Request(
            "GET",
            self._endpoint,
            headers={"accept": KEY_MEDIA_TYPE, "accept-encoding": "identity", "cache-control": "no-store"},
        )
        request.extensions["timeout"] = self._http.timeout.as_dict()
        try:
            response = await self._http.send(request, stream=True, auth=None, follow_redirects=False)
        except httpx.HTTPError as error:
            raise TransportError("discovery_network", "key GET failed") from error
        try:
            validate_key_response(
                response.status_code,
                response.headers.get_list("content-type"),
                response.headers.get_list("content-encoding"),
            )
            return await read_key_record(_raw_chunks(response))
        except httpx.HTTPError as error:
            raise TransportError("discovery_network", "key GET failed") from error
        finally:
            await response.aclose()

    async def get_key(self) -> KeyLease:
        """Get a lease and share a needed GET with other callers.

        The lease has ``key_id``, ``public_key``, and ``valid()``.
        Its remaining life can end before the caller starts a POST.
        """
        return await self._key.get()

    async def aclose(self) -> None:
        """Stop new calls and close the outer pool."""
        if self._closed:
            return
        self._closed = True
        await self._key.aclose()
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()


class HPKEAsyncClient:
    """Compose ``httpx.AsyncClient`` with one fixed key endpoint.

    Requests are bounded. Live SSE replies yield one checked block at a time.
    Redirects are never followed for the outer exchange.

    A ``DiscoveredEndpoint`` shares one key GET and owns the outer connection.
    It also supplies the endpoint URL. Set TLS and pool options on that source.
    ``PinnedKey`` uses its fixed key and requires ``endpoint=``;
    ``transport``, TLS, and pool options then set the outer connection.
    Default headers and params become logical request fields and target bytes.
    Logical headers are authenticated after transport fields are removed.
    Ambient auth, cookies, event hooks, redirects, and environment
    credentials are rejected or disabled. Every body uses protected records.
    """

    def __init__(
        self,
        key_source: DiscoveredEndpoint | PinnedKey,
        psk: bytes,
        psk_id: bytes,
        *,
        endpoint: str | None = None,
        target_origin: str | None = None,
        limits: Limits = _DEFAULT_LIMITS,
        transport: httpx.AsyncBaseTransport | None = None,
        **client_options: Any,
    ) -> None:
        if type(key_source) not in (DiscoveredEndpoint, PinnedKey):
            raise TypeError("key_source must be DiscoveredEndpoint or PinnedKey")
        if isinstance(key_source, DiscoveredEndpoint):
            if endpoint is not None:
                raise ValueError("the shared source owns the endpoint")
            endpoint = key_source.endpoint
            if transport is not None:
                raise ValueError("the shared source owns the outer transport")
        else:
            if endpoint is None:
                raise ValueError("endpoint is required with PinnedKey")
            endpoint = validate_endpoint(endpoint)
        target_key = validate_target_origin(target_origin, endpoint)
        validate_client_configuration(psk, psk_id, limits)
        forbidden = {
            "auth",
            "cookies",
            "event_hooks",
            "follow_redirects",
            "transport",
            "base_url",
            "transport_endpoint",
            "proxy",
        }.intersection(client_options)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            msg = f"outer transport cannot use ambient request state: {names}"
            raise ValueError(msg)
        if client_options.pop("trust_env", False) is not False:
            raise ValueError("outer transport cannot use ambient proxy credentials: trust_env")
        if isinstance(key_source, DiscoveredEndpoint):
            unused = set(client_options) - {"headers", "params", "timeout"}
            if unused:
                raise ValueError(f"the shared source owns outer transport options: {', '.join(sorted(unused))}")

        with ExitStack() as cleanup:
            client = (
                Client(key_source.public_key, key_source.key_id, psk, psk_id, limits=limits)
                if isinstance(key_source, PinnedKey)
                else None
            )
            if client is not None:
                cleanup.callback(client.close)
            http = httpx.AsyncClient(
                transport=_RequestOnlyTransport() if isinstance(key_source, DiscoveredEndpoint) else transport,
                cookies=CookieJar(policy=_RejectAllCookiePolicy()),
                follow_redirects=False,
                trust_env=False,
                **client_options,
            )
            cleanup.pop_all()
        self._http = http
        self._source = key_source if isinstance(key_source, DiscoveredEndpoint) else None
        self._outer_http = (
            self._source._http if self._source is not None else http  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        )
        self._client = client
        self._endpoint = httpx.URL(endpoint)
        self._target_origin = target_key
        self._psk = bytes(psk) if self._source is not None else b""
        self._psk_id = bytes(psk_id)
        self._limits = limits
        self._closed = False
        self._streams: set[HPKEStreamResponse] = set()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release credentials and close this client's HTTPX resources.

        A shared source owns the outer pool; close that source separately.
        """
        self._closed = True
        for stream in tuple(self._streams):
            await stream.aclose()
        self._psk = b""
        if self._client is not None:
            self._client.close()
        await self._http.aclose()

    def stream(self, method: str, url: str | httpx.URL, **request_options: Any) -> _StreamContext:
        """Open a context-owned response after its START record passes."""
        return _StreamContext(self, method, url, request_options)

    async def request(
        self,
        method: str,
        url: str | httpx.URL,
        **request_options: Any,
    ) -> httpx.Response:
        """Protect one logical request and authenticate its complete response.

        The method must be supported by ``hpke-http/3`` and the resolved target
        must be absolute HTTPS without embedded credentials. The returned
        ``httpx.Response`` is fully buffered and authenticated. ``TransportError``
        reports target, network, outer-response, content-coding, and request-body
        size errors. ``ProtocolError`` reports protected-message failures,
        including response record limits.
        """
        stream = await self._open_stream(method, url, request_options, live=False)
        try:
            if stream.mode == "sse":
                raise StateError("use stream() for an SSE response")
            body = await stream.read()
            return httpx.Response(
                status_code=stream.status_code,
                headers=stream.headers,
                content=body,
                request=stream.request,
            )
        finally:
            await stream.aclose()

    async def _open_stream(  # noqa: PLR0912, PLR0915
        self,
        method: str,
        url: str | httpx.URL,
        request_options: dict[str, Any],
        *,
        live: bool,
    ) -> HPKEStreamResponse:
        if self._closed:
            raise StateError("client is closed")
        logical = self._http.build_request(method, url, **request_options)
        target = _https_url(logical.url)
        if not same_origin(str(target), self._target_origin):
            raise TransportError("invalid_target", "logical target has the wrong HTTPS origin")
        try:
            protocol_method = Method(logical.method.upper())
        except ValueError as error:
            raise ProtocolError("unsupported_method", "request method is not supported by hpke-http") from error
        logical_headers = filter_request_headers(logical.headers.multi_items())

        protected: ProtectedRequest | None = None
        sealer: StreamRequestSealer | None = None
        response_right: list[ProtectedRequest] = []
        discovered: Client | None = None
        lease: KeyLease | None = None
        try:
            client = self._client
            if client is None:
                discovered, lease = await self._discover_client()
                client = discovered
            head = RequestHead(
                method=protocol_method,
                authority=target.netloc.decode("ascii"),
                path=target.raw_path.decode("ascii"),
                headers=logical_headers,
            )

            async def begin(selected: Client) -> tuple[StreamRequestSealer, bytes]:
                try:
                    return await run_native(selected.begin_stream, head)
                except ProtocolError as error:
                    if discovered is not None and error.code in {"invalid_configuration", "crypto_failure"}:
                        raise TransportError(
                            "discovery_response", "key endpoint returned an unusable public key"
                        ) from error
                    raise

            sealer, first = await begin(client)
            if lease is not None and not lease.valid():
                sealer.close()
                if discovered is not None:
                    discovered.close()
                discovered, lease = await self._discover_client()
                sealer, first = await begin(discovered)
            writer = sealer
            source = logical.stream

            async def encoded_request() -> AsyncIterator[bytes]:
                try:
                    if lease is not None and not lease.valid():
                        raise TransportError("discovery_expired", "key lifetime ended before protected POST")
                    yield first
                    async for chunk in _request_chunks(source):
                        async for frame in seal_request_chunk(writer, chunk):
                            yield frame
                    end, right = await run_native(writer.finish)
                    response_right.append(right)
                    yield end
                finally:
                    writer.close()

            outer_content: AsyncIterator[bytes] = encoded_request()
        except BaseException:
            await _close_request_stream(logical.stream)
            if sealer is not None:
                sealer.close()
            raise
        finally:
            if discovered is not None:
                discovered.close()
        try:
            if self._closed:
                raise StateError("client is closed")
            outer = httpx.Request(
                "POST",
                self._endpoint,
                headers={
                    "accept": RESPONSE_MEDIA_TYPE,
                    "accept-encoding": "identity",
                    "cache-control": "no-store",
                    "content-type": REQUEST_MEDIA_TYPE,
                },
                content=outer_content,
            )
            timeout = dict(logical.extensions.get("timeout", self._http.timeout.as_dict()))
            if live and "timeout" not in request_options:
                timeout["read"] = None
            outer.extensions["timeout"] = timeout
            try:
                response = await self._outer_http.send(outer, stream=True, auth=None, follow_redirects=False)
            except httpx.HTTPError as error:
                raise TransportError("network_error", "protected HTTP request failed") from error
            driver: CheckedStream | None = None
            try:
                _check_outer(response)
                if not response_right:
                    raise TransportError("network_error", "request ended before its protected END")
                protected = response_right.pop()
                driver = CheckedStream(protected.into_opener(), _raw_chunks(response), response.aclose)
                await driver.start()
                headers = filter_response_headers((field.name, field.value) for field in driver.headers)
                handle = HPKEStreamResponse(
                    driver,
                    logical,
                    driver.status,
                    httpx.Headers([(field.name, field.value) for field in headers]),
                    self._streams,
                )
                self._streams.add(handle)
                return handle
            except BaseException:
                if driver is not None:
                    await driver.aclose()
                else:
                    await response.aclose()
                raise
        finally:
            sealer.close()
            if protected is not None:
                protected.close()
            for right in response_right:
                right.close()
            await _close_request_stream(logical.stream)

    async def _discover_client(self) -> tuple[Client, KeyLease]:
        source = self._source
        if source is None:
            raise StateError("client has no discovered endpoint")
        lease = await source.get_key()
        client = make_discovered_client(lease.public_key, lease.key_id, self._psk, self._psk_id, self._limits)
        return client, lease

    async def get(self, url: str | httpx.URL, **options: Any) -> httpx.Response:
        """Send one protected ``GET`` request."""
        return await self.request("GET", url, **options)

    async def options(self, url: str | httpx.URL, **options: Any) -> httpx.Response:
        """Send one protected ``OPTIONS`` request."""
        return await self.request("OPTIONS", url, **options)

    async def head(self, url: str | httpx.URL, **options: Any) -> httpx.Response:
        """Send one protected ``HEAD`` request."""
        return await self.request("HEAD", url, **options)

    async def post(self, url: str | httpx.URL, **options: Any) -> httpx.Response:
        """Send one protected ``POST`` request."""
        return await self.request("POST", url, **options)

    async def put(self, url: str | httpx.URL, **options: Any) -> httpx.Response:
        """Send one protected ``PUT`` request."""
        return await self.request("PUT", url, **options)

    async def patch(self, url: str | httpx.URL, **options: Any) -> httpx.Response:
        """Send one protected ``PATCH`` request."""
        return await self.request("PATCH", url, **options)

    async def delete(self, url: str | httpx.URL, **options: Any) -> httpx.Response:
        """Send one protected ``DELETE`` request."""
        return await self.request("DELETE", url, **options)


class HPKEStreamResponse:
    """Checked response head plus one context-owned body reader."""

    def __init__(
        self,
        driver: CheckedStream,
        request: httpx.Request,
        status_code: int,
        headers: httpx.Headers,
        registry: set[HPKEStreamResponse],
    ) -> None:
        self._driver = driver
        self._registry = registry
        self.request = request
        self.status_code = status_code
        self.headers = headers
        self.mode = driver.mode

    async def read(self) -> bytes:
        """Return the finite body after DATA, END, and outer EOF pass."""
        return await self._driver.read()

    async def iter_sse(self) -> AsyncIterator[bytes]:
        """Yield clear bytes for each complete checked SSE block."""
        async for block in self._driver.iter_sse():
            yield block

    async def aclose(self) -> None:
        """Release the outer socket and native opener."""
        self._registry.discard(self)
        await self._driver.aclose()


class _StreamContext:
    def __init__(self, client: HPKEAsyncClient, method: str, url: str | httpx.URL, options: dict[str, Any]) -> None:
        self._client = client
        self._method = method
        self._url = url
        self._options = options
        self._handle: HPKEStreamResponse | None = None

    async def __aenter__(self) -> HPKEStreamResponse:
        self._handle = await self._client._open_stream(self._method, self._url, self._options, live=True)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        return self._handle

    async def __aexit__(self, *_exc_info: object) -> None:
        if self._handle is not None:
            await self._handle.aclose()


def _check_outer(response: httpx.Response) -> None:
    validate_outer_response(
        response.status_code,
        response.headers.get_list("content-type"),
        response.headers.get_list("content-encoding"),
    )


async def _raw_chunks(response: httpx.Response) -> AsyncIterator[bytes]:
    if response.is_stream_consumed:
        yield response.content
    else:
        async for chunk in response.aiter_raw():
            yield chunk


async def _request_chunks(source: httpx.SyncByteStream | httpx.AsyncByteStream) -> AsyncIterator[bytes]:
    if isinstance(source, httpx.AsyncByteStream):
        async for chunk in source:
            yield chunk
    else:
        for chunk in source:
            yield chunk


async def _close_request_stream(source: httpx.SyncByteStream | httpx.AsyncByteStream) -> None:
    if isinstance(source, httpx.AsyncByteStream):
        await source.aclose()
    else:
        source.close()


def _https_url(value: httpx.URL) -> httpx.URL:
    if not value.is_absolute_url or value.scheme != "https" or value.userinfo:
        raise TransportError(
            "invalid_target",
            "protected requests require an absolute HTTPS URL without embedded credentials",
        )
    return value.copy_with(fragment=None)


__all__ = ["DiscoveredEndpoint", "HPKEAsyncClient", "HPKEStreamResponse"]

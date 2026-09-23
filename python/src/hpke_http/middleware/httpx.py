"""Buffered httpx client adapter for the native hpke-http protocol."""

from __future__ import annotations

from collections.abc import AsyncIterable, Iterable, Mapping
from contextlib import ExitStack
from http.cookiejar import Cookie, CookieJar, DefaultCookiePolicy
from typing import Any, Literal, cast

import httpx
from typing_extensions import Self

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
_OUTER_OK_STATUS = 200


class _RejectAllCookiePolicy(DefaultCookiePolicy):
    def set_ok(self, cookie: Cookie, request: Any) -> bool:
        del cookie, request
        return False

    def return_ok(self, cookie: Cookie, request: Any) -> bool:
        del cookie, request
        return False


class HPKEAsyncClient:
    """Compose ``httpx.AsyncClient`` with one explicit recipient configuration.

    Requests and responses are buffered because protocol version 1 authenticates
    one bounded message at a time. Redirects are never followed for the outer
    exchange, and client cookies or authorization are not copied to it.

    ``base_url`` resolves logical relative targets. ``transport_endpoint`` can
    select one fixed HTTPS envelope endpoint. ``transport`` and other accepted
    client options configure only the dedicated outer connection pool. Default
    and per-request logical headers are authenticated after transport fields are
    removed. Ambient auth, cookies, event hooks, redirects, and environment
    credentials are rejected or disabled. ``compression`` selects optional
    Rust protocol body coding, not HTTP ``Content-Encoding``.
    """

    def __init__(
        self,
        recipient_public_key: bytes,
        recipient_key_id: bytes,
        psk: bytes,
        psk_id: bytes,
        *,
        base_url: str | httpx.URL = "",
        transport_endpoint: str | httpx.URL | None = None,
        limits: Limits = _DEFAULT_LIMITS,
        compression: Literal["gzip", "zstd"] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        **client_options: Any,
    ) -> None:
        forbidden = {"auth", "cookies", "event_hooks", "follow_redirects", "transport"}.intersection(client_options)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            msg = f"outer transport cannot use ambient request state: {names}"
            raise ValueError(msg)
        if client_options.pop("trust_env", False) is not False:
            raise ValueError("outer transport cannot use ambient proxy credentials: trust_env")

        with ExitStack() as cleanup:
            client = Client(recipient_public_key, recipient_key_id, psk, psk_id, limits=limits, compression=compression)
            cleanup.callback(client.close)
            http = httpx.AsyncClient(
                base_url=base_url,
                transport=transport,
                cookies=CookieJar(policy=_RejectAllCookiePolicy()),
                follow_redirects=False,
                trust_env=False,
                **client_options,
            )
            cleanup.pop_all()
        self._http = http
        self._client = client
        self._transport_endpoint = transport_endpoint
        self._max_request_len = max_body_len(limits)
        self._max_response_envelope_len = max_envelope_len(limits)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Release credentials and close the HTTP connection pool."""
        self._client.close()
        await self._http.aclose()

    async def request(
        self,
        method: str,
        url: str | httpx.URL,
        **request_options: Any,
    ) -> httpx.Response:
        """Protect one logical request and authenticate its complete response.

        The method must be supported by ``hpke-http/1`` and the resolved target
        must be absolute HTTPS without embedded credentials. The returned
        ``httpx.Response`` is fully buffered and authenticated. ``TransportError``
        reports target, network, outer-response, content-coding, and size errors;
        ``ProtocolError`` reports protected-message failures.
        """
        logical = self._http.build_request(method, url, **request_options)
        target = _https_url(logical.url)
        body = await _read_request_body(logical, self._max_request_len)
        try:
            protocol_method = Method(logical.method.upper())
        except ValueError as error:
            raise ProtocolError("unsupported_method", "request method is not supported by hpke-http") from error

        protected = await run_native(
            self._client.protect,
            Request(
                method=protocol_method,
                authority=target.netloc.decode("ascii"),
                path=target.raw_path.decode("ascii"),
                headers=filter_request_headers(logical.headers.multi_items()),
                body=body,
            ),
        )
        try:
            endpoint = target if self._transport_endpoint is None else self._resolve(self._transport_endpoint)
            envelope = await self._exchange(endpoint, protected.envelope)
            authenticated = await run_native(protected.open_response, envelope)
            headers = filter_response_headers((field.name, field.value) for field in authenticated.headers)
            return httpx.Response(
                status_code=authenticated.status,
                headers=[(field.name, field.value) for field in headers],
                content=authenticated.body,
                request=logical,
            )
        finally:
            protected.close()

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

    def _resolve(self, value: str | httpx.URL) -> httpx.URL:
        candidate = httpx.URL(value)
        if candidate.is_relative_url:
            base = self._http.base_url
            if base.is_relative_url:
                raise TransportError("invalid_target", "transport endpoint must be an absolute HTTPS URL")
            candidate = base.join(candidate)
        return _https_url(candidate)

    async def _exchange(self, endpoint: httpx.URL, envelope: bytes) -> bytes:
        outer = httpx.Request(
            "POST",
            endpoint,
            headers={
                "accept": RESPONSE_MEDIA_TYPE,
                "accept-encoding": "identity",
                "cache-control": "no-store",
                "content-type": REQUEST_MEDIA_TYPE,
            },
            content=envelope,
        )
        try:
            response = await self._http.send(outer, stream=True, auth=None, follow_redirects=False)
        except httpx.HTTPError as error:
            raise TransportError("network_error", "protected HTTP request failed") from error

        try:
            try:
                if response.status_code != _OUTER_OK_STATUS:
                    raise TransportError(
                        "outer_status",
                        f"protected endpoint returned outer status {response.status_code}",
                        status_code=response.status_code,
                    )
                if media_type(response.headers.get("content-type")) != RESPONSE_MEDIA_TYPE:
                    raise TransportError(
                        "outer_content_type",
                        f"protected endpoint must return {RESPONSE_MEDIA_TYPE}",
                    )
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise TransportError("outer_content_encoding", "protected envelope must not use content encoding")
                return await _read_response_body(response, self._max_response_envelope_len)
            finally:
                await response.aclose()
        except httpx.HTTPError as error:
            raise TransportError("network_error", "protected HTTP response failed") from error


async def _read_request_body(request: httpx.Request, maximum: int) -> bytes:
    _reject_declared_oversize(request.headers, maximum, "request_too_large")
    try:
        body = request.content
    except httpx.RequestNotRead:
        body = await _collect_stream(request.stream, maximum, "request_too_large")
    if len(body) > maximum:
        raise TransportError("request_too_large", "buffered request body exceeds the configured limit")
    return body


async def _read_response_body(response: httpx.Response, maximum: int) -> bytes:
    _reject_declared_oversize(response.headers, maximum, "response_too_large")
    try:
        body = response.content
    except httpx.ResponseNotRead:
        pass
    else:
        if len(body) > maximum:
            raise TransportError("response_too_large", "buffered response exceeds the configured limit")
        return body
    body = bytearray()
    length = 0
    async for chunk in response.aiter_raw():
        length += len(chunk)
        if length > maximum:
            raise TransportError("response_too_large", "buffered response exceeds the configured limit")
        body.extend(chunk)
    return bytes(body)


async def _collect_stream(stream: object, maximum: int, code: str) -> bytes:
    body = bytearray()
    length = 0
    if isinstance(stream, AsyncIterable):
        iterator = cast(AsyncIterable[bytes], stream)
        async for chunk in iterator:
            data = bytes(chunk)
            length += len(data)
            if length > maximum:
                raise TransportError(code, "buffered request body exceeds the configured limit")
            body.extend(data)
    elif isinstance(stream, Iterable):
        for chunk in cast(Iterable[bytes], stream):
            data = bytes(chunk)
            length += len(data)
            if length > maximum:
                raise TransportError(code, "buffered request body exceeds the configured limit")
            body.extend(data)
    else:
        raise TypeError("httpx request body is not iterable")
    return bytes(body)


def _reject_declared_oversize(headers: Mapping[str, str], maximum: int, code: str) -> None:
    declared = headers.get("content-length")
    if declared is not None and declared.isdecimal() and int(declared) > maximum:
        raise TransportError(code, "buffered HTTP body exceeds the configured limit")


def _https_url(value: httpx.URL) -> httpx.URL:
    if not value.is_absolute_url or value.scheme != "https" or value.userinfo:
        raise TransportError(
            "invalid_target",
            "protected requests require an absolute HTTPS URL without embedded credentials",
        )
    return value.copy_with(fragment=None)


__all__ = ["HPKEAsyncClient"]

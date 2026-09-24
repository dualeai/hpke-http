"""ASGI middleware with checked live SSE response records."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from urllib.parse import unquote

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from hpke_http.middleware._discovery import KEY_MEDIA_TYPE, encode_key_record, https_origin
from hpke_http.middleware._native_async import run_native
from hpke_http.protocol import (
    Limits,
    OpenedRequest,
    ProtocolError,
    Request,
    Response,
    ResponseSealer,
    Server,
    SseSplitter,
)
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

# Resolve one public PSK ID without receiving authenticated plaintext.
PSKResolver = Callable[[bytes, Scope], bytes | Awaitable[bytes]]
# Atomically reserve one replay ID through its exclusive Unix deadline.
ReplayAdmitter = Callable[[bytes, int, Scope], bool | Awaitable[bool]]

_DEFAULT_LIMITS = Limits()
_OK_STATUS = 200
_METHOD_NOT_ALLOWED_STATUS = 405


def _seal_next_sse_block(
    splitter: SseSplitter,
    sealer: ResponseSealer,
    chunk: bytes,
    offset: int,
) -> tuple[int, bytes | None]:
    used, block = splitter.feed(chunk, offset, 64 * 1024)
    return used, None if block is None else sealer.seal_sse_block(block)


class HPKEMiddleware:
    """Authenticate protected requests before invoking an ASGI application.

    ``replay_admitter`` must atomically reserve a replay ID if absent, retain it
    until the supplied exclusive Unix deadline, and return ``True`` only for
    that first reservation. Errors and uncertain results fail closed. The
    middleware never releases plaintext before this callback accepts the
    request.

    ``psk_resolver`` raises ``LookupError`` for an unknown public ID. Other
    resolver exceptions indicate an unavailable credential source.

    ``transport_path`` must match the full ASGI ``scope["path"]``, including any
    mount prefix. It serves a public key through GET and accepts a protected
    request through POST. Other paths go to the host application.

    ``expected_authority`` fixes the authenticated authority accepted by this
    endpoint. Without it, the middleware compares the inner authority with the
    outer ASGI ``Host`` field.

    The middleware closes its native server after the application's ASGI
    lifespan ends. With a host that does not send lifespan events, construct
    this wrapper directly so the application can retain it and call
    :meth:`close` during host shutdown. ``compression=True`` accepts the
    optional Rust protocol body-coding extension, not HTTP representation coding.
    """

    def __init__(
        self,
        app: ASGIApp,
        recipient_private_key: bytes,
        recipient_key_id: bytes,
        psk_resolver: PSKResolver,
        replay_admitter: ReplayAdmitter,
        *,
        transport_path: str,
        limits: Limits = _DEFAULT_LIMITS,
        compression: bool = False,
        expected_authority: str | None = None,
    ) -> None:
        if not transport_path.startswith("/") or "?" in transport_path or "#" in transport_path:
            raise ValueError("transport_path must be one local path")
        self.app = app
        self._server = Server(recipient_private_key, recipient_key_id, limits=limits, compression=compression)
        self._key_record = encode_key_record(recipient_key_id, self._server.public_key)
        self._psk_resolver = psk_resolver
        self._replay_admitter = replay_admitter
        self._transport_path = transport_path
        self._expected_authority = expected_authority
        self._max_request_envelope_len = max_envelope_len(limits)
        self._max_response_body_len = max_body_len(limits)

    def close(self) -> None:
        """Release the native recipient-key configuration."""
        self._server.close()

    @property
    def closed(self) -> bool:
        """Return whether the native recipient-key configuration is closed."""
        return self._server.closed

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            try:
                await self.app(scope, receive, send)
            finally:
                self.close()
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path") != self._transport_path:
            await self.app(scope, receive, send)
            return
        try:
            if scope.get("query_string"):
                raise _HTTPBoundaryError(400, b"key endpoint does not accept a query")
            method = scope.get("method")
            if method not in ("GET", "POST"):
                raise _HTTPBoundaryError(405, b"key endpoint accepts GET and POST")
            if self.closed:
                raise _HTTPBoundaryError(503, b"key endpoint is closed")
            if method == "GET":
                await _send_key_record(send, self._key_record)
                return
            headers = _validated_outer_headers(scope)
            try:
                envelope = await _read_body(receive, headers, self._max_request_envelope_len)
            except ValueError as error:
                raise _HTTPBoundaryError(400, b"invalid protected request") from error
            opened = await self._open(envelope, scope)
            try:
                await self._run_application(opened.request, opened, scope, receive, send)
            finally:
                opened.close()
        except _HTTPBoundaryError as failure:
            await _send_error(send, failure.status, failure.body)
            return

    async def _open(self, envelope: bytes, scope: Scope) -> OpenedRequest:
        try:
            preparsed = self._server.preparse(envelope)
        except ProtocolError as error:
            raise _HTTPBoundaryError(400, b"invalid protected request") from error
        try:
            try:
                psk_result = self._psk_resolver(preparsed.psk_id, scope)
                psk = await psk_result if inspect.isawaitable(psk_result) else psk_result
            except LookupError as error:
                raise _HTTPBoundaryError(400, b"invalid protected request") from error
            except Exception as error:
                raise _HTTPBoundaryError(503, b"credential resolution unavailable") from error
            try:
                authenticated = await run_native(preparsed.authenticate, psk)
            except ProtocolError as error:
                status, body = {
                    "clock_unavailable": (503, b"trusted clock unavailable"),
                    "invalid_request_time": (409, b"protected request time rejected"),
                }.get(error.code, (400, b"invalid protected request"))
                raise _HTTPBoundaryError(status, body) from error
        finally:
            preparsed.close()

        try:
            try:
                admission_result = self._replay_admitter(
                    authenticated.replay_id,
                    authenticated.retain_until_exclusive,
                    scope,
                )
                accepted = await admission_result if inspect.isawaitable(admission_result) else admission_result
            except Exception as error:
                raise _HTTPBoundaryError(503, b"replay admission unavailable") from error
            if accepted is not True:
                raise _HTTPBoundaryError(409, b"protected request replay rejected")
            try:
                return authenticated.admit(accepted=True)
            except ProtocolError as error:
                if error.code == "clock_unavailable":
                    raise _HTTPBoundaryError(503, b"trusted clock unavailable") from error
                if error.code == "invalid_request_time":
                    raise _HTTPBoundaryError(409, b"protected request expired before dispatch") from error
                raise _HTTPBoundaryError(400, b"invalid replay admission") from error
        finally:
            authenticated.close()

    async def _run_application(
        self, request: Request, opened: OpenedRequest, scope: Scope, receive: Receive, send: Send
    ) -> None:
        try:
            target_allowed = _target_is_allowed(
                scope,
                request.authority,
                self._expected_authority,
            )
        except (UnicodeError, ValueError, TransportError):
            target_allowed = False
        if not target_allowed:
            raise _HTTPBoundaryError(421, b"authenticated target does not match this endpoint")

        response_done = asyncio.Event()
        sink = _ResponseSink(opened, send, self._max_response_body_len, request.method.value, response_done)
        try:
            inner_scope = _inner_scope(scope, request)
        except TransportError as error:
            raise _HTTPBoundaryError(400, b"unsupported authenticated request content coding") from error
        peer_disconnected = asyncio.Event()
        inner_receive = _checked_receive(request.body, peer_disconnected, response_done)
        watcher = asyncio.create_task(_watch_disconnect(receive, peer_disconnected))
        app = asyncio.ensure_future(self.app(inner_scope, inner_receive, sink.send))
        try:
            done, _ = await asyncio.wait((app, watcher), return_when=asyncio.FIRST_COMPLETED)
            if watcher in done and peer_disconnected.is_set() and not sink.complete:
                app.cancel()
                await asyncio.gather(app, return_exceptions=True)
                raise _ResponseCaptureError("peer disconnected before response end")
            await app
            if not sink.complete:
                raise _ResponseCaptureError("application did not send a final body part")
        except Exception as error:
            if sink.outer_start_attempted:
                await sink.abort()
                raise RuntimeError("protected response ended without END") from error
            if isinstance(error, TransportError) and error.code == "inner_content_encoding":
                raise _HTTPBoundaryError(500, b"application returned unsupported content coding") from error
            if (
                isinstance(error, _ResponseCaptureError)
                and str(error) == "application returned a forbidden response body"
            ):
                raise _HTTPBoundaryError(500, b"application returned a forbidden response body") from error
            raise _HTTPBoundaryError(500, b"application returned an invalid ASGI response") from error
        finally:
            response_done.set()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            sink.close()


class _ResponseSink:
    def __init__(
        self, opened: OpenedRequest, outer_send: Send, maximum: int, method: str, response_done: asyncio.Event
    ) -> None:
        self._opened = opened
        self._outer_send = outer_send
        self._maximum = maximum
        self._method = method
        self._response_done = response_done
        self._status: int | None = None
        self._headers: list[tuple[str, str]] = []
        self._body = bytearray()
        self._sealer: ResponseSealer | None = None
        self._splitter: SseSplitter | None = None
        self._lock = asyncio.Lock()
        self.complete = False
        self.outer_start_attempted = False

    async def send(self, message: Message) -> None:
        async with self._lock:
            await self._send(message)

    async def _send(self, message: Message) -> None:
        message_type = message["type"]
        if message_type == "http.response.start":
            await self._start_response(message)
            return
        if message_type != "http.response.body" or self._status is None or self.complete:
            raise _ResponseCaptureError("application emitted an invalid ASGI response sequence")
        chunk = bytes(message.get("body", b""))
        if self._sealer is not None:
            await self._send_sse_body(chunk, more_body=bool(message.get("more_body", False)))
            return
        await self._send_finite_body(chunk, more_body=bool(message.get("more_body", False)))

    async def _start_response(self, message: Message) -> None:
        if self._status is not None:
            raise _ResponseCaptureError("application sent response start more than once")
        if message.get("trailers", False):
            raise _ResponseCaptureError("application response trailers are unsupported")
        self._status = int(message["status"])
        try:
            self._headers = _decode_headers(message.get("headers", []))
        except UnicodeError as error:
            raise _ResponseCaptureError("application emitted non-ASCII response headers") from error
        filtered = filter_response_headers(self._headers)
        content_types = [field.value for field in filtered if field.name == "content-type"]
        if len(content_types) > 1:
            raise _ResponseCaptureError("duplicate content type")
        if content_types and media_type(content_types[0]) == "text/event-stream":
            if self._method == "HEAD" or self._status != _OK_STATUS:
                raise _ResponseCaptureError("SSE requires a body and status 200")
            filtered = tuple(field for field in filtered if field.name not in {"content-length", "content-encoding"})
            sealer, first = await run_native(self._opened.into_sealer, self._status, filtered)
            self._sealer = sealer
            self._splitter = SseSplitter(self._maximum)
            await self._start_outer()
            await self._outer_send({"type": "http.response.body", "body": first, "more_body": True})
        else:
            self._headers = [(field.name, field.value) for field in filtered]

    async def _send_sse_body(self, chunk: bytes, *, more_body: bool) -> None:
        splitter = self._splitter
        sealer = self._sealer
        if splitter is None or sealer is None:
            raise _ResponseCaptureError("SSE response is closed")
        offset = 0
        while offset < len(chunk):
            used, frame = await run_native(_seal_next_sse_block, splitter, sealer, chunk, offset)
            if used == 0:
                raise _ResponseCaptureError("SSE splitter made no progress")
            offset += used
            if frame is not None:
                await self._outer_send({"type": "http.response.body", "body": frame, "more_body": True})
        if not more_body:
            splitter.finish()
            end = await run_native(sealer.finish)
            await self._outer_send({"type": "http.response.body", "body": end, "more_body": False})
            self.complete = True
            self._response_done.set()

    async def _send_finite_body(self, chunk: bytes, *, more_body: bool) -> None:
        if self._status is None:
            raise _ResponseCaptureError("application body has no response start")
        if len(self._body) + len(chunk) > self._maximum:
            raise _ResponseCaptureError("application response exceeds the configured body limit")
        self._body.extend(chunk)
        if not more_body:
            body = bytes(self._body)
            if self._method == "HEAD":
                body = b""
            elif self._status in {204, 205, 304} and body:
                raise _ResponseCaptureError("application returned a forbidden response body")
            headers = tuple(filter_response_headers(self._headers))
            envelope = await run_native(
                self._opened.protect_response,
                Response(status=self._status, headers=headers, body=body),
            )
            await self._start_outer()
            await self._outer_send({"type": "http.response.body", "body": envelope, "more_body": False})
            self.complete = True
            self._response_done.set()

    async def _start_outer(self) -> None:
        self.outer_start_attempted = True
        await self._outer_send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", RESPONSE_MEDIA_TYPE.encode()),
                    (b"cache-control", b"no-store, no-transform"),
                ],
            }
        )

    async def abort(self) -> None:
        if self.outer_start_attempted and not self.complete:
            with suppress(Exception):
                await self._outer_send({"type": "http.response.body", "body": b"", "more_body": False})

    def close(self) -> None:
        if self._sealer is not None:
            self._sealer.close()
        if self._splitter is not None:
            self._splitter.close()


class _ResponseCaptureError(RuntimeError):
    """The downstream application violated the bounded ASGI response contract."""


class _HTTPBoundaryError(RuntimeError):
    def __init__(self, status: int, body: bytes) -> None:
        super().__init__(body.decode("ascii"))
        self.status = status
        self.body = body


def _scope_headers(scope: Scope) -> list[tuple[str, str]]:
    return _decode_headers(scope.get("headers", []))


def _validated_outer_headers(scope: Scope) -> list[tuple[str, str]]:
    try:
        headers = _scope_headers(scope)
        content_type = _single_header(headers, "content-type")
        content_encoding = _single_header(headers, "content-encoding")
    except (UnicodeError, ValueError) as error:
        raise _HTTPBoundaryError(400, b"ambiguous protected request headers") from error
    if media_type(content_type) != REQUEST_MEDIA_TYPE:
        raise _HTTPBoundaryError(415, b"unsupported protected request media type")
    if (content_encoding or "identity").lower() != "identity":
        raise _HTTPBoundaryError(415, b"protected envelope must not use content encoding")
    return headers


def _decode_headers(fields: Iterable[tuple[bytes, bytes]]) -> list[tuple[str, str]]:
    return [(name.decode("ascii").lower(), value.decode("ascii")) for name, value in fields]


def _single_header(fields: Iterable[tuple[str, str]], name: str) -> str | None:
    values = [value for field_name, value in fields if field_name == name]
    if len(values) > 1:
        raise ValueError(f"duplicate {name} header")
    return values[0] if values else None


async def _read_body(receive: Receive, headers: list[tuple[str, str]], maximum: int) -> bytes:
    declared = _single_header(headers, "content-length")
    if declared is not None and declared.isdecimal() and int(declared) > maximum:
        raise ValueError("protected request exceeds the configured limit")
    body = bytearray()
    length = 0
    more = True
    while more:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise ValueError("client disconnected")
        if message["type"] != "http.request":
            raise ValueError("invalid ASGI request sequence")
        chunk = bytes(message.get("body", b""))
        length += len(chunk)
        if length > maximum:
            raise ValueError("protected request exceeds the configured limit")
        body.extend(chunk)
        more = bool(message.get("more_body", False))
    return bytes(body)


def _target_is_allowed(
    scope: Scope,
    authority: str,
    expected_authority: str | None,
) -> bool:
    expected = expected_authority or _single_header(_scope_headers(scope), "host")
    return expected is not None and https_origin(f"https://{authority}/") == https_origin(f"https://{expected}/")


def _inner_scope(scope: Scope, request: Request) -> Scope:
    raw_path, separator, query = request.path.partition("?")
    logical_fields = filter_request_headers((field.name, field.value) for field in request.headers)
    fields = [(field.name.encode("ascii"), field.value.encode("ascii")) for field in logical_fields]
    fields.append((b"host", request.authority.encode("ascii")))
    fields.append((b"content-length", str(len(request.body)).encode()))
    inner = dict(scope)
    inner.update(
        {
            "method": request.method.value,
            "scheme": "https",
            "path": unquote(raw_path, errors="strict"),
            "raw_path": raw_path.encode("ascii"),
            "query_string": query.encode("ascii") if separator else b"",
            "headers": fields,
        }
    )
    return inner


def _checked_receive(body: bytes, peer_disconnected: asyncio.Event, response_done: asyncio.Event) -> Receive:
    delivered = False

    async def receive() -> Message:
        nonlocal delivered
        if delivered:
            wait_peer = asyncio.create_task(peer_disconnected.wait())
            wait_done = asyncio.create_task(response_done.wait())
            try:
                await asyncio.wait((wait_peer, wait_done), return_when=asyncio.FIRST_COMPLETED)
            finally:
                wait_peer.cancel()
                wait_done.cancel()
                await asyncio.gather(wait_peer, wait_done, return_exceptions=True)
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def _watch_disconnect(outer_receive: Receive, peer_disconnected: asyncio.Event) -> None:
    while True:
        message = await outer_receive()
        if message["type"] == "http.disconnect":
            peer_disconnected.set()
            return


async def _send_error(send: Send, status: int, body: bytes) -> None:
    headers = [
        (b"content-type", b"text/plain; charset=utf-8"),
        (b"cache-control", b"no-store"),
        (b"content-length", str(len(body)).encode()),
    ]
    if status == _METHOD_NOT_ALLOWED_STATUS:
        headers.append((b"allow", b"GET, POST"))
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


async def _send_key_record(send: Send, record: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", KEY_MEDIA_TYPE.encode()),
                (b"cache-control", b"no-store"),
                (b"content-length", str(len(record)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": record, "more_body": False})


__all__ = ["HPKEMiddleware", "PSKResolver", "ReplayAdmitter"]

"""Buffered ASGI middleware for FastAPI and Starlette."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from urllib.parse import unquote

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from hpke_http.middleware._native_async import run_native
from hpke_http.protocol import Limits, OpenedRequest, ProtocolError, Request, Response, Server
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
_METHOD_NOT_ALLOWED_STATUS = 405


class HPKEMiddleware:
    """Authenticate protected requests before invoking an ASGI application.

    ``replay_admitter`` must atomically reserve a replay ID if absent, retain it
    until the supplied exclusive Unix deadline, and return ``True`` only for
    that first reservation. Errors and uncertain results fail closed. The
    middleware never releases plaintext before this callback accepts the
    request.

    ``psk_resolver`` raises ``LookupError`` for an unknown public ID. Other
    resolver exceptions indicate an unavailable credential source.

    With ``transport_path=None`` every HTTP route is protected and the inner
    target must equal the outer target. A fixed ``transport_path`` protects only
    that route and dispatches the authenticated inner path inside the app.

    ``expected_authority`` fixes the authenticated authority accepted by this
    endpoint. Without it, the middleware compares the inner authority with the
    outer ASGI ``Host`` field. Fixed-transport mode skips only the path match.

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
        limits: Limits = _DEFAULT_LIMITS,
        compression: bool = False,
        transport_path: str | None = None,
        expected_authority: str | None = None,
    ) -> None:
        if transport_path is not None and not transport_path.startswith("/"):
            raise ValueError("transport_path must start with '/'")
        self.app = app
        self._server = Server(recipient_private_key, recipient_key_id, limits=limits, compression=compression)
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
        if self._transport_path is not None and scope.get("path") != self._transport_path:
            await self.app(scope, receive, send)
            return
        try:
            headers = _validated_outer_headers(scope)
            try:
                envelope = await _read_body(receive, headers, self._max_request_envelope_len)
            except ValueError as error:
                raise _HTTPBoundaryError(400, b"invalid protected request") from error
            opened = await self._open(envelope, scope)
            try:
                response_envelope = await self._run_application(opened.request, opened, scope, receive)
            finally:
                opened.close()
        except _HTTPBoundaryError as failure:
            await _send_error(send, failure.status, failure.body)
            return
        await _send_envelope(send, response_envelope)

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

    async def _run_application(self, request: Request, opened: OpenedRequest, scope: Scope, receive: Receive) -> bytes:
        try:
            target_allowed = _target_is_allowed(
                scope,
                request.authority,
                request.path,
                self._expected_authority,
                fixed_transport=self._transport_path is not None,
            )
        except (UnicodeError, ValueError):
            target_allowed = False
        if not target_allowed:
            raise _HTTPBoundaryError(421, b"authenticated target does not match this endpoint")

        capture = _ResponseCapture(self._max_response_body_len)
        try:
            inner_scope = _inner_scope(scope, request)
        except TransportError as error:
            raise _HTTPBoundaryError(400, b"unsupported authenticated request content coding") from error
        try:
            await self.app(inner_scope, _receive_once(request.body, receive), capture.send)
            status, headers, body = capture.finish()
        except _ResponseCaptureError as error:
            raise _HTTPBoundaryError(500, b"application returned an invalid ASGI response") from error
        if request.method.value == "HEAD":
            body = b""
        elif status in {204, 205, 304} and body:
            raise _HTTPBoundaryError(500, b"application returned a forbidden response body")
        try:
            filtered_headers = filter_response_headers(headers)
            return await run_native(
                opened.protect_response, Response(status=status, headers=filtered_headers, body=body)
            )
        except TransportError as error:
            raise _HTTPBoundaryError(500, b"application returned unsupported content coding") from error
        except ProtocolError as error:
            raise _HTTPBoundaryError(500, b"failed to protect application response") from error


class _ResponseCapture:
    def __init__(self, maximum: int) -> None:
        self._maximum = maximum
        self._status: int | None = None
        self._headers: list[tuple[str, str]] = []
        self._body = bytearray()
        self._length = 0
        self._complete = False

    async def send(self, message: Message) -> None:
        message_type = message["type"]
        if message_type == "http.response.start":
            if self._status is not None:
                raise _ResponseCaptureError("application sent response start more than once")
            self._status = int(message["status"])
            try:
                self._headers = _decode_headers(message.get("headers", []))
            except UnicodeError as error:
                raise _ResponseCaptureError("application emitted non-ASCII response headers") from error
            return
        if message_type != "http.response.body" or self._status is None or self._complete:
            raise _ResponseCaptureError("application emitted an invalid ASGI response sequence")
        chunk = bytes(message.get("body", b""))
        self._length += len(chunk)
        if self._length > self._maximum:
            raise _ResponseCaptureError("application response exceeds the configured body limit")
        self._body.extend(chunk)
        self._complete = not bool(message.get("more_body", False))

    def finish(self) -> tuple[int, list[tuple[str, str]], bytes]:
        if self._status is None or not self._complete:
            raise _ResponseCaptureError("application did not complete its ASGI response")
        return self._status, self._headers, bytes(self._body)


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
    if scope.get("method") != "POST":
        raise _HTTPBoundaryError(405, b"protected endpoint requires POST")
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
    path: str,
    expected_authority: str | None,
    *,
    fixed_transport: bool,
) -> bool:
    expected = expected_authority or _single_header(_scope_headers(scope), "host")
    if expected is None or authority.lower() != expected.lower():
        return False
    if fixed_transport:
        return True
    if scope.get("path") is None:
        return False
    raw_path = scope.get("raw_path")
    if raw_path is None:
        raw_path = str(scope["path"]).encode()
    outer_path = bytes(raw_path).decode("ascii")
    query = bytes(scope.get("query_string", b"")).decode("ascii")
    outer_target = outer_path + (f"?{query}" if query else "")
    return outer_target == path


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


def _receive_once(body: bytes, outer_receive: Receive) -> Receive:
    delivered = False

    async def receive() -> Message:
        nonlocal delivered
        if delivered:
            return await outer_receive()
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def _send_error(send: Send, status: int, body: bytes) -> None:
    headers = [
        (b"content-type", b"text/plain; charset=utf-8"),
        (b"cache-control", b"no-store"),
        (b"content-length", str(len(body)).encode()),
    ]
    if status == _METHOD_NOT_ALLOWED_STATUS:
        headers.append((b"allow", b"POST"))
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": headers,
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


async def _send_envelope(send: Send, envelope: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", RESPONSE_MEDIA_TYPE.encode()),
                (b"cache-control", b"no-store"),
                (b"content-length", str(len(envelope)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": envelope, "more_body": False})


__all__ = ["HPKEMiddleware", "PSKResolver", "ReplayAdmitter"]

"""FastAPI/Starlette middleware coverage at the ASGI boundary."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Literal, NoReturn, cast

import pytest
from starlette.types import Message, Receive, Scope, Send

from hpke_http import (
    CheckedRecord,
    Client,
    Header,
    Limits,
    Method,
    PreparsedRequest,
    ProtocolError,
    Request,
    Response,
    Server,
    generate_key_pair,
)
from hpke_http.middleware.fastapi import HPKEMiddleware
from hpke_http.transport import REQUEST_MEDIA_TYPE, RESPONSE_MEDIA_TYPE

KEY_ID = b"primary-2026-09"
PSK = b"a 32-byte minimum test credential!"
PSK_ID = b"tenant-42"


async def _resolve_psk(psk_id: bytes, _scope: Scope) -> bytes:
    assert psk_id == PSK_ID
    return PSK


class _ReplayStore:
    def __init__(self) -> None:
        self.values: dict[bytes, int] = {}

    async def admit(self, replay_id: bytes, retain_until_exclusive: int, _scope: Scope) -> bool:
        if replay_id in self.values:
            return False
        self.values[replay_id] = retain_until_exclusive
        return True


@pytest.mark.asyncio
@pytest.mark.parametrize("compression_coding", [None, "zstd"])
async def test_asgi_middleware_round_trip_with_canonical_ascii_headers(
    compression_coding: Literal["zstd"] | None,
) -> None:
    request_body = b'{"name":"' + b"Ada" * 1024 + b'"}'
    response_body = b'{"id":"' + b"item-1" * 1024 + b'"}'

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        assert scope["method"] == "POST"
        assert (b"x-name", b"Ada") in scope["headers"]
        assert scope["headers"].count((b"content-length", str(len(request_body)).encode())) == 1
        request = await receive()
        assert request["body"] == request_body
        await send(
            {
                "type": "http.response.start",
                "status": 201,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(response_body)).encode()),
                    (b"set-cookie", b"first=1; Path=/"),
                    (b"set-cookie", b"second=2; Path=/"),
                    (b"connection", b"x-hop"),
                    (b"x-hop", b"remove-me"),
                    (b"x-greeting", b"hello"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": response_body, "more_body": False})

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID, compression=compression_coding)
    replay_store = _ReplayStore()
    middleware = HPKEMiddleware(
        app, key_pair.private_key, KEY_ID, _resolve_psk, replay_store.admit, compression=compression_coding is not None
    )
    protected = client.protect(
        Request(
            method=Method.POST,
            authority="api.example.test",
            path="/items",
            headers=(Header("x-name", "Ada"), Header("content-length", str(len(request_body)))),
            body=request_body,
        )
    )
    if compression_coding is not None:
        assert len(protected.envelope) < len(request_body)

    before_admission = int(time.time())
    messages = await _invoke(middleware, protected.envelope)
    assert messages[0]["status"] == 200
    assert (b"content-type", RESPONSE_MEDIA_TYPE.encode()) in messages[0]["headers"]
    assert (b"x-content-type-options", b"nosniff") not in messages[0]["headers"]
    if compression_coding is not None:
        assert len(cast(bytes, messages[1]["body"])) < len(response_body)
    response = protected.open_response(cast(bytes, messages[1]["body"]))
    assert response == Response(
        status=201,
        headers=(
            Header("content-type", "application/json"),
            Header("content-length", str(len(response_body))),
            Header("set-cookie", "first=1; Path=/"),
            Header("set-cookie", "second=2; Path=/"),
            Header("x-greeting", "hello"),
        ),
        body=response_body,
    )
    assert len(replay_store.values) == 1
    assert next(iter(replay_store.values.values())) > before_admission
    client.close()
    middleware.close()


@pytest.mark.asyncio
async def test_asgi_sends_each_complete_normalized_block_and_drops_tail() -> None:
    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/event-stream; charset=utf-8"),
                    (b"content-length", b"999"),
                    (b"content-encoding", b"identity"),
                    (b"connection", b"x-hop"),
                    (b"x-hop", b"private"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b": keepalive\r", "more_body": True})
        await send({"type": "http.response.body", "body": b"\n\r", "more_body": True})
        await send({"type": "http.response.body", "body": b"\nid: 7\n\n: control\n\npartial", "more_body": False})

    keys = generate_key_pair()
    client = Client(keys.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, keys.private_key, KEY_ID, _resolve_psk, _ReplayStore().admit)
    protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/items"))
    try:
        messages = await _invoke(middleware, protected.envelope)
        assert messages[0]["status"] == 200
        assert (b"content-length", b"999") not in messages[0]["headers"]
        assert (b"cache-control", b"no-store, no-transform") in messages[0]["headers"]
        reader = protected.into_opener()
        checked: list[CheckedRecord] = []
        for message in messages[1:]:
            part = cast(bytes, message["body"])
            offset = 0
            while offset < len(part):
                used, record = reader.feed(part[offset:])
                offset += used
                if record is not None:
                    checked.append(record)
        assert [record.kind for record in checked] == ["start", "data", "data", "data", "end"]
        assert checked[0].headers == (Header("content-type", "text/event-stream; charset=utf-8"),)
        assert [record.block for record in checked[1:4]] == [b": keepalive\n\n", b"id: 7\n\n", b": control\n\n"]
        assert reader.finish_eof() is None
    finally:
        protected.close()
        client.close()
        middleware.close()


@pytest.mark.asyncio
async def test_asgi_middleware_forwards_disconnect_after_the_logical_body() -> None:
    async def app(_scope: Scope, receive: Receive, send: Send) -> None:
        assert (await receive())["body"] == b"logical body"
        assert (await receive())["type"] == "http.disconnect"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, _resolve_psk, _ReplayStore().admit)
    protected = client.protect(
        Request(method=Method.POST, authority="api.example.test", path="/items", body=b"logical body")
    )

    messages = await _invoke(middleware, protected.envelope, disconnect_after_body=True)
    assert messages[0]["status"] == 500
    client.close()
    middleware.close()


@pytest.mark.parametrize("mode", ["finite", "sse"])
@pytest.mark.asyncio
async def test_asgi_receive_finishes_after_final_response_body(mode: Literal["finite", "sse"]) -> None:
    resumed = asyncio.Event()

    async def app(_scope: Scope, receive: Receive, send: Send) -> None:
        assert (await receive())["type"] == "http.request"
        headers = [(b"content-type", b"text/event-stream")] if mode == "sse" else []
        body = b"data: done\n\n" if mode == "sse" else b"done"
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})
        assert (await receive())["type"] == "http.disconnect"
        resumed.set()

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, _resolve_psk, _ReplayStore().admit)
    try:
        protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/items"))
        messages = await asyncio.wait_for(_invoke(middleware, protected.envelope), 2)
        assert resumed.is_set()
        assert messages[0]["status"] == 200
        assert messages[-1]["more_body"] is False
    finally:
        client.close()
        middleware.close()


@pytest.mark.asyncio
async def test_asgi_middleware_discards_head_body_and_preserves_length_metadata() -> None:
    async def app(scope: Scope, _receive: Receive, send: Send) -> None:
        assert scope["method"] == "HEAD"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", b"7")],
            }
        )
        await send({"type": "http.response.body", "body": b"ignored", "more_body": False})

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, _resolve_psk, _ReplayStore().admit)
    protected = client.protect(Request(method=Method.HEAD, authority="api.example.test", path="/items"))

    messages = await _invoke(middleware, protected.envelope)
    response = protected.open_response(cast(bytes, messages[1]["body"]))
    assert response == Response(status=200, headers=(Header("content-length", "7"),), body=b"")
    client.close()
    middleware.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [204, 205, 304])
async def test_asgi_middleware_rejects_nonempty_bodyless_statuses(status: int) -> None:
    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b"invalid", "more_body": False})

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, _resolve_psk, _ReplayStore().admit)
    protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/items"))

    messages = await _invoke(middleware, protected.envelope)
    assert messages[0]["status"] == 500
    assert messages[1]["body"] == b"application returned a forbidden response body"
    protected.close()
    client.close()
    middleware.close()


@pytest.mark.asyncio
async def test_asgi_middleware_accepts_none_raw_path() -> None:
    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, _resolve_psk, _ReplayStore().admit)
    protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/items"))

    messages = await _invoke(middleware, protected.envelope, raw_path=None)
    assert messages[0]["status"] == 200
    assert protected.open_response(cast(bytes, messages[1]["body"])) == Response(status=200)
    client.close()
    middleware.close()


@pytest.mark.asyncio
async def test_asgi_middleware_advertises_post_on_method_rejection() -> None:
    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        raise AssertionError("application must not be called")

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, _resolve_psk, _ReplayStore().admit)
    protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/items"))

    messages = await _invoke(middleware, protected.envelope, outer_method="GET")
    assert messages[0]["status"] == 405
    assert (b"allow", b"POST") in messages[0]["headers"]
    protected.close()
    client.close()
    middleware.close()


@pytest.mark.asyncio
async def test_asgi_middleware_fails_closed_when_replay_store_errors() -> None:
    application_called = False

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        nonlocal application_called
        application_called = True

    async def unavailable_store(_replay_id: bytes, _deadline: int, _scope: Scope) -> bool:
        raise RuntimeError("store unavailable")

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, _resolve_psk, unavailable_store)
    protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/items"))

    messages = await _invoke(middleware, protected.envelope)
    assert messages[0]["status"] == 503
    assert messages[1]["body"] == b"replay admission unavailable"
    assert not application_called
    protected.close()
    client.close()
    middleware.close()


@pytest.mark.asyncio
async def test_asgi_middleware_rejects_a_duplicate_without_dispatch() -> None:
    application_calls = 0

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        nonlocal application_calls
        application_calls += 1
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, _resolve_psk, _ReplayStore().admit)
    protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/items"))

    first = await _invoke(middleware, protected.envelope)
    duplicate = await _invoke(middleware, protected.envelope)
    assert first[0]["status"] == 200
    assert duplicate[0]["status"] == 409
    assert duplicate[1]["body"] == b"protected request replay rejected"
    assert application_calls == 1
    protected.close()
    client.close()
    middleware.close()


@pytest.mark.asyncio
async def test_asgi_middleware_bounds_unannounced_outer_request_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        pytest.fail("oversized envelope must not reach the application")

    key_pair = generate_key_pair()
    middleware = HPKEMiddleware(
        app,
        key_pair.private_key,
        KEY_ID,
        _resolve_psk,
        _ReplayStore().admit,
        limits=Limits(max_body_len=1),
    )
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    protected = client.protect(
        Request(method=Method.POST, authority="api.example.test", path="/items", body=b"x" * 100_000)
    )

    def fail_preparse(_server: Server, _envelope: bytes) -> NoReturn:
        pytest.fail("oversized outer body must be rejected before native parsing")

    monkeypatch.setattr(Server, "preparse", fail_preparse)

    messages = await _invoke(
        middleware,
        protected.envelope,
        include_content_length=False,
        incomplete_body=True,
    )
    assert messages[0]["status"] == 400
    assert messages[1]["body"] == b"invalid protected request"
    protected.close()
    client.close()
    middleware.close()


@pytest.mark.asyncio
async def test_asgi_middleware_hides_unknown_psk_ids() -> None:
    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        pytest.fail("application must not receive an unresolved request")

    def missing_psk(_psk_id: bytes, _scope: Scope) -> bytes:
        raise KeyError("private credential lookup detail")

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, missing_psk, _ReplayStore().admit)
    protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/items"))

    messages = await _invoke(middleware, protected.envelope)
    assert messages[0]["status"] == 400
    assert messages[1]["body"] == b"invalid protected request"
    protected.close()
    client.close()
    middleware.close()


@pytest.mark.asyncio
async def test_asgi_middleware_maps_psk_resolver_outages_to_503() -> None:
    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        pytest.fail("application must not receive a request without credentials")

    async def unavailable_psk(_psk_id: bytes, _scope: Scope) -> bytes:
        raise RuntimeError("private resolver outage detail")

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, unavailable_psk, _ReplayStore().admit)
    protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/items"))

    messages = await _invoke(middleware, protected.envelope)
    assert messages[0]["status"] == 503
    assert messages[1]["body"] == b"credential resolution unavailable"
    protected.close()
    client.close()
    middleware.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "expected_status", "expected_body"),
    [
        ("invalid_request_time", 409, b"protected request time rejected"),
        ("clock_unavailable", 503, b"trusted clock unavailable"),
        ("authentication_failed", 400, b"invalid protected request"),
    ],
)
async def test_asgi_middleware_maps_authentication_failures(
    monkeypatch: pytest.MonkeyPatch, code: str, expected_status: int, expected_body: bytes
) -> None:
    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        pytest.fail("application must not receive an unauthenticated request")

    def fail_authenticate(_stage: PreparsedRequest, _psk: bytes) -> NoReturn:
        raise ProtocolError(code, "injected authentication failure")

    monkeypatch.setattr(PreparsedRequest, "authenticate", fail_authenticate)
    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, _resolve_psk, _ReplayStore().admit)
    protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/items"))

    messages = await _invoke(middleware, protected.envelope)
    assert messages[0]["status"] == expected_status
    assert messages[1]["body"] == expected_body
    protected.close()
    client.close()
    middleware.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_body"),
    [
        ("duplicate_host", 421, b"authenticated target does not match this endpoint"),
        ("encoded_response", 500, b"application returned unsupported content coding"),
        ("invalid_header", 500, b"application returned an invalid ASGI response"),
        ("invalid_sequence", 500, b"application returned an invalid ASGI response"),
    ],
)
async def test_asgi_middleware_turns_boundary_failures_into_controlled_outer_errors(
    failure: str, expected_status: int, expected_body: bytes
) -> None:
    application_called = False

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        nonlocal application_called
        application_called = True
        if failure == "encoded_response":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-encoding", b"gzip")],
                }
            )
            await send({"type": "http.response.body", "body": b"compressed", "more_body": False})
            return
        if failure == "invalid_header":
            await send({"type": "http.response.start", "status": 200, "headers": [(b"x-invalid", b"\xff")]})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})
            return
        if failure == "invalid_sequence":
            await send({"type": "http.response.body", "body": b"bad", "more_body": False})
            return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, _resolve_psk, _ReplayStore().admit)
    protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/items"))
    extra_headers = [(b"host", b"other.example.test")] if failure == "duplicate_host" else []
    messages = await _invoke(middleware, protected.envelope, extra_headers=extra_headers)
    assert messages[0]["status"] == expected_status
    assert messages[1]["body"] == expected_body
    assert messages[1]["more_body"] is False
    assert application_called is (failure != "duplicate_host")
    protected.close()
    client.close()
    middleware.close()


@pytest.mark.asyncio
async def test_asgi_middleware_rejects_nonidentity_authenticated_request_content() -> None:
    application_called = False

    async def app(_scope: Scope, _receive: Receive, _send: Send) -> None:
        nonlocal application_called
        application_called = True

    key_pair = generate_key_pair()
    client = Client(key_pair.public_key, KEY_ID, PSK, PSK_ID)
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, _resolve_psk, _ReplayStore().admit)
    protected = client.protect(
        Request(
            method=Method.POST,
            authority="api.example.test",
            path="/items",
            headers=(Header("content-encoding", "gzip"),),
            body=b"compressed",
        )
    )

    messages = await _invoke(middleware, protected.envelope)
    assert messages[0]["status"] == 400
    assert not application_called
    protected.close()
    client.close()
    middleware.close()


@pytest.mark.asyncio
async def test_asgi_lifespan_closes_native_server() -> None:
    received = iter(
        (
            {"type": "lifespan.startup"},
            {"type": "lifespan.shutdown"},
        )
    )
    sent: list[Message] = []

    async def app(_scope: Scope, receive: Receive, send: Send) -> None:
        assert (await receive())["type"] == "lifespan.startup"
        await send({"type": "lifespan.startup.complete"})
        assert (await receive())["type"] == "lifespan.shutdown"
        await send({"type": "lifespan.shutdown.complete"})

    async def receive() -> Message:
        return next(received)

    async def send(message: Message) -> None:
        sent.append(message)

    key_pair = generate_key_pair()
    middleware = HPKEMiddleware(app, key_pair.private_key, KEY_ID, _resolve_psk, _ReplayStore().admit)
    scope = cast(Scope, {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}})

    await middleware(scope, receive, send)

    assert middleware.closed
    assert [message["type"] for message in sent] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]


async def _invoke(
    middleware: Callable[[Scope, Receive, Send], Awaitable[None]],
    envelope: bytes,
    *,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
    raw_path: bytes | None = b"/items",
    outer_method: str = "POST",
    include_content_length: bool = True,
    incomplete_body: bool = False,
    disconnect_after_body: bool = False,
) -> list[Message]:
    delivered = False
    disconnect = asyncio.Event()
    messages: list[Message] = []
    headers = [
        (b"host", b"api.example.test"),
        (b"content-type", REQUEST_MEDIA_TYPE.encode()),
    ]
    if include_content_length:
        headers.append((b"content-length", str(len(envelope)).encode()))
    headers.extend(extra_headers or [])
    scope = cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": outer_method,
            "scheme": "https",
            "path": "/items",
            "raw_path": raw_path,
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "server": ("api.example.test", 443),
            "client": ("127.0.0.1", 50000),
        },
    )

    async def receive() -> Message:
        nonlocal delivered
        if delivered:
            if incomplete_body:
                pytest.fail("oversized body must be rejected before requesting another chunk")
            if not disconnect_after_body:
                await disconnect.wait()
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": envelope, "more_body": incomplete_body}

    async def send(message: Message) -> None:
        messages.append(message)

    await middleware(scope, receive, send)
    return messages

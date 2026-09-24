"""aiohttp adapter coverage through a real ephemeral TLS transport."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import socket
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import aiohttp
import pytest
import trustme
from aiohttp import web
from starlette.datastructures import UploadFile
from starlette.requests import Request as StarletteRequest
from starlette.types import Receive, Scope, Send
from yarl import URL

from hpke_http import Header, Limits, ProtocolError, Response, Server, StateError, TransportError, generate_key_pair
from hpke_http.middleware import Discover, PinnedKey
from hpke_http.middleware.aiohttp import HPKEClientSession, HPKEResponse
from hpke_http.middleware.fastapi import HPKEMiddleware
from hpke_http.transport import REQUEST_MEDIA_TYPE, RESPONSE_MEDIA_TYPE
from tests.stream_request import open_stream_request
from tests.test_live_sse import live_host

KEY_ID = b"primary-2026-09"
PSK = b"a 32-byte minimum test credential!"
PSK_ID = b"tenant-42"
KEY_RECORD_FIXTURE = Path(__file__).resolve().parents[2] / "rust/hpke-http/tests/vectors/key-discovery-v1.json"

_Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


@asynccontextmanager
async def _https_endpoint(handler: _Handler) -> AsyncIterator[tuple[URL, aiohttp.TCPConnector]]:
    """Serve one aiohttp handler over a trusted ephemeral TLS connection."""
    authority = trustme.CA()
    certificate = authority.issue_cert("127.0.0.1", "localhost")
    server_tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    certificate.configure_cert(server_tls)  # pyright: ignore[reportUnknownMemberType]
    client_tls = ssl.create_default_context()
    authority.configure_trust(client_tls)  # pyright: ignore[reportUnknownMemberType]

    application = web.Application()
    application.router.add_post("/protected", handler)
    application.router.add_get("/protected", handler)
    runner = web.AppRunner(application)
    await runner.setup()
    connector: aiohttp.TCPConnector | None = None
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0, ssl_context=server_tls)
        await site.start()
        address = cast(tuple[str, int], runner.addresses[0])
        endpoint = URL.build(scheme="https", host="127.0.0.1", port=address[1], path="/protected")
        connector = aiohttp.TCPConnector(ssl=client_tls)
        yield endpoint, connector
    finally:
        if connector is not None and not connector.closed:
            await connector.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_aiohttp_adapter_protects_buffered_json_exchange() -> None:
    key_pair = generate_key_pair()
    server = Server(key_pair.private_key, KEY_ID)
    request_name = "Ada" * 1024
    request_body = b'{"name":"' + request_name.encode() + b'"}'
    response_value = "yes" * 1024
    response_body = b'{"ok":"' + response_value.encode() + b'"}'

    async def transport(request: web.Request) -> web.Response:
        assert request.method == "POST"
        assert request.headers["accept"] == RESPONSE_MEDIA_TYPE
        assert request.headers["accept-encoding"] == "identity"
        assert request.headers["cache-control"] == "no-store"
        assert request.headers["content-type"] == REQUEST_MEDIA_TYPE
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        outer_envelope = await request.read()
        opened = open_stream_request(server, outer_envelope, PSK)
        assert opened.request.path == "/items"
        assert opened.request.body == request_body
        assert Header("content-type", "application/json") in opened.request.headers
        envelope = opened.protect_response(
            Response(
                status=200,
                headers=(
                    Header("content-type", "application/json; charset=utf-8"),
                    Header("content-length", str(len(response_body))),
                    Header("set-cookie", "first=1; Path=/"),
                    Header("set-cookie", "second=2; Path=/"),
                ),
                body=response_body,
            )
        )
        return web.Response(body=envelope, headers={"content-type": RESPONSE_MEDIA_TYPE})

    try:
        async with _https_endpoint(transport) as (endpoint, connector):
            session = HPKEClientSession(
                str(endpoint),
                PinnedKey(key_pair.public_key, KEY_ID),
                PSK,
                PSK_ID,
                target_origin="https://api.example.test",
                connector=connector,
            )
            async with session:
                async with session.post("https://api.example.test/items", json={"name": request_name}) as response:
                    assert response.status == 200
                    assert await response.json() == {"ok": response_value}
                    assert response.headers["content-length"] == str(len(response_body))
                    assert response.headers.getall("set-cookie") == ["first=1; Path=/", "second=2; Path=/"]
            assert session.closed
            with pytest.raises(StateError):
                await session.get("https://api.example.test/after-close")
    finally:
        server.close()


@pytest.mark.asyncio
async def test_aiohttp_formdata_stream_upload_over_tls(tmp_path: Path) -> None:
    keys = generate_key_pair()
    seen: list[tuple[str, str, bytes]] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        assert scope["path"] == "/upload"
        request = StarletteRequest(scope, receive)
        async with request.form(max_files=1, max_fields=1) as form:
            upload = form["upload"]
            assert isinstance(upload, UploadFile)
            assert upload.filename is not None
            seen.append((cast(str, form["note"]), upload.filename, await upload.read()))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        lambda _id, _scope: PSK,
        lambda _id, _deadline, _scope: True,
        transport_path="/protected",
        expected_authority="api.example.test",
    )
    async with live_host(middleware, tmp_path) as (endpoint, tls):
        connector = aiohttp.TCPConnector(ssl=tls)
        async with HPKEClientSession(
            endpoint,
            PinnedKey(keys.public_key, KEY_ID),
            PSK,
            PSK_ID,
            target_origin="https://api.example.test",
            connector=connector,
        ) as session:
            form = aiohttp.FormData()
            form.add_field("note", "one")
            form.add_field("upload", io.BytesIO(b"file-content"), filename="one.bin")
            async with session.post("https://api.example.test/upload", data=form) as response:
                assert await response.read() == b"ok"
    assert seen == [("one", "one.bin", b"file-content")]


@pytest.mark.asyncio
async def test_aiohttp_file_form_over_8_mib_over_tls(tmp_path: Path) -> None:
    keys = generate_key_pair()
    source_path = tmp_path / "form-upload.bin"
    expected_hash = hashlib.sha256()
    with source_path.open("wb") as output:
        for _ in range(9):
            chunk = os.urandom(1024 * 1024)
            expected_hash.update(chunk)
            output.write(chunk)
    seen: list[tuple[str, int, bytes]] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        assert scope["path"] == "/upload"
        request = StarletteRequest(scope, receive)
        async with request.form(max_files=1, max_fields=0) as form:
            upload = form["file"]
            assert isinstance(upload, UploadFile)
            assert upload.filename is not None
            digest = hashlib.sha256()
            size = 0
            while chunk := await upload.read(64 * 1024):
                size += len(chunk)
                digest.update(chunk)
            seen.append((upload.filename, size, digest.digest()))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        lambda _id, _scope: PSK,
        lambda _id, _deadline, _scope: True,
        transport_path="/protected",
        expected_authority="api.example.test",
    )
    async with live_host(middleware, tmp_path) as (endpoint, tls):
        connector = aiohttp.TCPConnector(ssl=tls)
        async with HPKEClientSession(
            endpoint,
            PinnedKey(keys.public_key, KEY_ID),
            PSK,
            PSK_ID,
            target_origin="https://api.example.test",
            connector=connector,
        ) as session:
            form = aiohttp.FormData()
            with source_path.open("rb") as source:
                form.add_field("file", source, filename="form-upload.bin")
                async with session.post("https://api.example.test/upload", data=form) as response:
                    assert await response.read() == b"ok"
    assert seen == [("form-upload.bin", 9 * 1024 * 1024, expected_hash.digest())]


@pytest.mark.asyncio
async def test_aiohttp_pinned_form_and_async_data_use_one_post() -> None:
    keys = generate_key_pair()
    server = Server(keys.private_key, KEY_ID)
    bodies: list[bytes] = []
    methods: list[str] = []

    async def source() -> AsyncIterator[bytes]:
        yield b"raw-"
        yield b"body"

    async def transport(request: web.Request) -> web.Response:
        methods.append(request.method)
        assert request.method == "POST"
        assert request.headers["content-type"] == REQUEST_MEDIA_TYPE
        opened = open_stream_request(server, await request.read(), PSK)
        bodies.append(opened.request.body)
        return web.Response(
            body=opened.protect_response(Response(200, (), b"ok")),
            headers={"content-type": RESPONSE_MEDIA_TYPE},
        )

    try:
        async with _https_endpoint(transport) as (endpoint, connector):
            async with HPKEClientSession(
                str(endpoint),
                PinnedKey(keys.public_key, KEY_ID),
                PSK,
                PSK_ID,
                target_origin="https://api.example.test",
                connector=connector,
            ) as session:
                form = aiohttp.FormData()
                form.add_field("upload", io.BytesIO(b"file-content"), filename="one.bin")
                async with session.post("https://api.example.test/upload", data=form) as response:
                    assert await response.read() == b"ok"
                async with session.post("https://api.example.test/upload", data=source()) as response:
                    assert await response.read() == b"ok"
        assert methods == ["POST", "POST"]
        assert b'filename="one.bin"' in bodies[0]
        assert b"file-content" in bodies[0]
        assert bodies[1] == b"raw-body"
    finally:
        server.close()


@pytest.mark.asyncio
async def test_buffered_aiohttp_response_accessors() -> None:
    response = HPKEResponse(
        status=404,
        headers=(("content-type", "application/json; charset=utf-8"),),
        body=b'{"detail":"missing"}',
        url=URL("https://api.example.test/missing"),
        method="GET",
    )
    assert not response.ok
    assert response.closed
    assert response.content_type == "application/json"
    assert response.reason == "Not Found"
    assert await response.read() == b'{"detail":"missing"}'
    assert await response.text() == '{"detail":"missing"}'
    assert await response.json() == {"detail": "missing"}
    with pytest.raises(aiohttp.ClientResponseError, match="Not Found"):
        response.raise_for_status()
    async with response as entered:
        assert entered is response
    response.release()
    response.close()


@pytest.mark.asyncio
async def test_aiohttp_adapter_rejects_nonidentity_authenticated_content() -> None:
    key_pair = generate_key_pair()
    server = Server(key_pair.private_key, KEY_ID)

    async def transport(request: web.Request) -> web.Response:
        opened = open_stream_request(server, await request.read(), PSK)
        envelope = opened.protect_response(
            Response(
                status=200,
                headers=(Header("content-encoding", "gzip"),),
                body=b"compressed",
            )
        )
        return web.Response(body=envelope, headers={"content-type": RESPONSE_MEDIA_TYPE})

    try:
        async with _https_endpoint(transport) as (endpoint, connector):
            async with HPKEClientSession(
                str(endpoint),
                PinnedKey(key_pair.public_key, KEY_ID),
                PSK,
                PSK_ID,
                target_origin="https://api.example.test",
                connector=connector,
            ) as session:
                with pytest.raises(TransportError) as captured:
                    await session.get("https://api.example.test/items")
        assert captured.value.code == "inner_content_encoding"
    finally:
        server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "headers", "expected_code"),
    [
        (503, {"content-type": RESPONSE_MEDIA_TYPE}, "outer_status"),
        (200, {"content-type": "application/octet-stream"}, "outer_content_type"),
        (
            200,
            {"content-type": RESPONSE_MEDIA_TYPE, "content-encoding": "gzip"},
            "outer_content_encoding",
        ),
    ],
)
async def test_aiohttp_adapter_rejects_invalid_outer_responses(
    status: int,
    headers: dict[str, str],
    expected_code: str,
) -> None:
    async def transport(_request: web.Request) -> web.Response:
        return web.Response(status=status, headers=headers, body=b"not an envelope")

    key_pair = generate_key_pair()
    async with _https_endpoint(transport) as (endpoint, connector):
        async with HPKEClientSession(
            str(endpoint),
            PinnedKey(key_pair.public_key, KEY_ID),
            PSK,
            PSK_ID,
            target_origin="https://api.example.test",
            connector=connector,
        ) as session:
            with pytest.raises(TransportError) as captured:
                await session.get("https://api.example.test/items")
    assert captured.value.code == expected_code


@pytest.mark.asyncio
async def test_aiohttp_adapter_bounds_the_actual_outer_response_body() -> None:
    key_pair = generate_key_pair()
    server = Server(key_pair.private_key, KEY_ID)

    async def transport(request: web.Request) -> web.StreamResponse:
        opened = open_stream_request(server, await request.read(), PSK)
        envelope = opened.protect_response(Response(status=200, body=b"ab"))
        response = web.StreamResponse(headers={"content-type": RESPONSE_MEDIA_TYPE})
        response.enable_chunked_encoding()
        await response.prepare(request)
        assert "content-length" not in response.headers
        await response.write(envelope)
        await response.write_eof()
        return response

    async with _https_endpoint(transport) as (endpoint, connector):
        async with HPKEClientSession(
            str(endpoint),
            PinnedKey(key_pair.public_key, KEY_ID),
            PSK,
            PSK_ID,
            limits=Limits(max_body_len=1),
            target_origin="https://api.example.test",
            connector=connector,
        ) as session:
            with pytest.raises(ProtocolError) as captured:
                await session.get("https://api.example.test/items")
    assert captured.value.code == "limit_exceeded"
    server.close()


@pytest.mark.asyncio
async def test_aiohttp_adapter_maps_connection_failures() -> None:
    key_pair = generate_key_pair()
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        port = cast(tuple[str, int], unavailable.getsockname())[1]
        async with HPKEClientSession(
            f"https://127.0.0.1:{port}/protected",
            PinnedKey(key_pair.public_key, KEY_ID),
            PSK,
            PSK_ID,
            target_origin="https://api.example.test",
            connector=aiohttp.TCPConnector(ssl=False),
        ) as session:
            with pytest.raises(TransportError) as captured:
                await session.get("https://api.example.test/items", timeout=aiohttp.ClientTimeout(total=1))
    assert captured.value.code == "network_error"


@pytest.mark.asyncio
async def test_aiohttp_json_accepts_structured_suffix_but_not_substring_match() -> None:
    problem = HPKEResponse(
        status=400,
        headers=(("content-type", "application/problem+json"),),
        body=b'{"detail":"bad"}',
        url=URL("https://api.example.test/problem"),
        method="GET",
    )
    assert await problem.json() == {"detail": "bad"}

    not_json = HPKEResponse(
        status=200,
        headers=(("content-type", "application/json-seq"),),
        body=b"{}",
        url=URL("https://api.example.test/not-json"),
        method="GET",
    )
    with pytest.raises(aiohttp.ContentTypeError, match="unexpected content type"):
        await not_json.json()


@pytest.mark.parametrize("trust_env", [True, 1, "yes"])
def test_aiohttp_adapter_rejects_environment_credentials(trust_env: object) -> None:
    key_pair = generate_key_pair()
    with pytest.raises(ValueError, match="trust_env"):
        HPKEClientSession(
            "https://api.example.test/protected",
            PinnedKey(key_pair.public_key, KEY_ID),
            PSK,
            PSK_ID,
            trust_env=trust_env,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["status", "type", "coding", "size", "shape", "point"])
async def test_aiohttp_bad_key_get_never_sends_post(fault: str) -> None:
    valid_record = bytes.fromhex(json.loads(KEY_RECORD_FIXTURE.read_text())["record"])
    methods: list[str] = []

    async def transport(request: web.Request) -> web.Response:
        methods.append(request.method)
        assert request.method == "GET"
        record = valid_record
        status = 503 if fault == "status" else 200
        headers = {"content-type": "text/plain" if fault == "type" else "application/octet-stream"}
        if fault == "coding":
            headers["content-encoding"] = "gzip"
        if fault == "size":
            record = b"x" * 294
        elif fault == "shape":
            record += b"x"
        elif fault == "point":
            record = valid_record[:-32] + bytes(32)
        return web.Response(status=status, headers=headers, body=record)

    async with _https_endpoint(transport) as (endpoint, connector):
        async with HPKEClientSession(str(endpoint), Discover(), PSK, PSK_ID, connector=connector) as client:
            with pytest.raises(TransportError) as captured:
                await client.get(str(endpoint.with_path("/items")))
    assert methods == ["GET"]
    assert captured.value.code == ("discovery_status" if fault == "status" else "discovery_response")
    assert captured.value.status_code == (503 if fault == "status" else None)


@pytest.mark.asyncio
async def test_aiohttp_rejects_local_headers_before_key_get() -> None:
    methods: list[str] = []

    async def transport(request: web.Request) -> web.Response:
        methods.append(request.method)
        return web.Response(status=503)

    async with _https_endpoint(transport) as (endpoint, connector):
        async with HPKEClientSession(str(endpoint), Discover(), PSK, PSK_ID, connector=connector) as client:
            with pytest.raises(TransportError) as captured:
                await client.get(str(endpoint.with_path("/items")), headers={"content-encoding": "gzip"})
    assert captured.value.code == "inner_content_encoding"
    assert methods == []


@pytest.mark.asyncio
async def test_aiohttp_wrong_origin_fails_before_body_read_or_get() -> None:
    touched = False

    async def body() -> AsyncIterator[bytes]:
        nonlocal touched
        touched = True
        yield b"x"

    async with HPKEClientSession(
        "https://api.example.test/protected",
        Discover(),
        PSK,
        PSK_ID,
        connector=aiohttp.TCPConnector(ssl=False),
    ) as client:
        with pytest.raises(TransportError) as captured:
            await client.post("https://other.example.test/items", data=body())
    assert captured.value.code == "invalid_target"
    assert not touched


@pytest.mark.asyncio
async def test_aiohttp_cancelled_key_get_sends_no_post() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    methods: list[str] = []

    async def transport(request: web.Request) -> web.Response:
        methods.append(request.method)
        entered.set()
        await release.wait()
        return web.Response(status=503)

    async with _https_endpoint(transport) as (endpoint, connector):
        async with HPKEClientSession(str(endpoint), Discover(), PSK, PSK_ID, connector=connector) as client:
            task = asyncio.ensure_future(client.get(str(endpoint.with_path("/items"))))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 5)
            finally:
                release.set()
    assert methods == ["GET"]

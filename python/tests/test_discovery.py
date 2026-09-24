"""Fixed one-key discovery checks at the HTTP boundary."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Message, Receive, Scope, Send

from hpke_http import Response, Server, TransportError, generate_key_pair
from hpke_http.middleware import Discover
from hpke_http.middleware._discovery import encode_key_record, https_origin, parse_key_record, validate_endpoint
from hpke_http.middleware.fastapi import HPKEMiddleware
from hpke_http.middleware.httpx import HPKEAsyncClient
from hpke_http.transport import RESPONSE_MEDIA_TYPE

KEY_ID = b"primary-2026-09"
PSK = b"a 32-byte minimum test credential!"
PSK_ID = b"tenant-42"
ENDPOINT = "https://api.example.test/protected"
FIXTURE = Path(__file__).resolve().parents[2] / "rust/hpke-http/tests/vectors/key-discovery-v1.json"


def test_shared_key_record_fixture_and_exact_parser() -> None:
    source = json.loads(FIXTURE.read_text())
    key_id = bytes.fromhex(source["key_id"])
    public_key = bytes.fromhex(source["public_key"])
    record = bytes.fromhex(source["record"])
    assert source["schema"] == "hpke-http-key-discovery/1"
    assert encode_key_record(key_id, public_key) == record
    assert parse_key_record(record) == (key_id, public_key)
    for edge_id, edge_record, size in (
        (b"k", b"HHKD\x01\x01k" + public_key, 39),
        (b"k" * 255, b"HHKD\x01\xff" + b"k" * 255 + public_key, 293),
    ):
        assert len(edge_record) == size
        assert encode_key_record(edge_id, public_key) == edge_record
        assert parse_key_record(edge_record) == (edge_id, public_key)
    for invalid in (
        b"",
        record[:-1],
        record + b"x",
        b"BAD!" + record[4:],
        record[:4] + b"\x02" + record[5:],
        record[:5] + b"\x00" + record[6:],
    ):
        with pytest.raises(TransportError) as captured:
            parse_key_record(invalid)
        assert captured.value.code == "discovery_response"
        assert captured.value.status_code is None


def test_https_origin_normalizes_host_forms_and_rejects_other_sources() -> None:
    assert https_origin("https://API.example.test:443/") == https_origin("https://api.example.test")
    assert https_origin("https://bücher.example/") == https_origin("https://xn--bcher-kva.example")
    assert https_origin("https://faß.example/") == https_origin("https://xn--fa-hia.example/")
    assert https_origin("https://faß.example/") != https_origin("https://fass.example/")
    assert https_origin("https://[2001:0db8::1]:443/") == https_origin("https://[2001:db8::1]/")
    assert https_origin("https://api.example.test:8443") != https_origin("https://api.example.test")
    for invalid in (
        "http://api.example.test/protected",
        "https://user@api.example.test/protected",
        "https://api.example.test/protected?x=1",
        "https://api.example.test/protected#fragment",
        "https://api.example.test:0/protected",
    ):
        with pytest.raises(TransportError) as captured:
            validate_endpoint(invalid)
        assert captured.value.code == "invalid_target"


@pytest.mark.asyncio
async def test_httpx_discovers_one_key_for_each_call_then_posts_to_same_url() -> None:
    keys = generate_key_pair()
    server = Server(keys.private_key, KEY_ID)
    methods: list[str] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        assert str(request.url) == ENDPOINT
        if request.method == "GET":
            assert request.headers["accept"] == "application/octet-stream"
            assert request.headers["accept-encoding"] == "identity"
            assert "authorization" not in request.headers
            assert "cookie" not in request.headers
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream", "cache-control": "no-store"},
                content=b"HHKD\x01\x0fprimary-2026-09" + server.public_key,
            )
        opened = server.preparse(await request.aread()).authenticate(PSK).admit(accepted=True)
        assert opened.request.path == "/items"
        return httpx.Response(
            200,
            headers={"content-type": RESPONSE_MEDIA_TYPE},
            content=opened.protect_response(Response(status=200, body=b"ok")),
        )

    async with HPKEAsyncClient(ENDPOINT, Discover(), PSK, PSK_ID, transport=httpx.MockTransport(transport)) as client:
        assert (await client.get("https://API.example.test:443/items")).content == b"ok"
        assert (await client.get("https://api.example.test/items")).content == b"ok"
    assert methods == ["GET", "POST", "GET", "POST"]
    server.close()


@pytest.mark.asyncio
async def test_httpx_unicode_origin_matches_the_wire_host() -> None:
    calls: list[tuple[str, str]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        return httpx.Response(503)

    async with HPKEAsyncClient(
        "https://faß.example/protected", Discover(), PSK, PSK_ID, transport=httpx.MockTransport(transport)
    ) as client:
        with pytest.raises(TransportError) as wrong:
            await client.get("https://fass.example/items")
        assert wrong.value.code == "invalid_target"
        assert calls == []

        with pytest.raises(TransportError) as right:
            await client.get("https://faß.example/items")
        assert right.value.code == "discovery_status"
        assert calls == [("GET", "https://xn--fa-hia.example/protected")]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["status", "type", "coding", "size", "shape", "point", "network"])
async def test_httpx_bad_key_get_never_sends_post(fault: str) -> None:
    valid_record = bytes.fromhex(json.loads(FIXTURE.read_text())["record"])
    calls: list[str] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        assert request.method == "GET"
        if fault == "network":
            raise httpx.ConnectError("key source unavailable")
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
        if fault == "coding":
            return httpx.Response(status, headers=headers, stream=httpx.ByteStream(record))
        return httpx.Response(status, headers=headers, content=record)

    async with HPKEAsyncClient(ENDPOINT, Discover(), PSK, PSK_ID, transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(TransportError) as captured:
            await client.get("https://api.example.test/items")
    assert calls == ["GET"]
    expected = (
        "discovery_status" if fault == "status" else "discovery_network" if fault == "network" else "discovery_response"
    )
    assert captured.value.code == expected
    assert captured.value.status_code == (503 if fault == "status" else None)


@pytest.mark.asyncio
async def test_httpx_rejects_local_headers_before_key_get() -> None:
    calls: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        return httpx.Response(503)

    async with HPKEAsyncClient(ENDPOINT, Discover(), PSK, PSK_ID, transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(TransportError) as captured:
            await client.get("https://api.example.test/items", headers={"content-encoding": "gzip"})
    assert captured.value.code == "inner_content_encoding"
    assert calls == []


@pytest.mark.asyncio
async def test_wrong_logical_origin_fails_before_body_read_or_key_get() -> None:
    touched = False

    async def body() -> AsyncIterator[bytes]:
        nonlocal touched
        touched = True
        yield b"x"

    calls = 0

    def transport(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with HPKEAsyncClient(ENDPOINT, Discover(), PSK, PSK_ID, transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(TransportError) as captured:
            await client.post("https://other.example.test/items", content=body())
    assert captured.value.code == "invalid_target"
    assert not touched
    assert calls == 0


@pytest.mark.asyncio
async def test_httpx_cancellation_during_key_get_sends_no_post() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        entered.set()
        await release.wait()
        return httpx.Response(500)

    async with HPKEAsyncClient(ENDPOINT, Discover(), PSK, PSK_ID, transport=httpx.MockTransport(transport)) as client:
        task = asyncio.create_task(client.get("https://api.example.test/items"))
        await entered.wait()
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
    assert calls == ["GET"]


@pytest.mark.asyncio
async def test_asgi_key_get_is_public_and_uses_current_server_key() -> None:
    keys = generate_key_pair()
    app_calls = 0
    private_calls = 0

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        nonlocal app_calls
        app_calls += 1
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    def resolver(_id: bytes, _scope: Scope) -> bytes:
        nonlocal private_calls
        private_calls += 1
        return PSK

    def replay(_id: bytes, _deadline: int, _scope: Scope) -> bool:
        nonlocal private_calls
        private_calls += 1
        return True

    middleware = HPKEMiddleware(app, keys.private_key, KEY_ID, resolver, replay, transport_path="/protected")

    async def invoke(method: str, path: str = "/protected", query: bytes = b"") -> list[Message]:
        sent: list[Message] = []

        async def receive() -> Message:
            raise AssertionError("key GET must not read a request body")

        async def send(message: Message) -> None:
            sent.append(message)

        scope = cast(
            Scope,
            {
                "type": "http",
                "method": method,
                "path": path,
                "query_string": query,
                "headers": [(b"host", b"api.example.test")],
            },
        )
        await middleware(scope, receive, send)
        return sent

    record = await invoke("GET")
    assert record[0]["status"] == 200
    assert (b"cache-control", b"no-store") in record[0]["headers"]
    assert cast(bytes, record[1]["body"]) == b"HHKD\x01\x0fprimary-2026-09" + keys.public_key
    assert (await invoke("GET", query=b"x=1"))[0]["status"] == 400
    method = await invoke("PUT")
    assert method[0]["status"] == 405
    assert (b"allow", b"GET, POST") in method[0]["headers"]
    assert (await invoke("GET", path="/other"))[0]["status"] == 204
    middleware.close()
    assert (await invoke("GET"))[0]["status"] == 503
    assert app_calls == 1
    assert private_calls == 0


@pytest.mark.asyncio
async def test_asgi_key_route_works_inside_starlette_mount() -> None:
    keys = generate_key_pair()
    app_calls = 0

    async def app(scope: Scope, _receive: Receive, send: Send) -> None:
        nonlocal app_calls
        app_calls += 1
        assert scope["path"] == "/items"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"mounted", "more_body": False})

    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        lambda _id, _scope: PSK,
        lambda _id, _deadline, _scope: True,
        transport_path="/api/protected",
    )
    host = Starlette(routes=[Mount("/api", app=middleware)])
    try:
        async with HPKEAsyncClient(
            "https://api.example.test/api/protected",
            Discover(),
            PSK,
            PSK_ID,
            transport=httpx.ASGITransport(app=host),
        ) as client:
            assert (await client.get("https://api.example.test/items")).content == b"mounted"
        assert app_calls == 1
    finally:
        middleware.close()

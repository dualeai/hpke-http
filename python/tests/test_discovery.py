"""Fixed one-key discovery checks at the HTTP boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Message, Receive, Scope, Send

from hpke_http import Client, Response, Server, StateError, TransportError, generate_key_pair
from hpke_http.middleware import PinnedKey
from hpke_http.middleware._discovery import (
    encode_key_record,
    https_origin,
    parse_key_record,
    validate_endpoint,
)
from hpke_http.middleware._shared_key import KeyLease
from hpke_http.middleware.fastapi import HPKEMiddleware
from hpke_http.middleware.httpx import DiscoveredEndpoint, HPKEAsyncClient
from hpke_http.transport import RESPONSE_MEDIA_TYPE
from tests.stream_request import open_stream_request

KEY_ID = b"primary-2026-09"
PSK = b"a 32-byte minimum test credential!"
PSK_ID = b"tenant-42"
ENDPOINT = "https://api.example.test/protected"
FIXTURE = Path(__file__).resolve().parents[2] / "rust/hpke-http/tests/vectors/key-discovery-v2.json"


def test_shared_key_record_fixture_and_exact_parser() -> None:
    source = json.loads(FIXTURE.read_text())
    key_id = bytes.fromhex(source["key_id"])
    public_key = bytes.fromhex(source["public_key"])
    record = bytes.fromhex(source["record"])
    use_for_s = source["use_for_s"]
    assert source["schema"] == "hpke-http-key-discovery/2"
    assert encode_key_record(key_id, public_key, use_for_s) == record
    assert parse_key_record(record) == (key_id, public_key, use_for_s)
    long_record = bytes.fromhex(source["long_record"])
    assert source["long_use_for_s"] == 257
    assert long_record == record[:-4] + b"\x00\x00\x01\x01"
    assert encode_key_record(key_id, public_key, 257) == long_record
    assert parse_key_record(long_record) == (key_id, public_key, 257)
    for edge_id, edge_record, size in (
        (b"k", b"HHKD\x02\x01k" + public_key + b"\x00\x00\x00\x3c", 43),
        (b"k" * 255, b"HHKD\x02\xff" + b"k" * 255 + public_key + b"\x00\x00\x00\x3c", 297),
    ):
        assert len(edge_record) == size
        assert encode_key_record(edge_id, public_key, use_for_s) == edge_record
        assert parse_key_record(edge_record) == (edge_id, public_key, use_for_s)
    for invalid in (
        b"",
        record[:-1],
        record + b"x",
        b"BAD!" + record[4:],
        record[:4] + b"\x01" + record[5:],
        record[:5] + b"\x00" + record[6:],
        record[:-4] + bytes(4),
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
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("verify", False),
        ("cert", "unused.pem"),
        ("http2", True),
        ("http1", False),
        ("mounts", {}),
    ],
)
async def test_httpx_shared_source_owns_transport_options(option: str, value: object) -> None:
    async with DiscoveredEndpoint(
        ENDPOINT, transport=httpx.MockTransport(lambda _request: httpx.Response(503))
    ) as source:
        with pytest.raises(ValueError, match="shared source owns outer transport options"):
            HPKEAsyncClient(source, PSK, PSK_ID, **cast(Any, {option: value}))


@pytest.mark.asyncio
async def test_httpx_endpoint_has_one_owner() -> None:
    async with DiscoveredEndpoint(
        ENDPOINT, transport=httpx.MockTransport(lambda _request: httpx.Response(503))
    ) as source:
        with pytest.raises(ValueError, match="shared source owns the endpoint"):
            HPKEAsyncClient(source, PSK, PSK_ID, endpoint=ENDPOINT)
    with pytest.raises(ValueError, match="endpoint is required with PinnedKey"):
        HPKEAsyncClient(PinnedKey(bytes(32), b"key"), PSK, PSK_ID)


@pytest.mark.asyncio
async def test_httpx_shared_source_applies_logical_defaults_and_timeout() -> None:
    keys = generate_key_pair()
    server = Server(keys.private_key, KEY_ID)
    methods: list[str] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=encode_key_record(KEY_ID, keys.public_key, 60),
            )
        assert request.extensions["timeout"] == httpx.Timeout(2).as_dict()
        opened = open_stream_request(server, await request.aread(), PSK)
        assert opened.request.path == "/items?q=1"
        assert any(field.name == "x-logical" and field.value == "yes" for field in opened.request.headers)
        return httpx.Response(
            200,
            headers={"content-type": RESPONSE_MEDIA_TYPE},
            content=opened.protect_response(Response(status=200, body=b"ok")),
        )

    try:
        async with DiscoveredEndpoint(ENDPOINT, transport=httpx.MockTransport(transport)) as source:
            async with HPKEAsyncClient(
                source, PSK, PSK_ID, headers={"x-logical": "yes"}, params={"q": "1"}, timeout=2
            ) as client:
                assert (await client.get("https://api.example.test/items")).content == b"ok"
        assert methods == ["GET", "POST"]
    finally:
        server.close()


@pytest.mark.asyncio
async def test_httpx_shares_one_key_across_clients_then_posts_to_same_url() -> None:
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
                content=encode_key_record(KEY_ID, server.public_key, 60),
            )
        opened = open_stream_request(server, await request.aread(), PSK)
        assert opened.request.path == "/items"
        return httpx.Response(
            200,
            headers={"content-type": RESPONSE_MEDIA_TYPE},
            content=opened.protect_response(Response(status=200, body=b"ok")),
        )

    async with DiscoveredEndpoint(ENDPOINT, transport=httpx.MockTransport(transport)) as source:
        async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
            assert (await client.get("https://API.example.test:443/items")).content == b"ok"
        async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
            assert (await client.get("https://api.example.test/items")).content == b"ok"
    assert methods == ["GET", "POST", "POST"]
    server.close()


@pytest.mark.asyncio
async def test_httpx_concurrent_first_calls_share_one_get_after_a_waiter_stops() -> None:
    keys = generate_key_pair()
    server = Server(keys.private_key, KEY_ID)
    entered = asyncio.Event()
    release = asyncio.Event()
    methods: list[str] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "GET":
            entered.set()
            await release.wait()
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=encode_key_record(KEY_ID, keys.public_key, 60),
            )
        opened = open_stream_request(server, await request.aread(), PSK)
        return httpx.Response(
            200,
            headers={"content-type": RESPONSE_MEDIA_TYPE},
            content=opened.protect_response(Response(status=200, body=b"ok")),
        )

    try:
        async with DiscoveredEndpoint(ENDPOINT, transport=httpx.MockTransport(transport)) as source:
            async with HPKEAsyncClient(source, PSK, PSK_ID) as first:
                async with HPKEAsyncClient(source, PSK, PSK_ID) as second:
                    stopped = asyncio.create_task(first.get("https://api.example.test/items"))
                    active = asyncio.create_task(second.get("https://api.example.test/items"))
                    await entered.wait()
                    stopped.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await stopped
                    release.set()
                    assert (await active).content == b"ok"
                    assert (await first.get("https://api.example.test/items")).content == b"ok"
        assert methods == ["GET", "POST", "POST"]
    finally:
        release.set()
        server.close()


@pytest.mark.asyncio
async def test_httpx_rotation_keeps_old_key_until_lease_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [time.monotonic(), time.time()]
    monkeypatch.setattr(
        "hpke_http.middleware._shared_key.time",
        SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[1]),
    )
    old = generate_key_pair()
    new = generate_key_pair()
    new_id = b"next-2026-09"
    old_server = Server(old.private_key, KEY_ID)
    new_server = Server(new.private_key, new_id, accepted_keys=[(old.private_key, KEY_ID)])
    advertised = old_server
    methods: list[str] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=encode_key_record(new_id if advertised is new_server else KEY_ID, advertised.public_key, 60),
            )
        raw = await request.aread()
        opened = open_stream_request(advertised, raw, PSK)
        return httpx.Response(
            200,
            headers={"content-type": RESPONSE_MEDIA_TYPE},
            content=opened.protect_response(Response(status=200, body=b"ok")),
        )

    try:
        async with DiscoveredEndpoint(ENDPOINT, transport=httpx.MockTransport(transport)) as source:
            async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
                assert (await client.get("https://api.example.test/items")).content == b"ok"
                advertised = new_server
                assert (await client.get("https://api.example.test/items")).content == b"ok"
                now[0] += 61
                now[1] += 61
                assert (await client.get("https://api.example.test/items")).content == b"ok"
        assert methods == ["GET", "POST", "POST", "GET", "POST"]
    finally:
        old_server.close()
        new_server.close()


@pytest.mark.asyncio
async def test_httpx_lost_reply_does_not_repeat_an_admitted_post() -> None:
    keys = generate_key_pair()
    server = Server(keys.private_key, KEY_ID)
    methods: list[str] = []
    lose_reply = True

    async def transport(request: httpx.Request) -> httpx.Response:
        nonlocal lose_reply
        methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=encode_key_record(KEY_ID, keys.public_key, 60),
            )
        opened = open_stream_request(server, await request.aread(), PSK)
        if lose_reply:
            lose_reply = False
            raise httpx.ConnectError("reply lost after admission")
        return httpx.Response(
            200,
            headers={"content-type": RESPONSE_MEDIA_TYPE},
            content=opened.protect_response(Response(status=200, body=b"ok")),
        )

    try:
        async with DiscoveredEndpoint(ENDPOINT, transport=httpx.MockTransport(transport)) as source:
            async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
                with pytest.raises(TransportError) as captured:
                    await client.get("https://api.example.test/items")
                assert captured.value.code == "network_error"
                assert methods == ["GET", "POST"]
                assert (await client.get("https://api.example.test/items")).content == b"ok"
        assert methods == ["GET", "POST", "POST"]
    finally:
        server.close()


@pytest.mark.asyncio
async def test_httpx_refreshes_a_lease_spent_before_post(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [time.monotonic(), time.time()]
    monkeypatch.setattr(
        "hpke_http.middleware._shared_key.time",
        SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[1]),
    )
    keys = generate_key_pair()
    server = Server(keys.private_key, KEY_ID)
    methods: list[str] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=encode_key_record(KEY_ID, keys.public_key, 60),
            )
        opened = open_stream_request(server, await request.aread(), PSK)
        return httpx.Response(
            200,
            headers={"content-type": RESPONSE_MEDIA_TYPE},
            content=opened.protect_response(Response(status=200, body=b"ok")),
        )

    try:
        async with DiscoveredEndpoint(ENDPOINT, transport=httpx.MockTransport(transport)) as source:
            async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
                original = client._discover_client  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                calls = 0

                async def expire_once() -> tuple[Client, KeyLease]:
                    nonlocal calls
                    selected, lease = await original()
                    calls += 1
                    if calls == 1:
                        now[1] += 61
                    return selected, lease

                monkeypatch.setattr(client, "_discover_client", expire_once)
                assert (await client.get("https://api.example.test/items")).content == b"ok"
        assert methods == ["GET", "GET", "POST"]
    finally:
        server.close()


@pytest.mark.asyncio
async def test_httpx_expired_lease_stops_before_post_start(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [time.monotonic(), time.time()]
    monkeypatch.setattr(
        "hpke_http.middleware._shared_key.time",
        SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[1]),
    )
    keys = generate_key_pair()
    methods: list[str] = []
    sent: list[bytes] = []
    body_reads = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal body_reads
        body_reads += 1
        yield b"one-use"

    async def transport(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=encode_key_record(KEY_ID, keys.public_key, 60),
            )
        now[1] += 61
        async for part in cast(httpx.AsyncByteStream, request.stream):
            sent.append(part)  # noqa: PERF401  # Keep bytes sent before a later stream fault.
        raise AssertionError("expired lease sent a complete protected POST")

    class BeforeBodyTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return await transport(request)

    async with DiscoveredEndpoint(ENDPOINT, transport=BeforeBodyTransport()) as source:
        async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
            with pytest.raises(TransportError) as captured:
                await client.post("https://api.example.test/upload", content=body())
    assert captured.value.code == "discovery_expired"
    assert methods == ["GET", "POST"]
    assert sent == []
    assert body_reads == 0


@pytest.mark.asyncio
async def test_httpx_source_close_during_post_never_resends_it() -> None:
    keys = generate_key_pair()
    server = Server(keys.private_key, KEY_ID)
    entered = asyncio.Event()
    release = asyncio.Event()
    methods: list[str] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=encode_key_record(KEY_ID, keys.public_key, 60),
            )
        opened = open_stream_request(server, await request.aread(), PSK)
        entered.set()
        await release.wait()
        return httpx.Response(
            200,
            headers={"content-type": RESPONSE_MEDIA_TYPE},
            content=opened.protect_response(Response(status=200, body=b"ok")),
        )

    source = DiscoveredEndpoint(ENDPOINT, transport=httpx.MockTransport(transport))
    client = HPKEAsyncClient(source, PSK, PSK_ID)
    try:
        pending = asyncio.create_task(client.get("https://api.example.test/items"))
        await entered.wait()
        await source.aclose()
        release.set()
        with suppress(TransportError, httpx.HTTPError):
            assert (await pending).content == b"ok"
        assert methods == ["GET", "POST"]
        with pytest.raises(StateError):
            await client.get("https://api.example.test/items")
    finally:
        release.set()
        await client.aclose()
        await source.aclose()
        server.close()


@pytest.mark.asyncio
async def test_httpx_bridge_and_library_keep_distinct_keys_and_protected_bearer() -> None:
    bridge = generate_key_pair()
    library = generate_key_pair()
    bridge_id = b"bridge-key"
    library_id = b"library-key"
    token = b"a complete API token with more than 32 bytes"
    psk_id = hashlib.sha512(token).digest()
    bridge_server = Server(bridge.private_key, bridge_id)
    library_server = Server(library.private_key, library_id)
    methods: list[tuple[str, str]] = []
    bridge_endpoint = "https://api.example.test/http-bridge/v1/hpke"
    library_endpoint = "https://api.example.test/libraries/v1/hpke"

    async def transport(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        methods.append((path, request.method))
        server, key_id = (bridge_server, bridge_id) if path == "/http-bridge/v1/hpke" else (library_server, library_id)
        if request.method == "GET":
            assert "authorization" not in request.headers
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=encode_key_record(key_id, server.public_key, 60),
            )
        raw = await request.aread()
        start_length = server.stream_start_length(raw)
        assert start_length is not None
        public = server.preparse_stream(raw[:start_length])
        assert public.psk_id == psk_id
        public.close()
        opened = open_stream_request(server, raw, token)
        if path == "/libraries/v1/hpke":
            assert any(
                h.name.lower() == "authorization" and h.value == "Bearer library" for h in opened.request.headers
            )
        return httpx.Response(
            200,
            headers={"content-type": RESPONSE_MEDIA_TYPE},
            content=opened.protect_response(Response(status=200, body=b"ok")),
        )

    try:
        async with DiscoveredEndpoint(bridge_endpoint, transport=httpx.MockTransport(transport)) as bridge_source:
            async with DiscoveredEndpoint(library_endpoint, transport=httpx.MockTransport(transport)) as library_source:
                for source in (bridge_source, library_source):
                    for _ in range(2):
                        async with HPKEAsyncClient(source, token, psk_id) as client:
                            headers = {"authorization": "Bearer library"} if source is library_source else {}
                            assert (
                                await client.get("https://api.example.test/items", headers=headers)
                            ).content == b"ok"
        assert methods == [
            ("/http-bridge/v1/hpke", "GET"),
            ("/http-bridge/v1/hpke", "POST"),
            ("/http-bridge/v1/hpke", "POST"),
            ("/libraries/v1/hpke", "GET"),
            ("/libraries/v1/hpke", "POST"),
            ("/libraries/v1/hpke", "POST"),
        ]
    finally:
        bridge_server.close()
        library_server.close()


@pytest.mark.asyncio
async def test_httpx_unicode_origin_matches_the_wire_host() -> None:
    calls: list[tuple[str, str]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url)))
        return httpx.Response(503)

    async with DiscoveredEndpoint("https://faß.example/protected", transport=httpx.MockTransport(transport)) as source:
        async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
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
            record = b"x" * 298
        elif fault == "shape":
            record += b"x"
        elif fault == "point":
            record = valid_record[:-36] + bytes(32) + valid_record[-4:]
        if fault == "coding":
            return httpx.Response(status, headers=headers, stream=httpx.ByteStream(record))
        return httpx.Response(status, headers=headers, content=record)

    async with DiscoveredEndpoint(ENDPOINT, transport=httpx.MockTransport(transport)) as source:
        async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
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

    async with DiscoveredEndpoint(ENDPOINT, transport=httpx.MockTransport(transport)) as source:
        async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
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

    async with DiscoveredEndpoint(ENDPOINT, transport=httpx.MockTransport(transport)) as source:
        async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
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

    async with DiscoveredEndpoint(ENDPOINT, transport=httpx.MockTransport(transport)) as source:
        async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
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

    middleware = HPKEMiddleware(
        app, keys.private_key, KEY_ID, resolver, replay, key_use_for_s=257, transport_path="/protected"
    )

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
    key_record = cast(bytes, record[1]["body"])
    assert key_record == b"HHKD\x02" + bytes((len(KEY_ID),)) + KEY_ID + keys.public_key + b"\x00\x00\x01\x01"
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
        key_use_for_s=60,
        transport_path="/api/protected",
    )
    host = Starlette(routes=[Mount("/api", app=middleware)])
    try:
        endpoint = "https://api.example.test/api/protected"
        async with DiscoveredEndpoint(endpoint, transport=httpx.ASGITransport(app=host)) as source:
            async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
                assert (await client.get("https://api.example.test/items")).content == b"mounted"
        assert app_calls == 1
    finally:
        middleware.close()

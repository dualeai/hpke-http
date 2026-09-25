"""Real TLS checks that a complete SSE block arrives before app completion."""

from __future__ import annotations

import asyncio
import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import pytest
import trustme
import uvicorn
from starlette.types import Receive, Scope, Send

from hpke_http import TransportError, generate_key_pair
from hpke_http.middleware import PinnedKey
from hpke_http.middleware.aiohttp import DiscoveredEndpoint as AiohttpDiscoveredEndpoint
from hpke_http.middleware.aiohttp import HPKEClientSession
from hpke_http.middleware.fastapi import HPKEMiddleware
from hpke_http.middleware.httpx import DiscoveredEndpoint as HttpxDiscoveredEndpoint
from hpke_http.middleware.httpx import HPKEAsyncClient

KEY_ID = b"live-sse-key"
PSK = b"a 32-byte minimum test credential!"
PSK_ID = b"live-tenant"


@asynccontextmanager
async def live_host(app: HPKEMiddleware, path: Path) -> AsyncIterator[tuple[str, ssl.SSLContext]]:
    ca = trustme.CA()
    cert = ca.issue_cert("127.0.0.1", "localhost")
    cert_path = path / "cert.pem"
    key_path = path / "key.pem"
    cert.cert_chain_pems[0].write_to_path(cert_path)
    cert.private_key_pem.write_to_path(key_path)
    client_tls = ssl.create_default_context()
    ca.configure_trust(client_tls)  # pyright: ignore[reportUnknownMemberType]
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        ssl_certfile=str(cert_path),
        ssl_keyfile=str(key_path),
        lifespan="off",
        log_level="error",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:

        async def wait_until_started() -> None:
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_until_started(), 5)
        listener = server.servers[0].sockets[0]
        port = listener.getsockname()[1]
        yield f"https://127.0.0.1:{port}/protected", client_tls
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        app.close()


@pytest.mark.asyncio
async def test_both_clients_discover_over_https_and_reject_untrusted_tls(tmp_path: Path) -> None:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        assert scope["path"] == "/items"
        assert (await receive())["type"] == "http.request"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"discovered", "more_body": False})

    keys = generate_key_pair()
    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        lambda _id, _scope: PSK,
        lambda _id, _deadline, _scope: True,
        key_use_for_s=60,
        transport_path="/protected",
        expected_authority="api.example.test",
    )
    async with live_host(middleware, tmp_path) as (endpoint, client_tls):
        async with HttpxDiscoveredEndpoint(endpoint, verify=client_tls) as source:
            async with HPKEAsyncClient(source, PSK, PSK_ID, target_origin="https://api.example.test") as client:
                assert (await client.get("https://api.example.test/items")).content == b"discovered"
        connector = aiohttp.TCPConnector(ssl=client_tls)
        async with AiohttpDiscoveredEndpoint(endpoint, connector=connector) as source:
            async with HPKEClientSession(source, PSK, PSK_ID, target_origin="https://api.example.test") as client:
                assert await (await client.get("https://api.example.test/items")).read() == b"discovered"
        async with HttpxDiscoveredEndpoint(endpoint) as source:
            async with HPKEAsyncClient(source, PSK, PSK_ID, target_origin="https://api.example.test") as client:
                with pytest.raises(TransportError) as captured:
                    await client.get("https://api.example.test/items")
                assert captured.value.code == "discovery_network"
        async with AiohttpDiscoveredEndpoint(endpoint) as source:
            async with HPKEClientSession(source, PSK, PSK_ID, target_origin="https://api.example.test") as client:
                with pytest.raises(TransportError) as captured:
                    await client.get("https://api.example.test/items")
                assert captured.value.code == "discovery_network"


@pytest.mark.asyncio
@pytest.mark.parametrize("library", ["httpx", "aiohttp"])
async def test_each_checked_block_arrives_before_app_finishes(library: str, tmp_path: Path) -> None:
    first_sent = asyncio.Event()
    release_second = asyncio.Event()
    release_end = asyncio.Event()
    app_finished = asyncio.Event()

    async def app(_scope: Scope, receive: Receive, send: Send) -> None:
        request = await receive()
        assert request["type"] == "http.request"
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream; charset=utf-8"), (b"content-length", b"99")],
            }
        )
        await send({"type": "http.response.body", "body": b": keepalive\r\n\r\n", "more_body": True})
        first_sent.set()
        await release_second.wait()
        await send({"type": "http.response.body", "body": b"data: two\n\n", "more_body": True})
        await release_end.wait()
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        app_finished.set()

    keys = generate_key_pair()
    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        lambda _id, _scope: PSK,
        lambda _id, _deadline, _scope: True,
        key_use_for_s=60,
        transport_path="/protected",
        expected_authority="api.example.test",
    )
    async with live_host(middleware, tmp_path) as (endpoint, client_tls):
        try:
            if library == "httpx":
                async with HPKEAsyncClient(
                    PinnedKey(keys.public_key, KEY_ID),
                    PSK,
                    PSK_ID,
                    target_origin="https://api.example.test",
                    verify=client_tls,
                    endpoint=endpoint,
                ) as client:
                    async with client.stream("GET", "https://api.example.test/events") as response:
                        assert response.status_code == 200
                        assert response.headers["content-type"].startswith("text/event-stream")
                        blocks = response.iter_sse()
                        assert await asyncio.wait_for(anext(blocks), 5) == b": keepalive\n\n"
                        await asyncio.wait_for(first_sent.wait(), 5)
                        assert not app_finished.is_set()
                        release_second.set()
                        assert await asyncio.wait_for(anext(blocks), 5) == b"data: two\n\n"
                        assert not app_finished.is_set()
                        release_end.set()
                        with pytest.raises(StopAsyncIteration):
                            await asyncio.wait_for(anext(blocks), 5)
            else:
                connector = aiohttp.TCPConnector(ssl=client_tls)
                async with HPKEClientSession(
                    PinnedKey(keys.public_key, KEY_ID),
                    PSK,
                    PSK_ID,
                    target_origin="https://api.example.test",
                    connector=connector,
                    endpoint=endpoint,
                ) as client:
                    async with client.stream("GET", "https://api.example.test/events") as response:
                        assert response.status == 200
                        assert response.headers["content-type"].startswith("text/event-stream")
                        blocks = response.iter_sse()
                        assert await asyncio.wait_for(anext(blocks), 5) == b": keepalive\n\n"
                        await asyncio.wait_for(first_sent.wait(), 5)
                        assert not app_finished.is_set()
                        release_second.set()
                        assert await asyncio.wait_for(anext(blocks), 5) == b"data: two\n\n"
                        assert not app_finished.is_set()
                        release_end.set()
                        with pytest.raises(StopAsyncIteration):
                            await asyncio.wait_for(anext(blocks), 5)
        finally:
            release_second.set()
            release_end.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("library", ["httpx", "aiohttp"])
async def test_live_context_exit_stops_the_server_call(library: str, tmp_path: Path) -> None:
    app_stopped = asyncio.Event()

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
        await send({"type": "http.response.body", "body": b": ready\n\n", "more_body": True})
        try:
            await asyncio.Event().wait()
        finally:
            app_stopped.set()

    keys = generate_key_pair()
    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        lambda _id, _scope: PSK,
        lambda _id, _deadline, _scope: True,
        key_use_for_s=60,
        transport_path="/protected",
        expected_authority="api.example.test",
    )
    async with live_host(middleware, tmp_path) as (endpoint, client_tls):
        if library == "httpx":
            async with HPKEAsyncClient(
                PinnedKey(keys.public_key, KEY_ID),
                PSK,
                PSK_ID,
                target_origin="https://api.example.test",
                verify=client_tls,
                endpoint=endpoint,
            ) as client:
                async with client.stream("GET", "https://api.example.test/events") as response:
                    assert await asyncio.wait_for(anext(response.iter_sse()), 5) == b": ready\n\n"
        else:
            connector = aiohttp.TCPConnector(ssl=client_tls)
            async with HPKEClientSession(
                PinnedKey(keys.public_key, KEY_ID),
                PSK,
                PSK_ID,
                target_origin="https://api.example.test",
                connector=connector,
                endpoint=endpoint,
            ) as client:
                async with client.stream("GET", "https://api.example.test/events") as response:
                    assert await asyncio.wait_for(anext(response.iter_sse()), 5) == b": ready\n\n"
        await asyncio.wait_for(app_stopped.wait(), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("library", ["httpx", "aiohttp"])
async def test_finite_status_is_checked_at_stream_entry(library: str, tmp_path: Path) -> None:
    body_sent = asyncio.Event()

    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 422, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"invalid", "more_body": False})
        body_sent.set()

    keys = generate_key_pair()
    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        lambda _id, _scope: PSK,
        lambda _id, _deadline, _scope: True,
        key_use_for_s=60,
        transport_path="/protected",
        expected_authority="api.example.test",
    )
    async with live_host(middleware, tmp_path) as (endpoint, client_tls):
        if library == "httpx":
            async with HPKEAsyncClient(
                PinnedKey(keys.public_key, KEY_ID),
                PSK,
                PSK_ID,
                target_origin="https://api.example.test",
                verify=client_tls,
                endpoint=endpoint,
            ) as client:
                async with client.stream("GET", "https://api.example.test/invalid") as response:
                    assert response.status_code == 422 and response.mode == "finite"
                    assert await response.read() == b"invalid"
        else:
            connector = aiohttp.TCPConnector(ssl=client_tls)
            async with HPKEClientSession(
                PinnedKey(keys.public_key, KEY_ID),
                PSK,
                PSK_ID,
                target_origin="https://api.example.test",
                connector=connector,
                endpoint=endpoint,
            ) as client:
                async with client.stream("GET", "https://api.example.test/invalid") as response:
                    assert response.status == 422 and response.mode == "finite"
                    assert await response.read() == b"invalid"
        assert body_sent.is_set()

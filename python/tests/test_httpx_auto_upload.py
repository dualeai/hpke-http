"""HTTPX uses one v3 request stream for each logical body shape."""

from __future__ import annotations

import io
from collections.abc import AsyncIterator

import httpx
import pytest

from hpke_http import Response, Server, generate_key_pair
from hpke_http.middleware import PinnedKey
from hpke_http.middleware._discovery import encode_key_record
from hpke_http.middleware.httpx import DiscoveredEndpoint, HPKEAsyncClient
from hpke_http.transport import REQUEST_MEDIA_TYPE, RESPONSE_MEDIA_TYPE
from tests.stream_request import open_stream_request

KEY_ID = b"primary-2026-09"
PSK = b"a 32-byte minimum test credential!"
PSK_ID = b"tenant-42"
ENDPOINT = "https://api.example.test/protected"
LOGICAL_URL = "https://api.example.test/upload"


@pytest.mark.asyncio
async def test_pinned_key_uses_one_post_for_each_body_shape() -> None:
    keys = generate_key_pair()
    server = Server(keys.private_key, KEY_ID)
    calls: list[str] = []
    bodies: list[bytes] = []

    async def source() -> AsyncIterator[bytes]:
        yield b"first-"
        yield b"second"

    async def transport(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        assert request.method == "POST"
        assert request.headers["content-type"] == REQUEST_MEDIA_TYPE
        opened = open_stream_request(server, await request.aread(), PSK)
        bodies.append(opened.request.body)
        return httpx.Response(
            200,
            headers={"content-type": RESPONSE_MEDIA_TYPE},
            content=opened.protect_response(Response(status=200, body=b"ok")),
        )

    try:
        async with HPKEAsyncClient(
            PinnedKey(keys.public_key, KEY_ID),
            PSK,
            PSK_ID,
            transport=httpx.MockTransport(transport),
            endpoint=ENDPOINT,
        ) as client:
            assert (await client.post(LOGICAL_URL, content=b"bytes")).content == b"ok"
            assert (await client.post(LOGICAL_URL, content=source())).content == b"ok"
            assert (
                await client.post(LOGICAL_URL, files={"file": ("one.txt", io.BytesIO(b"file-content"))})
            ).content == b"ok"
        assert calls == ["POST", "POST", "POST"]
        assert bodies[:2] == [b"bytes", b"first-second"]
        assert b"file-content" in bodies[2]
    finally:
        server.close()


@pytest.mark.asyncio
async def test_discovery_reads_key_before_one_use_source() -> None:
    keys = generate_key_pair()
    server = Server(keys.private_key, KEY_ID)
    calls: list[str] = []
    source_reads = 0

    async def source() -> AsyncIterator[bytes]:
        nonlocal source_reads
        source_reads += 1
        yield b"body"

    async def transport(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "GET":
            assert source_reads == 0
            return httpx.Response(
                200,
                headers={"content-type": "application/octet-stream"},
                content=encode_key_record(KEY_ID, keys.public_key, 60),
            )
        opened = open_stream_request(server, await request.aread(), PSK)
        assert opened.request.body == b"body"
        return httpx.Response(
            200,
            headers={"content-type": RESPONSE_MEDIA_TYPE},
            content=opened.protect_response(Response(status=200, body=b"ok")),
        )

    try:
        async with DiscoveredEndpoint(ENDPOINT, transport=httpx.MockTransport(transport)) as key_source:
            async with HPKEAsyncClient(key_source, PSK, PSK_ID) as client:
                assert (await client.post(LOGICAL_URL, content=source())).content == b"ok"
        assert calls == ["GET", "POST"]
        assert source_reads == 1
    finally:
        server.close()

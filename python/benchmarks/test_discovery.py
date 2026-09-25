"""Measure the HTTPX adapter with an in-memory transport and no network delay."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import httpx
import pytest

from hpke_http import Response, Server, generate_key_pair
from hpke_http.middleware import PinnedKey
from hpke_http.middleware._discovery import encode_key_record
from hpke_http.middleware.httpx import DiscoveredEndpoint, HPKEAsyncClient
from hpke_http.transport import RESPONSE_MEDIA_TYPE

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("mode", ["pin", "discover"])
def test_httpx_adapter_roundtrip(benchmark: Any, mode: Literal["pin", "discover"]) -> None:
    keys = generate_key_pair()
    key_id = b"benchmark-key"
    psk = b"a 32-byte minimum benchmark credential"
    psk_id = b"benchmark-tenant"
    server = Server(keys.private_key, key_id)
    record = encode_key_record(key_id, server.public_key, 60)

    async def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, headers={"content-type": "application/octet-stream"}, content=record)
        opened = server.preparse(await request.aread()).authenticate(psk).admit(accepted=True)
        envelope = opened.protect_response(Response(status=200, body=b"ok"))
        return httpx.Response(200, headers={"content-type": RESPONSE_MEDIA_TYPE}, content=envelope)

    loop = asyncio.new_event_loop()
    endpoint = "https://api.example.test/protected"
    source = DiscoveredEndpoint(endpoint, transport=httpx.MockTransport(transport)) if mode == "discover" else None
    if source is None:
        client = HPKEAsyncClient(
            PinnedKey(keys.public_key, key_id),
            psk,
            psk_id,
            transport=httpx.MockTransport(transport),
            endpoint=endpoint,
        )
    else:
        client = HPKEAsyncClient(source, psk, psk_id)

    def roundtrip() -> bytes:
        return loop.run_until_complete(client.get("https://api.example.test/items")).content

    try:
        if benchmark(roundtrip) != b"ok":
            raise AssertionError("adapter benchmark returned the wrong body")
    finally:
        loop.run_until_complete(client.aclose())
        if source is not None:
            loop.run_until_complete(source.aclose())
        loop.close()
        server.close()


def test_httpx_shared_client_lifecycle(benchmark: Any) -> None:
    """Measure client setup and close when a long-lived source owns the pool."""
    endpoint = "https://api.example.test/protected"
    source = DiscoveredEndpoint(endpoint, transport=httpx.MockTransport(lambda _request: httpx.Response(503)))
    loop = asyncio.new_event_loop()

    def lifecycle() -> None:
        client = HPKEAsyncClient(source, b"a 32-byte minimum benchmark credential", b"benchmark-tenant")
        loop.run_until_complete(client.aclose())

    try:
        benchmark(lifecycle)
    finally:
        loop.run_until_complete(source.aclose())
        loop.close()

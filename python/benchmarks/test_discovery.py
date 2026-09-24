"""Measure the HTTPX adapter with an in-memory transport and no network delay."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import httpx
import pytest

from hpke_http import Response, Server, generate_key_pair
from hpke_http.middleware import Discover, PinnedKey
from hpke_http.middleware.httpx import HPKEAsyncClient
from hpke_http.transport import RESPONSE_MEDIA_TYPE

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("mode", ["pin", "discover"])
def test_httpx_adapter_roundtrip(benchmark: Any, mode: Literal["pin", "discover"]) -> None:
    keys = generate_key_pair()
    key_id = b"benchmark-key"
    psk = b"a 32-byte minimum benchmark credential"
    psk_id = b"benchmark-tenant"
    server = Server(keys.private_key, key_id)
    record = b"HHKD\x01" + bytes((len(key_id),)) + key_id + server.public_key

    async def transport(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, headers={"content-type": "application/octet-stream"}, content=record)
        opened = server.preparse(await request.aread()).authenticate(psk).admit(accepted=True)
        envelope = opened.protect_response(Response(status=200, body=b"ok"))
        return httpx.Response(200, headers={"content-type": RESPONSE_MEDIA_TYPE}, content=envelope)

    loop = asyncio.new_event_loop()
    client = HPKEAsyncClient(
        "https://api.example.test/protected",
        PinnedKey(keys.public_key, key_id) if mode == "pin" else Discover(),
        psk,
        psk_id,
        transport=httpx.MockTransport(transport),
    )

    def roundtrip() -> bytes:
        return loop.run_until_complete(client.get("https://api.example.test/items")).content

    try:
        if benchmark(roundtrip) != b"ok":
            raise AssertionError("adapter benchmark returned the wrong body")
    finally:
        loop.run_until_complete(client.aclose())
        loop.close()
        server.close()

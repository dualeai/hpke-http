"""CodSpeed measures a checked upload through the public ASGI middleware."""

from __future__ import annotations

import asyncio
import os
from typing import Any, cast

import pytest
from starlette.types import Message, Receive, Scope, Send

from hpke_http import Client, Method, Request, Response, generate_key_pair
from hpke_http.middleware.fastapi import HPKEMiddleware
from hpke_http.transport import REQUEST_MEDIA_TYPE

pytestmark = pytest.mark.benchmark
_REPLY_STATUS = 200


@pytest.mark.parametrize("event_size", [64 * 1024, 8 * 1024 * 1024 + 4096], ids=["64KiB-events", "one-event"])
def test_asgi_upload_8_mib(benchmark: Any, event_size: int) -> None:
    keys = generate_key_pair()
    key_id = b"benchmark-key"
    psk = b"a 32-byte minimum benchmark credential"
    clear = os.urandom(8 * 1024 * 1024)
    client = Client(keys.public_key, key_id, psk, b"benchmark-tenant")

    async def resolve(_psk_id: bytes, _scope: Scope) -> bytes:
        return psk

    async def admit(_replay_id: bytes, _deadline: int, _scope: Scope) -> bool:
        return True

    async def app(_scope: Scope, receive: Receive, send: Send) -> None:
        received = 0
        while True:
            message = await receive()
            received += len(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": _REPLY_STATUS, "headers": []})
        await send({"type": "http.response.body", "body": received.to_bytes(8, "big"), "more_body": False})

    middleware = HPKEMiddleware(
        app, keys.private_key, key_id, resolve, admit, key_use_for_s=60, transport_path="/protected"
    )
    scope = cast(
        Scope,
        {
            "type": "http",
            "method": "POST",
            "path": "/protected",
            "raw_path": b"/protected",
            "query_string": b"",
            "scheme": "https",
            "http_version": "1.1",
            "headers": [(b"host", b"api.example.test"), (b"content-type", REQUEST_MEDIA_TYPE.encode())],
        },
    )

    async def exchange() -> int:
        protected = client.protect(
            Request(method=Method.POST, authority="api.example.test", path="/upload", body=clear)
        )
        envelope = protected.envelope
        events: list[Message] = []
        offset = 0

        async def receive() -> Message:
            nonlocal offset
            if offset < len(envelope):
                part = envelope[offset : offset + event_size]
                offset += len(part)
                return {"type": "http.request", "body": part, "more_body": offset < len(envelope)}
            await asyncio.Event().wait()
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            events.append(message)

        await middleware(scope, receive, send)
        if events[0]["status"] != _REPLY_STATUS:
            raise AssertionError("ASGI upload failed")
        reply = protected.open_response(cast(bytes, events[1]["body"]))
        if reply != Response(status=_REPLY_STATUS, body=len(clear).to_bytes(8, "big")):
            raise AssertionError("ASGI upload lost body bytes")
        return len(clear)

    try:
        if benchmark(lambda: asyncio.run(exchange())) != len(clear):
            raise AssertionError("ASGI upload returned the wrong body length")
    finally:
        client.close()
        middleware.close()

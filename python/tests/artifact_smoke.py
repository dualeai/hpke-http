"""Exercise the public Python API from an installed release artifact."""

from __future__ import annotations

import asyncio
import sys
from importlib.metadata import version as distribution_version

from hpke_http import (
    BINDING_ABI_VERSION,
    PACKAGE_VERSION,
    PROTOCOL_ID,
    Client,
    Header,
    Method,
    Request,
    Response,
    Server,
    generate_key_pair,
)

KEY_ID = b"artifact-key"
PSK = b"a 32-byte minimum artifact credential"
PSK_ID = b"artifact-tenant"


def main() -> None:
    expected_version = sys.argv[1] if len(sys.argv) > 1 else distribution_version("hpke_http")
    if expected_version != PACKAGE_VERSION or expected_version != distribution_version("hpke_http"):
        raise RuntimeError("installed Python package version does not match the release candidate")
    if PROTOCOL_ID != "hpke-http/3" or BINDING_ABI_VERSION != 8:
        raise RuntimeError("installed Python binding identity is incoherent")

    keys = generate_key_pair()
    with Server(keys.private_key, KEY_ID) as key_server:
        if key_server.public_key != keys.public_key:
            raise RuntimeError("installed Python server returned the wrong public key")
    with (
        Client(keys.public_key, KEY_ID, PSK, PSK_ID) as client,
        Server(keys.private_key, KEY_ID) as server,
    ):
        protected = client.protect(Request(method=Method.GET, authority="artifact.example.test", path="/smoke"))
        opened = server.preparse(protected.envelope).authenticate(PSK).admit(accepted=True)
        expected = Response(status=200, body=b"artifact-ok")
        actual = protected.open_response(opened.protect_response(expected))
        if actual != expected:
            raise RuntimeError("installed Python artifact failed a public transaction")

    body = b"artifact-compression-" * 1024
    with (
        Client(keys.public_key, KEY_ID, PSK, PSK_ID) as client,
        Server(keys.private_key, KEY_ID) as server,
    ):
        protected = client.protect(
            Request(method=Method.POST, authority="artifact.example.test", path="/compressed", body=body)
        )
        if len(protected.envelope) >= len(body):
            raise RuntimeError("installed Python artifact did not encode a compressed request")
        opened = server.preparse(protected.envelope).authenticate(PSK).admit(accepted=True)
        if opened.request.body != body:
            raise RuntimeError("installed Python artifact did not restore the compressed request")
        expected = Response(status=200, body=body)
        encrypted = opened.protect_response(expected)
        if len(encrypted) >= len(body) or protected.open_response(encrypted) != expected:
            raise RuntimeError("installed Python artifact failed a compressed transaction")

    with (
        Client(keys.public_key, KEY_ID, PSK, PSK_ID) as client,
        Server(keys.private_key, KEY_ID) as server,
    ):
        protected = client.protect(Request(method=Method.GET, authority="artifact.example.test", path="/events"))
        opened = server.preparse(protected.envelope).authenticate(PSK).admit(accepted=True)
        sealer, start = opened.into_sealer(200, (Header("content-type", "text/event-stream"),))
        opener = protected.into_opener()
        try:
            used, checked = opener.feed(start)
            if used != len(start) or checked is None or checked.kind != "start" or checked.mode != "sse":
                raise RuntimeError("installed Python artifact did not check the SSE head")
            first = sealer.seal_sse_block(b": ready\n\n")
            used, checked = opener.feed(first)
            if used != len(first) or checked is None or checked.kind != "data" or checked.block != b": ready\n\n":
                raise RuntimeError("installed Python artifact did not release one checked live block")
            # The writer has not made END yet. The checked block is already available.
            end = sealer.finish()
            used, checked = opener.feed(end)
            if used != len(end) or checked is None or checked.kind != "end" or opener.finish_eof() is not None:
                raise RuntimeError("installed Python artifact did not check SSE completion")
        finally:
            sealer.close()
            opener.close()

    asyncio.run(check_adapter_discovery(keys.private_key))


async def check_adapter_discovery(private_key: bytes) -> None:
    """Use the installed host and HTTPX adapter through their public APIs."""
    import httpx
    from starlette.types import Receive, Scope, Send

    from hpke_http.middleware.aiohttp import HPKEClientSession
    from hpke_http.middleware.fastapi import HPKEMiddleware
    from hpke_http.middleware.httpx import DiscoveredEndpoint, HPKEAsyncClient

    if not (
        HPKEClientSession.__module__.startswith("hpke_http.") and HPKEMiddleware.__module__.startswith("hpke_http.")
    ):
        raise RuntimeError("installed Python package omitted an adapter")
    calls: list[str] = []
    replay_ids: set[bytes] = set()

    def resolve(credential_id: bytes, _scope: Scope) -> bytes:
        if credential_id != PSK_ID:
            raise LookupError("unknown artifact credential")
        return PSK

    def admit(replay_id: bytes, _deadline: int, _scope: Scope) -> bool:
        if replay_id in replay_ids:
            return False
        replay_ids.add(replay_id)
        return True

    async def app(scope: Scope, _receive: Receive, send: Send) -> None:
        if scope["path"] != "/items":
            raise RuntimeError("installed Python host sent the wrong logical path")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"artifact-discovery-ok"})

    middleware = HPKEMiddleware(
        app,
        private_key,
        KEY_ID,
        resolve,
        admit,
        key_use_for_s=60,
        transport_path="/protected",
        expected_authority="artifact.example.test",
    )

    async def host(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            calls.append(scope["method"])
        await middleware(scope, receive, send)

    try:
        endpoint = "https://artifact.example.test/protected"
        async with DiscoveredEndpoint(endpoint, transport=httpx.ASGITransport(app=host)) as source:
            for _ in range(2):
                async with HPKEAsyncClient(source, PSK, PSK_ID) as client:
                    response = await client.get("https://artifact.example.test/items")
                    if response.content != b"artifact-discovery-ok":
                        raise RuntimeError("installed Python discovery call failed")
            if calls != ["GET", "POST", "POST"]:
                raise RuntimeError("installed Python shared discovery failed")
    finally:
        middleware.close()


if __name__ == "__main__":
    main()

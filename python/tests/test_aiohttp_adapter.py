"""aiohttp adapter coverage through a real ephemeral TLS transport."""

from __future__ import annotations

import socket
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal, cast

import aiohttp
import pytest
import trustme
from aiohttp import web
from yarl import URL

from hpke_http import Header, Limits, Response, Server, StateError, TransportError, generate_key_pair
from hpke_http.middleware.aiohttp import HPKEClientSession, HPKEResponse
from hpke_http.transport import REQUEST_MEDIA_TYPE, RESPONSE_MEDIA_TYPE

KEY_ID = b"primary-2026-09"
PSK = b"a 32-byte minimum test credential!"
PSK_ID = b"tenant-42"

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
@pytest.mark.parametrize("compression", [None, "gzip", "zstd"])
async def test_aiohttp_adapter_protects_buffered_json_exchange(compression: Literal["gzip", "zstd"] | None) -> None:
    key_pair = generate_key_pair()
    server = Server(key_pair.private_key, KEY_ID, compression=compression is not None)
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
        if compression is not None:
            assert len(outer_envelope) < len(request_body)
        preparsed = server.preparse(outer_envelope)
        assert preparsed.psk_id == PSK_ID
        opened = preparsed.authenticate(PSK).admit(accepted=True)
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
        if compression is not None:
            assert len(envelope) < len(response_body)
        return web.Response(body=envelope, headers={"content-type": RESPONSE_MEDIA_TYPE})

    try:
        async with _https_endpoint(transport) as (endpoint, connector):
            session = HPKEClientSession(
                key_pair.public_key,
                KEY_ID,
                PSK,
                PSK_ID,
                base_url="https://api.example.test/",
                transport_endpoint=endpoint,
                connector=connector,
                compression=compression,
            )
            async with session:
                async with session.post("/items", json={"name": request_name}) as response:
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
    with pytest.raises(Exception, match="Not Found"):
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
        opened = server.preparse(await request.read()).authenticate(PSK).admit(accepted=True)
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
                key_pair.public_key,
                KEY_ID,
                PSK,
                PSK_ID,
                transport_endpoint=endpoint,
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
            key_pair.public_key,
            KEY_ID,
            PSK,
            PSK_ID,
            transport_endpoint=endpoint,
            connector=connector,
        ) as session:
            with pytest.raises(TransportError) as captured:
                await session.get("https://api.example.test/items")
    assert captured.value.code == expected_code


@pytest.mark.asyncio
async def test_aiohttp_adapter_bounds_the_actual_outer_response_body() -> None:
    async def transport(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"content-type": RESPONSE_MEDIA_TYPE})
        response.enable_chunked_encoding()
        await response.prepare(request)
        assert "content-length" not in response.headers
        await response.write(b"x" * 100_000)
        await response.write_eof()
        return response

    key_pair = generate_key_pair()
    async with _https_endpoint(transport) as (endpoint, connector):
        async with HPKEClientSession(
            key_pair.public_key,
            KEY_ID,
            PSK,
            PSK_ID,
            limits=Limits(max_body_len=1),
            transport_endpoint=endpoint,
            connector=connector,
        ) as session:
            with pytest.raises(TransportError) as captured:
                await session.get("https://api.example.test/items")
    assert captured.value.code == "response_too_large"


@pytest.mark.asyncio
async def test_aiohttp_adapter_maps_connection_failures() -> None:
    key_pair = generate_key_pair()
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        port = cast(tuple[str, int], unavailable.getsockname())[1]
        async with HPKEClientSession(
            key_pair.public_key,
            KEY_ID,
            PSK,
            PSK_ID,
            transport_endpoint=f"https://127.0.0.1:{port}/protected",
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
    with pytest.raises(Exception, match="unexpected content type"):
        await not_json.json()


def test_aiohttp_adapter_rejects_environment_credentials() -> None:
    key_pair = generate_key_pair()
    with pytest.raises(ValueError, match="trust_env"):
        HPKEClientSession(
            key_pair.public_key,
            KEY_ID,
            PSK,
            PSK_ID,
            trust_env=True,
        )

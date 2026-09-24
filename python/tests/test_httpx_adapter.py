"""httpx adapter coverage against the native server binding."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from hpke_http import (
    Header,
    Limits,
    ProtocolError,
    Request,
    Response,
    Server,
    StateError,
    TransportError,
    generate_key_pair,
)
from hpke_http.middleware import PinnedKey
from hpke_http.middleware.httpx import HPKEAsyncClient
from hpke_http.transport import RESPONSE_MEDIA_TYPE, filter_request_headers, filter_response_headers
from tests.stream_request import open_stream_request

KEY_ID = b"primary-2026-09"
PSK = b"a 32-byte minimum test credential!"
PSK_ID = b"tenant-42"


@pytest.mark.asyncio
async def test_httpx_adapter_protects_complete_exchange_without_ambient_headers() -> None:
    key_pair = generate_key_pair()
    server = Server(key_pair.private_key, KEY_ID)
    seen_replays: set[bytes] = set()
    request_name = "Ada" * 1024
    response_id = "item-" + "1" * 1024
    response_body = b'{"id":"' + response_id.encode() + b'"}'

    async def transport(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        assert "x-default" not in request.headers
        assert "x-logical" not in request.headers
        outer_envelope = await request.aread()
        first_length = server.stream_start_length(outer_envelope)
        assert first_length is not None
        preparsed = server.preparse_stream(outer_envelope[:first_length])
        assert preparsed.psk_id == PSK_ID
        authenticated = preparsed.authenticate(PSK)
        assert authenticated.replay_id not in seen_replays
        seen_replays.add(authenticated.replay_id)
        opened_stream = authenticated.admit(accepted=True)
        parts: list[bytes] = []
        offset = first_length
        while offset < len(outer_envelope):
            used, record = opened_stream.feed(outer_envelope, offset)
            offset += used
            if record is not None and record[0] == "data":
                parts.append(record[1])
        opened = opened_stream.finish_eof()
        checked_request = Request(
            opened_stream.head.method,
            opened_stream.head.authority,
            opened_stream.head.path,
            opened_stream.head.headers,
            b"".join(parts),
        )
        assert checked_request.path == "/items?limit=2"
        assert json.loads(checked_request.body) == {"name": request_name}
        assert all(field.name != "accept-encoding" for field in checked_request.headers)
        assert Header("authorization", "Bearer inner-secret") in checked_request.headers
        assert Header("cookie", "session=inner-secret") in checked_request.headers
        assert Header("x-default", "logical-default") in checked_request.headers
        assert Header("x-logical", "yes") in checked_request.headers
        envelope = opened.protect_response(
            Response(
                status=201,
                headers=(
                    Header("content-type", "application/json"),
                    Header("content-length", str(len(response_body))),
                    Header("set-cookie", "first=1; Path=/"),
                    Header("set-cookie", "second=2; Path=/"),
                ),
                body=response_body,
            )
        )
        return httpx.Response(200, headers={"content-type": RESPONSE_MEDIA_TYPE}, content=envelope)

    adapter = HPKEAsyncClient(
        "https://api.example.test/protected",
        PinnedKey(key_pair.public_key, KEY_ID),
        PSK,
        PSK_ID,
        headers={"x-default": "logical-default"},
        transport=httpx.MockTransport(transport),
    )
    async with adapter:
        response = await adapter.post(
            "https://api.example.test/items?limit=2",
            headers={
                "authorization": "Bearer inner-secret",
                "cookie": "session=inner-secret",
                "x-logical": "yes",
            },
            json={"name": request_name},
        )
        assert response.status_code == 201
        assert response.json() == {"id": response_id}
        assert response.headers["content-length"] == str(len(response_body))
        assert response.headers.get_list("set-cookie") == ["first=1; Path=/", "second=2; Path=/"]

    with pytest.raises(StateError):
        await adapter.get("https://api.example.test/after-close")
    server.close()


@pytest.mark.asyncio
async def test_httpx_adapter_rejects_ambient_client_state() -> None:
    key_pair = generate_key_pair()
    with pytest.raises(ValueError, match="ambient request state"):
        HPKEAsyncClient(
            "https://api.example.test/protected",
            PinnedKey(key_pair.public_key, KEY_ID),
            PSK,
            PSK_ID,
            cookies={"session": "outer-state"},
        )
    with pytest.raises(ValueError, match="trust_env"):
        HPKEAsyncClient(
            "https://api.example.test/protected", PinnedKey(key_pair.public_key, KEY_ID), PSK, PSK_ID, trust_env=True
        )
    async with HPKEAsyncClient(
        "https://api.example.test/protected", PinnedKey(key_pair.public_key, KEY_ID), PSK, PSK_ID
    ) as adapter:
        assert adapter._http.trust_env is False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    async with HPKEAsyncClient(
        "https://api.example.test/protected", PinnedKey(key_pair.public_key, KEY_ID), PSK, PSK_ID, trust_env=False
    ) as adapter:
        assert adapter._http.trust_env is False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]


def test_transport_filters_are_direction_aware_and_honor_connection_nominations() -> None:
    for name in (
        "connection",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authentication-info",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ):
        fields = ((name, "field-value"), ("x-logical", "yes"))
        assert filter_request_headers(fields) == (Header("x-logical", "yes"),)
        assert filter_response_headers(fields) == (Header("x-logical", "yes"),)

    for name in ("accept-encoding", "expect"):
        fields = ((name, "12"), ("x-logical", "yes"))
        assert filter_request_headers(fields) == (Header("x-logical", "yes"),)
        assert filter_response_headers(fields) == (Header(name, "12"), Header("x-logical", "yes"))

    assert filter_request_headers((("content-length", "12"),)) == (Header("content-length", "12"),)

    for filter_headers in (filter_request_headers, filter_response_headers):
        with pytest.raises(TransportError) as captured:
            filter_headers((("content-encoding", "gzip"),))
        assert captured.value.code == "inner_content_encoding"

    fields = (
        ("Connection", "X-Hop, X-Other"),
        ("X-Hop", "remove-me"),
        ("x-other", "remove-me-too"),
        ("content-length", "12"),
        ("expect", "100-continue"),
        ("set-cookie", "first=1"),
        ("set-cookie", "second=2"),
    )
    assert filter_request_headers(fields) == (
        Header("content-length", "12"),
        Header("set-cookie", "first=1"),
        Header("set-cookie", "second=2"),
    )
    assert filter_response_headers(fields) == (
        Header("content-length", "12"),
        Header("expect", "100-continue"),
        Header("set-cookie", "first=1"),
        Header("set-cookie", "second=2"),
    )


@pytest.mark.asyncio
async def test_httpx_outer_set_cookie_never_reaches_a_later_exchange() -> None:
    key_pair = generate_key_pair()
    server = Server(key_pair.private_key, KEY_ID)
    calls = 0

    async def transport(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert "cookie" not in request.headers
        opened = open_stream_request(server, await request.aread(), PSK)
        envelope = opened.protect_response(Response(status=200, body=b"ok"))
        headers = [("content-type", RESPONSE_MEDIA_TYPE)]
        if calls == 1:
            headers.append(("set-cookie", "outer-secret=must-not-persist; Path=/"))
        return httpx.Response(200, headers=headers, content=envelope)

    async with HPKEAsyncClient(
        "https://api.example.test/protected",
        PinnedKey(key_pair.public_key, KEY_ID),
        PSK,
        PSK_ID,
        transport=httpx.MockTransport(transport),
    ) as adapter:
        assert (await adapter.get("https://api.example.test/first")).content == b"ok"
        assert (await adapter.get("https://api.example.test/second")).content == b"ok"
    assert calls == 2
    server.close()


@pytest.mark.asyncio
async def test_httpx_adapter_rejects_nonidentity_authenticated_content() -> None:
    key_pair = generate_key_pair()
    server = Server(key_pair.private_key, KEY_ID)

    async def transport(request: httpx.Request) -> httpx.Response:
        opened = open_stream_request(server, await request.aread(), PSK)
        envelope = opened.protect_response(
            Response(
                status=200,
                headers=(Header("content-encoding", "gzip"),),
                body=b"compressed",
            )
        )
        return httpx.Response(200, headers={"content-type": RESPONSE_MEDIA_TYPE}, content=envelope)

    async with HPKEAsyncClient(
        "https://api.example.test/protected",
        PinnedKey(key_pair.public_key, KEY_ID),
        PSK,
        PSK_ID,
        transport=httpx.MockTransport(transport),
    ) as adapter:
        with pytest.raises(TransportError) as captured:
            await adapter.get("https://api.example.test/items")
    assert captured.value.code == "inner_content_encoding"
    server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "headers", "expected_code"),
    [
        (503, {"content-type": RESPONSE_MEDIA_TYPE}, "outer_status"),
        (200, {"content-type": "application/octet-stream"}, "outer_content_type"),
        (200, {"content-type": RESPONSE_MEDIA_TYPE, "content-encoding": "gzip"}, "outer_content_encoding"),
    ],
)
async def test_httpx_adapter_rejects_invalid_outer_responses(
    status: int,
    headers: dict[str, str],
    expected_code: str,
) -> None:
    async def transport(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=headers, content=b"")

    key_pair = generate_key_pair()
    async with HPKEAsyncClient(
        "https://gateway.example.test/protected",
        PinnedKey(key_pair.public_key, KEY_ID),
        PSK,
        PSK_ID,
        target_origin="https://api.example.test",
        transport=httpx.MockTransport(transport),
    ) as adapter:
        with pytest.raises(TransportError) as captured:
            await adapter.get("https://api.example.test/items")
    assert captured.value.code == expected_code


@pytest.mark.asyncio
async def test_httpx_adapter_bounds_unannounced_outer_response_bytes() -> None:
    closed = False
    key_pair = generate_key_pair()
    server = Server(key_pair.private_key, KEY_ID)

    class OversizedStream(httpx.AsyncByteStream):
        def __init__(self, envelope: bytes) -> None:
            self.envelope = envelope

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield self.envelope

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    async def transport(request: httpx.Request) -> httpx.Response:
        opened = open_stream_request(server, await request.aread(), PSK)
        envelope = opened.protect_response(Response(status=200, body=b"ab"))
        response = httpx.Response(200, headers={"content-type": RESPONSE_MEDIA_TYPE}, stream=OversizedStream(envelope))
        assert "content-length" not in response.headers
        return response

    async with HPKEAsyncClient(
        "https://api.example.test/protected",
        PinnedKey(key_pair.public_key, KEY_ID),
        PSK,
        PSK_ID,
        limits=Limits(max_body_len=1),
        transport=httpx.MockTransport(transport),
    ) as adapter:
        with pytest.raises(ProtocolError) as captured:
            await adapter.get("https://api.example.test/items")
    assert captured.value.code == "limit_exceeded"
    assert closed
    server.close()


@pytest.mark.asyncio
async def test_httpx_adapter_holds_finite_body_until_outer_eof() -> None:
    key_pair = generate_key_pair()
    server = Server(key_pair.private_key, KEY_ID)
    waiting_for_eof = asyncio.Event()
    release_eof = asyncio.Event()

    class HeldStream(httpx.AsyncByteStream):
        def __init__(self, envelope: bytes) -> None:
            self.envelope = envelope

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield self.envelope
            waiting_for_eof.set()
            await release_eof.wait()

        async def aclose(self) -> None:
            release_eof.set()

    async def transport(request: httpx.Request) -> httpx.Response:
        opened = open_stream_request(server, await request.aread(), PSK)
        envelope = opened.protect_response(Response(status=200, body=b"complete"))
        return httpx.Response(200, headers={"content-type": RESPONSE_MEDIA_TYPE}, stream=HeldStream(envelope))

    async with HPKEAsyncClient(
        "https://api.example.test/protected",
        PinnedKey(key_pair.public_key, KEY_ID),
        PSK,
        PSK_ID,
        transport=httpx.MockTransport(transport),
    ) as adapter:
        waiting = asyncio.create_task(adapter.get("https://api.example.test/items"))
        try:
            await asyncio.wait_for(waiting_for_eof.wait(), 2)
            assert not waiting.done()
            release_eof.set()
            response = await asyncio.wait_for(waiting, 2)
            assert response.content == b"complete"
        finally:
            release_eof.set()
            if not waiting.done():
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
    server.close()


@pytest.mark.asyncio
async def test_httpx_adapter_maps_response_stream_failure_to_network_error() -> None:
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"partial envelope"
            raise httpx.ReadError("response stream failed")

        async def aclose(self) -> None:
            pass

    async def transport(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": RESPONSE_MEDIA_TYPE}, stream=BrokenStream())

    key_pair = generate_key_pair()
    async with HPKEAsyncClient(
        "https://api.example.test/protected",
        PinnedKey(key_pair.public_key, KEY_ID),
        PSK,
        PSK_ID,
        transport=httpx.MockTransport(transport),
    ) as adapter:
        with pytest.raises(TransportError) as captured:
            await adapter.get("https://api.example.test/items")
    assert captured.value.code == "network_error"


@pytest.mark.asyncio
async def test_httpx_adapter_enforces_https_and_request_limit() -> None:
    stream_read = False

    async def oversized_stream() -> AsyncIterator[bytes]:
        nonlocal stream_read
        stream_read = True
        yield b"xx"

    key_pair = generate_key_pair()
    async with HPKEAsyncClient(
        "https://api.example.test/protected",
        PinnedKey(key_pair.public_key, KEY_ID),
        PSK,
        PSK_ID,
        limits=Limits(max_request_bytes=1),
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
    ) as adapter:
        with pytest.raises(TransportError) as insecure:
            await adapter.get("http://api.example.test/items")
        assert insecure.value.code == "invalid_target"
        with pytest.raises(ProtocolError) as oversized:
            await adapter.post("https://api.example.test/items", content=b"xx")
        assert oversized.value.code == "limit_exceeded"
        with pytest.raises(ProtocolError) as streamed_oversized:
            await adapter.post("https://api.example.test/items", content=oversized_stream())
        assert streamed_oversized.value.code == "limit_exceeded"
        assert stream_read

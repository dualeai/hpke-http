"""httpx adapter coverage against the native server binding."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Literal

import httpx
import pytest

from hpke_http import Header, Limits, Response, Server, StateError, TransportError, generate_key_pair
from hpke_http.middleware.httpx import HPKEAsyncClient
from hpke_http.transport import RESPONSE_MEDIA_TYPE, filter_request_headers, filter_response_headers

KEY_ID = b"primary-2026-09"
PSK = b"a 32-byte minimum test credential!"
PSK_ID = b"tenant-42"


@pytest.mark.asyncio
@pytest.mark.parametrize("compression", [None, "gzip", "zstd"])
async def test_httpx_adapter_protects_complete_exchange_without_ambient_headers(
    compression: Literal["gzip", "zstd"] | None,
) -> None:
    key_pair = generate_key_pair()
    server = Server(key_pair.private_key, KEY_ID, compression=compression is not None)
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
        preparsed = server.preparse(outer_envelope)
        assert preparsed.psk_id == PSK_ID
        authenticated = preparsed.authenticate(PSK)
        assert authenticated.replay_id not in seen_replays
        seen_replays.add(authenticated.replay_id)
        opened = authenticated.admit(accepted=True)
        assert opened.request.path == "/items?limit=2"
        assert json.loads(opened.request.body) == {"name": request_name}
        if compression is not None:
            assert len(outer_envelope) < len(opened.request.body)
        assert all(field.name != "accept-encoding" for field in opened.request.headers)
        assert Header("authorization", "Bearer inner-secret") in opened.request.headers
        assert Header("cookie", "session=inner-secret") in opened.request.headers
        assert Header("x-default", "logical-default") in opened.request.headers
        assert Header("x-logical", "yes") in opened.request.headers
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
        if compression is not None:
            assert len(envelope) < len(response_body)
        return httpx.Response(200, headers={"content-type": RESPONSE_MEDIA_TYPE}, content=envelope)

    adapter = HPKEAsyncClient(
        key_pair.public_key,
        KEY_ID,
        PSK,
        PSK_ID,
        base_url="https://api.example.test",
        headers={"x-default": "logical-default"},
        transport=httpx.MockTransport(transport),
        compression=compression,
    )
    async with adapter:
        response = await adapter.post(
            "/items?limit=2",
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
            key_pair.public_key,
            KEY_ID,
            PSK,
            PSK_ID,
            cookies={"session": "outer-state"},
        )
    with pytest.raises(ValueError, match="trust_env"):
        HPKEAsyncClient(key_pair.public_key, KEY_ID, PSK, PSK_ID, trust_env=True)
    async with HPKEAsyncClient(key_pair.public_key, KEY_ID, PSK, PSK_ID) as adapter:
        assert adapter._http.trust_env is False  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    async with HPKEAsyncClient(key_pair.public_key, KEY_ID, PSK, PSK_ID, trust_env=False) as adapter:
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

    for name in ("accept-encoding", "content-length", "expect"):
        fields = ((name, "12"), ("x-logical", "yes"))
        assert filter_request_headers(fields) == (Header("x-logical", "yes"),)
        assert filter_response_headers(fields) == (Header(name, "12"), Header("x-logical", "yes"))

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
        preparsed = server.preparse(await request.aread())
        opened = preparsed.authenticate(PSK).admit(accepted=True)
        envelope = opened.protect_response(Response(status=200, body=b"ok"))
        headers = [("content-type", RESPONSE_MEDIA_TYPE)]
        if calls == 1:
            headers.append(("set-cookie", "outer-secret=must-not-persist; Path=/"))
        return httpx.Response(200, headers=headers, content=envelope)

    async with HPKEAsyncClient(
        key_pair.public_key,
        KEY_ID,
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
        opened = server.preparse(await request.aread()).authenticate(PSK).admit(accepted=True)
        envelope = opened.protect_response(
            Response(
                status=200,
                headers=(Header("content-encoding", "gzip"),),
                body=b"compressed",
            )
        )
        return httpx.Response(200, headers={"content-type": RESPONSE_MEDIA_TYPE}, content=envelope)

    async with HPKEAsyncClient(
        key_pair.public_key,
        KEY_ID,
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
        key_pair.public_key,
        KEY_ID,
        PSK,
        PSK_ID,
        transport_endpoint="https://gateway.example.test/protected",
        transport=httpx.MockTransport(transport),
    ) as adapter:
        with pytest.raises(TransportError) as captured:
            await adapter.get("https://api.example.test/items")
    assert captured.value.code == expected_code


@pytest.mark.asyncio
async def test_httpx_adapter_bounds_unannounced_outer_response_bytes() -> None:
    closed = False

    class OversizedStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"x" * 100_000

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    async def transport(_request: httpx.Request) -> httpx.Response:
        response = httpx.Response(200, headers={"content-type": RESPONSE_MEDIA_TYPE}, stream=OversizedStream())
        assert "content-length" not in response.headers
        return response

    key_pair = generate_key_pair()
    async with HPKEAsyncClient(
        key_pair.public_key,
        KEY_ID,
        PSK,
        PSK_ID,
        limits=Limits(max_body_len=1),
        transport=httpx.MockTransport(transport),
    ) as adapter:
        with pytest.raises(TransportError) as captured:
            await adapter.get("https://api.example.test/items")
    assert captured.value.code == "response_too_large"
    assert closed


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
        key_pair.public_key,
        KEY_ID,
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
        key_pair.public_key,
        KEY_ID,
        PSK,
        PSK_ID,
        limits=Limits(max_body_len=1),
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
    ) as adapter:
        with pytest.raises(TransportError) as insecure:
            await adapter.get("http://api.example.test/items")
        assert insecure.value.code == "invalid_target"
        with pytest.raises(TransportError) as oversized:
            await adapter.post("https://api.example.test/items", content=b"xx")
        assert oversized.value.code == "request_too_large"
        with pytest.raises(TransportError) as streamed_oversized:
            await adapter.post("https://api.example.test/items", content=oversized_stream())
        assert streamed_oversized.value.code == "request_too_large"
        assert stream_read

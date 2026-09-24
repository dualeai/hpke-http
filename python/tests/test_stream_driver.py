"""Check live reader state across raw byte cuts and late failures."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from hpke_http import (
    Client,
    Header,
    Method,
    ProtocolError,
    Request,
    ResponseOpener,
    Server,
    StateError,
    generate_key_pair,
)
from hpke_http.middleware._records import CheckedStream

KEY_ID = b"stream-driver"
PSK = b"a 32-byte minimum driver credential"
PSK_ID = b"driver-tenant"


def _response_parts() -> tuple[ResponseOpener, bytes, bytes, bytes]:
    keys = generate_key_pair()
    with (
        Client(keys.public_key, KEY_ID, PSK, PSK_ID) as client,
        Server(keys.private_key, KEY_ID) as server,
    ):
        protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/events"))
        opened = server.preparse(protected.envelope).authenticate(PSK).admit(accepted=True)
        writer, start = opened.into_sealer(200, (Header("content-type", "text/event-stream"),))
        try:
            first = writer.seal_sse_block(b": ready\n\n")
            end = writer.finish()
            return protected.into_opener(), start, first, end
        finally:
            writer.close()


def _finite_response() -> tuple[ResponseOpener, bytes]:
    keys = generate_key_pair()
    with (
        Client(keys.public_key, KEY_ID, PSK, PSK_ID) as client,
        Server(keys.private_key, KEY_ID) as server,
    ):
        protected = client.protect(Request(method=Method.GET, authority="api.example.test", path="/finite"))
        opened = server.preparse(protected.envelope).authenticate(PSK).admit(accepted=True)
        writer, start = opened.into_sealer(200, (Header("content-type", "text/plain"),))
        try:
            data = writer.seal_finite_body(b"complete")
            assert data is not None
            return protected.into_opener(), start + data + writer.finish()
        finally:
            writer.close()


@pytest.mark.asyncio
async def test_driver_yields_checked_block_then_waits_for_outer_eof() -> None:
    opener, start, first, end = _response_parts()
    waiting_for_eof = asyncio.Event()
    release_eof = asyncio.Event()
    closed = False

    async def chunks() -> AsyncIterator[bytes]:
        for byte in start + first + end:
            yield bytes((byte,))
        waiting_for_eof.set()
        await release_eof.wait()

    async def close_outer() -> None:
        nonlocal closed
        closed = True

    driver = CheckedStream(opener, chunks(), close_outer)
    try:
        await driver.start()
        assert driver.status == 200
        assert driver.mode == "sse"
        iterator = driver.iter_sse()
        assert await anext(iterator) == b": ready\n\n"
        waiting = asyncio.ensure_future(anext(iterator))
        await asyncio.wait_for(waiting_for_eof.wait(), 2)
        assert not waiting.done()
        release_eof.set()
        with pytest.raises(StopAsyncIteration):
            await waiting
        with pytest.raises(StateError):
            await anext(driver.iter_sse())
    finally:
        release_eof.set()
        await driver.aclose()
    assert closed


@pytest.mark.asyncio
async def test_driver_holds_finite_body_until_outer_eof() -> None:
    opener, wire = _finite_response()
    waiting_for_eof = asyncio.Event()
    release_eof = asyncio.Event()

    async def chunks() -> AsyncIterator[bytes]:
        yield wire
        waiting_for_eof.set()
        await release_eof.wait()

    async def close_outer() -> None:
        release_eof.set()

    driver = CheckedStream(opener, chunks(), close_outer)
    try:
        await driver.start()
        waiting = asyncio.create_task(driver.read())
        await asyncio.wait_for(waiting_for_eof.wait(), 2)
        assert not waiting.done()
        release_eof.set()
        assert await asyncio.wait_for(waiting, 2) == b"complete"
    finally:
        release_eof.set()
        await driver.aclose()


@pytest.mark.asyncio
async def test_driver_keeps_first_block_visible_before_late_bad_tag() -> None:
    opener, start, first, end = _response_parts()
    bad_end = bytearray(end)
    bad_end[-1] ^= 1
    closed = False

    async def chunks() -> AsyncIterator[bytes]:
        yield start + first + bytes(bad_end)

    async def close_outer() -> None:
        nonlocal closed
        closed = True

    driver = CheckedStream(opener, chunks(), close_outer)
    try:
        await driver.start()
        iterator = driver.iter_sse()
        assert await anext(iterator) == b": ready\n\n"
        with pytest.raises(ProtocolError) as failure:
            await anext(iterator)
        assert failure.value.code == "authentication_failed"
    finally:
        await driver.aclose()
    assert closed

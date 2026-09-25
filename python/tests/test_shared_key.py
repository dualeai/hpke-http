"""Key leases and shared GET tasks at the async boundary."""

from __future__ import annotations

import asyncio
import time
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from hpke_http import StateError, TransportError
from hpke_http.middleware._shared_key import KeyLease, SharedKey


def test_lease_stays_expired_after_wall_clock_moves_back(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [time.monotonic(), time.time()]
    monkeypatch.setattr(
        "hpke_http.middleware._shared_key.time",
        SimpleNamespace(monotonic=lambda: now[0], time=lambda: now[1]),
    )
    lease = KeyLease(b"key", bytes(32), 60, now[0], now[1])
    assert lease.valid()
    with pytest.raises(FrozenInstanceError):
        lease.key_id = b"other"  # pyright: ignore[reportAttributeAccessIssue]
    now[1] += 61
    assert not lease.valid()
    now[1] -= 61
    assert not lease.valid()


@pytest.mark.asyncio
async def test_failed_get_clears_singleflight_for_next_caller() -> None:
    calls = 0

    async def fetch() -> tuple[bytes, bytes, int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransportError("discovery_network", "GET failed")
        return b"key", bytes(32), 60

    source = SharedKey(fetch, get_timeout_s=1)
    try:
        with pytest.raises(TransportError, match="GET failed"):
            await source.get()
        assert (await source.get()).key_id == b"key"
        assert (await source.get()).key_id == b"key"
        assert calls == 2
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_slow_get_cannot_return_an_expired_key() -> None:
    async def fetch() -> tuple[bytes, bytes, int]:
        await asyncio.sleep(1.05)
        return b"key", bytes(32), 1

    source = SharedKey(fetch, get_timeout_s=2)
    try:
        with pytest.raises(TransportError) as captured:
            await source.get()
        assert captured.value.code == "discovery_response"
    finally:
        await source.aclose()


@pytest.mark.asyncio
async def test_source_close_stops_one_shared_get() -> None:
    entered = asyncio.Event()

    async def fetch() -> tuple[bytes, bytes, int]:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("GET did not stop")

    source = SharedKey(fetch, get_timeout_s=10)
    first = asyncio.create_task(source.get())
    second = asyncio.create_task(source.get())
    await entered.wait()
    await source.aclose()
    for caller in (first, second):
        with pytest.raises(asyncio.CancelledError):
            await caller
    with pytest.raises(StateError):
        await source.get()

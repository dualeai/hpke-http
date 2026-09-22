"""Async adapters must not close a native stage while its worker still uses it."""

from __future__ import annotations

import asyncio
from threading import Event

import pytest

from hpke_http.middleware._native_async import run_native


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure"])
async def test_native_work_finishes_before_cancellation_is_delivered(outcome: str) -> None:
    started = Event()
    release = Event()

    def native_work() -> int:
        started.set()
        assert release.wait(timeout=3)
        if outcome == "failure":
            raise ValueError("worker failed after cancellation")
        return 7

    task = asyncio.create_task(run_native(native_work))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
async def test_native_work_returns_and_reports_errors() -> None:
    assert await run_native(lambda: 7) == 7

    def fail() -> int:
        raise ValueError("worker failed")

    with pytest.raises(ValueError, match="worker failed"):
        await run_native(fail)

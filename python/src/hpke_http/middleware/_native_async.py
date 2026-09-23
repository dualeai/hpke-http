"""Run bounded native work without blocking an async transport's event loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")


async def run_native(operation: Callable[..., _T], *args: object, **kwargs: object) -> _T:
    """Wait for native work to finish before cancellation can close its state."""
    task = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:  # noqa: PERF203 - cancellation may repeat while native work finishes
                continue
            except Exception:  # noqa: BLE001 - preserve the caller's cancellation after worker failure
                break
        raise

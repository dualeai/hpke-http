"""Run bounded native work without blocking an async transport's event loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import copy_context
from typing import TypeVar

_T = TypeVar("_T")


async def run_native(operation: Callable[..., _T], *args: object, **kwargs: object) -> _T:
    """Wait for native work to finish before cancellation can close its state."""
    context = copy_context()

    def invoke() -> _T:
        return context.run(operation, *args, **kwargs)

    future = asyncio.get_running_loop().run_in_executor(None, invoke)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:  # noqa: PERF203 - cancellation may repeat while native work finishes
                continue
            except Exception:  # noqa: BLE001 - preserve the caller's cancellation after worker failure
                break
        raise

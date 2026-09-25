"""One key lease and one key GET for a trusted outer endpoint."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from hpke_http.protocol import StateError
from hpke_http.transport import TransportError


@dataclass(frozen=True, slots=True)
class KeyLease:
    """One discovered public key with a service-set lease.

    The source starts the lease before GET. Call ``valid()`` when you use
    the key, since the lease can end after this object is returned.
    """

    key_id: bytes
    public_key: bytes
    _use_for_s: int = field(repr=False)
    _started_monotonic: float = field(repr=False)
    _started_wall: float = field(repr=False)
    _expired: bool = field(default=False, init=False, repr=False)

    def valid(self) -> bool:
        """Keep a key expired after either clock has passed its lease."""
        if self._expired:
            return False
        if (
            time.monotonic() - self._started_monotonic >= self._use_for_s
            or time.time() - self._started_wall >= self._use_for_s
        ):
            object.__setattr__(self, "_expired", True)
            return False
        return True


class SharedKey:
    """Keep one checked key and share one GET among callers on one event loop."""

    def __init__(
        self,
        fetch: Callable[[], Awaitable[tuple[bytes, bytes, int]]],
        *,
        get_timeout_s: float,
    ) -> None:
        if not math.isfinite(get_timeout_s) or get_timeout_s <= 0:
            raise ValueError("key GET timeout must be positive")
        self._fetch = fetch
        self._get_timeout_s = get_timeout_s
        self._key: KeyLease | None = None
        self._task: asyncio.Task[KeyLease] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif loop is not self._loop:
            raise StateError("shared key source belongs to another event loop")

    async def get(self) -> KeyLease:
        """Return a live lease, with one GET for all cold callers."""
        self._check_loop()
        if self._closed:
            raise StateError("shared key source is closed")
        key = self._key
        if key is not None and key.valid():
            return key
        task = self._task
        if task is None:
            task = asyncio.create_task(self._load())
            self._task = task
        return await asyncio.shield(task)

    async def _load(self) -> KeyLease:
        started_monotonic = time.monotonic()
        started_wall = time.time()
        try:
            try:
                key_id, public_key, use_for_s = await asyncio.wait_for(self._fetch(), self._get_timeout_s)
            except TimeoutError as error:
                raise TransportError("discovery_network", "key GET timed out") from error
            if self._closed:
                raise StateError("shared key source is closed")
            key = KeyLease(key_id, public_key, use_for_s, started_monotonic, started_wall)
            if not key.valid():
                raise TransportError("discovery_response", "key GET consumed its lifetime")
            self._key = key
            return key
        finally:
            if self._task is asyncio.current_task():
                self._task = None

    async def aclose(self) -> None:
        """Stop new callers and the source-owned GET task."""
        self._check_loop()
        if self._closed:
            return
        self._closed = True
        self._key = None
        task = self._task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._task = None


__all__ = ["KeyLease", "SharedKey"]

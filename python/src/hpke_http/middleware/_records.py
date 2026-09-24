"""Shared request writer and response reader for HTTPX and aiohttp."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from hpke_http.middleware._native_async import run_native
from hpke_http.protocol import (
    STREAM_DATA_LEN,
    CheckedRecord,
    Header,
    ProtocolError,
    Response,
    ResponseOpener,
    StateError,
    StreamRequestSealer,
)
from hpke_http.transport import TransportError


async def seal_request_chunk(sealer: StreamRequestSealer, chunk: bytes) -> AsyncIterator[bytes]:
    """Send one source chunk to the native writer and yield complete records."""
    data = bytes(chunk)
    offset = 0
    while offset < len(data):
        used, frame = await run_native(sealer.push, data[offset : offset + STREAM_DATA_LEN])
        if used == 0 and frame is None:
            raise RuntimeError("native request writer made no progress")
        offset += used
        if frame is not None:
            yield frame


class CheckedStream:
    """Own a raw HTTP body and one native response opener."""

    def __init__(
        self,
        opener: ResponseOpener,
        chunks: AsyncIterator[bytes],
        close_outer: Callable[[], Awaitable[None]],
    ) -> None:
        self._opener = opener
        self._chunks = chunks
        self._close_outer = close_outer
        self._pending = b""
        self._offset = 0
        self._closed = False
        self._claimed = False
        self._eof = False
        self._finite: Response | None = None
        self.status = 0
        self.headers: tuple[Header, ...] = ()
        self.mode = ""

    async def start(self) -> None:
        """Read until the first checked START record."""
        record = await self._next_record()
        if record is None or record.kind != "start":
            raise ProtocolError("malformed_envelope", "response START is missing")
        self.status = record.status
        self.headers = record.headers
        self.mode = record.mode

    async def _next_record(self) -> CheckedRecord | None:
        if self._closed:
            raise StateError("response stream is closed")
        while True:
            if self._offset < len(self._pending):
                used, record = self._opener.feed(self._pending, self._offset)
                self._offset += used
                if record is not None:
                    return record
                if used == 0:
                    raise ProtocolError("malformed_envelope", "record reader made no progress")
                continue
            self._pending = b""
            self._offset = 0
            try:
                chunk = await anext(self._chunks)
            except StopAsyncIteration:
                self._finite = self._opener.finish_eof()
                self._eof = True
                return None
            except asyncio.CancelledError:
                raise
            except (ProtocolError, TransportError):
                raise
            except Exception as error:
                raise TransportError("network_error", "protected HTTP response failed") from error
            self._pending = bytes(chunk)

    async def read(self) -> bytes:
        """Check finite DATA, END, and true outer body EOF."""
        if self.mode != "finite":
            raise StateError("read() requires a finite response")
        if self._claimed:
            raise StateError("response body already has a reader")
        self._claimed = True
        while not self._eof:
            record = await self._next_record()
            if record is not None and record.kind == "data":
                raise ProtocolError("malformed_envelope", "SSE data in finite response")
        if self._finite is None:
            raise ProtocolError("malformed_envelope", "finite response body is missing")
        return self._finite.body

    async def iter_sse(self) -> AsyncIterator[bytes]:
        """Yield one checked clear SSE block at a time."""
        if self.mode != "sse":
            raise StateError("iter_sse() requires an SSE response")
        if self._claimed:
            raise StateError("response body already has a reader")
        self._claimed = True
        while not self._eof:
            record = await self._next_record()
            if record is None:
                return
            if record.kind == "data":
                yield record.block

    async def aclose(self) -> None:
        """Close without claiming a checked END or body EOF."""
        if self._closed:
            return
        self._closed = True
        self._opener.close()
        await self._close_outer()

"""E2E test server for granian.

This module is loaded by granian via `tests.e2e_server:app`.
Reads configuration from environment variables set by the test fixtures.
"""

import hashlib
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

# HPKE_DISABLE_ZSTD=true simulates zstd being unavailable for testing
# Must be patched before importing middleware (which caches availability)
if os.environ.get("HPKE_DISABLE_ZSTD") == "true":
    import hpke_http.streaming

    def _mock_import_zstd() -> Any:
        raise ImportError("zstd disabled for testing (HPKE_DISABLE_ZSTD=true)")

    hpke_http.streaming.import_zstd = _mock_import_zstd
    # Also clear any cached module
    hpke_http.streaming._zstd_module = None  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from hpke_http.constants import KemId
from hpke_http.middleware.fastapi import HPKEMiddleware

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")


async def health(_request: Request) -> JSONResponse:
    """Health check endpoint (plaintext, no encryption)."""
    _logger.debug("GET /health")
    return JSONResponse({"status": "ok"})


async def echo(request: Request) -> JSONResponse:
    """Echo endpoint - returns the decrypted body."""
    _logger.debug("%s /echo: reading body...", request.method)
    body = await request.body()
    _logger.debug("%s /echo: body read, %d bytes", request.method, len(body))
    return JSONResponse(
        {
            "path": request.url.path,
            "method": request.method,
            "echo": body.decode("utf-8", errors="replace"),
        }
    )


async def echo_headers(request: Request) -> JSONResponse:
    """Echo headers endpoint - returns Content-Type and other headers as seen by the app.

    Used to verify X-HPKE-Content-Type restoration after decryption.
    """
    _logger.debug("%s /echo-headers", request.method)
    body = await request.body()
    return JSONResponse(
        {
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
            "body_size": len(body),
            "all_headers": dict(request.headers),
        }
    )


async def stream(request: Request) -> StreamingResponse:
    """SSE streaming endpoint - encryption is automatic via middleware."""
    _logger.debug("POST /stream: reading body...")
    body = await request.body()
    _logger.debug("POST /stream: body read, %d bytes", len(body))

    async def sse_generator() -> AsyncGenerator[bytes]:
        """Generate SSE chunks: 3 progress + 1 complete."""
        _logger.debug("POST /stream: SSE generator started")
        for i in range(1, 4):
            _logger.debug("POST /stream: yielding progress %d/3", i)
            yield f"event: progress\ndata: {json.dumps({'step': i, 'total': 3})}\n\n".encode()
        _logger.debug("POST /stream: yielding complete")
        yield f"event: complete\ndata: {json.dumps({'result': 'success'})}\n\n".encode()
        _logger.debug("POST /stream: SSE generator done")

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


async def stream_delayed(request: Request) -> StreamingResponse:
    """SSE endpoint with delayed events (tests real streaming)."""
    import asyncio

    _logger.debug("POST /stream-delayed: reading body...")
    await request.body()
    _logger.debug("POST /stream-delayed: starting SSE")

    async def sse_generator() -> AsyncGenerator[bytes]:
        """Generate events with 100ms delays between them."""
        for i in range(5):
            yield f"event: tick\ndata: {json.dumps({'count': i})}\n\n".encode()
            await asyncio.sleep(0.1)  # 100ms delay
        yield f"event: done\ndata: {json.dumps({'total': 5})}\n\n".encode()
        _logger.debug("POST /stream-delayed: SSE done")

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


async def stream_large(request: Request) -> StreamingResponse:
    """SSE endpoint with large payload events."""
    _logger.debug("POST /stream-large: reading body...")
    await request.body()
    _logger.debug("POST /stream-large: starting SSE (3 x 10KB events)")

    async def sse_generator() -> AsyncGenerator[bytes]:
        """Generate events with ~10KB payloads."""
        for i in range(3):
            large_payload = {"index": i, "data": "x" * 10000}  # ~10KB
            yield f"event: large\ndata: {json.dumps(large_payload)}\n\n".encode()
        yield f"event: complete\ndata: {json.dumps({'sizes': [10000, 10000, 10000]})}\n\n".encode()
        _logger.debug("POST /stream-large: SSE done")

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


async def stream_many(request: Request) -> StreamingResponse:
    """SSE endpoint with 50+ events."""
    _logger.debug("POST /stream-many: reading body...")
    await request.body()
    _logger.debug("POST /stream-many: starting SSE (50 events)")

    async def sse_generator() -> AsyncGenerator[bytes]:
        """Generate 50 events."""
        for i in range(50):
            yield f"event: event\ndata: {json.dumps({'index': i})}\n\n".encode()
        yield f"event: complete\ndata: {json.dumps({'count': 50})}\n\n".encode()
        _logger.debug("POST /stream-many: SSE done")

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


async def echo_chunks(request: Request) -> JSONResponse:
    """Diagnostic endpoint reporting how request body arrived in chunks."""
    import time

    start_time = time.monotonic()
    chunks_received: list[dict[str, int | float]] = []

    async for chunk in request.stream():
        offset_ms = (time.monotonic() - start_time) * 1000
        chunks_received.append({"size": len(chunk), "offset_ms": round(offset_ms, 2)})

    total_bytes = sum(int(c["size"]) for c in chunks_received)
    return JSONResponse(
        {
            "chunk_count": len(chunks_received),
            "total_bytes": total_bytes,
            "chunks": chunks_received,
            "appears_streamed": len(chunks_received) > 1,
        }
    )


async def upload(request: Request) -> JSONResponse:
    """Handle multipart upload, return SHA256 hash + metadata for each part.

    Streams each part to compute hash without storing full content in memory.
    Client validates by comparing expected vs actual hashes.
    """
    _logger.debug("POST /upload: parsing multipart form...")
    form = await request.form()
    parts: list[dict[str, Any]] = []

    for key in form:
        value = form[key]
        # Check if it's an UploadFile (not a plain string field)
        if isinstance(value, UploadFile):
            # Stream hash computation - O(1) memory per part
            hasher = hashlib.sha256()
            size = 0
            while chunk := await value.read(64 * 1024):  # 64KB chunks
                hasher.update(chunk)
                size += len(chunk)

            parts.append(
                {
                    "field": key,
                    "filename": value.filename,
                    "content_type": value.content_type,
                    "size": size,
                    "sha256": hasher.hexdigest(),
                }
            )
            _logger.debug("POST /upload: processed part %s, size=%d", key, size)

    _logger.debug("POST /upload: completed, %d parts", len(parts))
    return JSONResponse({"parts": parts})


def _create_app() -> HPKEMiddleware:
    """Create ASGI app wrapped with HPKEMiddleware.

    Reads configuration from environment variables:
    - TEST_HPKE_PRIVATE_KEY: Hex-encoded X25519 private key
    - TEST_PSK: Hex-encoded pre-shared key
    - TEST_PSK_ID: Hex-encoded PSK ID
    - TEST_COMPRESS: Enable Zstd compression for SSE responses ("true"/"false")
    """
    # Read config from environment
    private_key_hex = os.environ.get("TEST_HPKE_PRIVATE_KEY", "")
    psk_hex = os.environ.get("TEST_PSK", "")
    psk_id_hex = os.environ.get("TEST_PSK_ID", "")
    compress_enabled = os.environ.get("TEST_COMPRESS", "").lower() == "true"

    if not private_key_hex:
        raise ValueError("TEST_HPKE_PRIVATE_KEY environment variable required")

    private_key = bytes.fromhex(private_key_hex)
    psk = bytes.fromhex(psk_hex) if psk_hex else b""
    psk_id = bytes.fromhex(psk_id_hex) if psk_id_hex else b""

    # Create base Starlette app
    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/echo", echo, methods=["POST", "PUT", "PATCH", "DELETE"]),
        Route("/echo-headers", echo_headers, methods=["POST"]),
        Route("/echo-chunks", echo_chunks, methods=["POST"]),
        Route("/stream", stream, methods=["POST"]),
        Route("/stream-delayed", stream_delayed, methods=["POST"]),
        Route("/stream-large", stream_large, methods=["POST"]),
        Route("/stream-many", stream_many, methods=["POST"]),
        Route("/upload", upload, methods=["POST"]),
    ]
    starlette_app = Starlette(routes=routes)

    # PSK resolver callback
    async def psk_resolver(_scope: dict[str, Any]) -> tuple[bytes, bytes]:
        """Return PSK and PSK ID for request."""
        return (psk, psk_id)

    # Wrap with HPKE middleware
    return HPKEMiddleware(
        app=starlette_app,
        private_keys={KemId.DHKEM_X25519_HKDF_SHA256: private_key},
        psk_resolver=psk_resolver,
        compress=compress_enabled,
    )


# Module-level app instance for granian
# Only create app if env vars are set (granian will set them, pytest won't)
_private_key_hex = os.environ.get("TEST_HPKE_PRIVATE_KEY", "")
app = _create_app() if _private_key_hex else None

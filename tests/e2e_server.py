"""E2E test server for granian.

This module is loaded by granian via `tests.e2e_server:app`.
Reads configuration from environment variables set by the test fixtures.
"""

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from hpke_http.constants import KemId
from hpke_http.middleware.fastapi import EncryptedSSEResponse, HPKEMiddleware


async def health(_request: Request) -> JSONResponse:
    """Health check endpoint (plaintext, no encryption)."""
    return JSONResponse({"status": "ok"})


async def echo(request: Request) -> JSONResponse:
    """Echo endpoint - returns the decrypted body."""
    body = await request.body()
    return JSONResponse(
        {
            "path": request.url.path,
            "method": request.method,
            "echo": body.decode("utf-8", errors="replace"),
        }
    )


async def stream(request: Request) -> EncryptedSSEResponse | StreamingResponse:
    """SSE streaming endpoint with encrypted events."""
    # Read and discard body (required by HPKE middleware)
    await request.body()

    async def sse_generator() -> AsyncGenerator[str]:
        """Generate SSE chunks: 3 progress + 1 complete."""
        for i in range(1, 4):
            yield f"event: progress\ndata: {json.dumps({'step': i, 'total': 3})}\n\n"
        yield f"event: complete\ndata: {json.dumps({'result': 'success'})}\n\n"

    # Get HPKE context from scope (set by middleware)
    ctx = request.scope.get("hpke_context")
    if ctx:
        return EncryptedSSEResponse(ctx, sse_generator())

    # Fallback: plaintext SSE (shouldn't happen in E2E tests)
    async def plaintext_sse() -> AsyncGenerator[bytes]:
        async for chunk in sse_generator():
            yield chunk.encode()

    return StreamingResponse(plaintext_sse(), media_type="text/event-stream")


async def stream_delayed(request: Request) -> EncryptedSSEResponse | StreamingResponse:
    """SSE endpoint with delayed events (tests real streaming)."""
    import asyncio

    await request.body()

    async def sse_generator() -> AsyncGenerator[str]:
        """Generate events with 100ms delays between them."""
        for i in range(5):
            yield f"event: tick\ndata: {json.dumps({'count': i})}\n\n"
            await asyncio.sleep(0.1)  # 100ms delay
        yield f"event: done\ndata: {json.dumps({'total': 5})}\n\n"

    ctx = request.scope.get("hpke_context")
    if ctx:
        return EncryptedSSEResponse(ctx, sse_generator())

    async def plaintext_sse() -> AsyncGenerator[bytes]:
        async for chunk in sse_generator():
            yield chunk.encode()

    return StreamingResponse(plaintext_sse(), media_type="text/event-stream")


async def stream_large(request: Request) -> EncryptedSSEResponse | StreamingResponse:
    """SSE endpoint with large payload events."""
    await request.body()

    async def sse_generator() -> AsyncGenerator[str]:
        """Generate events with ~10KB payloads."""
        for i in range(3):
            large_payload = {"index": i, "data": "x" * 10000}  # ~10KB
            yield f"event: large\ndata: {json.dumps(large_payload)}\n\n"
        yield f"event: complete\ndata: {json.dumps({'sizes': [10000, 10000, 10000]})}\n\n"

    ctx = request.scope.get("hpke_context")
    if ctx:
        return EncryptedSSEResponse(ctx, sse_generator())

    async def plaintext_sse() -> AsyncGenerator[bytes]:
        async for chunk in sse_generator():
            yield chunk.encode()

    return StreamingResponse(plaintext_sse(), media_type="text/event-stream")


async def stream_many(request: Request) -> EncryptedSSEResponse | StreamingResponse:
    """SSE endpoint with 50+ events."""
    await request.body()

    async def sse_generator() -> AsyncGenerator[str]:
        """Generate 50 events."""
        for i in range(50):
            yield f"event: event\ndata: {json.dumps({'index': i})}\n\n"
        yield f"event: complete\ndata: {json.dumps({'count': 50})}\n\n"

    ctx = request.scope.get("hpke_context")
    if ctx:
        return EncryptedSSEResponse(ctx, sse_generator())

    async def plaintext_sse() -> AsyncGenerator[bytes]:
        async for chunk in sse_generator():
            yield chunk.encode()

    return StreamingResponse(plaintext_sse(), media_type="text/event-stream")


def _create_app() -> HPKEMiddleware:
    """Create ASGI app wrapped with HPKEMiddleware.

    Reads configuration from environment variables:
    - TEST_HPKE_PRIVATE_KEY: Hex-encoded X25519 private key
    - TEST_PSK: Hex-encoded pre-shared key
    - TEST_PSK_ID: Hex-encoded PSK ID
    """
    # Read config from environment
    private_key_hex = os.environ.get("TEST_HPKE_PRIVATE_KEY", "")
    psk_hex = os.environ.get("TEST_PSK", "")
    psk_id_hex = os.environ.get("TEST_PSK_ID", "")

    if not private_key_hex:
        raise ValueError("TEST_HPKE_PRIVATE_KEY environment variable required")

    private_key = bytes.fromhex(private_key_hex)
    psk = bytes.fromhex(psk_hex) if psk_hex else b""
    psk_id = bytes.fromhex(psk_id_hex) if psk_id_hex else b""

    # Create base Starlette app
    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/echo", echo, methods=["POST"]),
        Route("/stream", stream, methods=["POST"]),
        Route("/stream-delayed", stream_delayed, methods=["POST"]),
        Route("/stream-large", stream_large, methods=["POST"]),
        Route("/stream-many", stream_many, methods=["POST"]),
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
    )


# Module-level app instance for granian
# Only create app if env vars are set (granian will set them, pytest won't)
_private_key_hex = os.environ.get("TEST_HPKE_PRIVATE_KEY", "")
app = _create_app() if _private_key_hex else None

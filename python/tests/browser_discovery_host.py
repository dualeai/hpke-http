"""Local HTTPS ASGI host for the browser discovery test."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import trustme
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.types import ASGIApp, Receive, Scope, Send

from hpke_http import generate_key_pair
from hpke_http.middleware.fastapi import HPKEMiddleware

KEY_ID = b"browser-key"
PSK = b"a 32-byte minimum browser credential"
PSK_ID = b"browser-tenant"


class Probe:
    """Count outer GET, preflight, and POST requests in this test only."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.key_get = 0
        self.preflight = 0
        self.protected_post = 0
        self.bad_get = 0
        self.key_get_had_credentials = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            method = scope.get("method")
            path = scope.get("path")
            if path == "/protected" and method == "GET" and not scope.get("query_string"):
                self.key_get += 1
                names = {name.lower() for name, _value in scope.get("headers", [])}
                self.key_get_had_credentials |= "authorization" in names or "cookie" in names
            elif path == "/protected" and method == "OPTIONS":
                self.preflight += 1
            elif path == "/protected" and method == "POST":
                self.protected_post += 1
            elif path == "/missing" and method == "GET":
                self.bad_get += 1
        await self.app(scope, receive, send)


async def main(directory: Path) -> None:
    ca = trustme.CA()
    cert = ca.issue_cert("127.0.0.1", "localhost")
    cert_path = directory / "cert.pem"
    key_path = directory / "key.pem"
    ca_path = directory / "ca.pem"
    cert.cert_chain_pems[0].write_to_path(cert_path)
    cert.private_key_pem.write_to_path(key_path)
    ca.cert_pem.write_to_path(ca_path)
    probe: Probe

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("path") == "/stats":
            body = json.dumps(
                {
                    "key_get": probe.key_get,
                    "preflight": probe.preflight,
                    "protected_post": probe.protected_post,
                    "bad_get": probe.bad_get,
                    "key_get_had_credentials": probe.key_get_had_credentials,
                }
            ).encode()
            status = 200
            headers = [(b"content-type", b"application/json")]
        elif scope.get("path") == "/items":
            if scope.get("method") == "POST":
                request = StarletteRequest(scope, receive)
                async with request.form(max_files=0, max_fields=1) as form:
                    body = b"browser-form-upload-ok" if form.get("note") == "browser-form-upload" else b"invalid form"
                status = 200 if body == b"browser-form-upload-ok" else 422
            else:
                body = b"browser-discovery-ok"
                status = 200
            headers = [(b"content-type", b"text/plain")]
        else:
            body = b"not found"
            status = 404
            headers = [(b"content-type", b"text/plain")]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    keys = generate_key_pair()
    seen: set[bytes] = set()

    def resolve(psk_id: bytes, _scope: Scope) -> bytes:
        if psk_id != PSK_ID:
            raise LookupError
        return PSK

    def admit(replay_id: bytes, _deadline: int, _scope: Scope) -> bool:
        if replay_id in seen:
            return False
        seen.add(replay_id)
        return True

    middleware = HPKEMiddleware(
        app,
        keys.private_key,
        KEY_ID,
        resolve,
        admit,
        transport_path="/protected",
        expected_authority="api.example.test",
    )
    cors = CORSMiddleware(
        middleware,
        allow_origin_regex=r"https://127\.0\.0\.1:\d+",
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    probe = Probe(cors)
    config = uvicorn.Config(
        probe,
        host="127.0.0.1",
        port=0,
        ssl_certfile=str(cert_path),
        ssl_keyfile=str(key_path),
        lifespan="off",
        log_level="error",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if task.done():
                await task
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        sys.stdout.write(
            json.dumps(
                {"endpoint": f"https://127.0.0.1:{port}/protected", "cert": str(cert_path), "key": str(key_path)}
            )
            + "\n"
        )
        sys.stdout.flush()
        await task
    finally:
        middleware.close()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))

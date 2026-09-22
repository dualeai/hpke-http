# hpke-http for Python

`hpke_http` is the typed Python binding and buffered HTTP integration layer for
the sole `hpke-http/1` implementation in this repository: the shared Rust
engine. The package contains no Python cryptographic implementation and has no
fallback when its private extension is absent or mismatched.

## Install and runtime support

The package supports CPython 3.10 through 3.14. Release wheels target Linux
x86-64 and AArch64, macOS universal2, and Windows x86-64. Other CPython targets
need a Rust source build. PyPy is not supported.

```sh
python -m pip install hpke_http
python -m pip install "hpke_http[httpx]"
python -m pip install "hpke_http[aiohttp]"
python -m pip install "hpke_http[fastapi]" fastapi
```

The `fastapi` extra installs the middleware's direct Starlette dependency; the
example application also declares FastAPI itself. The installed Python package
version, Rust engine version, protocol ID, and binding ABI are checked together
during import. Any skew is an import error.

## Protocol and credentials

The package authenticates complete bounded request and response messages. HTTPS
is still required: the protocol does not hide endpoints, recipient-key or PSK
identifiers, sizes, or timing.

Recipient keys use X25519 and are 32 bytes. Recipient-key and PSK identifiers
are public opaque values from 1 through 255 bytes. A PSK is at least 32 bytes,
and its public identifier must not equal the PSK.

Names such as `recipient_public_key`, `request_envelope`, `resolve_psk`, and
`replay_store` in the examples are application-provided key storage, transport,
and replay components; the package does not discover them.

```python
from hpke_http import generate_key_pair

key_pair = generate_key_pair()
store_recipient_key(key_pair.private_key, key_pair.public_key)
del key_pair
```

`store_recipient_key` is application storage in this example. Python
`bytes` are immutable and cannot be erased in place. Avoid unnecessary secret
copies and move generated private keys into protected storage promptly. Closing
a client or server releases native copies, not caller-owned byte strings.

| Limit | Default | Hard maximum |
| --- | ---: | ---: |
| Body bytes per message | 8 MiB | 64 MiB |
| Combined header-name and value bytes | 16 KiB | 64 KiB |
| Header fields per message | 64 | 256 |
| Combined authority and path bytes | 8 KiB | 8 KiB |

Pass `Limits` to a client, server, or adapter to make a limit stricter. A
`None` field uses the native default.

Protocol body compression is opt-in: set `compression="gzip"` or `"zstd"` on
`Client`, `HPKEAsyncClient`, or `HPKEClientSession`, and `compression=True` on
`Server` or `HPKEMiddleware`. The Rust engine codes only the body and
restores it before exposing the logical request or response. It enforces the
same body limit before and after decompression. A client that opts in requires
an extension-capable server; no silent fallback occurs. Ciphertext length can
leak information when attacker input and secrets share a body, so leave
compression disabled for those messages.

## Low-level client transaction

```python
from hpke_http import Client, Header, Method, Request

with Client(
    recipient_public_key,
    b"primary-2026-09",
    psk,
    b"tenant-42",
) as client:
    transaction = client.protect(
        Request(
            method=Method.POST,
            authority="api.example.test",
            path="/items",
            headers=(Header("content-type", "application/json"),),
            body=b'{"name":"Ada"}',
        )
    )
    try:
        response_envelope = send_envelope(transaction.envelope)
        response = transaction.open_response(response_envelope)
    finally:
        transaction.close()
```

`send_envelope` is the application's HTTPS transport in this low-level
example. `open_response` consumes the response continuation. A failed transport
attempt must not reuse an earlier envelope: call `Client.protect` again.

## Low-level server and replay admission

Server processing is staged so authenticated plaintext is not released before
one atomic replay-store decision:

```python
from hpke_http import Response, Server

with Server(recipient_private_key, b"primary-2026-09") as server:
    preparsed = server.preparse(request_envelope)
    try:
        request_psk = resolve_psk(preparsed.psk_id)
        authenticated = preparsed.authenticate(request_psk)
        try:
            accepted = replay_store.reserve_if_absent(
                authenticated.replay_id,
                authenticated.retain_until_exclusive,
            )
            opened = authenticated.admit(accepted=accepted)
            try:
                logical_response = dispatch(opened.request)
                response_envelope = opened.protect_response(
                    Response(
                        status=logical_response.status,
                        headers=logical_response.headers,
                        body=logical_response.body,
                    )
                )
            finally:
                opened.close()
        finally:
            authenticated.close()
    finally:
        preparsed.close()
```

The replay operation must atomically reserve an ID if absent across every worker
that can receive the same credentials. Keep it through the supplied exclusive
Unix deadline. Treat store errors and uncertain outcomes as rejected. The Rust
engine checks the trusted clock after the store operation and never releases
plaintext at or after the authenticated deadline.

## HTTP boundary rules

Authenticated headers are ordered fields with lower-case token names and
canonical ASCII values. Repeated names such as `set-cookie` keep their order.

The core rejects `Connection`, `Expect`, `Host`, `Keep-Alive`, proxy
authentication fields, `Proxy-Connection`, `TE`, `Trailer`,
`Transfer-Encoding`, and `Upgrade`. High-level adapters also remove each
field named by `Connection`, plus request `Accept-Encoding`,
`Content-Length`, and `Expect`, before they construct the logical message.

The adapters accept only absent or identity logical `Content-Encoding`. They
do not decompress authenticated logical bytes outside the body bound. Python
client adapters also require identity coding for outer envelopes. The opt-in
Rust protocol transform is independent of this HTTP representation field.

A single authenticated `Content-Length` must equal the body on requests and
ordinary responses. It is representation metadata for `HEAD` and 304, must be
zero when present on 205, and is forbidden for 204. `HEAD`, 204, 205, and 304
responses contain no logical body.

Authenticated `Set-Cookie` fields remain visible as response fields, but the
dedicated outer HTTP clients never store or resend cookies. Logical cookies or
authorization are encrypted fields and are never copied to the outer request.

## httpx

`HPKEAsyncClient` returns a fully buffered and authenticated
`httpx.Response`:

```python
from hpke_http.middleware.httpx import HPKEAsyncClient

async with HPKEAsyncClient(
    recipient_public_key,
    b"primary-2026-09",
    psk,
    b"tenant-42",
    base_url="https://api.example.test",
) as client:
    response = await client.post("/items", json={"name": "Ada"})
    response.raise_for_status()
```

Client default headers and per-request headers belong to the logical request,
except for adapter-owned transport fields. Set `transport_endpoint` to use one
fixed HTTPS envelope endpoint. The outer exchange is a separately constructed,
non-redirecting `POST` with `message/hpke-http-request`; it does not inherit
logical headers, cookies, authorization, event hooks, or redirects.
Ambient HTTPX proxy and CA settings are disabled; `trust_env=True` is rejected.

## aiohttp

`HPKEClientSession` is a supported buffered subset, not a drop-in replacement
for every `aiohttp.ClientSession` feature:

```python
from hpke_http.middleware.aiohttp import HPKEClientSession

async with HPKEClientSession(
    recipient_public_key,
    b"primary-2026-09",
    psk,
    b"tenant-42",
    base_url="https://api.example.test/",
) as session:
    async with session.post("/items", json={"name": "Ada"}) as response:
        payload = await response.json()
```

`HPKEResponse` exposes authenticated status, headers, URL, method, reason,
buffered `read`, `text`, and `json`, status checking, and context-manager
compatibility. It has no live socket, streaming body, redirect history, or
cookie-jar side effect. Set `transport_endpoint` to use a fixed envelope
endpoint.

## FastAPI and Starlette

The ASGI middleware accepts sync or async PSK resolvers and replay admitters. The
replay callback receives `(replay_id, retain_until_exclusive, scope)` and must
implement the atomic operation described above.
The PSK resolver raises `LookupError` for an unknown public ID; the middleware
returns a generic outer 400. Other resolver failures return outer 503 without
exposing callback details.

```python
from fastapi import FastAPI

from hpke_http.middleware.fastapi import HPKEMiddleware

app = FastAPI()
app.add_middleware(
    HPKEMiddleware,
    recipient_private_key=private_key,
    recipient_key_id=b"primary-2026-09",
    psk_resolver=resolve_psk,
    replay_admitter=admit_replay_id,
    expected_authority="api.example.test",
)
```

The middleware closes its native server when the application's ASGI lifespan
ends. With a host that does not send lifespan events, construct
`HPKEMiddleware(app, ...)` directly instead of using `add_middleware`; retain
that wrapper and call `close()` during host shutdown.

Set `transport_path="/_protected"` to use one fixed outer endpoint and dispatch
the authenticated inner path within the application. Other routes then remain
outside this middleware. Without it, every HTTP route is protected and the inner
target must match the outer target. In fixed-transport mode,
`expected_authority` is recommended so one endpoint cannot dispatch arbitrary
authenticated authorities.

Boundary failures use outer 400 for malformed or unauthenticated input, 405 for
a non-POST outer request, 409 for replay or invalid authenticated request time,
415 for invalid envelope media or content coding, 421 for a target mismatch,
500 for an invalid application response, and 503 when credential resolution,
replay storage, or the trusted clock is unavailable.

## Errors and lifecycle

`ProtocolError.code` uses stable language-neutral Rust codes:

- configuration and parsing: `invalid_configuration`, `limit_exceeded`,
  `malformed_envelope`, `unsupported_version`, `unsupported_suite`, and
  `unsupported_method`;
- credentials and authentication: `unknown_recipient_key`,
  `invalid_credential`, and `authentication_failed`;
- replay and time: `replay_rejected`, `invalid_request_time`,
  `replay_decision_mismatch`, and `clock_unavailable`;
- platform and local operations: `entropy_unavailable`, `crypto_failure`,
  and `compression_failure`.

`StateError` uses `state_consumed`. `TransportError.code` distinguishes
invalid targets, request and response bounds, network failure, invalid outer
status or media type, unsupported outer coding, and unsupported authenticated
content coding. `TransportError.status_code` is present only for an invalid
outer HTTP status.

`Client` and `Server` support context managers. Every one-shot continuation
supports idempotent `close()`. Close continuations in `finally` blocks when a
host operation can fail before the next stage consumes them.

There are no discovery, SSE, incremental-streaming, Flask, or Django adapters.
Whole-message buffering and explicit key configuration are protocol constraints.

## Development

From the repository root:

```sh
uv run --project python maturin develop --manifest-path python/native/Cargo.toml
uv run --project python pytest
```

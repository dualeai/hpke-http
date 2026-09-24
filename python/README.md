# hpke-http for Python

`hpke_http` is the typed Python binding and HTTP integration layer for
the sole `hpke-http/2` implementation in this repository: the shared Rust
engine. The package contains no Python cryptographic implementation and has no
fallback when its private extension is absent or mismatched.

## Install and runtime support

Build the `/2` Python binding from this checkout. From the repository root, run:

```sh
make install-deps-python build-python
```

This installs the binding in the project's `uv` environment.

The package supports CPython 3.10 through 3.14. Release wheels target Linux
x86-64 and AArch64 and macOS universal2. Windows is not supported. Other Linux
and macOS CPython targets need a Rust source build. PyPy is not supported.

For a built package, install the extra for the adapter you use:
`hpke_http[httpx]`, `hpke_http[aiohttp]`, or `hpke_http[fastapi]`. The `fastapi`
extra installs the middleware's direct Starlette dependency; the example
application also needs FastAPI. The checkout command above installs all extras.
The installed Python package version, Rust engine version, protocol ID, and
binding ABI are checked together during import. Any skew is an import error.

## Protocol and credentials

The package authenticates complete bounded requests and each response record. HTTPS
is still required: the protocol does not hide endpoints, recipient-key or PSK
identifiers, record sizes, counts, or timing. The client checks START before it
shows status or headers. It checks each SSE block before it yields its bytes.
A finite response needs checked END and outer body EOF before it is complete.

Recipient keys use X25519 and are 32 bytes. Recipient-key and PSK identifiers
are public opaque values from 1 through 255 bytes. A PSK needs at least 32
bytes of entropy, and its public identifier must not equal the PSK.

Names such as `recipient_public_key`, `request_envelope`, `resolve_psk`, and
`replay_store` in the examples are application-provided key storage, transport,
and replay components. The high-level clients can get a public key from a
fixed HTTPS endpoint.

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
| Request, finite body, or one SSE block | 8 MiB | 64 MiB |
| Combined header-name and value bytes | 16 KiB | 64 KiB |
| Header fields per message | 64 | 256 |
| Combined authority and path bytes | 8 KiB | 8 KiB |

Pass `Limits` to a client, server, or adapter to make a limit stricter. A
`None` field uses the native default. SSE has no whole-stream body cap.

Protocol body compression is opt-in: set `compression="gzip"` or `"zstd"` on
`Client`, `HPKEAsyncClient`, or `HPKEClientSession`, and `compression=True` on
`Server` or `HPKEMiddleware`. The Rust engine codes only the body and
restores it before exposing the logical request or response. It enforces the
same body limit before and after decompression. A client that opts in requires
an extension-capable server; no silent fallback occurs. Ciphertext length can
leak information when attacker input and secrets share a body, so leave
compression disabled for those messages.
SSE blocks use no private compression.

## httpx

`HPKEAsyncClient` returns a fully buffered and authenticated
`httpx.Response` for ordinary `request()` calls:

```python
from hpke_http.middleware import Discover
from hpke_http.middleware.httpx import HPKEAsyncClient

async with HPKEAsyncClient(
    "https://api.example.test/protected",
    Discover(),
    psk,
    b"tenant-42",
) as client:
    response = await client.post("https://api.example.test/items", json={"name": "Ada"})
    response.raise_for_status()

    async with client.stream("GET", "https://api.example.test/events") as response:
        if response.mode != "sse":
            body = await response.read()  # A complete checked finite reply.
            raise RuntimeError(f"expected SSE, got {response.status_code}: {body!r}")
        async for block in response.iter_sse():
            handle_sse_block(block)
```

Client default headers and per-request headers belong to the logical request,
except for adapter-owned transport fields. One fixed endpoint serves a key
through GET and accepts a protected POST. Discovery sends one GET per call. Use
`PinnedKey(recipient_public_key, key_id)` instead of `Discover()` to send no GET.
Both modes send POST to the configured endpoint. The outer exchange does not
inherit logical headers, cookies, authorization, event hooks, or redirects.
Set `target_origin="https://api.example.test"` when a gateway endpoint has a
different HTTPS origin from the logical target.
Ambient HTTPX proxy and CA settings are disabled; `trust_env=True` is rejected.

For a live SSE reply, enter `stream()` while the client is open. The caller's
`handle_sse_block` code parses `data`, `event`, `id`, `retry`, and comment lines.

`iter_sse()` yields one complete LF-ended `bytes` block after its own tag
passes. The context closes the outer socket on early exit. `read()` works for
finite replies, and a second body reader fails. `request()` rejects an SSE
reply at checked START; it never waits for the SSE stream to end. A caller
timeout still applies. With no per-call timeout, the protected POST has no
idle read timeout; a key GET uses the client's normal timeout.

Blocks contain raw bytes. Follow the
[SSE parsing rules](https://html.spec.whatwg.org/multipage/server-sent-events.html#interpreting-an-event-stream):
decode them as UTF-8 with replacement for bad byte sequences and ignore one
leading BOM at the start of the stream. A checked block can hold comments or
control fields without a dispatched data event.

## aiohttp

`HPKEClientSession` is a supported HTTP subset, not a drop-in replacement
for every `aiohttp.ClientSession` feature:

```python
from hpke_http.middleware import Discover
from hpke_http.middleware.aiohttp import HPKEClientSession

async with HPKEClientSession(
    "https://api.example.test/protected",
    Discover(),
    psk,
    b"tenant-42",
) as session:
    async with session.post("https://api.example.test/items", json={"name": "Ada"}) as response:
        payload = await response.json()

    async with session.stream("GET", "https://api.example.test/events") as response:
        if response.mode != "sse":
            body = await response.read()
            raise RuntimeError(f"expected SSE, got {response.status}: {body!r}")
        async for block in response.iter_sse():
            handle_sse_block(block)
```

`HPKEResponse` exposes authenticated status, headers, URL, method, reason,
buffered `read`, `text`, and `json`, status checking, and context-manager
support. It has no live socket, streaming body, redirect history, or
cookie-jar side effect. Set a trusted `connector` when a private CA serves the
endpoint; verified TLS remains the default.

The stream response's `status` and `headers` come from checked START.
`handle_sse_block` is application code that parses each checked byte block.
A caller timeout applies. With no per-call timeout, the protected POST has no
total or idle read timeout; a key GET uses the session's normal timeout. The
context closes the outer socket on early exit.

## Key GET and trust

`Discover()` gets one key before each protected call. The GET and POST use the
same configured HTTPS URL, with no query, fragment, or embedded credentials.
The client binds every logical request to one HTTPS origin and checks that
origin before it reads the request body or uses the network. The GET carries
no logical authorization, cookies, or PSK ID. It follows no redirects and
accepts only status 200, `application/octet-stream`, identity content coding,
and at most 293 raw bytes. It uses the transport's normal timeout. There is no
package key cache or extra timer.

The exact GET body is `"HHKD" || 0x01 || id_len:u8 || id || public_key[32]`.
The ID is 1 through 255 opaque bytes, so the record is 39 through 293 bytes.
The client rejects missing or extra bytes before it protects a request. Key
GET failures use `discovery_network`, `discovery_status`, or
`discovery_response`. Only `discovery_status` has a `status_code`. A bad GET
sends no POST. The adapters do not send plaintext or retry a protected POST.
After a lost POST reply, the caller does not know whether the app ran.

Both clients read the full logical request body before key GET. The body read,
GET, and protected POST run in order, so their wait times add. The HTTPX read
timeout applies to each read, and an aiohttp `ClientTimeout.total` starts again
for POST. Neither setting limits the full discovered call. Use a caller-owned
deadline around the full call when it needs one, such as `asyncio.wait_for` on
Python 3.10 or newer.

Keep the same one-key server key on every worker for an endpoint. A key change
while workers have different keys can make calls fail. Discovery adds one GET
round trip per call; choose a pinned key when that cost matters and the key is
known. Discovery does not make key changes seamless. HTTPX can use a
trusted `transport` or `verify` setting, and aiohttp can use a trusted
`connector`, for a private CA. The default checks the server certificate.

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
For a live reply, use `transaction.into_opener()` and feed raw outer bytes to
its `feed()` method. For each network chunk, call
`feed(chunk[offset : offset + 65536])`, add the returned byte count to `offset`,
and repeat until the whole chunk is used. The 64 KiB slice bounds temporary
copies.
Process each returned record; one chunk can hold several records. Finite DATA
returns no record. After END, call `finish_eof()` only when the outer body ends;
it then returns the finite response.

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
For a low-level SSE reply, pass status 200 and one `text/event-stream`
`Content-Type` to `opened.into_sealer(status, headers)`. Do not include logical
`Content-Length` or `Content-Encoding`. The call returns `(sealer, start_bytes)`.
Send `start_bytes`, then send the result of `seal_sse_block(block)` for each
complete LF-normalized block and `finish()` at the end. For a finite reply, call
`seal_finite_body(body)` once before `finish()`; an empty body gives no DATA
record. End the outer HTTP body after `finish()`.

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
    transport_path="/protected",
    expected_authority="api.example.test",
)
```

For cross-origin browser calls, wrap the completed app so CORS runs before
`HPKEMiddleware`:

```python
from starlette.middleware.cors import CORSMiddleware

app = CORSMiddleware(
    app,
    allow_origins=["https://web.example.test"],
    allow_methods=["GET", "POST"],
    allow_headers=["Accept", "Content-Type", "Cache-Control"],
)
```

The wrapper answers the browser's OPTIONS check and adds the allowed origin
to key GET, protected POST, and fault replies. A proxy can handle outer CORS
instead.

The middleware closes its native server when the application's ASGI lifespan
ends. With a host that does not send lifespan events, construct
`HPKEMiddleware(app, ...)` directly instead of using `add_middleware`; retain
that wrapper and call `close()` during host shutdown.

For `text/event-stream`, the middleware splits the app's ASGI body across any
chunk cuts, maps CR and CRLF line ends to LF, and sends each full blank-line
ended block in its own checked record. This includes comments and control
blocks. A complete comment block can serve as a heartbeat. It drops an
unfinished final block. A completed final ASGI body sends
checked END; an app failure or disconnect before that point leaves the reply
incomplete. Keep generic GZip middleware off the protected route, and turn off
response buffering in any proxy on that route. The outer reply uses identity
coding and `Cache-Control: no-store, no-transform`.

`transport_path` is required and must match the full ASGI `scope["path"]`.
Include any mount prefix, such as `/api/protected` for a mount at `/api`.
GET serves the public key record with `Cache-Control: no-store`; POST accepts a
protected envelope and dispatches the authenticated inner path. Other paths
pass to the app. A nonempty query on this path returns 400, another method
returns 405 with `Allow: GET, POST`, and GET
or POST after close returns 503. Set `expected_authority` for a gateway so
the endpoint accepts only one logical authority. Without it, the middleware
uses the outer `Host`.
When TLS ends at a proxy, that proxy must check the public HTTPS host and
forward the correct `Host` to ASGI. The middleware trusts that edge.

Boundary failures use outer 400 for malformed or unauthenticated input, 405 for
a method other than GET or POST at the endpoint, 409 for replay or invalid authenticated request time,
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
invalid targets, request body bounds, network failure, invalid outer
status or media type, unsupported outer coding, unsupported authenticated
content coding, and the three key GET failures above. `TransportError.status_code`
is present only for an invalid outer or discovery HTTP status.
Protected response record bounds use `ProtocolError`.

`Client` and `Server` support context managers. Every continuation
supports idempotent `close()`. Close continuations in `finally` blocks when a
host operation can fail before the next stage consumes them.

There is no automatic SSE reconnect or field parser. A new connection needs a
new protected request. There are no Flask or Django adapters.

## Development

From the repository root:

```sh
make install-deps-python build-python
make test-python
```

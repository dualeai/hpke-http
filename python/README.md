# hpke-http for Python

Send encrypted requests with HTTPX or aiohttp. Add `HPKEMiddleware` to check
requests before your FastAPI or Starlette app runs, then encrypt its replies.
Send JSON, forms, files, or byte streams. Read complete replies or live
server-sent events (SSE).

Start with [installation](#install-the-published-package) and the
[local demo](#run-a-local-round-trip). For an existing app, go to
[HTTPX](#send-requests-with-httpx), [aiohttp](#send-requests-with-aiohttp),
[server setup](#protect-an-asgi-app), or [troubleshooting](#troubleshooting).

## App URLs and the protected endpoint

```text
client.get("https://api.example.test/items?limit=2")
    -> POST https://api.example.test/protected
       with the app method, path, query, headers, and body encrypted inside
```

You choose the endpoint path; `/protected` and `/v1/hpke` are examples.
Every app method travels inside a POST to that endpoint. The guides call this
the **outer request**. Key discovery uses GET at the same endpoint.
Your app receives the original request, and your client receives the decrypted app status.

See [what observers can see](https://github.com/dualeai/hpke-http/blob/main/README.md#what-observers-can-see)
and the [protection limits](https://github.com/dualeai/hpke-http/blob/main/README.md#protection-limits).

## Install the published package

Install the v4 release with the HTTPX, aiohttp, and FastAPI adapters used below:

```sh
python -m pip install 'hpke-http[httpx,aiohttp,fastapi]~=4.0'
```

Select only the extras you need. The `fastapi` extra installs Starlette and
multipart support. Install FastAPI and an ASGI server separately if you use
them. A prebuilt package (wheel) includes the Rust engine and needs no Rust compiler.

Before upgrading an existing service, check
[HHKD v2 compatibility](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record).

## Run a local round trip

Save this as `demo.py` and run `python demo.py`. It sends JSON to a local app
and prints the checked reply. HTTPX connects to the app in the same process,
so you do not need to start a server or create a TLS certificate.

```python
import asyncio
import secrets

import httpx
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from hpke_http import generate_key_pair
from hpke_http.middleware.fastapi import HPKEMiddleware
from hpke_http.middleware.httpx import DiscoveredEndpoint, HPKEAsyncClient


async def main():
    keys = generate_key_pair()
    psk = secrets.token_bytes(32)
    psk_id = b"local-demo"
    seen = set()

    def resolve_psk(public_id, scope):
        if public_id != psk_id:
            raise LookupError("unknown PSK ID")
        return psk

    def admit_replay(replay_id, deadline, scope):
        # One-process demo: retain every ID until this program exits.
        if replay_id in seen:
            return False
        seen.add(replay_id)
        return True

    async def echo(request):
        return JSONResponse(await request.json(), status_code=201)

    app = Starlette(routes=[Route("/echo", echo, methods=["POST"])])
    protected_app = HPKEMiddleware(
        app, keys.private_key, b"demo-key", resolve_psk, admit_replay,
        transport_path="/protected", key_use_for_s=60,
    )
    try:
        transport = httpx.ASGITransport(app=protected_app)
        async with DiscoveredEndpoint(
            "https://api.example.test/protected", transport=transport,
        ) as key_source:
            async with HPKEAsyncClient(key_source, psk, psk_id) as client:
                reply = await client.post(
                    "https://api.example.test/echo", json={"message": "hello"},
                )
                reply.raise_for_status()
                print(reply.status_code, reply.json())
    finally:
        protected_app.close()


asyncio.run(main())
```

Expected output: `201 {'message': 'hello'}`. The in-memory replay set is for
this one-process demo. Follow [server setup](#protect-an-asgi-app) for stable
keys, HTTPS, and shared replay storage in a deployed service.

The remaining snippets use your app's config and callbacks. Run async
client code inside an `async def` function, called with `asyncio.run()`.
The upload examples read `large.bin`; replace it with a file on your machine.

## Protect an ASGI app

Store the server's recipient keys and pre-shared keys (PSKs) in your app's
secret store. Load the same recipient keys in each worker. Discovery supplies
only the recipient public key. Keep each PSK distinct from its public ID.

| Value | Who needs it | Format |
| --- | --- | --- |
| Recipient private key | Server only | 32 raw bytes from `generate_key_pair()` |
| Recipient public key and key ID | Client, by discovery or pin | 32 raw key bytes; public ID of 1..255 bytes |
| PSK | Client and server | At least 32 bytes of entropy; use `secrets.token_bytes(32)` for a new PSK |
| PSK ID | Client and server | The same public ID of 1..255 bytes on both sides |

Decode stored hex or Base64 values to raw bytes. The PSK authenticates the
protected message; your app still decides which actions to allow.
Supply the keys and app, choose a key lease in seconds (`lease_seconds`),
and implement the two callbacks below:

```python
from hpke_http.middleware.fastapi import HPKEMiddleware

protected_app = HPKEMiddleware(
    app,
    recipient_private_key,
    key_id,
    resolve_psk,
    admit_replay,
    key_use_for_s=lease_seconds,
    transport_path="/protected",
)
```

### Callbacks

- `resolve_psk(psk_id, scope)`: return the matching PSK, or raise `LookupError`
  for an unknown ID. The ID is untrusted until authentication passes.
- `admit_replay(replay_id, deadline, scope)`: atomically reserve the ID if
  it is unused. Return `True` only for its first use. Keep it until `deadline`,
  an exclusive Unix time in seconds, including across restarts.
  Reject admission if the store fails or its result is uncertain.

Both callbacks can be async. Share replay state across all workers and
endpoints that accept the same credentials. The deadline is exclusive:
the host rejects requests at or after that time. Keep clocks in sync;
requests have a five-minute lifetime with 30 seconds of allowed clock skew.

### Routing and host setup

- Match `transport_path` to the full ASGI path, including any mount prefix.
  That path serves key GET and protected POST.
- Enforce HTTPS. Other outer paths pass to the app; block ordinary calls if
  those routes must require HPKE. Your app still checks route permissions.
- Set `expected_authority` when the app host differs from outer `Host`.
  For a gateway, pair client `target_origin="https://service.example.test:8443"`
  with server `expected_authority="service.example.test:8443"` (no scheme or path).
- Use distinct recipient keys for separate services. The outer endpoint path
  does not bind a protected message to a service.
- Disable app and outer HTTP compression. Adapters accept only absent or
  identity `Content-Encoding`; internal zstd records are separate.
- Set `key_use_for_s` and a maximum time to deliver and parse the first protected
  request record (START). To change keys, follow the
  [key switch order](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record).
  `accepted_keys` takes other `(private_key, key_id)` pairs.
- Lifespan shutdown closes the native server. Without lifespan events, call
  `protected_app.close()` at shutdown.

Before it calls your app, the wrapper checks the full request and its END record.
It also waits for the actual end of the HTTP request body (EOF).
Set [upload and storage limits](#limits-and-payload-coding).

### Serve SSE

Return status 200 and `Content-Type: text/event-stream`. End each block with
a blank line; the wrapper drops an incomplete last block. Disable proxy
response buffering for live delivery.

Create the SSE route before wrapping the app:

```python
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route


async def events(request):
    async def blocks():
        yield b": ready\n\n"
        yield b"data: hello\n\n"

    return StreamingResponse(blocks(), media_type="text/event-stream")


app = Starlette(routes=[Route("/events", events)])
```

### Browser CORS

For browser calls from another origin, put CORS (cross-origin resource sharing)
middleware around the HPKE wrapper. It must answer OPTIONS and add headers to
GET, POST, and error replies:

```python
from starlette.middleware.cors import CORSMiddleware

browser_app = CORSMiddleware(
    protected_app,
    allow_origins=["https://app.example.test"],
    allow_methods=["GET", "POST"],
    allow_headers=["cache-control", "content-type"],
    expose_headers=["content-encoding"],
)
```

Expose outer `Content-Encoding` so the browser can check it, including when
a proxy adds the field.

## Send requests with HTTPX

Set `psk` to your app's shared secret bytes. This example uses `b"tenant-42"`
as its public PSK ID.

```python
from hpke_http.middleware.httpx import DiscoveredEndpoint, HPKEAsyncClient

endpoint = "https://api.example.test/protected"
async with DiscoveredEndpoint(endpoint) as key_source:
    async with HPKEAsyncClient(key_source, psk, b"tenant-42") as client:
        response = await client.post(
            "https://api.example.test/items", json={"name": "Ada"}
        )
        response.raise_for_status()
    async with HPKEAsyncClient(key_source, psk, b"tenant-42") as client:
        with open("large.bin", "rb") as file:
            response = await client.post(
                "https://api.example.test/upload",
                files={"upload": file},
            )
```

`request()` waits for the full reply and returns a checked `httpx.Response`.
Use HTTPX `content`, `data`, `json`, or `files`; `content` also accepts async
byte sources. The writer sends DATA parts of at most 64 KiB as the transport
reads the source.
The shared [key source](#keys-and-transport) avoids a new GET while its key lease is valid.

For an SSE reply, use `stream()` with an open client. Supply
`handle_sse_block` to decode and process each block:

```python
async with DiscoveredEndpoint(endpoint) as key_source:
    async with HPKEAsyncClient(key_source, psk, b"tenant-42") as client:
        async with client.stream(
            "GET", "https://api.example.test/events", timeout=30.0,
        ) as response:
            if response.mode != "sse":
                await response.read()
                raise RuntimeError(f"expected SSE; received status {response.status_code}")
            async for block in response.iter_sse():
                handle_sse_block(block)
```

The live reply is not an `httpx.Response`. See the
[shared stream API and reader rules](#sse-reader-rules).

## Send requests with aiohttp

Set `endpoint`, `logical_url`, `psk`, and `psk_id` from your app config.

```python
import aiohttp
from hpke_http.middleware.aiohttp import DiscoveredEndpoint, HPKEClientSession

form = aiohttp.FormData()
with open("large.bin", "rb") as source:
    form.add_field("upload", source, filename="large.bin")
    async with DiscoveredEndpoint(endpoint) as key_source:
        async with HPKEClientSession(key_source, psk, psk_id) as session:
            async with session.post(logical_url, data=form) as response:
                result = await response.read()
```

`HPKEClientSession` accepts bytes, text, forms, `FormData`, async byte sources,
and JSON. Its finite `HPKEResponse` has `status`, `headers`, `read()`, `text()`,
`json()`, and `raise_for_status()`. It is not an `aiohttp.ClientResponse` and
has no live `.content` reader.

Use a live context for SSE; supply `handle_sse_block` as above:

```python
async with DiscoveredEndpoint(endpoint) as key_source:
    async with HPKEClientSession(key_source, psk, psk_id) as session:
        async with session.stream(
            "GET", "https://api.example.test/events",
            timeout=aiohttp.ClientTimeout(total=None, sock_read=30),
        ) as response:
            if response.mode != "sse":
                await response.read()
                raise RuntimeError(f"expected SSE; received status {response.status}")
            async for block in response.iter_sse():
                handle_sse_block(block)
```

### SSE reader rules

Both live replies expose `headers`, `mode`, `read()`, and `iter_sse()`.
HTTPX uses `status_code`; aiohttp uses `status`.

Each reply allows one body reader. Check `mode`: use `iter_sse()` for SSE,
or `await response.read()` for a **finite reply** (a response other than SSE).
Each SSE block passes its checks before you receive it and uses line-feed (LF)
line endings. Parse its bytes under the
[SSE rules](https://html.spec.whatwg.org/multipage/server-sent-events.html#interpreting-an-event-stream).

A later read can fail after earlier blocks reached your app. Reading to the
end checks the END record and the end of the HTTP body (EOF).
Leaving the context early cancels the stream.
The clients do not parse SSE fields or reconnect. Reconnect with a fresh
request. Ordinary finite request methods reject SSE with `StateError`.

## Supported client options

Both adapters support GET, POST, PUT, PATCH, DELETE, HEAD, and OPTIONS.
Use absolute HTTPS app URLs on the configured origin (scheme, host, and port).
URL fragments do not travel in the protected request. Neither adapter follows
logical or outer redirects, and neither manages a login cookie jar.

HTTPX accepts normal body options and per-request `headers`, `params`, and
`timeout`. aiohttp's request options are `params`, `headers`, `data`,
`json`, and `timeout`; its adapter exposes a subset of `ClientSession`.
See [keys and transport](#keys-and-transport) for client defaults and pool setup.
Pass bearer tokens in the app's protected `headers`. Default `auth`, cookie settings,
and environment proxy credentials are disabled or rejected.
Transport-only fields are removed before protection. Logical
headers must satisfy the
[protocol field rules](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#data-coding-and-logical-http-checks).

## Technical details

### Keys and transport

Keep one `DiscoveredEndpoint(endpoint)` per full HTTPS endpoint URL. It owns
the HTTP connection pool and shares discovery GETs and key leases across clients.
Keep it open across calls, on one event loop.

| Setting | Where to set it |
| --- | --- |
| HTTPX TLS and pool options | HTTPX source |
| HTTPX logical `headers`, `params`, and `timeout` defaults | `HPKEAsyncClient` when it borrows a source |
| aiohttp connector and session options | aiohttp source |

Discovery trusts HTTPS; the key record has no separate signature. GET sends
no bearer token, cookies, or PSK ID. If one caller cancels, the shared GET
continues for other callers.

`get_key()` returns a `hpke_http.middleware.KeyLease` with `key_id`,
`public_key`, and `valid()`. A lease limits when you can start a POST with
that key. Check `valid()` when you use it. See the
[lease rules](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record)
and [gateway setup](#routing-and-host-setup).

#### Timeouts

`get_timeout_s` defaults to 10 seconds and limits only the discovery GET.
For POST, set `timeout` on the HTTPX client or request, or on the aiohttp
source session or request. Use a deadline in your app to limit the whole call,
including GET and POST.

Without an explicit per-request timeout, `stream()` disables HTTPX's read
timeout, or aiohttp's total and socket-read timeouts. The SSE examples set
an explicit idle bound.

#### Pinned keys

Use a pin when you already have the server's public key and matching key ID:

```python
from hpke_http.middleware import PinnedKey
from hpke_http.middleware.httpx import HPKEAsyncClient

pin = PinnedKey(recipient_public_key, key_id)
async with HPKEAsyncClient(pin, psk, psk_id, endpoint=endpoint) as client:
    response = await client.get("https://api.example.test/items")
    response.raise_for_status()
```

`HPKEClientSession` accepts the same pin and `endpoint=` arguments. A pin
skips GET and has no lease or automatic refresh; update it when keys change.

#### Cleanup and retries

Use `async with` to close clients, sources, and live replies. For manual cleanup:

- HTTPX: call `await client.aclose()`, then `await key_source.aclose()`.
- aiohttp: call `await session.close()`, then `await key_source.close()`.
- Low-level engine objects: call synchronous `close()`.

Keep secret copies to a minimum: Python cannot erase `bytes` in place.
Keep keys and clear data out of logs.

The clients never retry protected POSTs. See
[retry safety](https://github.com/dualeai/hpke-http/blob/main/README.md#retry-safety).

### Transport errors

`hpke_http.TransportError.code` tells callers why an adapter could not finish
a protected call:

| Discovery code | Meaning |
| --- | --- |
| `discovery_network` | GET failed or timed out. |
| `discovery_status` | GET returned a status other than 200. |
| `discovery_response` | GET headers or key record were invalid. |
| `discovery_expired` | The lease ended before POST START. |

`status_code` is set for `discovery_status` and `outer_status`.
App HTTP errors return a normal response; call `raise_for_status()` to raise
on them. Those exceptions can include the clear app URL and query.

- `ProtocolError.code`: configuration, method, authentication, framing, time,
  replay, or limit failure.
- `StateError`: a `ProtocolError` with `state_consumed`, including closed
  clients, consumed rights, or wrong body-reader use.
- `TypeError` or `ValueError`: bad arguments or unsupported options.
- `asyncio.CancelledError`: task cancellation.

### Troubleshooting

| Failure | Check |
| --- | --- |
| `invalid_target` | Use an absolute HTTPS app URL on the allowed origin. Check `target_origin`. |
| `discovery_status` or `discovery_response` | Check the full endpoint path, GET routing, HHKD v2 support, media type, and absence of outer compression. |
| `discovery_expired` | Check the key lease and delay before POST START; do not replay a saved envelope. |
| `outer_status` 400 | Check matching key/PSK IDs and raw key bytes, request limits, and host diagnostics. The generic fault does not identify one cause. |
| `outer_status` 409 | Check clock skew and replay-store decisions. |
| `outer_status` 421 | Match the logical authority with the host's `expected_authority` or outer Host. |
| `outer_status` 500 or 503 | Check the app response, credential/replay store, host lifecycle, and temporary storage. |
| `inner_content_encoding` or `outer_content_encoding` | Disable the relevant HTTP compression layer. |
| `ProtocolError` with `limit_exceeded` | Set compatible message limits on both peers. |
| `StateError` or `state_consumed` | Check object lifetime, stream mode, and whether a reader/right was already used. |

These statuses describe the Python host; a proxy can return its own errors.
Log codes without keys or clear request data. Follow the
[retry rules](#cleanup-and-retries) after an uncertain result.

### Limits and payload coding

Set compatible `Limits` on the client and server; they do not agree on limits
automatically. See the
[engine defaults and hard maxima](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#current-engine-limits)
for these Python fields:

| Setting | Applies to |
| --- | --- |
| `max_request_bytes` | Every request body |
| `max_body_len` | One-shot requests, finite replies, and each SSE block |
| `max_header_bytes` | Sum of header name and value bytes |
| `max_header_count` | Header pairs |
| `max_target_len` | Authority plus path bytes |

HTTPX and aiohttp use the streamed request limit. Calls that take a whole
body in memory (the one-shot `Client.protect()` and server readers) check
both body limits; the smaller applies.
SSE has no total body cap.

For example, create this value and pass `limits=limits` to both the client
and `HPKEMiddleware`:

```python
from hpke_http import Limits

limits = Limits(max_request_bytes=64 * 1024 * 1024, max_body_len=4 * 1024 * 1024)
```

Byte limits count data before compression and encryption. The ASGI host stores
checked upload bytes in memory, then moves them to a temporary file above 256 KiB.
That threshold does not limit total memory use; one ASGI event can exceed
64 KiB and use more memory. App form, file, and reply limits still apply.
Also bound concurrent uploads, time, and temporary disk use.

Read the
[compression limits](https://github.com/dualeai/hpke-http/blob/main/README.md#protection-limits)
before mixing secrets and attacker-chosen text. The
[DATA coding rules](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#data-coding-and-logical-http-checks)
define record compression separately from HTTP content coding.

### Low-level streamed requests

The HTTPX and aiohttp clients send records on their own. For a custom HTTP
transport, use the low-level writer. `send_part` writes one outer POST body part,
`end_outer_body` closes that body, and `source` supplies byte chunks:

```python
from contextlib import closing
from hpke_http import Client, Method, RequestHead

with closing(Client(recipient_public_key, key_id, psk, psk_id)) as client:
    head = RequestHead(method=Method.POST, authority="api.example.test", path="/upload")
    writer, start = client.begin_stream(head)
    response_right = None
    try:
        await send_part(start)
        async for chunk in source:
            offset = 0
            while offset < len(chunk):
                used, record = writer.push(chunk[offset:])
                if used == 0 and record is None:
                    raise RuntimeError("request writer made no progress")
                offset += used
                if record is not None:
                    await send_part(record)
        end, response_right = writer.finish()
        await send_part(end)
        await end_outer_body()
    except BaseException:
        if response_right is not None:
            response_right.close()
        raise
    finally:
        writer.close()
```

Use `response_right` to check one reply, then close it. The
[protocol terms](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#terms-used-below)
describe the records and response rights used by this API.

For the server side of a custom transport:

1. Feed the request to `OpenedStreamRequest.feed()` and store the checked DATA parts.
2. Wait for END and the actual end of the HTTP body, then call `finish_eof()`.
3. Check the app's host, port, and path before calling it with the stored body
   and `OpenedStreamRequest.head`.
4. Use the returned `StreamResponseRight` to encrypt one reply. It holds no
   request body.

### Runtime support

The package supports CPython 3.10 through 3.14. Release wheels target Linux
x86-64 and AArch64 with glibc 2.28 or newer (`manylinux_2_28`), and macOS
universal2. The [wheel platform tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/#manylinux)
define Linux compatibility. Other Linux and macOS CPython targets need a Rust
source build. Windows and PyPy are not supported.

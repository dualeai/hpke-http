# hpke-http for Python

`hpke_http` protects HTTP requests and replies with the Rust `hpke-http/3`
engine. Use the HTTPX or aiohttp client with an ASGI app.

## Build this checkout

Build this checkout with the `hpke-http/3` protocol:

```sh
make install-deps-python build-python
```

This installs the HTTPX, aiohttp, and FastAPI extras and builds the Rust binding
from this checkout.

## Protect an ASGI app

Wrap the app at one HTTPS path:

```python
from hpke_http.middleware.fastapi import HPKEMiddleware

protected_app = HPKEMiddleware(
    app,
    recipient_private_key,
    key_id,
    resolve_psk,
    admit_replay,
    transport_path="/protected",
)
```

`resolve_psk(psk_id, scope)` returns the PSK for a public ID. It raises
`LookupError` if the ID is unknown. `admit_replay(replay_id, deadline, scope)`
atomically stores a new replay ID until the given exclusive Unix second. It
returns `True` only for the first use. Both callbacks can be async.

The wrapper serves the public key at `GET /protected`. It accepts protected
requests at `POST /protected`. The `transport_path` must match the full ASGI
path. Set `expected_authority` if the logical host differs from the outer
`Host` header. A lifespan shutdown closes the native server; call
`protected_app.close()` at shutdown if the host sends no lifespan events.

The app receives its usual HTTP request fields and body. The wrapper checks
START, every DATA part, END, and the outer body end before it starts the app.
It keeps up to 256 KiB of checked body bytes in memory, then uses a temporary
file. The app can send a finite reply or server-sent events (SSE). For SSE, send
status 200 and `Content-Type: text/event-stream`. End each event with a blank
line; the wrapper drops an incomplete last block.

For a browser page on another origin, put CORS outside the HPKE wrapper so it
can answer the browser's OPTIONS request and add headers to fault replies:

```python
from starlette.middleware.cors import CORSMiddleware

browser_app = CORSMiddleware(
    protected_app,
    allow_origins=["https://app.example.test"],
    allow_methods=["GET", "POST"],
    allow_headers=["cache-control", "content-type"],
)
```

## Send requests with HTTPX

```python
from hpke_http.middleware import Discover
from hpke_http.middleware.httpx import HPKEAsyncClient

async with HPKEAsyncClient(
    "https://api.example.test/protected",
    Discover(),
    psk,
    b"tenant-42",
) as client:
    response = await client.post(
        "https://api.example.test/items",
        json={"name": "Ada"},
    )
    response.raise_for_status()

    with open("large.bin", "rb") as source:
        response = await client.post(
            "https://api.example.test/upload",
            files={"upload": source},
        )
```

`request()` returns a fully checked `httpx.Response`. Normal HTTPX `content`,
`data`, `json`, and `files` arguments work. The Rust writer reads source bytes
as the HTTP library sends them. It groups bytes into request DATA parts of at
most 64 KiB. You can also pass an async byte source as `content`.

For an SSE reply, use `stream()`:

```python
async with client.stream("GET", "https://api.example.test/events") as response:
    async for block in response.iter_sse():
        handle_sse_block(block)
```

`iter_sse()` yields each checked, LF-ended byte block. Decode and parse blocks
with the [SSE rules](https://html.spec.whatwg.org/multipage/server-sent-events.html#interpreting-an-event-stream).
For a finite reply in a stream context, call `await response.read()`.

## Send requests with aiohttp

```python
import aiohttp
from hpke_http.middleware import Discover
from hpke_http.middleware.aiohttp import HPKEClientSession

form = aiohttp.FormData()
with open("large.bin", "rb") as source:
    form.add_field("upload", source, filename="large.bin")
    async with HPKEClientSession(endpoint, Discover(), psk, psk_id) as session:
        async with session.post(logical_url, data=form) as response:
            result = await response.read()
```

`HPKEClientSession` accepts bytes, text, form mappings, `FormData`, async byte
sources, and JSON. It keeps the ordinary aiohttp request shape. The finite
`HPKEResponse` has `status`, `headers`, `read()`, `text()`, `json()`, and
`raise_for_status()`. Use `session.stream()` and `iter_sse()` for SSE.

## Technical details

### Keys and transport

`Discover()` gets the current public key before each call. Use
`PinnedKey(recipient_public_key, key_id)` to use a fixed key without a GET.
Both clients send the protected POST to their configured endpoint. Set
`target_origin="https://api.example.test"` if a gateway serves a different
logical host. The clients require HTTPS, check the logical target, and do not
follow outer redirects. They do not retry a protected POST.

Generate a recipient key pair with `generate_key_pair()`. Recipient keys use
X25519 and have 32 bytes. Key and PSK IDs are public opaque values of 1 to 255
bytes. A PSK needs at least 32 bytes of entropy. A Python `bytes` object
cannot be erased in place; keep secret copies to a minimum.

### Limits and payload coding

| Limit | Default | Hard maximum |
| --- | ---: | ---: |
| Request clear bytes | 1 GiB | 4 GiB |
| Finite reply body or one SSE block | 8 MiB | 64 MiB |
| Request DATA part | 64 KiB | 64 KiB |
| Header name and value bytes | 16 KiB | 64 KiB |
| Header fields | 64 | 256 |
| Authority and path bytes | 8 KiB | 8 KiB |

Set `Limits(max_request_bytes=...)` on both client and server for a service
request cap. The HTTPX and aiohttp clients use this stream limit for every
request. The low-level `Client.protect(Request(...))` helper holds a full body
in memory and uses `max_body_len` (8 MiB by default) as its cap.
`max_body_len` also sets the finite reply and per-SSE-block cap. A host must
also set limits for concurrent uploads, upload time, and temporary disk. An
ASGI host can deliver one outer event larger than 64 KiB, so that event can
use more memory.

Rust selects raw bytes or zstd for each request DATA part, finite reply body,
and SSE block, then encrypts it. The recipient checks each tag before it
decodes the part. This coding is part of the protected payload; it is not HTTP
`Content-Encoding`.

The protocol does not hide payload size, record count, or timing, and it adds
no random padding. HTTPS remains required. The request media type is
`message/hpke-http-request`, and the reply media type is
`message/hpke-http-response`.

### Low-level streamed requests

The HTTPX and aiohttp clients send records on their own. For a custom HTTP
transport, use the low-level writer. `send_part` writes one outer POST body part,
`end_outer_body` closes that body, and `source` supplies byte chunks:

```python
from hpke_http import Method, RequestHead

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

Use `response_right` to check the reply, then close it.

`OpenedStreamRequest.feed()` yields checked DATA parts. Store those parts until
END and the outer body end pass. Then `finish_eof()` returns a
`StreamResponseRight`, which can protect one reply. It has no `request` body;
use the checked parts and `OpenedStreamRequest.head` to dispatch the request.

### Runtime support

The package supports CPython 3.10 through 3.14. Wheels target Linux x86-64
and AArch64 and macOS universal2. Other Linux and macOS CPython targets need
a Rust source build. Windows and PyPy are not supported.

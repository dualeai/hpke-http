# hpke-http for Python

`hpke_http` protects HTTP requests and replies with the Rust `hpke-http/3`
engine. Use the HTTPX or aiohttp client with an ASGI app. The
[protocol specification](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md)
holds the shared wire bytes, key record, checks, and host steps.

## Install the published package

Install the v4 release with the HTTPX, aiohttp, and FastAPI adapters used below:

```sh
python -m pip install 'hpke-http[httpx,aiohttp,fastapi]~=4.0'
```

The v4 clients do not read HHKD v1 key records. Change clients and hosts
together, or use separate endpoints.

## Protect an ASGI app

Wrap the app at one HTTPS path. Your app supplies `app`,
`recipient_private_key`, `key_id`, and `lease_seconds`. It also implements
`resolve_psk` and `admit_replay`:

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

`resolve_psk(psk_id, scope)` returns the PSK for a public ID. It raises
`LookupError` if the ID is unknown. `admit_replay(replay_id, deadline, scope)`
atomically stores a new replay ID until the given exclusive Unix second. It
returns `True` only for the first use. Both callbacks can be async.

The wrapper serves the public key at `GET /protected`. It accepts protected
requests at `POST /protected`. The `transport_path` must match the full ASGI
path. Set `expected_authority` if the logical host differs from the outer
`Host` header. A lifespan shutdown closes the native server; call
`protected_app.close()` at shutdown if the host sends no lifespan events.
The service sets `lease_seconds` and a bound to deliver and parse POST START.
For a planned key change, follow the
[key switch order](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record).
Set `accepted_keys` to `(private_key, key_id)` pairs for other accepted keys.

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
    expose_headers=["content-encoding"],
)
```

The browser must read outer `Content-Encoding` on GET and POST replies. If a
proxy adds this header, CORS must expose it so the client can check it.

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

The first call sends a key GET and a protected POST. The second sends only a
POST while the lease is valid. Keep the source open across client lifetimes.

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

`HPKEClientSession` accepts bytes, text, form mappings, `FormData`, async byte
sources, and JSON. It keeps the ordinary aiohttp request shape. The finite
`HPKEResponse` has `status`, `headers`, `read()`, `text()`, `json()`, and
`raise_for_status()`. Use `session.stream()` and `iter_sse()` for SSE.

## Technical details

### Keys and transport

`DiscoveredEndpoint(endpoint)` shares one checked key and one GET among
callers. Keep a distinct source for each full endpoint URL, including Bridge
and Library paths. The [key record and lease rules](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record)
are in the central spec.

A source owns its outer HTTP pool and endpoint URL; short-lived credential
clients borrow it.
Set HTTPX TLS and pool options on the HTTPX source, and aiohttp connector and
session options on the aiohttp source. HTTPX client default headers, params,
and timeout apply to its logical requests. Each Python source belongs to one
event loop. If one caller cancels its wait for a shared GET, the GET continues
for other callers. Close each client before you close the source.

`get_timeout_s` defaults to 10 seconds and limits one key GET. It does not
limit the protected POST. Set an HTTPX client or request `timeout` for POST I/O.
With aiohttp, set a source session or request `timeout` for POST. To limit the
full GET and POST call, set an app deadline around the call.

`get_key()` returns a public `hpke_http.middleware.KeyLease` with `key_id`,
`public_key`, and `valid()`. Check `valid()` when you use the key, since the
lease can end after `get_key()` returns.

Use `PinnedKey(recipient_public_key, key_id)` to use a fixed key without a GET.
Pass `endpoint=` with a pin. A shared source supplies its own endpoint URL.
Both clients send the protected POST to that URL. Set
`target_origin="https://api.example.test"` if a gateway serves a different
logical host. The clients require HTTPS, check the logical target, and do not
follow outer redirects. They do not retry a protected POST. The GET has no
bearer, cookies, or PSK ID. Business headers stay in the protected request. A
lost reply leaves the request result unknown.

Generate a recipient key pair with `generate_key_pair()`. Follow the
[credential rules](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#request-hpke-steps)
when you provision keys and PSKs. A Python `bytes` object cannot be erased
in place; keep secret copies to a minimum.

### Transport errors

`hpke_http.TransportError.code` tells callers why an adapter could not finish
a protected call:

| Discovery code | Meaning |
| --- | --- |
| `discovery_network` | GET failed or timed out. |
| `discovery_status` | GET returned a status other than 200. |
| `discovery_response` | GET headers or key record were invalid. |
| `discovery_expired` | The lease ended before POST START. |

`status_code` is set for `discovery_status` and `outer_status`. Other outer
transport failures also raise `TransportError`; authenticated logical HTTP
errors return a normal response. Use `response.raise_for_status()` to raise on
those logical errors. A failed POST is not retried.

### Limits and payload coding

The [central limits and DATA rules](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#current-engine-limits)
give each default and hard cap, and explain raw and zstd record coding.
Set `Limits(max_request_bytes=...)` on both client and server for a service
request cap. The HTTPX and aiohttp clients use this stream limit for every
request. The low-level `Client.protect(Request(...))` helper holds a full body
in memory and uses `max_body_len` as its cap. That limit also sets the finite
reply and per-SSE-block cap.

A host must set limits for concurrent uploads, upload time, and temporary
disk. An ASGI host can deliver one outer event larger than 64 KiB, so that
event can use more memory.

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

Use `response_right` to check the reply, then close it.

`OpenedStreamRequest.feed()` yields checked DATA parts. Store those parts until
END and the outer body end pass. Then `finish_eof()` returns a
`StreamResponseRight`, which can protect one reply. It has no `request` body;
check the logical target policy before you use the stored parts and
`OpenedStreamRequest.head` to dispatch the request.

### Runtime support

The package supports CPython 3.10 through 3.14. Wheels target Linux x86-64
and AArch64 and macOS universal2. Other Linux and macOS CPython targets need
a Rust source build. Windows and PyPy are not supported.

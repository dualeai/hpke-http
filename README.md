# hpke-http

**Keep app requests and replies encrypted through your HTTPS proxies.**

`hpke-http` adds Hybrid Public Key Encryption (HPKE) inside HTTPS. It encrypts
the app method, path, query, headers, and body, plus the reply status, headers,
and body. A CDN or reverse proxy can route the traffic without reading those
fields, provided it has no HPKE secrets.
Use it when you control both the client and the service.

## What you get

- Send JSON, forms, files, and byte streams through one HTTPS endpoint.
- Reject altered or incomplete requests before your app runs.
- Read complete replies after all checks pass.
- Receive live server-sent events (SSE), one checked block at a time.
- Reject repeated encrypted requests with a shared replay store that you supply.
- Discover and reuse public keys, or pin a key to skip discovery.

| Language | Client | Server |
| --- | --- | --- |
| [Python](https://github.com/dualeai/hpke-http/blob/main/python/README.md) | HTTPX and aiohttp adapters | ASGI middleware for Starlette and FastAPI |
| [TypeScript](https://github.com/dualeai/hpke-http/blob/main/typescript/README.md) | Fetch for Node.js and browsers | Low-level engine; supply your HTTP host |
| [Rust](https://docs.rs/hpke-http/4.0.0/hpke_http/) | Low-level engine; supply your HTTP client | Low-level engine; supply your HTTP host |

## One endpoint, private app routes

The **app URL** names the route you want to call. The **outer endpoint**
is the HTTPS URL that carries the encrypted request:

```text
App call:         GET /users/42?full=true
HTTPS transport:  POST /protected
Encrypted inside: GET /users/42?full=true, app headers, and body
```

You choose the endpoint path. `/protected` and `/v1/hpke` are examples.
Every protected app call uses POST at that endpoint, including app GET and DELETE calls.
Key discovery uses GET at the same URL; browsers can also send OPTIONS.

The client returns the decrypted app status. For example, an app 404 arrives
inside a protected reply whose outer HTTP status is 200. Transport and
protocol errors can use ordinary HTTP error responses.

## What observers can see

This table assumes HTTPS remains secure. The proxy has no HPKE secrets.
Neither observer controls the client or the service that decrypts the request.

| Data | Proxy that terminates HTTPS | Passive network observer |
| --- | --- | --- |
| App method, path, query, headers, and body | Encrypted | Encrypted |
| App reply status, headers, and body | Encrypted | Encrypted |
| Outer HTTP path, method, headers, and status | Visible | Encrypted by HTTPS |
| Public recipient key ID, public pre-shared key (PSK) ID, and request timestamp | Visible | Encrypted by HTTPS |
| Public protocol fields and key discovery records | Visible | Encrypted by HTTPS |
| HPKE record lengths and count | Visible | Inside HTTPS encryption |
| Connection IP addresses, byte volume, and timing | Visible | Visible |

DNS or TLS settings can also reveal the hostname. Browsers can add outer
`Origin`, `User-Agent`, and request-context headers. The Fetch adapter omits
outer credentials and the referrer. A stable public PSK ID can link calls.

There is no padding. Size and timing can help an observer guess an operation.
See [protection limits](#protection-limits) for compression and secret handling.

## Use the library

Supply the server's encryption key pair, a PSK, its public ID, and a shared
atomic replay store. The PSK needs at least 32 bytes of entropy and must differ
from its public ID. Store it securely and give it only to clients allowed to
call your service. Key discovery supplies only the server's public key.

### Send a request with Python

```sh
python -m pip install 'hpke-http[httpx]~=4.0'
```

To run both client and server locally, follow the [Python quick start](https://github.com/dualeai/hpke-http/blob/main/python/README.md#install-the-published-package),
including its server extras. To call an existing service, use the
[HTTPX](https://github.com/dualeai/hpke-http/blob/main/python/README.md#send-requests-with-httpx)
or [aiohttp](https://github.com/dualeai/hpke-http/blob/main/python/README.md#send-requests-with-aiohttp) examples.

### Serve protected requests with Python

Install `hpke-http[fastapi]~=4.0`, then
[wrap your ASGI app](https://github.com/dualeai/hpke-http/blob/main/python/README.md#protect-an-asgi-app).
The Python guide covers [server dependencies](https://github.com/dualeai/hpke-http/blob/main/python/README.md#install-the-published-package),
PSK lookup, replay, and CORS.

### Use Fetch in TypeScript

```sh
npm install '@dualeai/hpke-http@^4.0.0'
```

Use the [first Fetch call](https://github.com/dualeai/hpke-http/blob/main/typescript/README.md#send-a-request-and-read-its-reply),
then the [upload](https://github.com/dualeai/hpke-http/blob/main/typescript/README.md#upload-a-file-or-body-stream)
or [SSE](https://github.com/dualeai/hpke-http/blob/main/typescript/README.md#read-an-sse-reply) example.
The guide covers Node, browsers, WebAssembly setup, and request deadlines.

### Use the Rust crate

```sh
cargo add hpke-http@4.0.0
```

Use `Client` and `Server` with your HTTP stack. The
[crate docs](https://docs.rs/hpke-http/4.0.0/hpke_http/) show complete and streamed calls.

### Supported HTTP behavior

- Methods: GET, POST, PUT, PATCH, DELETE, HEAD, and OPTIONS.
- Replies: complete bodies (called **finite replies**) or live SSE. WebSockets,
  CONNECT tunnels, and other live reply formats are not supported.
- Uploads: the ASGI app runs only after the whole request passes its checks.
- Default limits: 1 GiB per streamed request; 8 MiB per finite reply or SSE block.
  SSE has no total body cap. See the [limit table](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#current-engine-limits).

The adapters do not follow app redirects or manage login sessions. Send app
authorization in protected headers.

### Retry safety

The adapters do not retry protected POSTs. After a lost reply, the app may
have run. A new call creates a new replay ID, so replay checks cannot prevent
the operation from running again. Before retrying, make repeated calls for the
same operation have the same effect as one call (**idempotency**).

## Build and runtime support

| Package | Supported runtime |
| --- | --- |
| Rust | Rust 1.87 or newer |
| Python | CPython 3.10–3.14; see [platform and wheel support](https://github.com/dualeai/hpke-http/blob/main/python/README.md#runtime-support) |
| TypeScript | Node.js 24; see [browser requirements and entry points](https://github.com/dualeai/hpke-http/blob/main/typescript/README.md#build-and-runtime-support) |

The v4 packages use HHKD v2 discovery. Check
[discovery compatibility](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record)
before changing clients or hosts.
Use [published packages](https://github.com/dualeai/hpke-http/releases) or
[build this checkout](#development).

## Technical details

### Protocol specification

The [specification](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md)
defines the exact message bytes, discovery format, checks, limits, and test vectors.
Its [terms](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#terms-used-below)
explain the record names used in the host rules below.

### Host duties

- Look up PSKs and reserve replay IDs across all workers that accept the same
  credentials. Each reservation must succeed for only one caller, even when
  calls arrive together. Keep the ID until its supplied deadline.
- Check the full request, its END record, and the actual end of the HTTP body (EOF).
  Check the app's host, port, and path before you call the app.
- Enforce HTTPS and app route permissions. The Python wrapper passes other outer
  paths to the app; block those calls if the routes must require HPKE.
- Use distinct recipient keys for separate services. Cryptographic checks do not
  bind the outer endpoint URL; shared credentials need explicit routing rules.
- Keep outer GET and POST replies uncached, preserve their content types, and
  disable HTTP compression. For live SSE, disable proxy buffering and set idle timeouts.
- Bound concurrent uploads, time, and temporary disk use. See the
  [Python host's storage limits](https://github.com/dualeai/hpke-http/blob/main/python/README.md#limits-and-payload-coding).

Proxies cannot enforce rules that need encrypted app paths, headers, or bodies.
Check those rules after decryption. For recipient key changes, follow the
[worker and lease order](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record).

### Protection limits

The writer compresses body data with zstd when that reduces its size. If a
body contains both a secret and text chosen by an attacker, its encrypted
size can help the attacker test guesses about the secret. Keep these inputs
separate or add defenses in your app. The API has no option to disable
compression. Request headers use a separate, uncompressed record.

The client and decrypting host hold clear data and secrets. Keep credentials,
decrypted URLs, and bodies out of logs; protect temporary upload storage.
Saved HPKE messages can be decrypted if both the recipient private key and
matching PSK are later exposed.

### Native engine

Rust owns the wire format, cryptography, replay identity, and record checks.
It does no network I/O. Python uses PyO3; TypeScript uses WebAssembly.
There is no pure-Python or pure-TypeScript cryptographic fallback.

### Repository layout

```text
rust/hpke-http/       protocol engine and executable API examples
python/              Python API, PyO3 binding, and HTTP adapters
typescript/          TypeScript API, WASM binding, and Fetch adapter
cicd/                coordinated release tooling
```

The language adapters own HTTP I/O and runtime cleanup. Generated binding APIs are private.

## Development

Install Rust with `rustup`, then use its `cargo` and `rustc` for the WASM target.
Put the Rust toolchain ahead of any Rust tools from another package manager.
Install `uv` and Node.js 24 before you run these commands:

```sh
rustup update stable
export PATH="$(dirname "$(rustup which cargo --toolchain stable)"):$PATH"
rustup target add --toolchain stable wasm32-unknown-unknown
make install
make install-wasm-bindgen
make build
make test
```

Use `make build-rust`, `make build-python`, or `make build-typescript` to build
one language. Use `make test-rust`, `make test-python`, or `make test-typescript`
to check one language. `make package` writes the Rust crate to `target/package/`,
Python packages to `artifacts/python/`, and the npm archive to `artifacts/npm/`.
Use the matching `smoke-python-wheel`, `smoke-python-sdist`, or
`smoke-typescript` target to check a package. Local Python targets can update
`uv.lock` when dependencies change; CI uses the locked versions.

`wasm-bindgen-cli` 0.2.128 and the Rust `wasm32-unknown-unknown` target are
required for the TypeScript build. Installing the pinned CLI from source needs
Rust 1.88 or newer. Use the current stable Rust release to build all bindings.
The TypeScript runtime test needs Chrome or Chromium. Set `CHROME_BIN` if the
browser is not in a standard path.

## Security

See the [security policy](https://github.com/dualeai/hpke-http/blob/main/SECURITY.md)
for supported releases and private reporting.

## License

The library uses the [Apache 2.0 license](https://github.com/dualeai/hpke-http/blob/main/LICENSE).

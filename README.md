# hpke-http

`hpke-http` protects HTTP requests and replies. Send them with Python HTTPX,
aiohttp, or TypeScript Fetch. Serve them with a Python ASGI app. The shared
Rust engine checks each payload.

The published v4 packages include the shared key source and HHKD v2
discovery used below. The [protocol specification](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md)
gives the exact bytes, checks, and client and server steps.
The Python and TypeScript v4 clients do not read HHKD v1 key records. Change
clients and hosts together, or use separate endpoints.

## Use the library

### Send a request with Python

Set `endpoint`, `psk`, and `psk_id` from your app config. Keep one key source
open for each full protected endpoint path while your app sends calls. Use
separate sources for Bridge and Library. Short-lived clients can share a source.

```python
from hpke_http.middleware.httpx import DiscoveredEndpoint, HPKEAsyncClient

async with DiscoveredEndpoint(endpoint) as key_source:
    async with HPKEAsyncClient(key_source, psk, psk_id) as client:
        response = await client.post("https://api.example.test/items", json={"name": "Ada"})
        print(response.status_code, response.json())
    async with HPKEAsyncClient(key_source, psk, psk_id) as client:
        response = await client.post("https://api.example.test/items", json={"name": "Grace"})
        print(response.status_code, response.json())
```

The first call sends GET then POST. The second sends POST alone while the key
lease is valid. The client also accepts normal HTTPX file input; see the
[Python file example](https://github.com/dualeai/hpke-http/blob/main/python/README.md#send-requests-with-httpx).

### Serve protected requests with Python

Wrap a FastAPI or Starlette app. Supply the app, recipient private key, public
key ID, and key lease. The app also supplies PSK lookup and an atomic replay
check shared by all server workers.

```python
from hpke_http.middleware.fastapi import HPKEMiddleware

protected_app = HPKEMiddleware(
    app, private_key, key_id, resolve_psk, admit_replay,
    key_use_for_s=lease_seconds,
    transport_path="/protected",
)
```

The host checks the full request before it calls the app. The app reads files
and other bodies through its normal request API. See the [Python guide](https://github.com/dualeai/hpke-http/blob/main/python/README.md)
for aiohttp, forms, SSE, and cross-origin browser setup.

### Use Fetch in TypeScript

Set `psk` and `pskId` as `Uint8Array` values from your app config. The PSK
needs at least 32 bytes of entropy. Its public ID must differ from the PSK.

```ts
import { DiscoveredEndpoint, createHpkeFetch, initialize } from "@dualeai/hpke-http/browser";

await initialize();
const keySource = new DiscoveredEndpoint("https://api.example.test/protected");
const hpkeFetch = createHpkeFetch({
  key: keySource,
  psk,
  pskId,
});

try {
  const response = await hpkeFetch("https://api.example.test/items", {
    method: "POST",
    body: "payload",
  });
  console.log(response.status);
  const next = await hpkeFetch("https://api.example.test/items", {
    method: "POST",
    body: "another payload",
  });
  console.log(next.status);
} finally {
  hpkeFetch.close();
  keySource.close();
}
```

The first call sends GET then POST. The second sends POST alone while the key
lease is valid. Other clients can use `keySource` while it stays open. Use a
different source for Library's endpoint path.

Fetch keeps its normal body types. Large uploads need a runtime that can send
a request stream; browser support and HTTP/1.x routes can limit that path. A
small buffered outer POST uses the same protected wire. See the
[TypeScript guide](https://github.com/dualeai/hpke-http/blob/main/typescript/README.md) for Node, browser, and SSE calls.

### Use the Rust crate

Add the published crate:

```sh
cargo add hpke-http@4.0.0
```

Use `Client` and `Server` with your HTTP stack. The crate does no network I/O.
The [crate docs](https://docs.rs/hpke-http/4.0.0/hpke_http/) show complete and
streamed transactions.

## Build and runtime support

Use [published releases](https://github.com/dualeai/hpke-http/releases)
for package builds, or use [Development](#development) to build this checkout.

The Rust crate requires Rust 1.87 or newer. Python supports CPython 3.10 through
3.14; release wheels target Linux x86-64 and AArch64 and macOS universal2.
Windows Python builds are not supported. The npm package supports Node.js 24
and browsers with WebAssembly, Fetch, Web Crypto, and Web Streams. It has
explicit `./node` and `./browser` exports and no root export.

## Technical details

### Protocol specification

The [central specification](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md)
defines the hpke-http/3 request and response bytes, the HHKD v2 key record,
the HTTPS exchange, checks, limits, and pseudocode. It links the frozen
test vectors. Use it when you make another implementation or inspect the
wire format. The Rust crate docs show executable API transactions.

A client gets one key by HTTPS GET or uses a pinned key, then sends one
protected HTTPS POST to the same endpoint. The inner method, target, fields,
body, and response status are checked protected data. The endpoint, public
IDs, size, record count, and timing remain visible; the format adds no
padding.

### Host duties

A server resolves a public PSK ID and makes one atomic replay decision
shared by all workers that accept the same credentials. It keeps clear
request data from the app until the full request, END, and outer body EOF
pass. It also checks its logical target policy before app dispatch. Each
request gives one response right. The client checks each SSE block before
it gives that block to the caller; it gives a finite reply
only after END and outer EOF. Use HTTPS and give each PSK at least 32 bytes
of entropy.

The Python ASGI host keeps up to 256 KiB of checked request data in memory,
then uses a temporary file. An incoming ASGI event can use more memory.
Set service limits for concurrent uploads, temporary disk, and time, as
well as the protocol byte limit. For a planned recipient key change, follow
the worker and lease steps in the
[specification](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record).

### Native engine

The protocol identifier is hpke-http/3. The Rust crate owns the wire
format, HPKE state, replay identity, limits, and response record checks.
It does no network I/O. Python calls it through PyO3; TypeScript calls it
through WebAssembly. There is no pure-Python or pure-TypeScript
cryptographic fallback.

### Repository layout

```text
rust/hpke-http/       safe protocol engine and executable API examples
python/               Python API, PyO3 boundary, and HTTP framework adapters
typescript/           TypeScript API, WASM boundary, and native Fetch adapter
cicd/                 tag-derived coordinated release tooling
```

The language projects own key GET and record handling, runtime lifecycle,
and HTTP integrations. Generated binding APIs are private.

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
to check one language. `make package` writes Rust, Python, and npm packages to
`artifacts/`. Use the matching `smoke-python-wheel`, `smoke-python-sdist`, or
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
Give each PSK at least 32 bytes of entropy. Do not use it as its public PSK ID.
Use an atomic replay store shared by all
server workers that can process the same credentials.

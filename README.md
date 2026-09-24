# hpke-http

`hpke-http` protects HTTP requests and replies. Send them with Python HTTPX,
aiohttp, or TypeScript Fetch. Serve them with a Python ASGI app. The shared
Rust engine checks each payload.

## Use the library

### Send a request with Python

Set `endpoint`, `psk`, and `psk_id` from your app config. `Discover()` gets the
server's public key from the HTTPS endpoint.

```python
from hpke_http.middleware import Discover
from hpke_http.middleware.httpx import HPKEAsyncClient

async with HPKEAsyncClient(endpoint, Discover(), psk, psk_id) as client:
    response = await client.post("https://api.example.test/items", json={"name": "Ada"})
    print(response.status_code, response.json())
```

The same call takes normal HTTPX file input. The library sends the file through
checked request records without a mode setting.

```python
async with HPKEAsyncClient(endpoint, Discover(), psk, psk_id) as client:
    with open("report.bin", "rb") as source:
        response = await client.post(
            "https://api.example.test/uploads", files={"file": source}
        )
```

### Serve protected requests with Python

Wrap a FastAPI or Starlette app. The app supplies PSK lookup and an atomic
replay check shared by all server workers.

```python
from hpke_http.middleware.fastapi import HPKEMiddleware

protected_app = HPKEMiddleware(
    app, private_key, key_id, resolve_psk, admit_replay,
    transport_path="/protected",
)
```

The host checks the full request before it calls the app. The app reads files
and other bodies through its normal request API. See the [Python guide](python/README.md)
for aiohttp, forms, SSE, and cross-origin browser setup.

### Use Fetch in TypeScript

```ts
import { createHpkeFetch, initialize } from "@dualeai/hpke-http/browser";

await initialize();
const hpkeFetch = createHpkeFetch({
  endpoint: "https://api.example.test/protected",
  key: { kind: "discover" },
  psk,
  pskId,
});

try {
  const response = await hpkeFetch("https://api.example.test/items", {
    method: "POST",
    body: "payload",
  });
  console.log(response.status);
} finally {
  hpkeFetch.close();
}
```

Fetch keeps its normal body types. Large uploads need a runtime that can send
a request stream; browser support and HTTP/1.x routes can limit that path. A
small buffered outer POST uses the same protected wire. See the
[TypeScript guide](typescript/README.md) for Node, browser, and SSE calls.

### Use the Rust crate

Use `Client` and `Server` from `hpke-http` with your HTTP stack. The
[crate docs](rust/hpke-http/src/lib.rs) show complete and streamed transactions.

## Build and runtime support

Build this checkout with the commands in [Development](#development) to use
`hpke-http/3`. [Published releases](https://github.com/dualeai/hpke-http/releases)
link to their package builds.

The Rust crate requires Rust 1.87 or newer. Python supports CPython 3.10 through
3.14; release wheels target Linux x86-64 and AArch64 and macOS universal2.
Windows Python builds are not supported. The npm package supports Node.js 24
and browsers with WebAssembly, Fetch, Web Crypto, and Web Streams. It has
explicit `./node` and `./browser` exports and no root export.

## Technical details

### Request and response checks

Protocol version 3 uses:

- RFC 9180 HPKE PSK mode with X25519, HKDF-SHA256, and ChaCha20-Poly1305;
- one checked START/DATA/END request form for every body size;
- a project-specific request envelope and request-bound response keys; and
- one checked response START, a checked DATA record per SSE block, and a checked END.

HTTPS is required. The protocol encrypts the payload. It does not hide payload
size, record count, or timing, and it adds no random padding.

Servers resolve a public PSK ID and then make one atomic replay-admission
decision. The authenticated plaintext remains inside the engine until that
decision succeeds. The Python ASGI host checks every request DATA record, END,
and the outer body end before it calls the app. Each request can create and
open only one protected response. A client uses the START fields only after
its tag passes. It gets each SSE block after that block's tag passes, before
the server sends END. A finite response becomes complete only after END and
the real outer body EOF.

The Python ASGI spool keeps up to 256 KiB of checked request data in memory,
then uses a temporary file. An incoming ASGI event can use more memory. Each
request has a byte limit. A service must also set its own concurrent upload,
temporary disk, and request time limits.

### Limits and payload coding

The engine accepts request parts, finite replies, and live SSE blocks. The
default streamed request body limit is 1 GiB, with a 4 GiB hard cap. The
low-level one-shot request helper uses the 8 MiB body limit. Each request DATA
record carries at most 64 KiB of clear data. The default finite reply or SSE
block limit is 8 MiB; a live SSE stream has no total body cap. Rust uses zstd
for parts of at least 64 bytes when it saves bytes. Shorter parts and parts
that do not shrink stay raw. Rust then encrypts the chosen form and checks
decoded limits on receive. The outer HTTP `Content-Encoding` stays identity.
Python and TypeScript do not run a second compressor. The engine does no
network I/O. The Python and Fetch adapters can discover one key. There is no
suite or codec setting.

Recipient keys use X25519. Public recipient-key and PSK identifiers are
non-empty and at most 255 bytes; PSKs must be at least 32 bytes long, contain
at least 32 bytes of entropy, and differ from their public IDs. Native objects
copy credentials. Closing them releases native copies, but it cannot erase
caller-owned Python `bytes` or JavaScript `Uint8Array` values.

### Key discovery

One configured HTTPS endpoint serves the public key through GET and receives
protected requests through POST. A client gets the key before each discovered
call. The GET sends no logical authorization, cookies, or PSK ID. The client
accepts no redirect and stores no app key cache. A pinned key uses the same
POST endpoint and sends no GET. The client accepts logical requests at one
fixed HTTPS origin; a gateway can set a different `target_origin` or
`targetOrigin` and the ASGI server's `expected_authority`.
Origin checks fold DNS case and IDNA names, normalize IPv6 and the default
port 443, and keep other ports distinct. A wrong logical origin fails before
the client reads its body or sends GET.

The GET body is exactly `"HHKD" || 0x01 || id_len:u8 || id || x25519_public_key[32]`.
The ID length is 1 through 255, so the full record is 39 through 293 bytes.
The server sends `application/octet-stream` and `Cache-Control: no-store`.
Clients reject any other record form or extra bytes. Version `0x01` describes
this key record; request and response bytes use `hpke-http/3`.

The server holds one recipient key. All workers for an endpoint must use that
key; a key change across mixed workers can make a request fail. The adapters
do not retry a protected POST. If its reply is lost, the caller does not know
whether the app ran. The added GET is one more network round trip on each
discovered call. For browser calls, outer CORS must cover GET, POST preflight,
POST, and fault replies.

### Native engine

The protocol identifier is `hpke-http/3`. It is the only protocol implemented
in this repository. There is no pure-Python or pure-TypeScript cryptographic
fallback.

The Rust crate owns the wire format, HPKE state, replay identity, limits, and
response record checks. It does no network I/O. Python calls it through PyO3;
TypeScript calls it through WebAssembly. The crate docs give the exact bytes
and an executable transaction.

### Repository layout

```text
rust/hpke-http/       safe protocol engine and executable protocol contract
python/               Python API, PyO3 boundary, and HTTP framework adapters
typescript/           TypeScript API, WASM boundary, and native Fetch adapter
cicd/                 tag-derived coordinated release tooling
```

The language projects own key GET and record handling, runtime lifecycle, and
HTTP integrations. Generated binding APIs are private.

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

See the [security policy](SECURITY.md)
for supported releases and private reporting.
Give each PSK at least 32 bytes of entropy. Do not use it as its public PSK ID.
Use an atomic replay store shared by all
server workers that can process the same credentials.

The core cryptographic standard is [RFC 9180](https://www.rfc-editor.org/rfc/rfc9180.html).
The payload coding is [RFC 8878 zstd](https://www.rfc-editor.org/rfc/rfc8878.html).
Response-key
derivation follows the pattern in
[RFC 9458 section 4.4](https://www.rfc-editor.org/rfc/rfc9458.html#section-4.4),
but `hpke-http/3` is not an Oblivious HTTP profile.

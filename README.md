# hpke-http

`hpke-http` protects HTTP requests and replies. Send them with Python HTTPX,
aiohttp, or TypeScript Fetch. Serve them with a Python ASGI app. The shared
Rust engine checks each payload.

This guide describes the current checkout. Its shared key source and HHKD v2
discovery record are not in the published 3.0.0 packages. Build this checkout
before you use the examples below; see
[Build and runtime support](#build-and-runtime-support).

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
[Python file example](python/README.md#send-requests-with-httpx).

### Serve protected requests with Python

Wrap a FastAPI or Starlette app. The app supplies PSK lookup and an atomic
replay check shared by all server workers.

```python
from hpke_http.middleware.fastapi import HPKEMiddleware

protected_app = HPKEMiddleware(
    app, private_key, key_id, resolve_psk, admit_replay,
    key_use_for_s=lease_seconds,
    transport_path="/protected",
)
```

The host checks the full request before it calls the app. The app reads files
and other bodies through its normal request API. See the [Python guide](python/README.md)
for aiohttp, forms, SSE, and cross-origin browser setup.

### Use Fetch in TypeScript

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
[TypeScript guide](typescript/README.md) for Node, browser, and SSE calls.

### Use the Rust crate

Use `Client` and `Server` from `hpke-http` with your HTTP stack. The
[crate docs](rust/hpke-http/src/lib.rs) show complete and streamed transactions.

## Build and runtime support

Build this checkout with the commands in [Development](#development) to use
the shared key source and HHKD v2 record shown here.
[Published releases](https://github.com/dualeai/hpke-http/releases) link to
their package builds.

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
protected requests through POST. A shared source gets the key once and uses it
until its lease ends. The client starts the lease clock before GET, so a slow
GET leaves less time for POST. The client checks the lease again before it
sends POST START. It can get a new key if it can still use the request body;
otherwise it reports `discovery_expired` before it yields POST START. The
service's POST START delivery bound covers time after the last lease check.
Concurrent first calls share one GET. Keep separate sources for Bridge and
Library, even when they have one HTTPS origin. The GET sends no logical
authorization, cookies, or PSK ID. The client accepts no
redirect. A pinned key uses the same POST endpoint and sends no GET. The client
accepts logical requests at one fixed HTTPS origin. For a gateway, set
`target_origin` or `targetOrigin` on the client and `expected_authority` on the
ASGI server. Origin checks fold DNS case and IDNA names, normalize IPv6 and
the default port 443, and keep other ports distinct. A wrong logical origin
fails before the client reads its body or sends GET.

The GET body is exactly `"HHKD" || 0x02 || id_len:u8 || id || x25519_public_key[32] || use_for_s:u32be`.
The ID length is 1 through 255 and `use_for_s` is a positive number of seconds.
The full record is 43 through 297 bytes. The service sets the lease.
The server sends `application/octet-stream` and `Cache-Control: no-store`.
Clients reject any other record form or extra bytes. Version `0x02` describes
this key record; request and response bytes still use `hpke-http/3`. There is
no v1 discovery fallback. A client that reads only HHKD v1 cannot use a v2
host, and a v2 client cannot use a v1 host. Switch clients and hosts together,
or use separate endpoints during the switch. HTTP `no-store` controls HTTP
caches; the explicit source keeps one checked key for its lease.

Each worker advertises one key and can accept other keys. For a planned switch
from A to B, use these worker states in order: advertise A and accept B;
advertise B and accept A; then advertise B alone. Complete each state on all
workers before the next state. Keep A accepted after its last advertisement
for its full last lease plus the bound to deliver and parse POST START and a
worker clock margin. Never reuse a KID while keys overlap. The service sets
the lease and POST START delivery bound. A failed POST stays failed; the
adapters do not resend it. If a reply is lost, the caller does not know whether
the app ran. An emergency key removal can cause calls to fail until their
leases end. For browser calls, outer CORS must cover GET, POST preflight, POST,
and fault replies, and expose `Content-Encoding` on GET and POST replies.

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

# hpke-http

`hpke-http` protects complete, bounded HTTP requests and responses with one
shared Rust protocol engine. Python calls that engine through PyO3. TypeScript
calls the same engine through WebAssembly and includes an adapter for the
runtime's native Fetch API.

The protocol identifier is `hpke-http/1`. It is the only protocol implemented
in this repository. There is no pure-Python or pure-TypeScript cryptographic
fallback.

## Repository layout

```text
rust/hpke-http/       safe protocol engine and executable protocol contract
python/               Python API, PyO3 boundary, and HTTP framework adapters
typescript/           TypeScript API, WASM boundary, and native Fetch adapter
cicd/                 tag-derived coordinated release tooling
```

The Rust crate owns byte encoding, HPKE operations, limits, validation, replay
identity, and one-shot response state. The language projects own their runtime
lifecycle and HTTP integrations. Generated binding APIs are private.

## Install and runtime support

```sh
cargo add hpke-http
python -m pip install hpke_http
npm install @dualeai/hpke-http
```

The Rust crate requires Rust 1.87 or newer. Python supports CPython 3.10 through
3.14; release wheels target Linux x86-64 and AArch64, macOS universal2, and
Windows x86-64. The npm package supports Node.js 24 and browsers with
WebAssembly, Fetch, Web Crypto, and Web Streams. It has explicit `./node` and
`./browser` exports and no root export.

## Protocol properties

Protocol version 1 uses:

- RFC 9180 HPKE PSK mode with X25519, HKDF-SHA256, and ChaCha20-Poly1305;
- canonical known-length Binary HTTP Messages from RFC 9292;
- a project-specific request envelope and request-bound response keys; and
- complete-message authentication before application plaintext is released.

The protocol is not wire-compatible with RFC 9458 Oblivious HTTP. HTTPS is
still required because endpoints, public key and PSK identifiers, message
sizes, and timing remain visible.

Servers resolve a public PSK ID and then make one atomic replay-admission
decision. The authenticated plaintext remains inside the engine until that
decision succeeds. Each request can create and open only one protected
response.

The engine accepts whole messages, with an 8 MiB default body limit and hard
resource ceilings. It supports opt-in gzip or zstd body compression in the Rust
protocol layer, with bounded decompression in every binding. This is separate
from HTTP `Content-Encoding`: high-level adapters still require identity
logical content coding. The engine does not implement incremental streaming,
discovery, or suite negotiation. Compression can leak body information through
ciphertext length; enable it only when attacker-controlled data cannot be
combined with secrets in the same body.

Recipient keys use X25519. Public recipient-key and PSK identifiers are
non-empty and at most 255 bytes; PSKs are at least 32 bytes and must not equal
their public IDs. Native objects copy credentials. Closing them releases native
copies, but it cannot erase caller-owned Python `bytes` or JavaScript
`Uint8Array` values.

## Rust

The `hpke-http` crate exposes the network-independent client/server state
machine. Its crate-level rustdoc contains an executable complete transaction,
the exact protocol bytes, replay contract, limits, and stable error categories.

## Python

The Python package provides the low-level state machine plus buffered adapters
for `httpx`, `aiohttp`, and FastAPI/Starlette. See the
[Python package README](https://github.com/dualeai/hpke-http/blob/main/python/README.md).

```python
from hpke_http import Client, Method, Request

with Client(server_public_key, key_id, psk, psk_id) as client:
    transaction = client.protect(
        Request(Method.POST, "api.example.test", "/items", body=b"payload")
    )
    try:
        # Send transaction.envelope, then authenticate the returned envelope.
        response = transaction.open_response(protected_response)
    finally:
        transaction.close()
```

## TypeScript

The TypeScript package provides explicit browser and Node entry points and a
buffered `createHpkeFetch` adapter over native Fetch. See the
[TypeScript package README](https://github.com/dualeai/hpke-http/blob/main/typescript/README.md).

```ts
import { createHpkeFetch, initialize } from "@dualeai/hpke-http/browser";

await initialize();
const hpkeFetch = createHpkeFetch({
  recipientPublicKey: serverPublicKey,
  recipientKeyId: keyId,
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

Axios and TanStack Query adapters are medium-term work. Java, Kotlin, and Swift
bindings are dormant options. Any activated binding uses this Rust engine and
joins the same coordinated release pipeline.

## Development

Install the Python and TypeScript dependencies, then build the local Python
extension:

```sh
make install
```

Run all checks:

```sh
make test
```

The separate CodSpeed workflow benchmarks complete public-API transactions in
Rust, Python, and Node/WASM at empty, 1 KiB, 1 MiB, and 8 MiB body sizes. It prepares
keys and message bodies outside the measured operation; only the native Rust
suite enables CodSpeed's allocation-memory mode. There are no local stopwatch
scripts or hardware-dependent timing thresholds.

`wasm-bindgen-cli` 0.2.128 and the Rust `wasm32-unknown-unknown` target are
required for the TypeScript build. CI installs the pinned Rust toolchain and
uses a shared Rust build cache. The TypeScript runtime test also requires Chrome
or Chromium; set `CHROME_BIN` for a nonstandard installation path.

## Versions and releases

Package versions, the protocol ID, and the binding ABI are separate values.
Python and TypeScript reject a native engine whose three-part identity does not
match their package.

Release versions come from an exact `vMAJOR.MINOR.PATCH` Git tag through
`cicd/version.sh`. The release workflow injects that one version into Cargo,
Python, and npm package metadata before it builds or tests any artifact. The
crate, Python distributions, npm package, and GitHub release then use the same
tag and workflow.

The supported 2.x line is the Rust-backed product described here. Published
1.x versions used the removed Python protocol and are not API- or
wire-compatible fallbacks. The coordinated release notes are the migration
record; no old implementation remains in the package.

The protocol details live in Rust module documentation, public API docs, and
frozen known-answer tests owned by the Rust core. This repository does not
maintain a separate specification or architecture-decision document tree.

## Security

See the [security policy](https://github.com/dualeai/hpke-http/blob/main/SECURITY.md)
for supported releases and private reporting.
Do not use a PSK as its public PSK ID. Use an atomic replay store shared by all
server workers that can process the same credentials.

The core standards are [RFC 9180](https://www.rfc-editor.org/rfc/rfc9180.html)
and [RFC 9292](https://www.rfc-editor.org/rfc/rfc9292.html). Response-key
derivation follows the pattern in
[RFC 9458 section 4.4](https://www.rfc-editor.org/rfc/rfc9458.html#section-4.4),
but `hpke-http/1` is not an Oblivious HTTP profile.

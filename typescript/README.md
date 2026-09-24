# hpke-http for TypeScript

Use `@dualeai/hpke-http` to protect Fetch requests and check replies with the
Rust engine in WebAssembly.

## Build and runtime support

Build the package from this checkout. From the repository root, run:

```sh
make install-deps-typescript install-wasm-bindgen build-typescript
```

Use one explicit entry point:

- `@dualeai/hpke-http/node` supports Node.js 24;
- `@dualeai/hpke-http/browser` requires WebAssembly, Fetch, Web Crypto, and Web
  Streams.

There is no root package export. Applications must initialize the selected WASM
target before they create keys, clients, or servers.

```ts
import { initialize } from "@dualeai/hpke-http/node";

await initialize();
```

Use the `/browser` entry point in a browser. Its `initialize()` call loads the
WASM module.

## Send a request and read its reply

```ts
import { createHpkeFetch, initialize } from "@dualeai/hpke-http/browser";

await initialize();
const hpkeFetch = createHpkeFetch({
  endpoint: "https://api.example.test/protected",
  key: { kind: "discover" },
  psk,
  pskId,
});

const reply = await hpkeFetch("https://api.example.test/items", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ name: "Ada" }),
});
if (!reply.ok) throw new Error(`request failed: ${reply.status}`);
const item = await reply.json();
```

The call accepts the same URL, method, headers, body, and abort signal as
Fetch. It returns a Fetch `Response` with checked status, headers, and body.
In Node.js, use the same call with imports from `@dualeai/hpke-http/node`.
The next examples use this open `hpkeFetch` client. Call `hpkeFetch.close()`
when the application no longer needs it.

## Upload a file or body stream

```ts
const form = new FormData();
form.set("file", file); // A browser File from an input element.
const reply = await hpkeFetch("https://api.example.test/upload", {
  method: "POST",
  body: form,
});
```

`Blob`, `FormData`, and body streams use the same call. The adapter reads the
source once. Browser bodies up to 8 MiB by default use a bounded byte POST.
Larger bodies and Node.js streams send protected records as bytes arrive.

For a direct `ReadableStream` body in a browser or Node.js, pass
`duplex: "half"` in the request options, as Fetch requires. Normal small
`FormData` uploads work over HTTP/1.x. Browser bodies above the bounded
threshold need Fetch upload streaming and an HTTP/2 or HTTP/3 endpoint.
[Chrome rejects streamed requests over HTTP/1.x](https://developer.chrome.com/docs/capabilities/web-apis/fetch-streaming-requests).
The Rust engine handles record coding and checks.

## Read an SSE reply

```ts
const reply = await hpkeFetch("https://api.example.test/events");
if (reply.body === null) throw new Error("missing SSE body");
const reader = reply.body.getReader();
try {
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    handleSseBlock(value);
  }
} finally {
  await reader.cancel();
}
```

Each `value` is one checked SSE block. The caller parses fields and comments.

## Technical details

### Credentials and limits

Recipient keys use X25519 and are 32 bytes. Recipient-key and PSK identifiers
are public opaque values from 1 through 255 bytes. A PSK needs at least 32
bytes of entropy, and its public identifier must not equal the PSK.

Names such as `serverPublicKey`, `requestEnvelope`, `resolvePsk`, and
`replayStore` in the examples are application-provided key storage, transport,
and replay components. The Fetch adapter can get a public key from a fixed
HTTPS endpoint.

```ts
import { generateKeyPair, initialize } from "@dualeai/hpke-http/node";

await initialize();
const keys = generateKeyPair();
try {
  await storeRecipientKey(keys.privateKey, keys.publicKey);
} finally {
  keys.privateKey.fill(0);
}
```

`storeRecipientKey` is application storage in this example. Native objects copy
their inputs. Calling `close()` clears or releases native copies, but it cannot
clear caller-owned `Uint8Array` values. Clear secret arrays when the application
no longer needs them.

| Limit | Default | Hard maximum |
| --- | ---: | ---: |
| One-shot request, finite reply, or one SSE block | 8 MiB | 64 MiB |
| Total request body across DATA records | 1 GiB | 4 GiB |
| Combined header-name and value bytes | 16 KiB | 64 KiB |
| Header fields per message | 64 | 256 |
| Combined authority and path bytes | 8 KiB | 8 KiB |

Pass a `Limits` object to `Client`, `Server`, or `createHpkeFetch` to change a
limit within its hard maximum. Omitted fields use the defaults. SSE has no
whole-stream body cap.

Rust uses zstd for clear parts of at least 64 bytes when it shrinks them;
other parts stay raw. It encrypts either form. Rust checks each record and
restores the clear bytes on receipt. The application sees the original body.
The protocol adds no padding; an observer can see record size, count, and
timing.

### Fetch behavior

The first browser `initialize(input)` call selects a WASM URL, `Request`,
`Response`, byte buffer, or compiled module. Later calls share it. A site with
a Content Security Policy must allow the JS module, WASM load, and WASM
instantiation.

The adapter checks the logical HTTPS origin before it reads a body. In a
browser, it waits for body EOF or the first source chunk that crosses
`maxBodyLength` before POST. This can hold the limit plus one source chunk; a
source can supply a chunk of any size. It bounds finite replies by
`maxBodyLength`. For SSE, it returns after checked START and yields each
checked block as it arrives. A finite reply waits for END and real outer EOF.
A complete comment block can serve as a heartbeat.

Blocks contain raw bytes. Follow the
[SSE parsing rules](https://html.spec.whatwg.org/multipage/server-sent-events.html#interpreting-an-event-stream):
decode them as UTF-8 with replacement for bad byte sequences and ignore one
leading BOM at the start of the stream. A checked block can hold comments or
control fields without a dispatched data event.

The `endpoint` is a string HTTPS URL with no query, fragment, or credentials.
GET and POST use that same URL. Set `targetOrigin` when a gateway endpoint has
a different HTTPS origin from the logical request. An injected `fetch`
function must honor the abort signal and deliver decoded response body bytes
like native Fetch. The adapter checks browser request stream support before
it sends a large body.

Discovery sends one GET per call with no logical authorization, cookies, or
PSK ID. It uses `credentials: "omit"`, `redirect: "error"`, and
`cache: "no-store"`. It accepts only status 200, `application/octet-stream`,
identity coding, and an exact key record of at most 293 bytes. See
[key discovery](../README.md#key-discovery) for the record bytes. A pinned key
skips GET. The caller's abort signal covers GET, upload, and reply. The adapter
keeps no key cache and does not retry a protected POST.

For cross-origin browser calls, the outer endpoint must allow the public GET,
the browser's OPTIONS check, the protected POST, and fault replies. The outer
POST uses `Content-Type: message/hpke-http-request` and `Cache-Control: no-store`.
Configure CORS before the HPKE handler or at a proxy. See the
[Python CORS example](../python/README.md#protect-an-asgi-app) and the
[Fetch CORS rules](https://fetch.spec.whatwg.org/#http-cors-protocol).

The synthetic response represents authenticated status, headers, and body. It
does not preserve outer URL history, redirect history, timing, or a cookie-jar
side effect. `Set-Cookie` and `Set-Cookie2` are hidden at this boundary. The
platform `Headers` object can combine other repeated fields. The low-level
`Client` API preserves the exact ordered authenticated field list.
`Response.clone()` and `ReadableStream.tee()` can buffer branches that a caller
does not read. Consume or cancel each branch. An abort after Fetch returns,
an early body cancel, or `hpkeFetch.close()` closes the protected stream.

### Low-level client transaction

```ts
import { Client, initialize } from "@dualeai/hpke-http/node";

await initialize();
const client = new Client(serverPublicKey, keyId, psk, pskId);
try {
  const transaction = client.protect({
    method: "POST",
    authority: "api.example.test",
    path: "/items",
    headers: [{ name: "content-type", value: "application/json" }],
    body: new TextEncoder().encode('{"name":"Ada"}'),
  });
  try {
    const responseEnvelope = await sendEnvelope(transaction.envelope);
    const response = transaction.openResponse(responseEnvelope);
    console.log(response.status);
  } finally {
    transaction.close();
  }
} finally {
  client.close();
}
```

`sendEnvelope` is the application's HTTPS transport in this low-level example.
`openResponse` consumes the response capability. A failed transport attempt
must not reuse the same envelope: call `Client.protect` again to create a fresh
cryptographic attempt.

To send a large or one-use body, call `client.beginStream(head)`. Send
`writer.start`, pass each source byte cut to `writer.push(bytes)`, and send each
returned `record`. Use `consumed` to pass any unconsumed bytes again. On source
EOF, call `writer.finish()`, send its `end` bytes, then end the outer HTTP body.
Its `intoOpener()` checks the reply. Close the writer and response right on
failure. Rust keeps request record and zstd state; JavaScript handles only byte
cuts and transport flow.

For a low-level live reply, call `transaction.intoOpener()`. For each network
chunk, call `feed(chunk.subarray(offset, offset + 65536))`, add `consumed` to
`offset`, and repeat until the whole chunk is used. This 64 KiB slice bounds
the copy into WASM. Process each returned record; one chunk can hold several
records. Finite DATA returns no record; `finishEof()` returns the finite response
after END and real outer body EOF. Each `sse_data` record holds one complete
clear block. For an SSE reply, the server calls
`opened.startResponse(200, headers)` with one
`text/event-stream` `Content-Type` and no logical `Content-Length` or
`Content-Encoding`. Send `sealer.start`, then the result of `sealSseBlock(block)`
for each complete LF-normalized block, then `finish()`. End the outer HTTP body
after `finish()`. For a finite reply, call `sealFiniteBody(body)` once before
`finish()`; an empty body gives no DATA record. Close both state objects on an
early end.

### Low-level server and replay admission

The TypeScript package does not provide a web-framework server adapter. A host
can connect the staged `Server` API to its own HTTP stack:

```ts
import { Server, initialize } from "@dualeai/hpke-http/node";

await initialize();
const server = new Server(recipientPrivateKey, keyId);
try {
  const preparsed = server.preparse(requestEnvelope);
  try {
    const requestPsk = await resolvePsk(preparsed.pskId);
    const authenticated = preparsed.authenticate(requestPsk);
    try {
      const accepted = await replayStore.reserveIfAbsent(
        authenticated.replayId,
        authenticated.retainUntilExclusive,
      );
      const opened = authenticated.admit({ accepted });
      try {
        const logicalResponse = await dispatch(opened.request);
        const responseEnvelope = opened.protectResponse(logicalResponse);
        await sendProtectedResponse(responseEnvelope);
      } finally {
        opened.close();
      }
    } finally {
      authenticated.close();
    }
  } finally {
    preparsed.close();
  }
} finally {
  server.close();
}
```

The replay operation must be one atomic reserve-if-absent decision shared by
all workers that can receive the same credentials. Keep the reservation through
the supplied exclusive Unix deadline. Report errors and uncertain outcomes as
rejected. The engine checks the clock again after the store operation and never
releases plaintext at or after the authenticated deadline.

For a large upload, use the staged server reader. Collect the public prefix
until `server.streamStartLength(prefix)` returns a length. Pass exactly that
many bytes to `server.preparseStream(first)`. Resolve `pskId`, call
`authenticate(psk)`, reserve `replayId`, then call `admit({ accepted })`.
Feed the remaining outer bytes through `opened.feed(bytes)` and store each
checked `data` block in temporary storage. Each call accepts at most 64 KiB;
use `consumed` until all bytes pass. After `end` and true outer EOF, call
`opened.finishEof()`. Only then dispatch the stored body to the application.
The returned right can call `protectResponse` or `startResponse`. Close each
stage on failure. This path uses `maxRequestBytes`, not the one-shot body limit.

### HTTP boundary rules

Requests use absolute HTTPS URLs without embedded credentials. Protocol headers
are ordered lower-case token names with canonical ASCII values. Repeated fields
remain separate in the low-level API.

The Fetch adapter removes transport-only request fields: `Connection` and every
field it names, `Content-Length`, `Expect`, `Host`, `Keep-Alive`, proxy
authentication fields, `Proxy-Connection`, `TE`, `Trailer`, `Transfer-Encoding`,
`Upgrade`, and `Accept-Encoding`.

Logical requests and responses accept only absent or identity
`Content-Encoding`. The adapter does not decompress authenticated logical bytes
outside the body bound. Native zstd record coding is separate from this HTTP
representation field. Native Fetch can decode an outer response while it keeps
encoded response metadata, so the adapter bounds the bytes actually yielded by
Fetch instead of trusting outer `Content-Length`.

`HEAD` responses and statuses 204, 205, and 304 expose no body. A single
authenticated `Content-Length` remains valid metadata for `HEAD` and 304, must
be zero when present on 205, and is forbidden for 204.

### Errors and lifecycle

`ProtocolError.code` uses the Rust engine's stable language-neutral codes:

- configuration and parsing: `invalid_configuration`, `limit_exceeded`,
  `malformed_envelope`, `unsupported_version`, `unsupported_suite`, and
  `unsupported_method`;
- credentials and authentication: `unknown_recipient_key`,
  `invalid_credential`, and `authentication_failed`;
- replay and time: `replay_rejected`, `invalid_request_time`,
  `replay_decision_mismatch`, and `clock_unavailable`;
- platform or local operations: `entropy_unavailable` and `crypto_failure`.

`StateError` uses `state_consumed`. `InitializationError` reports an unavailable
or mismatched WASM module. `FetchTransportError.code` distinguishes invalid
targets, request body bounds, network failure, invalid outer status or media
type, unsupported authenticated content coding, and authenticated responses
that Web Fetch cannot represent. Key GET failures use `discovery_network`,
`discovery_status`, or `discovery_response`. Only `discovery_status` has a
`statusCode`. A bad GET sends no POST or plaintext fallback. Protected response
record bounds use `ProtocolError`.

Every live client, server, request stage, response right, and `HpkeFetch` has an idempotent
`close()` method. Close objects promptly; do not depend on JavaScript garbage
collection for credential or continuation cleanup.

The package does not parse SSE fields or reconnect on its own. A reconnect
needs a fresh protected request. After a lost reply, the caller does not know
whether the app ran.

## Development

Set up Rust and the WASM target as shown in the repository README. Then run
`make install-deps-typescript install-wasm-bindgen test-typescript` from the
repository root. The test target builds the WASM package, runs the Node facade
tests, and runs a browser smoke test. That test covers pinned and discovered
calls, checked SSE, aborts, and CORS over local HTTPS. It requires Chrome or
Chromium; set `CHROME_BIN` when it is not in a standard path.

Run `make package-typescript smoke-typescript` to check the packed npm archive
with Vite 8.2.2 and Chrome. This also checks browser initialization under a
restrictive test CSP. A deployed site's own CSP needs its own integration check.
The standalone `npm run check` builds the generated WASM types first and needs
the same Rust and wasm-bindgen tools as `build`.

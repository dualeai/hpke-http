# hpke-http for TypeScript

Use `@dualeai/hpke-http` to protect Fetch requests and check replies with the
Rust engine in WebAssembly. The
[protocol specification](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md)
holds the shared wire bytes, key record, checks, and host steps.

## Build and runtime support

Install the published v4 package:

```sh
npm install '@dualeai/hpke-http@^4.0.0'
```

The v4 client does not read HHKD v1 key records. Change clients and hosts
together, or use separate endpoints.

To build this checkout, use the commands in [Development](#development).

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
  const reply = await hpkeFetch("https://api.example.test/items", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name: "Ada" }),
  });
  if (!reply.ok) throw new Error(`request failed: ${reply.status}`);
  const item = await reply.json();

  const next = await hpkeFetch("https://api.example.test/items", {
    method: "POST",
    body: "another item",
  });
  if (!next.ok) throw new Error(`request failed: ${next.status}`);
} finally {
  hpkeFetch.close();
  keySource.close();
}
```

The call accepts the same URL, method, headers, body, and abort signal as
Fetch. It returns a Fetch `Response` with checked status and headers. A finite
body is complete and checked when the call returns; each SSE block is checked
when the caller reads it.
In Node.js, use the same call with imports from `@dualeai/hpke-http/node`.
The first call sends GET then POST. The second sends POST alone while the key
lease is valid. Other clients can use `keySource` while it stays open. Keep a
different source for each full endpoint path, including Bridge and Library.
Put the next examples inside the `try` block above, before it closes the client.

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

Use the [credential rules](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#request-hpke-steps)
when you provision recipient keys and PSKs.

Names such as `serverPublicKey`, `requestEnvelope`, `resolvePsk`,
`replayStore`, and `enforceTargetPolicy` in the examples are host-provided key
storage, transport, replay, and target checks. The Fetch adapter can get a
public key from a fixed HTTPS endpoint.

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

The [central limits and DATA rules](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#current-engine-limits)
give each default and hard cap, and explain raw and zstd record coding.
Pass a `Limits` object to `Client`, `Server`, or `createHpkeFetch` to
change a limit within its hard maximum. Omitted fields use the defaults.

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

One `DiscoveredEndpoint` shares a key and one GET among clients for the same
full endpoint URL. Keep Bridge and Library sources separate. The
[key record and lease rules](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record)
define when the source can start a protected POST.

`getTimeoutMs` defaults to 10,000 milliseconds and limits only the key GET.
It does not limit the protected POST. Pass a Fetch abort signal with a deadline
when a caller needs to limit the full GET, upload, and reply. One caller's
abort does not stop a shared GET needed by other callers.

Discovery GET uses `credentials: "omit"`, `redirect: "error"`, and
`cache: "no-store"`. A pinned key skips GET. The
[specification](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record)
gives the exact GET checks and record bytes.

`getKey(signal)` returns a `DiscoveredKeyLease` with copies of `keyId` and
`publicKey`, plus `valid()`. Its lease can end after `getKey()` returns. An
aborted caller stops waiting for a shared GET; other callers can still use
that GET. The abort signal also covers that caller's upload and reply.

The adapter does not retry a protected POST. A lost reply does not prove that
the server skipped the operation.

For cross-origin browser calls, the outer endpoint must allow the public GET,
the browser's OPTIONS check, the protected POST, and fault replies. The outer
POST uses `Content-Type: message/hpke-http-request` and `Cache-Control: no-store`.
The browser must read outer `Content-Encoding` on GET and POST replies to
check their coding. It is [not a CORS-safelisted response header](https://fetch.spec.whatwg.org/#cors-safelisted-response-header-name).
Set `Access-Control-Expose-Headers: Content-Encoding` on these replies.
Configure CORS before the HPKE handler or at a proxy. See the
[Python CORS example](https://github.com/dualeai/hpke-http/blob/main/python/README.md#protect-an-asgi-app) and the
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
can connect the staged `Server` API to its own HTTP stack. Read the full
bounded outer POST body through real EOF before this one-shot call. The host
must check the authenticated logical target before it calls its app:

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
        enforceTargetPolicy(opened.request);
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

The host supplies `enforceTargetPolicy` to check the request authority and
path against its configured logical origin and route. The engine checks the
protected fields but does not set that host policy.

Follow the [key switch order](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record)
for a planned change. Pass each other accepted private key and its public ID
as a pair:

```ts
const server = new Server(bPrivateKey, bKeyId, {}, [[aPrivateKey, aKeyId]]);
```

The HTTP host serves the HHKD v2 record for the advertised key; the Rust
engine does no HTTP I/O.

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
`opened.finishEof()`. Check the logical target before dispatching the stored
body to the application.
The returned right can call `protectResponse` or `startResponse`. Close each
stage on failure. This path uses `maxRequestBytes`, not the one-shot body limit.

### HTTP boundary rules

The [central specification](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#data-coding-and-logical-http-checks)
defines logical HTTP fields and target checks. The low-level API keeps
ordered repeated fields; Fetch has its own header view.

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
that Web Fetch cannot represent. Key failures use `discovery_network`,
`discovery_status`, `discovery_response`, or `discovery_expired`. Only
`discovery_status` has a `statusCode`. A bad GET sends no POST or plaintext
fallback. Protected response
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

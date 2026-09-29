# hpke-http for TypeScript

Use Fetch in Node.js or a browser to send encrypted requests. Read complete
replies or live server-sent events (SSE), with checks before each body or block
reaches your code. Connect to the
[Python ASGI host](https://github.com/dualeai/hpke-http/blob/main/python/README.md#protect-an-asgi-app)
or build a host with the low-level `Server` API. This package has no web-framework
server adapter.

Start with [installation](#build-and-runtime-support) and the
[first call](#send-a-request-and-read-its-reply). Then see
[uploads](#upload-a-file-or-body-stream), [SSE](#read-an-sse-reply),
[pinned keys](#pinned-keys-and-call-deadlines), or [errors](#errors-and-lifecycle).

## App URLs and the protected endpoint

```text
hpkeFetch("https://api.example.test/items?limit=2")
    -> POST https://api.example.test/protected
       with the app method, path, query, headers, and body encrypted inside
```

You choose the endpoint path; `/protected` and `/v1/hpke` are examples.
Every app method travels inside a POST to that endpoint: the **outer request**.
Key discovery uses GET at the same endpoint. Browsers can also send OPTIONS.
A valid protected reply uses outer HTTP 200; the returned `Response` holds the app status.

See [what observers can see](https://github.com/dualeai/hpke-http/blob/main/README.md#what-observers-can-see)
and the [protection limits](https://github.com/dualeai/hpke-http/blob/main/README.md#protection-limits).

## Build and runtime support

Install the published v4 package:

```sh
npm install '@dualeai/hpke-http@^4.0.0'
```

Choose the import for your runtime:

| Runtime | Import from |
| --- | --- |
| Node.js 24 | `@dualeai/hpke-http/node` |
| Browser with WebAssembly, Fetch, Web Crypto, and Web Streams | `@dualeai/hpke-http/browser` |

There is no root export. Call `initialize()` before creating keys, clients,
or servers, as shown below. Both WebAssembly (WASM) builds ship in the package;
you do not need Rust or `wasm-bindgen` to use them.

For an existing host, check
[HHKD v2 compatibility](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record).
For custom assets, see [browser WASM setup](#browser-wasm-setup);
to build the package, see [Development](#development).

## Send a request and read its reply

Set the pre-shared key (`psk`) and its public ID (`pskId`) as `Uint8Array`
values from your app config. The PSK needs at least 32 bytes of entropy.
Its public ID must be 1..255 bytes and differ from the PSK.
Discovery does not supply PSKs. Give each client its secret through your app;
keep service-wide PSKs out of public browser bundles.

This example uses the browser entry point. In Node.js, change the import to
`@dualeai/hpke-http/node`.

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
} finally {
  hpkeFetch.close();
  keySource.close();
}
```

The call returns a Fetch `Response`. For a **finite reply** (a response other
than SSE), it waits for the HTTP body to end and checks the reply in full.

Put the upload and SSE snippets inside this `try` block, before the client
closes. They use the same `hpkeFetch` and config.

## Upload a file or body stream

```ts
const form = new FormData();
form.set("file", file); // A browser File from an input element.
const uploadReply = await hpkeFetch("https://api.example.test/upload", {
  method: "POST",
  body: form,
});
```

Use `Blob`, `FormData`, or a body stream. The adapter reads the source once.
Browser bodies up to `maxBodyLength` (8 MiB by default) use a buffered POST.
Larger bodies and Node.js streams use streamed POSTs. Before a browser POST,
the adapter reads until the input ends or a chunk takes the total past the limit.
It can hold the limit plus that last chunk in memory. The chunk can be any size.

For a direct `ReadableStream` body in a browser or Node.js, pass
`duplex: "half"` in the request options. Extend the DOM type if its
`RequestInit` does not declare that field:

```ts
const uploadOptions: RequestInit & { duplex: "half" } = {
  method: "POST",
  body: file.stream(),
  duplex: "half",
};
const streamedReply = await hpkeFetch("https://api.example.test/upload", uploadOptions);
```

Small buffered uploads work over HTTP/1.x. Larger browser uploads need
Fetch stream support and an HTTP/2 or HTTP/3 endpoint;
[Chrome rejects request streams over HTTP/1.x](https://developer.chrome.com/docs/capabilities/web-apis/fetch-streaming-requests).
The adapter checks runtime stream support before sending the large body.

## Read an SSE reply

```ts
const eventReply = await hpkeFetch("https://api.example.test/events");
const mediaType = eventReply.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
if (!eventReply.ok || mediaType !== "text/event-stream") {
  await eventReply.body?.cancel();
  throw new Error(`expected SSE; received status ${eventReply.status}`);
}
if (eventReply.body === null) throw new Error("missing SSE body");
const reader = eventReply.body.getReader();
try {
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    handleSseBlock(value);
  }
} finally {
  await reader.cancel().catch(() => undefined);
  reader.releaseLock();
}
```

Supply `handleSseBlock` to parse each checked byte block under the
[SSE rules](https://html.spec.whatwg.org/multipage/server-sent-events.html#interpreting-an-event-stream).
Decode UTF-8 with replacement for bad sequences and ignore one leading
byte-order mark at stream start. Blocks can contain comments or control fields
without a data event; a complete comment block can serve as a heartbeat.

The call returns after the initial response record (START) passes its checks.
Later reads can fail after earlier blocks arrived. Normal completion requires
a valid END record and the end of the HTTP body (EOF). Cancelling early does
not prove completion. The package does not parse fields or reconnect.
Reconnect with a fresh request. Native `EventSource` cannot use this adapter.

## Pinned keys and call deadlines

Use the server's public key and matching key ID when you want to skip GET:

```ts
import { createHpkeFetch, initialize } from "@dualeai/hpke-http/browser";

await initialize();
const pinnedFetch = createHpkeFetch({
  key: { kind: "pin", publicKey: serverPublicKey, keyId },
  endpoint: "https://api.example.test/protected",
  psk,
  pskId,
});
try {
  const reply = await pinnedFetch("https://api.example.test/items", {
    signal: AbortSignal.timeout(10_000),
  });
  if (!reply.ok) throw new Error(`request failed: ${reply.status}`);
} finally {
  pinnedFetch.close();
}
```

Pins have no lease or automatic refresh. Update them when the server key
changes. Put `endpoint` and optional `fetch` on the pin config;
for discovery, put them on the source.

The same `signal` option works with a discovered key. It covers that caller's
wait for GET, upload, and reply, including reads after an SSE call returns.
The adapter sets no timeout for the whole POST. Choose a deadline or use your own
`AbortController` for long live streams.

## Technical details

### Browser WASM setup

If your bundler omits the WASM asset, serve
`_wasm/browser/hpke_http_wasm_bg.wasm` from the package and pass its URL to
the first `initialize()` call:

```ts
import { initialize } from "@dualeai/hpke-http/browser";

await initialize(new URL("/assets/hpke_http_wasm_bg.wasm", location.origin));
```

Serve it as `application/wasm` and keep the package's JavaScript modules available.
Your Content Security Policy (CSP) must allow the JS module and any WASM download.
Use `'wasm-unsafe-eval'` in `script-src` to allow WASM without JavaScript
`eval`; see the
[CSP rules](https://www.w3.org/TR/CSP3/#can-compile-wasm-bytes).

`initialize(input)` also accepts a `Request`, `Response`, byte buffer, or
compiled module. Calls share the first result. After failure, retry with a
new input; after success, later inputs do not replace the module.

### Credentials and limits

Recipient public keys are 32 raw bytes; public key and PSK IDs are 1..255 bytes.
Decode hex or Base64 storage forms before use. Follow the
[credential rules](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#request-hpke-steps).
For a new server key pair:

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

Supply `storeRecipientKey` from your app. Native objects copy their inputs;
`close()` cannot clear your `Uint8Array` values. Clear secret arrays after use.

Pass `Limits` to `Client`, `Server`, or `createHpkeFetch`. Omitted fields use
the [engine defaults](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#current-engine-limits);
the same table gives hard maxima. Set compatible limits on the client and
server; they do not agree on limits automatically.

| Setting | Applies to |
| --- | --- |
| `maxRequestBytes` | Every request body |
| `maxBodyLength` | One-shot requests, finite replies, and each SSE block |
| `maxHeaderBytes` | Sum of header name and value bytes |
| `maxHeaderCount` | Header pairs |
| `maxTargetLength` | Authority plus path bytes |

For example: `limits: { maxRequestBytes: 64 * 1024 * 1024,
maxBodyLength: 4 * 1024 * 1024 }`. Byte limits count data before compression and encryption.

Calls that take a whole body in memory (one-shot requests) must fit both body
limits. The streamed API uses `maxRequestBytes` alone. `maxBodyLength` also selects the
[browser upload buffer threshold](#upload-a-file-or-body-stream).
SSE has no total body cap. Set app time and resource limits.

Read the
[compression limits](https://github.com/dualeai/hpke-http/blob/main/README.md#protection-limits)
before mixing secrets and attacker-chosen text. See
[DATA coding](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#data-coding-and-logical-http-checks)
for the record format.

### Fetch behavior

Supported methods: GET, POST, PUT, PATCH, DELETE, HEAD, and OPTIONS. Native
Fetch forbids a GET or HEAD body. The adapter uses the URL, method, headers,
body, and signal; it controls outer credentials, redirects, cache, and referrer.
It does not follow app redirects, manage login cookies, or open WebSockets.
Put bearer tokens in protected headers.

#### Endpoints and key sources

`endpoint` must be a full HTTPS URL with no query, fragment, or credentials.
The adapter checks the app's HTTPS origin before reading its body. For a
gateway, pair `targetOrigin: "https://service.example.test:8443"` with the
Python host's `expected_authority="service.example.test:8443"`.
An injected `fetch` must honor aborts and yield decoded bytes like native Fetch.

Keep one `DiscoveredEndpoint` per full endpoint URL. Clients share discovery
GETs and reuse the key while its lease is valid. Discovery trusts HTTPS;
the key record has no separate signature. GET uses `credentials: "omit"`,
`redirect: "error"`, and `cache: "no-store"`.

`getTimeoutMs` defaults to 10,000 ms and limits only the discovery GET. Use a
[call deadline](#pinned-keys-and-call-deadlines) for upload and reply too.
If one caller aborts, the shared GET continues for other callers.

`getKey(signal)` returns a `DiscoveredKeyLease` with copies of `keyId` and
`publicKey`, plus `valid()`. The lease limits when you can start a POST with
that key. Check `valid()` when you use it. See the
[lease rules](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record).

The adapter never retries protected POSTs. See
[retry safety](https://github.com/dualeai/hpke-http/blob/main/README.md#retry-safety).

#### Browser CORS and proxies

For browser calls across origins, configure cross-origin resource sharing
(CORS) on the outer endpoint.
Allow public GET, preflight OPTIONS, protected POST, and error replies.
POST uses `Content-Type: message/hpke-http-request` and `Cache-Control: no-store`.
Expose outer `Content-Encoding` on GET and POST with
`Access-Control-Expose-Headers: Content-Encoding`; it is
[not CORS-safelisted](https://fetch.spec.whatwg.org/#cors-safelisted-response-header-name).
Put CORS before the HPKE handler or at the proxy; see the
[Python example](https://github.com/dualeai/hpke-http/blob/main/python/README.md#browser-cors).
Disable outer HTTP compression and proxy buffering for SSE.

#### Response view

The returned `Response` has no outer URL history, redirect history, or timing.
The adapter does not update a cookie jar.
`Set-Cookie` and `Set-Cookie2` are hidden; `Headers` can combine other repeated
fields. The low-level API preserves their exact order and values.

`Response.clone()` and `ReadableStream.tee()` can buffer unread branches.
Consume or cancel each branch. Abort, body cancellation, or `hpkeFetch.close()`
closes a live protected stream, including after the Fetch call returns.

### Low-level client transaction

Your app supplies `serverPublicKey`, `keyId`, and `sendEnvelope`.
The transport must use HTTPS and read the complete HTTP response body.

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

`openResponse` uses the transaction's **response right**, which can check
one matching reply. After a transport failure, create a new transaction with
`Client.protect()`; never resend the old envelope. See the
[protocol terms](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#terms-used-below)
for the record names used below.

#### Send a streamed request

To send a large body or a source you can read only once:

1. Call `const writer = client.beginStream(head)` and send `writer.start`.
2. Pass source chunks to `writer.push(bytes)`. Send each returned `record`,
   if present. Advance by `consumed` bytes and push any remaining bytes again.
3. When the source ends, call `const finished = writer.finish()`. Send all
   `finished.end` bytes; they contain any final DATA record and END.
4. End the outer HTTP body. Use `finished.intoOpener()` to read the reply as
   described below.

Close the writer, finished response right, and reader when you no longer need them.

#### Read a reply in chunks

To read records as they arrive, use `transaction.intoOpener()` instead of
`openResponse()`. For a streamed upload, use `finished.intoOpener()`.

1. For each network chunk, start with `offset = 0`.
2. Call the reader's `feed(chunk.subarray(offset, offset + 65536))`. The 64 KiB
   slice limits the copy into WASM.
3. Process any returned record and add `consumed` to `offset`. Repeat until
   the whole chunk is used; one chunk can hold several records.
4. After the HTTP body ends, call `finishEof()` to check END and complete the reply.

Each `sse_data` record holds one complete decrypted SSE block. The reader
holds finite DATA internally and returns the finite reply from `finishEof()`.
Close the reader if you stop early or a read fails.

### Low-level server and replay admission

Read the complete outer POST body, within your size limit, before this one-shot call.
Supply the private key, PSK resolver, replay store, dispatch, and transport
functions from your app:

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

`enforceTargetPolicy` must check the app authority (host and port) and path
before calling the app.
Enforce HTTPS and decide which routes allow ordinary calls. Use distinct
recipient keys for separate services: the outer endpoint URL is not bound
by cryptographic checks. Shared credentials need explicit routing rules.

Follow the [key switch order](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#hhkd-v2-key-record)
for a planned change. Pass each other accepted private key and its public ID
as a pair:

```ts
const server = new Server(bPrivateKey, bKeyId, {}, [[aPrivateKey, aKeyId]]);
```

The host serves an HHKD v2 record for the public key it advertises.
Use one atomic store operation to reserve each replay ID across all workers
that accept the same credentials. Keep the reservation until its exclusive
Unix deadline in seconds. Reject requests if the store fails or its result is
uncertain. The engine checks time again at admission and rejects requests at
or after the deadline.

#### Receive a streamed request

For a large upload, use the staged server reader:

1. Collect the public prefix until `server.streamStartLength(prefix)` returns
   the total prefix and START length. Collect that many bytes, then pass them
   to `server.preparseStream(first)`.
2. Look up the PSK using `pskId`, then call `authenticate(psk)`.
3. Atomically reserve `replayId` and call `admit({ accepted })` with the result.
4. Pass the remaining bytes to `opened.feed(bytes)`, at most 64 KiB per call.
   Use `consumed` to advance through each chunk. Store each checked `data`
   block in temporary storage.
5. After `end` and the actual end of the HTTP body, call `opened.finishEof()`.
6. Check the app's host, port, and path before calling it with `opened.head`
   and the stored body.

The returned response right can call `protectResponse` or `startResponse`.
Close each stage on failure. This path uses `maxRequestBytes`, not the one-shot body limit.

#### Write a reply

Use the response right from one-shot admission or the streamed `finishEof()`.
Call its `startResponse(status, headers)` method to get `sealer`, then send `sealer.start`.

- For SSE, use status 200, one `text/event-stream` `Content-Type`, and no logical
  `Content-Length` or `Content-Encoding`. Send `sealer.sealSseBlock(block)`
  for each complete block with line-feed (LF) line endings.
- For a finite reply, call `sealer.sealFiniteBody(body)` once, even for an empty
  body. Send its result if present; an empty body gives no DATA record.

Send the END bytes from `sealer.finish()`, then end the outer HTTP body.
Close the response right and sealer on an early end.

### HTTP boundary rules

The [field rules](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md#data-coding-and-logical-http-checks)
define allowed app headers and targets.

The Fetch adapter removes transport-only request fields: `Connection` and every
field it names, `Content-Length`, `Expect`, `Host`, `Keep-Alive`, proxy
authentication fields, `Proxy-Connection`, `TE`, `Trailer`, `Transfer-Encoding`,
`Upgrade`, and `Accept-Encoding`.

The Fetch adapter accepts only absent or identity logical `Content-Encoding`.
The low-level API leaves content coding to the host; it does not decode that
representation. SSE forbids logical `Content-Encoding`, including identity.
Native zstd record coding is separate from this HTTP field.
Native Fetch can decode an outer response while it keeps
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

| Failure | Meaning |
| --- | --- |
| `StateError` (`state_consumed`) | Closed object, response right already used, or wrong response mode |
| `InitializationError` | WASM load or version mismatch |
| `FetchTransportError` | Target, buffered-body size, network, outer status/media type, content coding, or unrepresentable response |
| `TypeError` | Invalid native Fetch input |
| Signal reason, often `AbortError` or `TimeoutError` | Caller abort or deadline |

Discovery codes are `discovery_network`, `discovery_status`,
`discovery_response`, and `discovery_expired`. Only `discovery_status` sets
`statusCode`; `outer_status` leaves it undefined. A bad GET sends no POST or
plaintext fallback. Native request or reply limits raise `ProtocolError`
with `limit_exceeded`, including during uploads.

An app 4xx or 5xx returns a normal `Response`; check `ok` or `status`.
For live reply completion, follow the [SSE reader rules](#read-an-sse-reply).
See the
[host troubleshooting table](https://github.com/dualeai/hpke-http/blob/main/python/README.md#troubleshooting)
for endpoint faults.

Clients, servers, request stages, response rights, and `HpkeFetch` have an
idempotent `close()`: calling it again has no further effect.
Close them promptly; do not rely on garbage collection.

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

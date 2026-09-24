# hpke-http for TypeScript

`@dualeai/hpke-http` is the TypeScript binding and runtime facade for the sole
`hpke-http/2` implementation in this repository: the shared Rust engine compiled
to WebAssembly. The generated wasm-bindgen modules are private package details.

The package provides a low-level record state machine and a Fetch adapter with
live SSE bodies. It does not contain a JavaScript
cryptographic fallback.

## Install and runtime support

Build the `/2` TypeScript package from this checkout. From the repository root,
run:

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

The first browser `initialize(input)` call selects the WASM input: a URL string,
`URL`, `Request`, `Response`, byte buffer, or compiled module. Later calls share
that result. A new call can try again after a failure. `initialize` loads the
browser JS glue before it uses the input, so an explicit URL cannot replace
missing glue.
If the site uses a Content Security Policy, allow the emitted JS under
`script-src`, allow `'wasm-unsafe-eval'` there for WASM instantiation, and allow
any WASM URL under `connect-src`.

## Credentials and limits

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
| Request, finite body, or one SSE block | 8 MiB | 64 MiB |
| Combined header-name and value bytes | 16 KiB | 64 KiB |
| Header fields per message | 64 | 256 |
| Combined authority and path bytes | 8 KiB | 8 KiB |

Pass a `Limits` object to `Client`, `Server`, or `createHpkeFetch` to make a
limit stricter. Omitted fields use the defaults. SSE has no whole-stream body cap.

Protocol body compression is opt-in: pass `"gzip"` or `"zstd"` as the sixth
`Client` constructor argument (after `Limits`), set `compression` in
`createHpkeFetch`, and pass `true` as the fourth `Server` argument. The Rust
engine codes only the body and applies the body limit both before and after
decompression. An opt-in client requires an extension-capable server; there is
no silent fallback. Ciphertext length can leak information when attacker
input and secrets share a body, so leave compression disabled for those messages.
SSE blocks use no private compression.

## Native Fetch adapter

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
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name: "Ada" }),
  });
  if (!response.ok) {
    throw new Error(`logical request failed with ${response.status}`);
  }
} finally {
  hpkeFetch.close();
}
```

The adapter uses only the URL, method, headers, body, and abort signal from the
logical Fetch input. It checks the logical HTTPS origin before it reads a body.
Discovery sends one GET per call; `{ kind: "pin", publicKey, keyId }` sends no
GET and keeps a reusable native client. Both modes send POST to the fixed
endpoint. The adapter buffers requests and finite replies within the
configured limits, uses `redirect: "error"`, omits ambient credentials, and
performs no automatic retry. For SSE it returns a synthetic `Response` after
checked START. Each pull on its body yields one checked clear SSE block before
the server ends the reply. For finite replies, it waits for END and real outer
body EOF before it returns. A complete comment block can serve as a heartbeat.

```ts
const sseFetch = createHpkeFetch({
  endpoint: "https://api.example.test/protected",
  key: { kind: "discover" },
  psk,
  pskId,
});
try {
  const response = await sseFetch("https://api.example.test/events");
  const mediaType = response.headers.get("content-type")?.split(";")[0]?.trim().toLowerCase();
  if (mediaType !== "text/event-stream") throw new Error("expected SSE response");
  if (response.body === null) throw new Error("missing SSE body");
  const reader = response.body.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      handleSseBlock(value); // The caller parses SSE fields and comments.
    }
  } finally {
    await reader.cancel(); // Also close on an early loop exit.
  }
} finally {
  sseFetch.close();
}
```

Blocks contain raw bytes. Follow the
[SSE parsing rules](https://html.spec.whatwg.org/multipage/server-sent-events.html#interpreting-an-event-stream):
decode them as UTF-8 with replacement for bad byte sequences and ignore one
leading BOM at the start of the stream. A checked block can hold comments or
control fields without a dispatched data event.

The `endpoint` is a string HTTPS URL with no query, fragment, or credentials.
GET and POST use that same URL. Set `targetOrigin` when a gateway endpoint has
a different HTTPS origin from the logical request. An injected `fetch`
function owns its TLS, abort, and retry behavior and must deliver decoded
response body bytes like native Fetch.

GET sends no logical authorization, cookies, or PSK ID. It uses
`credentials: "omit"`, `redirect: "error"`, and `cache: "no-store"`.
It accepts only status 200, `application/octet-stream`, identity content
coding, and at most 293 decoded bytes. The exact body is
`"HHKD" || 0x01 || id_len:u8 || id || public_key[32]`, with a 1 through 255
byte opaque ID. Any missing or extra bytes fail before protection. There is
no app key cache or package timer. The caller's Fetch signal sets the limit
on a pending GET.

The adapter reads the full logical request body before key GET. The body read,
GET, and protected POST run in order, so their wait times add. Use a caller
signal with a deadline when the full call needs one. An injected Fetch function
must honor that signal. A pinned key skips GET when the key is known and this
extra round trip matters.

For cross-origin browser calls, the outer endpoint must allow the public GET,
the browser's OPTIONS check, the protected POST, and fault replies. The outer POST uses
`Content-Type: message/hpke-http-request` and `Cache-Control: no-store`.
Configure CORS for these headers and the outer response before the HPKE
handler or at a proxy. See the [Fetch CORS rules](https://fetch.spec.whatwg.org/#http-cors-protocol).

The synthetic response represents authenticated status, headers, and body. It
does not preserve outer URL history, redirect history, timing, or a cookie-jar
side effect. `Set-Cookie` and `Set-Cookie2` are hidden at this boundary. The
platform `Headers` object can combine other repeated fields. The low-level
`Client` API preserves the exact ordered authenticated field list.
`Response.clone()` and `ReadableStream.tee()` can buffer branches that a caller
does not read. Consume or cancel each branch. An abort after Fetch returns,
an early body cancel, or `hpkeFetch.close()` closes the protected stream.

## Low-level client transaction

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

## Low-level server and replay admission

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

## HTTP boundary rules

Requests use absolute HTTPS URLs without embedded credentials. Protocol headers
are ordered lower-case token names with canonical ASCII values. Repeated fields
remain separate in the low-level API.

The Fetch adapter removes transport-only request fields: `Connection` and every
field it names, `Content-Length`, `Expect`, `Host`, `Keep-Alive`, proxy
authentication fields, `Proxy-Connection`, `TE`, `Trailer`, `Transfer-Encoding`,
`Upgrade`, and `Accept-Encoding`.

Logical requests and responses accept only absent or identity
`Content-Encoding`. The adapter does not decompress authenticated logical bytes
outside the body bound. Opt-in Rust protocol body coding is separate from this
HTTP representation field. Native Fetch can decode an outer response while it keeps
encoded response metadata, so the adapter bounds the bytes actually yielded by
Fetch instead of trusting outer `Content-Length`.

`HEAD` responses and statuses 204, 205, and 304 expose no body. A single
authenticated `Content-Length` remains valid metadata for `HEAD` and 304, must
be zero when present on 205, and is forbidden for 204.

## Errors and lifecycle

`ProtocolError.code` uses the Rust engine's stable language-neutral codes:

- configuration and parsing: `invalid_configuration`, `limit_exceeded`,
  `malformed_envelope`, `unsupported_version`, `unsupported_suite`, and
  `unsupported_method`;
- credentials and authentication: `unknown_recipient_key`,
  `invalid_credential`, and `authentication_failed`;
- replay and time: `replay_rejected`, `invalid_request_time`,
  `replay_decision_mismatch`, and `clock_unavailable`;
- platform or local operations: `entropy_unavailable`, `crypto_failure`, and
  `compression_failure`.

`StateError` uses `state_consumed`. `InitializationError` reports an unavailable
or mismatched WASM module. `FetchTransportError.code` distinguishes invalid
targets, request body bounds, network failure, invalid outer status or media
type, unsupported authenticated content coding, and authenticated responses
that Web Fetch cannot represent. Key GET failures use `discovery_network`,
`discovery_status`, or `discovery_response`. Only `discovery_status` has a
`statusCode`. A bad GET sends no POST or plaintext fallback. Protected response
record bounds use `ProtocolError`.

Every live `Client`, `Server`, `ProtectedRequest`, `PreparsedRequest`,
`AuthenticatedRequest`, `OpenedRequest`, and `HpkeFetch` has an idempotent
`close()` method. Close objects promptly; do not depend on JavaScript garbage
collection for credential or continuation cleanup.

The package does not parse SSE fields or reconnect on its own. A reconnect
needs a fresh protected request. Discovery adds one GET round trip per call.
The server holds one key, so mixed worker keys during a change can make calls
fail. The adapter does not retry a protected POST; after a lost reply, the
caller does not know whether the app ran. Cookie-jar integration, Axios,
TanStack Query, and more framework adapters are outside this package.

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

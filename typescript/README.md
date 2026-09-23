# hpke-http for TypeScript

`@dualeai/hpke-http` is the TypeScript binding and runtime facade for the sole
`hpke-http/1` implementation in this repository: the shared Rust engine compiled
to WebAssembly. The generated wasm-bindgen modules are private package details.

The package provides a low-level one-shot protocol state machine and a buffered
adapter for the runtime's native Fetch API. It does not contain a JavaScript
cryptographic fallback.

## Install and runtime support

```sh
npm install @dualeai/hpke-http
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

Browser initialization accepts an explicit WASM URL, `Response`, byte buffer, or
compiled module when the default adjacent asset does not fit the deployment's
asset path or Content Security Policy.

## Credentials and limits

Recipient keys use X25519 and are 32 bytes. Recipient-key and PSK identifiers
are public opaque values from 1 through 255 bytes. A PSK is at least 32 bytes,
and its public identifier must not equal the PSK.

Names such as `serverPublicKey`, `requestEnvelope`, `resolvePsk`, and
`replayStore` in the examples are application-provided key storage, transport,
and replay components; the package does not discover them.

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
| Body bytes per message | 8 MiB | 64 MiB |
| Combined header-name and value bytes | 16 KiB | 64 KiB |
| Header fields per message | 64 | 256 |
| Combined authority and path bytes | 8 KiB | 8 KiB |

Pass a `Limits` object to `Client`, `Server`, or `createHpkeFetch` to make a
limit stricter. Omitted fields use the defaults.

Protocol body compression is opt-in: pass `"gzip"` or `"zstd"` as the sixth
`Client` constructor argument (after `Limits`), set `compression` in
`createHpkeFetch`, and pass `true` as the fourth `Server` argument. The Rust
engine codes only the body and applies the body limit both before and after
decompression. An opt-in client requires an extension-capable server; there is
no silent fallback. Ciphertext length can leak information when attacker
input and secrets share a body, so leave compression disabled for those messages.

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

## Native Fetch adapter

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
logical Fetch input. It buffers within the configured protocol limits, sends a
fresh outer `POST`, uses `redirect: "error"`, omits ambient credentials, and
performs no automatic retry. It returns a synthetic `Response` only after the
complete protected response authenticates.

Set `transportEndpoint` to one fixed HTTPS envelope endpoint. Otherwise, the
logical target URL is also the outer endpoint. An injected `fetch` function must
follow native Fetch response semantics, including delivery of decoded response
body bytes.

The synthetic response represents authenticated status, headers, and body. It
does not preserve outer URL history, redirect history, timing, or a cookie-jar
side effect. `Set-Cookie` and `Set-Cookie2` are hidden at this boundary. The
platform `Headers` object can combine other repeated fields. The low-level
`Client` API preserves the exact ordered authenticated field list.

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
targets, request/response bounds, network failure, invalid outer status or media
type, unsupported authenticated content coding, and authenticated responses
that Web Fetch cannot represent.

Every live `Client`, `Server`, `ProtectedRequest`, `PreparsedRequest`,
`AuthenticatedRequest`, `OpenedRequest`, and `HpkeFetch` has an idempotent
`close()` method. Close objects promptly; do not depend on JavaScript garbage
collection for credential or continuation cleanup.

Streaming, discovery, cookie-jar integration, Axios, TanStack
Query, and additional framework adapters are not part of the current package.

## Development

Set up Rust and the WASM target as shown in the repository README. Then run
`make install-deps-typescript install-wasm-bindgen test-typescript` from the
repository root. The test target builds the WASM package, runs the Node facade
tests, and runs one real browser transaction. The browser test requires Chrome
or Chromium; set `CHROME_BIN` when it is not in a standard path.

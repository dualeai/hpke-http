import assert from "node:assert/strict";
import test from "node:test";

import {
  Client,
  FetchTransportError,
  ProtocolError,
  REQUEST_MEDIA_TYPE,
  RESPONSE_MEDIA_TYPE,
  Server,
  StateError,
  createHpkeFetch,
  generateKeyPair,
  initialize,
} from "../dist/node.js";

const KEY_ID = new TextEncoder().encode("primary-2026-09");
const PSK = new TextEncoder().encode("a 32-byte minimum test credential!");
const PSK_ID = new TextEncoder().encode("tenant-42");

test("native Fetch adapter authenticates request and response before exposure", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID);
  let observedOuter;

  const transport = async (input, init) => {
    observedOuter = new Request(input, init);
    const envelope = new Uint8Array(await observedOuter.arrayBuffer());
    const opened = server
      .preparse(envelope)
      .authenticate(PSK)
      .admit({ accepted: true });
    assert.equal(opened.request.method, "POST");
    assert.equal(opened.request.authority, "api.example.test");
    assert.equal(opened.request.path, "/items?limit=2");
    assert.deepEqual(opened.request.headers, [
      { name: "content-type", value: "application/json" },
      { name: "x-trace", value: "public" },
    ]);
    assert.equal(new TextDecoder().decode(opened.request.body), '{"name":"Ada"}');

    const protectedResponse = opened.protectResponse({
      status: 201,
      headers: [
        { name: "content-type", value: "application/json" },
        { name: "x-result", value: "authenticated" },
      ],
      body: new TextEncoder().encode('{"id":"item-1"}'),
    });
    return new Response(protectedResponse, {
      status: 200,
      headers: { "content-type": `${RESPONSE_MEDIA_TYPE}; version=1` },
    });
  };

  const hpkeFetch = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    fetch: transport,
  });
  const response = await hpkeFetch("https://api.example.test/items?limit=2#ignored", {
    method: "POST",
    headers: {
      "accept-encoding": "gzip",
      connection: "X-Hop, keep-alive",
      "content-length": "999",
      "content-type": "application/json",
      expect: "100-continue",
      host: "wrong.example.test",
      "keep-alive": "timeout=5",
      "proxy-authenticate": 'Basic realm="proxy"',
      "proxy-authentication-info": "nextnonce=abc",
      "proxy-authorization": "Basic dGVzdA==",
      "proxy-connection": "keep-alive",
      te: "trailers",
      trailer: "x-trailer",
      "transfer-encoding": "chunked",
      upgrade: "websocket",
      "x-hop": "private",
      "x-trace": "public",
    },
    body: '{"name":"Ada"}',
  });

  assert.equal(observedOuter.method, "POST");
  assert.equal(observedOuter.url, "https://api.example.test/items?limit=2");
  assert.equal(observedOuter.headers.get("content-type"), REQUEST_MEDIA_TYPE);
  assert.equal(observedOuter.headers.get("accept"), RESPONSE_MEDIA_TYPE);
  assert.equal(observedOuter.credentials, "omit");
  assert.equal(observedOuter.redirect, "error");
  assert.equal(response.status, 201);
  assert.equal(response.headers.get("x-result"), "authenticated");
  assert.equal(await response.text(), '{"id":"item-1"}');

  hpkeFetch.close();
  server.close();
});

test("Fetch uses the runtime transport when no transport is injected", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID);
  const originalFetch = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async (input, init) => {
    calls += 1;
    const outer = new Request(input, init);
    const opened = server
      .preparse(new Uint8Array(await outer.arrayBuffer()))
      .authenticate(PSK)
      .admit({ accepted: true });
    assert.equal(opened.request.path, "/default-fetch");
    return new Response(opened.protectResponse({ status: 204 }), {
      status: 200,
      headers: { "content-type": RESPONSE_MEDIA_TYPE },
    });
  };
  try {
    const hpkeFetch = createHpkeFetch({
      recipientPublicKey: keys.publicKey,
      recipientKeyId: KEY_ID,
      psk: PSK,
      pskId: PSK_ID,
    });
    try {
      const response = await hpkeFetch("https://api.example.test/default-fetch");
      assert.equal(response.status, 204);
      assert.equal(calls, 1);
    } finally {
      hpkeFetch.close();
    }
  } finally {
    globalThis.fetch = originalFetch;
    server.close();
  }
});

test("native Fetch opts into Rust body coding without HTTP content coding", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID, {}, true);
  const body = "a".repeat(16 * 1024);
  const transport = async (input, init) => {
    const outer = new Request(input, init);
    const envelope = new Uint8Array(await outer.arrayBuffer());
    assert.ok(envelope.byteLength < body.length);
    const opened = server.preparse(envelope).authenticate(PSK).admit({ accepted: true });
    assert.equal(new TextDecoder().decode(opened.request.body), body);
    assert.equal(opened.request.headers.some((field) => field.name === "content-encoding"), false);
    const protectedResponse = opened.protectResponse({
      status: 200,
      body: new TextEncoder().encode(body),
    });
    return new Response(protectedResponse, {
      status: 200,
      headers: { "content-type": RESPONSE_MEDIA_TYPE },
    });
  };
  const hpkeFetch = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    compression: "zstd",
    fetch: transport,
  });
  const response = await hpkeFetch("https://api.example.test/compressed", {
    method: "POST",
    body,
  });
  assert.equal(await response.text(), body);
  hpkeFetch.close();
  server.close();
});

test("Fetch adapter rejects method spellings that Fetch does not normalize", async () => {
  await initialize();
  const keys = generateKeyPair();
  let transportCalled = false;
  const hpkeFetch = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    fetch: async () => {
      transportCalled = true;
      throw new Error("transport must not run");
    },
  });

  await assert.rejects(
    hpkeFetch("https://api.example.test/items", { method: "patch" }),
    (error) => error instanceof ProtocolError && error.code === "unsupported_method",
  );
  assert.equal(transportCalled, false);
  hpkeFetch.close();
});

test("Fetch adapter rejects nonidentity logical content coding", async () => {
  await initialize();
  const keys = generateKeyPair();
  let transportCalled = false;
  const requestClient = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    fetch: async () => {
      transportCalled = true;
      throw new Error("transport must not run");
    },
  });
  await assert.rejects(
    requestClient("https://api.example.test/items", {
      method: "POST",
      headers: { "content-encoding": "gzip" },
      body: "compressed",
    }),
    (error) => error instanceof FetchTransportError && error.code === "inner_content_encoding",
  );
  assert.equal(transportCalled, false);
  requestClient.close();

  const server = new Server(keys.privateKey, KEY_ID);
  const responseClient = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    fetch: async (input, init) => {
      const request = new Request(input, init);
      const opened = server
        .preparse(new Uint8Array(await request.arrayBuffer()))
        .authenticate(PSK)
        .admit({ accepted: true });
      const envelope = opened.protectResponse({
        status: 200,
        headers: [{ name: "content-encoding", value: "gzip" }],
        body: new TextEncoder().encode("compressed"),
      });
      return new Response(envelope, {
        status: 200,
        headers: { "content-type": RESPONSE_MEDIA_TYPE },
      });
    },
  });
  await assert.rejects(
    responseClient("https://api.example.test/items"),
    (error) => error instanceof FetchTransportError && error.code === "inner_content_encoding",
  );
  responseClient.close();
  server.close();
});

test("Fetch facade hides cookie fields without changing low-level responses", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID);
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID);
  const cookieFields = [
    { name: "set-cookie", value: "session=one; Path=/; HttpOnly" },
    { name: "set-cookie", value: "theme=dark; Path=/" },
    { name: "set-cookie2", value: "legacy=one; Version=1" },
  ];

  const lowLevelRequest = client.protect({
    method: "GET",
    authority: "api.example.test",
    path: "/cookies",
  });
  const lowLevelOpened = server
    .preparse(lowLevelRequest.envelope)
    .authenticate(PSK)
    .admit({ accepted: true });
  const lowLevelEnvelope = lowLevelOpened.protectResponse({
    status: 200,
    headers: cookieFields,
  });
  const lowLevelResponse = lowLevelRequest.openResponse(lowLevelEnvelope);
  assert.deepEqual(lowLevelResponse.headers, cookieFields);
  client.close();

  const hpkeFetch = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    fetch: async (input, init) => {
      const request = new Request(input, init);
      const envelope = new Uint8Array(await request.arrayBuffer());
      const opened = server
        .preparse(envelope)
        .authenticate(PSK)
        .admit({ accepted: true });
      const protectedResponse = opened.protectResponse({
        status: 200,
        headers: [...cookieFields, { name: "x-result", value: "visible" }],
      });
      return new Response(protectedResponse, {
        status: 200,
        headers: { "content-type": RESPONSE_MEDIA_TYPE },
      });
    },
  });

  const response = await hpkeFetch("https://api.example.test/cookies");
  assert.equal(response.headers.get("set-cookie"), null);
  assert.equal(response.headers.get("set-cookie2"), null);
  assert.equal(response.headers.get("x-result"), "visible");

  hpkeFetch.close();
  server.close();
});

test("each caller-approved Fetch attempt uses a fresh protected envelope", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID);
  const envelopes = [];
  let attempt = 0;
  const transport = async (input, init) => {
    const request = new Request(input, init);
    const envelope = new Uint8Array(await request.arrayBuffer());
    envelopes.push(envelope);
    attempt += 1;
    if (attempt === 1) {
      return new Response(null, { status: 503 });
    }
    const opened = server
      .preparse(envelope)
      .authenticate(PSK)
      .admit({ accepted: true });
    return new Response(opened.protectResponse({ status: 204 }), {
      status: 200,
      headers: { "content-type": RESPONSE_MEDIA_TYPE },
    });
  };
  const hpkeFetch = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    fetch: transport,
  });

  await assert.rejects(
    hpkeFetch("https://api.example.test/health"),
    (error) => error instanceof FetchTransportError && error.code === "outer_status",
  );
  const response = await hpkeFetch("https://api.example.test/health");
  assert.equal(response.status, 204);
  assert.equal(envelopes.length, 2);
  assert.notDeepEqual(envelopes[0], envelopes[1]);

  hpkeFetch.close();
  await assert.rejects(hpkeFetch("https://api.example.test/health"), StateError);
  server.close();
});

test("Fetch adapter bounds bodies and never exposes a tampered response", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID, { maxBodyLength: 4 });
  const transport = async (input, init) => {
    const request = new Request(input, init);
    const envelope = new Uint8Array(await request.arrayBuffer());
    const opened = server
      .preparse(envelope)
      .authenticate(PSK)
      .admit({ accepted: true });
    const protectedResponse = opened.protectResponse({
      status: 200,
      body: new TextEncoder().encode("pong"),
    });
    protectedResponse[protectedResponse.length - 1] ^= 1;
    return new Response(protectedResponse, {
      status: 200,
      headers: { "content-type": RESPONSE_MEDIA_TYPE },
    });
  };
  const hpkeFetch = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    limits: { maxBodyLength: 4 },
    fetch: transport,
  });

  await assert.rejects(
    hpkeFetch("https://api.example.test/items", { method: "POST", body: "12345" }),
    (error) => error instanceof FetchTransportError && error.code === "request_too_large",
  );
  await assert.rejects(
    hpkeFetch("https://api.example.test/ping", { method: "POST", body: "ping" }),
    (error) => error instanceof ProtocolError && error.code === "authentication_failed",
  );

  hpkeFetch.close();
  server.close();
});

test("Fetch limits reject values that WebAssembly integer coercion could change", async () => {
  await initialize();
  const keys = generateKeyPair();
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY, -1, 1.5, 2 ** 32]) {
    assert.throws(
      () =>
        createHpkeFetch({
          recipientPublicKey: keys.publicKey,
          recipientKeyId: KEY_ID,
          psk: PSK,
          pskId: PSK_ID,
          limits: { maxBodyLength: value },
        }),
      (error) => error instanceof ProtocolError && error.code === "invalid_configuration",
    );
  }
  for (const [name, value] of [
    ["maxBodyLength", 64 * 1024 * 1024 + 1],
    ["maxHeaderBytes", 64 * 1024 + 1],
    ["maxHeaderCount", 257],
    ["maxTargetLength", 8 * 1024 + 1],
  ]) {
    assert.throws(
      () =>
        createHpkeFetch({
          recipientPublicKey: keys.publicKey,
          recipientKeyId: KEY_ID,
          psk: PSK,
          pskId: PSK_ID,
          limits: { [name]: value },
        }),
      (error) => error instanceof ProtocolError && error.code === "invalid_configuration",
    );
  }
});

test("Fetch adapter cancels rejected outer response bodies", async () => {
  await initialize();
  const keys = generateKeyPair();
  let canceled = false;
  const body = new ReadableStream({
    pull(controller) {
      controller.enqueue(new Uint8Array([1]));
    },
    cancel() {
      canceled = true;
    },
  });
  const hpkeFetch = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    fetch: async () => new Response(body, { status: 503 }),
  });

  await assert.rejects(
    hpkeFetch("https://api.example.test/health"),
    (error) => error instanceof FetchTransportError && error.code === "outer_status",
  );
  assert.equal(canceled, true);
  hpkeFetch.close();
});

test("Fetch abort cancels a pending outer response body", { timeout: 5000 }, async () => {
  await initialize();
  const keys = generateKeyPair();
  const controller = new AbortController();
  const reason = new Error("caller aborted");
  let canceled = false;
  let transportReturned;
  const transportDone = new Promise((resolve) => {
    transportReturned = resolve;
  });
  const body = new ReadableStream({
    cancel() {
      canceled = true;
    },
  });
  const hpkeFetch = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    fetch: async () => {
      transportReturned();
      return new Response(body, {
        status: 200,
        headers: { "content-type": RESPONSE_MEDIA_TYPE },
      });
    },
  });

  try {
    const pending = hpkeFetch("https://api.example.test/abort", {
      signal: controller.signal,
    });
    await transportDone;
    setTimeout(() => controller.abort(reason), 0);
    await assert.rejects(pending, (error) => error === reason);
    assert.equal(canceled, true);
  } finally {
    hpkeFetch.close();
  }
});

test("Fetch adapter bounds actual bytes instead of trusting declared length", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID, { maxBodyLength: 1 });
  const hpkeFetch = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    limits: { maxBodyLength: 1 },
    fetch: async (input, init) => {
      const request = new Request(input, init);
      const envelope = new Uint8Array(await request.arrayBuffer());
      const opened = server
        .preparse(envelope)
        .authenticate(PSK)
        .admit({ accepted: true });
      return new Response(opened.protectResponse({ status: 204 }), {
        status: 200,
        headers: {
          "content-length": "99999999",
          "content-type": RESPONSE_MEDIA_TYPE,
        },
      });
    },
  });

  const response = await hpkeFetch("https://api.example.test/health");
  assert.equal(response.status, 204);
  hpkeFetch.close();
  server.close();
});

test("Fetch adapter rejects and cancels one oversized response chunk before copying it", async () => {
  await initialize();
  const keys = generateKeyPair();
  let canceled = false;
  const body = new ReadableStream({
    pull(controller) {
      controller.enqueue(new Uint8Array(100_000));
    },
    cancel() {
      canceled = true;
    },
  });
  const hpkeFetch = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    limits: { maxBodyLength: 1 },
    fetch: async () =>
      new Response(body, {
        status: 200,
        headers: { "content-type": RESPONSE_MEDIA_TYPE },
      }),
  });

  await assert.rejects(
    hpkeFetch("https://api.example.test/health"),
    (error) => error instanceof FetchTransportError && error.code === "response_too_large",
  );
  assert.equal(canceled, true);
  hpkeFetch.close();
});

test("Fetch adapter coalesces tiny and empty response chunks before authentication", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID);
  const expected = new Uint8Array(2048).fill(7);
  const hpkeFetch = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    fetch: async (input, init) => {
      const outer = new Request(input, init);
      const envelope = new Uint8Array(await outer.arrayBuffer());
      const opened = server.preparse(envelope).authenticate(PSK).admit({ accepted: true });
      const protectedResponse = opened.protectResponse({ status: 200, body: expected });
      let offset = 0;
      let empty = true;
      return new Response(
        new ReadableStream({
          pull(controller) {
            if (offset === protectedResponse.byteLength) {
              controller.close();
            } else if (empty) {
              controller.enqueue(new Uint8Array());
              empty = false;
            } else {
              controller.enqueue(protectedResponse.subarray(offset, offset + 1));
              offset += 1;
              empty = true;
            }
          },
        }),
        { status: 200, headers: { "content-type": RESPONSE_MEDIA_TYPE } },
      );
    },
  });

  const response = await hpkeFetch("https://api.example.test/tiny-chunks");
  assert.deepEqual(new Uint8Array(await response.arrayBuffer()), expected);
  hpkeFetch.close();
  server.close();
});

test("Fetch adapter maps response-stream failures to network errors", async () => {
  await initialize();
  const keys = generateKeyPair();
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array([1]));
      controller.error(new TypeError("response stream failed"));
    },
  });
  const hpkeFetch = createHpkeFetch({
    recipientPublicKey: keys.publicKey,
    recipientKeyId: KEY_ID,
    psk: PSK,
    pskId: PSK_ID,
    fetch: async () =>
      new Response(body, {
        status: 200,
        headers: { "content-type": RESPONSE_MEDIA_TYPE },
      }),
  });

  await assert.rejects(
    hpkeFetch("https://api.example.test/items"),
    (error) =>
      error instanceof FetchTransportError &&
      error.code === "network_error" &&
      error.cause instanceof TypeError,
  );
  hpkeFetch.close();
});

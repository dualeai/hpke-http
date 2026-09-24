import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
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
const DISCOVERY_FIXTURE = JSON.parse(readFileSync(new URL("../../rust/hpke-http/tests/vectors/key-discovery-v1.json", import.meta.url)));

function keyRecord(keyId, publicKey) {
  return new Uint8Array([0x48, 0x48, 0x4b, 0x44, 1, keyId.byteLength, ...keyId, ...publicKey]);
}

test("Fetch accepts the fixed record and both legal record sizes", async () => {
  await initialize();
  const fixtureBytes = Uint8Array.from(Buffer.from(DISCOVERY_FIXTURE.record, "hex"));
  assert.deepEqual(fixtureBytes, keyRecord(
    Uint8Array.from(Buffer.from(DISCOVERY_FIXTURE.key_id, "hex")),
    Uint8Array.from(Buffer.from(DISCOVERY_FIXTURE.public_key, "hex")),
  ));
  const keys = generateKeyPair();
  const oneByteId = Uint8Array.of(0x6b);
  const maxId = new Uint8Array(255).fill(0x6b);
  for (const { record, keyId, size } of [
    { record: fixtureBytes, keyId: undefined, size: 53 },
    { record: keyRecord(oneByteId, keys.publicKey), keyId: oneByteId, size: 39 },
    { record: keyRecord(maxId, keys.publicKey), keyId: maxId, size: 293 },
  ]) {
    assert.equal(record.byteLength, size);
    const server = keyId === undefined ? undefined : new Server(keys.privateKey, keyId);
    const calls = [];
    const client = createHpkeFetch({
      endpoint: "https://api.example.test/protected",
      key: { kind: "discover" }, psk: PSK, pskId: PSK_ID,
      fetch: async (input, init) => {
        const request = new Request(input, init);
        calls.push(request.method);
        if (request.method === "GET") {
          return new Response(record, { status: 200, headers: { "content-type": "application/octet-stream" } });
        }
        if (server === undefined) return new Response(null, { status: 503 });
        const opened = server.preparse(new Uint8Array(await request.arrayBuffer()))
          .authenticate(PSK).admit({ accepted: true });
        return new Response(opened.protectResponse({ status: 200, body: new TextEncoder().encode("ok") }), {
          status: 200, headers: { "content-type": RESPONSE_MEDIA_TYPE },
        });
      },
    });
    try {
      if (server === undefined) {
        await assert.rejects(client("https://api.example.test/items"),
          (error) => error instanceof FetchTransportError && error.code === "outer_status");
      } else {
        assert.equal(await (await client("https://api.example.test/items")).text(), "ok");
      }
      assert.deepEqual(calls, ["GET", "POST"]);
    } finally { client.close(); server?.close(); }
  }
});

test("Fetch reads a fresh key for each call", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID);
  const calls = [];
  const client = createHpkeFetch({
    endpoint: "https://api.example.test/protected",
    key: { kind: "discover" }, psk: PSK, pskId: PSK_ID,
    fetch: async (input, init) => {
      const request = new Request(input, init);
      calls.push(request.method);
      assert.equal(request.url, "https://api.example.test/protected");
      assert.equal(request.credentials, "omit");
      assert.equal(request.redirect, "error");
      if (request.method === "GET") {
        assert.equal(request.headers.get("accept"), "application/octet-stream");
        assert.equal(request.cache, "no-store");
        assert.equal(request.referrerPolicy, "no-referrer");
        return new Response(keyRecord(KEY_ID, keys.publicKey), {
          status: 200, headers: { "content-type": "application/octet-stream" },
        });
      }
      const opened = server.preparse(new Uint8Array(await request.arrayBuffer())).authenticate(PSK).admit({ accepted: true });
      assert.equal(opened.request.path, "/items");
      return new Response(opened.protectResponse({ status: 200, body: new TextEncoder().encode("ok") }), {
        status: 200, headers: { "content-type": RESPONSE_MEDIA_TYPE },
      });
    },
  });
  try {
    assert.equal(await (await client("https://API.example.test:443/items")).text(), "ok");
    assert.equal(await (await client("https://api.example.test/items")).text(), "ok");
    assert.deepEqual(calls, ["GET", "POST", "GET", "POST"]);
  } finally { client.close(); server.close(); }
});

test("Fetch rejects bad key GET before any protected POST", async () => {
  await initialize();
  const valid = Uint8Array.from(Buffer.from(DISCOVERY_FIXTURE.record, "hex"));
  for (const fault of ["status", "type", "coding", "size", "shape", "point", "network"]) {
    const calls = [];
    const client = createHpkeFetch({
      endpoint: "https://api.example.test/protected", key: { kind: "discover" }, psk: PSK, pskId: PSK_ID,
      fetch: async (input, init) => {
        const request = new Request(input, init);
        calls.push(request.method);
        assert.equal(request.method, "GET");
        if (fault === "network") { throw new TypeError("key source unavailable"); }
        const headers = { "content-type": fault === "type" ? "text/plain" : "application/octet-stream" };
        if (fault === "coding") { headers["content-encoding"] = "gzip"; }
        const record = fault === "size" ? new Uint8Array(294)
          : fault === "shape" ? new Uint8Array([...valid, 0])
          : fault === "point" ? keyRecord(KEY_ID, new Uint8Array(32)) : valid;
        return new Response(record, { status: fault === "status" ? 503 : 200, headers });
      },
    });
    try {
      await assert.rejects(client("https://api.example.test/items"), (error) => {
        assert.ok(error instanceof FetchTransportError);
        assert.equal(error.code, fault === "status" ? "discovery_status" : fault === "network" ? "discovery_network" : "discovery_response");
        assert.equal(error.statusCode, fault === "status" ? 503 : undefined);
        return true;
      });
      assert.deepEqual(calls, ["GET"]);
    } finally { client.close(); }
  }
});

test("Fetch key errors settle when body cancellation stays pending", async () => {
  await initialize();
  for (const fault of ["status", "type", "oversize"]) {
    let cancels = 0;
    const body = new ReadableStream({
      start(controller) {
        if (fault === "oversize") controller.enqueue(new Uint8Array(294));
      },
      cancel() {
        cancels += 1;
        return new Promise(() => {});
      },
    });
    const client = createHpkeFetch({
      endpoint: "https://api.example.test/protected", key: { kind: "discover" }, psk: PSK, pskId: PSK_ID,
      fetch: async () => new Response(body, {
        status: fault === "status" ? 503 : 200,
        headers: { "content-type": fault === "type" ? "text/plain" : "application/octet-stream" },
      }),
    });
    try {
      await assert.rejects(within(client("https://api.example.test/items"), 5000), (error) => {
        assert.ok(error instanceof FetchTransportError);
        assert.equal(error.code, fault === "status" ? "discovery_status" : "discovery_response");
        return true;
      });
      assert.equal(cancels, 1);
    } finally { client.close(); }
  }
});

test("Fetch outer error settles when body cancellation stays pending", async () => {
  await initialize();
  const keys = generateKeyPair();
  let cancels = 0;
  const client = createHpkeFetch({
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID }, psk: PSK, pskId: PSK_ID,
    fetch: async () => new Response(new ReadableStream({
      cancel() {
        cancels += 1;
        return new Promise(() => {});
      },
    }), { status: 503 }),
  });
  try {
    await assert.rejects(within(client("https://api.example.test/items"), 5000),
      (error) => error instanceof FetchTransportError && error.code === "outer_status");
    assert.equal(cancels, 1);
  } finally { client.close(); }
});

test("Fetch checks origin before body read and close blocks a late key GET", async () => {
  await initialize();
  let calls = 0;
  let release;
  const pending = new Promise((resolve) => { release = resolve; });
  const client = createHpkeFetch({
    endpoint: "https://api.example.test/protected", key: { kind: "discover" }, psk: PSK, pskId: PSK_ID,
    fetch: async () => {
      calls += 1;
      await pending;
      return new Response(keyRecord(KEY_ID, generateKeyPair().publicKey), {
        status: 200, headers: { "content-type": "application/octet-stream" },
      });
    },
  });
  let bodyReads = 0;
  const body = new ReadableStream({
    pull(controller) { bodyReads += 1; controller.enqueue(new Uint8Array([1])); controller.close(); },
  }, { highWaterMark: 0 });
  try {
    await assert.rejects(
      client("https://other.example.test/items", { method: "POST", body, duplex: "half" }),
      (error) => error instanceof FetchTransportError && error.code === "invalid_target",
    );
    assert.equal(calls, 0);
    assert.equal(bodyReads, 0);
    const inFlight = client("https://api.example.test/items");
    await Promise.resolve();
    client.close();
    release();
    await assert.rejects(inFlight, (error) => error instanceof StateError);
    assert.equal(calls, 1);
  } finally { release(); client.close(); }
});

test("Fetch caller abort during key GET sends no protected POST", async () => {
  await initialize();
  const controller = new AbortController();
  const reason = new Error("caller stopped key GET");
  let entered;
  const started = new Promise((resolve) => { entered = resolve; });
  let release;
  const pendingGet = new Promise((resolve) => { release = resolve; });
  const calls = [];
  const client = createHpkeFetch({
    endpoint: "https://api.example.test/protected", key: { kind: "discover" }, psk: PSK, pskId: PSK_ID,
    fetch: async (input, init) => {
      const request = new Request(input, init);
      calls.push(request.method);
      entered();
      await pendingGet;
      return new Response(Uint8Array.from(Buffer.from(DISCOVERY_FIXTURE.record, "hex")), {
        status: 200, headers: { "content-type": "application/octet-stream" },
      });
    },
  });
  try {
    const pending = client("https://api.example.test/items", { signal: controller.signal });
    await started;
    controller.abort(reason);
    release();
    await assert.rejects(pending, (error) => error === reason);
    assert.deepEqual(calls, ["GET"]);
  } finally { release(); client.close(); }
});

async function within(promise, milliseconds) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error("response read timed out")), milliseconds);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

test("Fetch live body cancel settles when outer cancellation stays pending", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID);
  let cancels = 0;
  const client = createHpkeFetch({
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID }, psk: PSK, pskId: PSK_ID,
    fetch: async (input, init) => {
      const request = new Request(input, init);
      const opened = server.preparse(new Uint8Array(await request.arrayBuffer())).authenticate(PSK).admit({ accepted: true });
      const writer = opened.startResponse(200, [{ name: "content-type", value: "text/event-stream" }]);
      const start = writer.start;
      writer.close();
      return new Response(new ReadableStream({
        start(controller) { controller.enqueue(start); },
        cancel() {
          cancels += 1;
          return new Promise(() => {});
        },
      }), { status: 200, headers: { "content-type": RESPONSE_MEDIA_TYPE } });
    },
  });
  try {
    const response = await client("https://api.example.test/events");
    assert.ok(response.body);
    await within(response.body.cancel(), 5000);
    assert.equal(cancels, 1);
  } finally {
    client.close();
    server.close();
  }
});

test("real Node Fetch yields each checked SSE block before server completion", async () => {
  await initialize();
  const keys = generateKeyPair();
  const serverEngine = new Server(keys.privateKey, KEY_ID);
  const first = new TextEncoder().encode(": ready\n\n");
  const second = new TextEncoder().encode("data: two\n\n");
  let releaseSecond;
  const gate = new Promise((resolve) => { releaseSecond = resolve; });
  let releaseEnd;
  const endGate = new Promise((resolve) => { releaseEnd = resolve; });
  let firstSent;
  const firstOnWire = new Promise((resolve) => { firstSent = resolve; });
  let serverFinished = false;
  const httpServer = createServer((request, response) => {
    void (async () => {
      const parts = [];
      for await (const part of request) { parts.push(part); }
      const envelope = Buffer.concat(parts);
      const opened = serverEngine.preparse(envelope).authenticate(PSK).admit({ accepted: true });
      const writer = opened.startResponse(200, [{ name: "content-type", value: "text/event-stream; charset=utf-8" }]);
      try {
        response.writeHead(200, { "content-type": RESPONSE_MEDIA_TYPE, "cache-control": "no-store" });
        response.write(writer.start);
        response.write(writer.sealSseBlock(first));
        firstSent();
        await gate;
        response.write(writer.sealSseBlock(second));
        await endGate;
        response.end(writer.finish());
        serverFinished = true;
      } finally {
        writer.close();
      }
    })().catch((error) => response.destroy(error));
  });
  await new Promise((resolve) => httpServer.listen(0, "127.0.0.1", resolve));
  const address = httpServer.address();
  assert.ok(address && typeof address !== "string");
  const relay = `http://127.0.0.1:${address.port}/protected`;
  const hpkeFetch = createHpkeFetch({
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID }, psk: PSK, pskId: PSK_ID,
    fetch: (_input, init) => fetch(relay, init),
  });
  try {
    const response = await hpkeFetch("https://api.example.test/events");
    assert.equal(response.status, 200);
    assert.ok(response.body);
    const reader = response.body.getReader();
    const firstRead = await within(reader.read(), 5000);
    await firstOnWire;
    assert.deepEqual(firstRead.value, first);
    assert.equal(serverFinished, false);
    releaseSecond();
    assert.deepEqual((await within(reader.read(), 5000)).value, second);
    assert.equal(serverFinished, false);
    releaseEnd();
    assert.equal((await within(reader.read(), 5000)).done, true);
  } finally {
    releaseSecond();
    releaseEnd();
    hpkeFetch.close();
    serverEngine.close();
    httpServer.closeAllConnections();
    await new Promise((resolve) => httpServer.close(resolve));
  }
});

test("Fetch keeps one checked block visible before a later bad record", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID);
  const client = createHpkeFetch({
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID }, psk: PSK, pskId: PSK_ID,
    fetch: async (input, init) => {
      const request = new Request(input, init);
      const opened = server.preparse(new Uint8Array(await request.arrayBuffer())).authenticate(PSK).admit({ accepted: true });
      const writer = opened.startResponse(200, [{ name: "content-type", value: "text/event-stream" }]);
      try {
        const first = writer.sealSseBlock(new TextEncoder().encode(": ready\n\n"));
        const badEnd = writer.finish();
        badEnd[badEnd.length - 1] ^= 1;
        const bytes = new Uint8Array([...writer.start, ...first, ...badEnd]);
        return new Response(new ReadableStream({ start(controller) { controller.enqueue(bytes); controller.close(); } }), {
          status: 200, headers: { "content-type": RESPONSE_MEDIA_TYPE },
        });
      } finally { writer.close(); }
    },
  });
  try {
    const response = await client("https://api.example.test/events");
    const reader = response.body.getReader();
    assert.deepEqual((await reader.read()).value, new TextEncoder().encode(": ready\n\n"));
    await assert.rejects(reader.read(), (error) => error instanceof ProtocolError && error.code === "authentication_failed");
  } finally {
    client.close();
    server.close();
  }
});

test("closing Fetch ends a pending live body read", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID);
  let outerCancelled = false;
  const client = createHpkeFetch({
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID }, psk: PSK, pskId: PSK_ID,
    fetch: async (input, init) => {
      const request = new Request(input, init);
      const opened = server.preparse(new Uint8Array(await request.arrayBuffer())).authenticate(PSK).admit({ accepted: true });
      const writer = opened.startResponse(200, [{ name: "content-type", value: "text/event-stream" }]);
      const first = writer.sealSseBlock(new TextEncoder().encode(": ready\n\n"));
      writer.close();
      return new Response(new ReadableStream({
        start(controller) { controller.enqueue(new Uint8Array([...writer.start, ...first])); },
        cancel() { outerCancelled = true; },
      }), { status: 200, headers: { "content-type": RESPONSE_MEDIA_TYPE } });
    },
  });
  try {
    const response = await client("https://api.example.test/events");
    const reader = response.body.getReader();
    assert.deepEqual((await reader.read()).value, new TextEncoder().encode(": ready\n\n"));
    const pending = reader.read();
    client.close();
    await assert.rejects(within(pending, 5000), (error) => error instanceof StateError);
    assert.equal(outerCancelled, true);
  } finally {
    client.close();
    server.close();
  }
});

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
      headers: { "content-type": RESPONSE_MEDIA_TYPE },
    });
  };

  const hpkeFetch = createHpkeFetch({
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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
  assert.equal(observedOuter.url, "https://api.example.test/protected");
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
      endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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

test("real Node Fetch sends checked upload data before its source ends", async () => {
  await initialize();
  const keys = generateKeyPair();
  const serverEngine = new Server(keys.privateKey, KEY_ID);
  let releaseSource;
  const sourceGate = new Promise((resolve) => { releaseSource = resolve; });
  let sawData;
  const dataOnWire = new Promise((resolve) => { sawData = resolve; });
  let sourceEnded = false;
  const source = new ReadableStream({
    start(controller) { controller.enqueue(new Uint8Array(64 * 1024).fill(0x41)); },
    async pull(controller) {
      await sourceGate;
      controller.enqueue(new Uint8Array([0x42]));
      controller.close();
      sourceEnded = true;
    },
  });
  const httpServer = createServer((request, response) => {
    void (async () => {
      assert.equal(request.method, "POST");
      assert.equal(request.headers["content-type"], REQUEST_MEDIA_TYPE);
      let start = Buffer.alloc(0);
      let opened;
      const clearParts = [];
      for await (const part of request) {
        let bytes = Buffer.from(part);
        if (opened === undefined) {
          start = Buffer.concat([start, bytes]);
          const firstLength = serverEngine.streamStartLength(start);
          if (firstLength === undefined || start.byteLength < firstLength) continue;
          opened = serverEngine.preparseStream(start.subarray(0, firstLength))
            .authenticate(PSK).admit({ accepted: true });
          assert.equal(opened.head.path, "/stream");
          bytes = start.subarray(firstLength);
        }
        for (let offset = 0; offset < bytes.byteLength;) {
          const result = opened.feed(bytes.subarray(offset));
          assert.ok(result.consumed > 0);
          offset += result.consumed;
          if (result.record?.kind === "data") {
            clearParts.push(Buffer.from(result.record.block));
            sawData();
          }
        }
      }
      assert.ok(opened);
      assert.deepEqual(Buffer.concat(clearParts), Buffer.concat([
        Buffer.alloc(64 * 1024, 0x41), Buffer.from([0x42]),
      ]));
      const right = opened.finishEof();
      response.writeHead(200, { "content-type": RESPONSE_MEDIA_TYPE });
      response.end(right.protectResponse({ status: 200, body: new TextEncoder().encode("ok") }));
    })().catch((error) => response.destroy(error));
  });
  await new Promise((resolve) => httpServer.listen(0, "127.0.0.1", resolve));
  const address = httpServer.address();
  assert.ok(address && typeof address !== "string");
  const relay = `http://127.0.0.1:${address.port}/protected`;
  const hpkeFetch = createHpkeFetch({
    endpoint: "https://api.example.test/protected",
    key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
    psk: PSK, pskId: PSK_ID,
    fetch: (_input, init) => fetch(relay, init),
  });
  try {
    const pending = hpkeFetch("https://api.example.test/stream", {
      method: "POST", body: source, duplex: "half",
    });
    await within(Promise.race([
      dataOnWire,
      pending.then(() => { throw new Error("Fetch returned before the upload source ended"); }),
    ]), 5000);
    assert.equal(sourceEnded, false);
    releaseSource();
    assert.equal(await (await within(pending, 5000)).text(), "ok");
  } finally {
    releaseSource();
    hpkeFetch.close();
    serverEngine.close();
    httpServer.closeAllConnections();
    await new Promise((resolve) => httpServer.close(resolve));
  }
});

test("native Fetch uses Rust zstd body coding without HTTP content coding", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID);
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
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
    psk: PSK,
    pskId: PSK_ID,
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
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
    psk: PSK,
    pskId: PSK_ID,
    limits: { maxBodyLength: 4, maxRequestBytes: 4 },
    fetch: transport,
  });

  await assert.rejects(
    hpkeFetch("https://api.example.test/items", { method: "POST", body: "12345" }),
    (error) => error instanceof ProtocolError && error.code === "limit_exceeded",
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
          endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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
    ["maxRequestBytes", 4 * 1024 * 1024 * 1024 + 1],
  ]) {
    assert.throws(
      () =>
        createHpkeFetch({
          endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
          psk: PSK,
          pskId: PSK_ID,
          limits: { [name]: value },
        }),
      (error) => error instanceof ProtocolError && error.code === "invalid_configuration",
    );
  }
  for (const value of [0, Number.NaN, 1.5]) {
    assert.throws(
      () => createHpkeFetch({
        endpoint: "https://api.example.test/protected",
        key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
        psk: PSK, pskId: PSK_ID,
        limits: { maxRequestBytes: value },
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
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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
  const server = new Server(keys.privateKey, KEY_ID);
  const hpkeFetch = createHpkeFetch({
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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
      const large = opened.request.path === "/large";
      return new Response(opened.protectResponse(large
        ? { status: 200, body: new TextEncoder().encode("ab") }
        : { status: 204 }), {
        status: 200,
        headers: {
          "content-length": large ? "0" : "99999999",
          "content-type": RESPONSE_MEDIA_TYPE,
        },
      });
    },
  });

  const response = await hpkeFetch("https://api.example.test/health");
  assert.equal(response.status, 204);
  await assert.rejects(
    hpkeFetch("https://api.example.test/large"),
    (error) => error instanceof ProtocolError && error.code === "limit_exceeded",
  );
  hpkeFetch.close();
  server.close();
});

test("Fetch waits for outer EOF before returning a finite response", async () => {
  await initialize();
  const keys = generateKeyPair();
  const server = new Server(keys.privateKey, KEY_ID);
  let reachedEofRead;
  const eofRead = new Promise((resolve) => { reachedEofRead = resolve; });
  let releaseEof;
  const eofGate = new Promise((resolve) => { releaseEof = resolve; });
  const hpkeFetch = createHpkeFetch({
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
    psk: PSK,
    pskId: PSK_ID,
    fetch: async (input, init) => {
      const request = new Request(input, init);
      const opened = server
        .preparse(new Uint8Array(await request.arrayBuffer()))
        .authenticate(PSK)
        .admit({ accepted: true });
      const envelope = opened.protectResponse({ status: 200, body: new TextEncoder().encode("complete") });
      return new Response(new ReadableStream({
        start(controller) { controller.enqueue(envelope); },
        async pull(controller) {
          reachedEofRead();
          await eofGate;
          controller.close();
        },
      }, { highWaterMark: 0 }), {
        status: 200,
        headers: { "content-type": RESPONSE_MEDIA_TYPE },
      });
    },
  });
  let eofReleased = false;
  let settledBeforeEof = false;
  const pending = hpkeFetch("https://api.example.test/finite");
  void pending.then(
    () => { if (!eofReleased) settledBeforeEof = true; },
    () => { if (!eofReleased) settledBeforeEof = true; },
  );
  try {
    const first = await within(Promise.race([
      eofRead.then(() => "eof_read"),
      pending.then(() => "returned", () => "failed"),
    ]), 5000);
    assert.equal(first, "eof_read");
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(settledBeforeEof, false);
    eofReleased = true;
    releaseEof();
    const response = await within(pending, 5000);
    assert.equal(await response.text(), "complete");
    assert.equal(settledBeforeEof, false);
  } finally {
    releaseEof();
    hpkeFetch.close();
    server.close();
  }
});

test("Fetch adapter rejects malformed record bytes and cancels the body", async () => {
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
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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
    (error) => error instanceof ProtocolError && error.code === "malformed_envelope",
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
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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
    endpoint: "https://api.example.test/protected", key: { kind: "pin", publicKey: keys.publicKey, keyId: KEY_ID },
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

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  BINDING_ABI_VERSION,
  PACKAGE_VERSION,
  PROTOCOL_ID,
  Client,
  ProtocolError,
  Server,
  StateError,
  generateKeyPair,
  initialize,
  isInitialized,
} from "../dist/node.js";

const packageMetadata = JSON.parse(
  await readFile(new URL("../package.json", import.meta.url), "utf8"),
);

const KEY_ID = new TextEncoder().encode("primary-2026-09");
const PSK = new TextEncoder().encode("a 32-byte minimum test credential!");
const PSK_ID = new TextEncoder().encode("tenant-42");

test("Node loader and complete hpke-http transaction", async () => {
  await initialize();
  assert.equal(isInitialized(), true);
  assert.equal(PACKAGE_VERSION, packageMetadata.version);
  assert.equal(PROTOCOL_ID, "hpke-http/3");
  assert.equal(BINDING_ABI_VERSION, 7);

  const keys = generateKeyPair();
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID);
  const server = new Server(keys.privateKey, KEY_ID);
  const request = {
    method: "POST",
    authority: "api.example.test",
    path: "/v3/items?limit=2",
    headers: [{ name: "content-type", value: "application/json" }],
    body: new TextEncoder().encode('{"name":"Ada"}'),
  };

  const protectedRequest = client.protect(request);
  const preparsed = server.preparse(protectedRequest.envelope);
  assert.deepEqual(preparsed.pskId, PSK_ID);
  const authenticated = preparsed.authenticate(PSK);
  assert.equal(authenticated.replayId.byteLength, 32);
  assert.equal(Number.isSafeInteger(authenticated.retainUntilExclusive), true);
  assert.ok(authenticated.retainUntilExclusive > Math.floor(Date.now() / 1000));
  server.close();
  const opened = authenticated.admit({ accepted: true });
  assert.equal(opened.request.method, request.method);
  assert.equal(opened.request.authority, request.authority);
  assert.deepEqual(opened.request.body, request.body);

  const expectedResponse = {
    status: 201,
    headers: [{ name: "content-type", value: "application/json" }],
    body: new TextEncoder().encode('{"id":"item-1"}'),
  };
  const protectedResponse = opened.protectResponse(expectedResponse);
  const response = protectedRequest.openResponse(protectedResponse);
  assert.equal(response.status, expectedResponse.status);
  assert.deepEqual(response.headers, expectedResponse.headers);
  assert.deepEqual(response.body, expectedResponse.body);
  assert.throws(() => protectedRequest.openResponse(protectedResponse), StateError);

  client.close();
  server.close();
});

test("native request writer accepts uneven cuts and one v3 wire form", async () => {
  await initialize();
  const keys = generateKeyPair();
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID, {
    maxRequestBytes: 512 * 1024,
  });
  const server = new Server(keys.privateKey, KEY_ID, { maxRequestBytes: 512 * 1024 });
  const body = new Uint8Array(256 * 1024).fill(0x41);
  const writer = client.beginStream({
    method: "POST", authority: "api.example.test", path: "/stream",
    headers: [{ name: "content-length", value: String(body.byteLength) }],
  });
  const records = [writer.start];
  try {
    for (let offset = 0; offset < body.byteLength;) {
      const result = writer.push(body.subarray(offset, Math.min(offset + 7777, body.byteLength)));
      assert.ok(result.consumed > 0);
      offset += result.consumed;
      if (result.record !== undefined) records.push(result.record);
    }
    const finished = writer.finish();
    records.push(finished.end);
    const envelope = Buffer.concat(records);
    assert.ok(envelope.byteLength < body.byteLength);
    const opened = server.preparse(envelope).authenticate(PSK).admit({ accepted: true });
    assert.deepEqual(opened.request.body, body);
    const protectedResponse = opened.protectResponse({ status: 200, body: new TextEncoder().encode("ok") });
    const opener = finished.intoOpener();
    try {
      let offset = 0;
      while (offset < protectedResponse.byteLength) {
        const result = opener.feed(protectedResponse.subarray(offset));
        assert.ok(result.consumed > 0);
        offset += result.consumed;
      }
      assert.equal(new TextDecoder().decode(opener.finishEof().body), "ok");
    } finally {
      opener.close();
    }
  } finally {
    writer.close();
    client.close();
    server.close();
  }
});

test("staged TypeScript server reads a request beyond the one-shot body limit", async () => {
  await initialize();
  const keys = generateKeyPair();
  const limits = { maxBodyLength: 32, maxRequestBytes: 256 * 1024 };
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID, limits);
  const server = new Server(keys.privateKey, KEY_ID, limits);
  const body = new Uint8Array(128 * 1024 + 7).fill(0x61);
  const writer = client.beginStream({ method: "POST", authority: "api.example.test", path: "/large" });
  const records = [];
  try {
    for (let offset = 0; offset < body.byteLength;) {
      const result = writer.push(body.subarray(offset));
      offset += result.consumed;
      if (result.record !== undefined) records.push(result.record);
    }
    const finished = writer.finish();
    records.push(finished.end);
    assert.equal(server.streamStartLength(writer.start.subarray(0, 4)), undefined);
    assert.equal(server.streamStartLength(writer.start), writer.start.byteLength);
    assert.throws(() => server.preparse(Buffer.concat([writer.start, ...records])).authenticate(PSK),
      (error) => error instanceof ProtocolError && error.code === "limit_exceeded");
    const preparsed = server.preparseStream(writer.start);
    assert.deepEqual(preparsed.pskId, PSK_ID);
    const authenticated = preparsed.authenticate(PSK);
    assert.equal(authenticated.replayId.byteLength, 32);
    const opened = authenticated.admit({ accepted: true });
    assert.equal(opened.head.path, "/large");
    const clearParts = [];
    let endSeen = false;
    for (const record of records) {
      for (let offset = 0; offset < record.byteLength;) {
        const result = opened.feed(record.subarray(offset, offset + 17));
        assert.ok(result.consumed > 0);
        offset += result.consumed;
        if (result.record?.kind === "data") clearParts.push(result.record.block);
        if (result.record?.kind === "end") endSeen = true;
      }
    }
    assert.equal(endSeen, true);
    assert.deepEqual(Uint8Array.from(Buffer.concat(clearParts)), body);
    const right = opened.finishEof();
    const encrypted = right.protectResponse({ status: 200, body: new TextEncoder().encode("ok") });
    const opener = finished.intoOpener();
    try {
      let offset = 0;
      while (offset < encrypted.byteLength) {
        offset += opener.feed(encrypted.subarray(offset)).consumed;
      }
      assert.equal(new TextDecoder().decode(opener.finishEof().body), "ok");
    } finally { opener.close(); }
  } finally {
    writer.close();
    client.close();
    server.close();
  }
});

test("low-level finite response sealer handles empty and nonempty bodies", async () => {
  await initialize();
  for (const body of [new Uint8Array(), new TextEncoder().encode("finite body")]) {
    const keys = generateKeyPair();
    const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID);
    const server = new Server(keys.privateKey, KEY_ID);
    try {
      const protectedRequest = client.protect({
        method: "POST", authority: "api.example.test", path: "/finite",
      });
      const opened = server.preparse(protectedRequest.envelope).authenticate(PSK).admit({ accepted: true });
      const headers = [{ name: "content-type", value: "text/plain" }];
      const writer = opened.startResponse(200, headers);
      const data = writer.sealFiniteBody(body);
      assert.equal(data === undefined, body.byteLength === 0);
      const end = writer.finish();
      const envelope = new Uint8Array(writer.start.byteLength + (data?.byteLength ?? 0) + end.byteLength);
      envelope.set(writer.start);
      if (data !== undefined) envelope.set(data, writer.start.byteLength);
      envelope.set(end, writer.start.byteLength + (data?.byteLength ?? 0));
      assert.deepEqual(protectedRequest.openResponse(envelope), { status: 200, headers, body });
    } finally {
      client.close();
      server.close();
    }
  }
});

test("native zstd request and response coding preserves logical HTTP", async () => {
    await initialize();
    const keys = generateKeyPair();
    const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID);
    const server = new Server(keys.privateKey, KEY_ID);
    const body = new Uint8Array(16 * 1024).fill(0x41);
    const request = {
      method: "POST",
      authority: "api.example.test",
      path: "/compressed",
      headers: [{ name: "content-length", value: String(body.byteLength) }],
      body,
    };
    const protectedRequest = client.protect(request);
    assert.ok(protectedRequest.envelope.byteLength < body.byteLength);
    const opened = server.preparse(protectedRequest.envelope).authenticate(PSK).admit({ accepted: true });
    assert.deepEqual(opened.request.body, body);
    assert.deepEqual(opened.request.headers, request.headers);
    const responseBody = new Uint8Array(16 * 1024).fill(0x42);
    const response = {
      status: 200,
      headers: [{ name: "content-length", value: String(responseBody.byteLength) }],
      body: responseBody,
    };
    const envelope = opened.protectResponse(response);
    assert.ok(envelope.byteLength < responseBody.byteLength);
    assert.deepEqual(protectedRequest.openResponse(envelope), response);
    client.close();
    server.close();
});

test("empty request can receive a native zstd response", async () => {
  await initialize();
  const keys = generateKeyPair();
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID);
  const server = new Server(keys.privateKey, KEY_ID);
  const protectedRequest = client.protect({ method: "GET", authority: "api.example.test", path: "/" });
  const opened = server.preparse(protectedRequest.envelope).authenticate(PSK).admit({ accepted: true });
  assert.deepEqual(opened.request.body, new Uint8Array());
  const body = new Uint8Array(16 * 1024).fill(0x42);
  const envelope = opened.protectResponse({ status: 200, body });
  assert.ok(envelope.byteLength < body.byteLength);
  assert.deepEqual(protectedRequest.openResponse(envelope).body, body);
  client.close();
  server.close();
});

test("binding size guards reject before copying and preserve one-shot consumption", async () => {
  await initialize();
  const keys = generateKeyPair();
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID, { maxBodyLength: 1, maxRequestBytes: 1 });
  const server = new Server(keys.privateKey, KEY_ID, { maxBodyLength: 1, maxRequestBytes: 1 });
  const oversizedBody = new Uint8Array(2);
  const oversizedEnvelope = new Uint8Array(100_000);

  assert.throws(
    () => client.protect({ method: "POST", authority: "api.example.test", path: "/", body: oversizedBody }),
    (error) => error instanceof ProtocolError && error.code === "limit_exceeded",
  );
  assert.throws(
    () => server.preparse(oversizedEnvelope),
    (error) => error instanceof ProtocolError && error.code === "limit_exceeded",
  );

  const protectedRequest = client.protect({ method: "POST", authority: "api.example.test", path: "/" });
  const opened = server.preparse(protectedRequest.envelope).authenticate(PSK).admit({ accepted: true });
  assert.throws(
    () => opened.protectResponse({ status: 200, body: oversizedBody }),
    (error) => error instanceof ProtocolError && error.code === "limit_exceeded",
  );
  assert.equal(opened.responseConsumed, true);
  const largeServer = new Server(keys.privateKey, KEY_ID);
  const largeOpened = largeServer.preparse(protectedRequest.envelope).authenticate(PSK).admit({ accepted: true });
  const validLargeResponse = largeOpened.protectResponse({ status: 200, body: oversizedBody });
  assert.throws(
    () => protectedRequest.openResponse(validLargeResponse),
    (error) => error instanceof ProtocolError && error.code === "limit_exceeded",
  );
  assert.equal(protectedRequest.consumed, true);
  client.close();
  server.close();
  largeServer.close();
});

test("preparse keeps room for many request records", async () => {
  await initialize();
  const keys = generateKeyPair();
  const limits = { maxBodyLength: 10_000, maxRequestBytes: 10_000 };
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID, limits);
  const server = new Server(keys.privateKey, KEY_ID, limits);
  const protectedRequest = client.protect({ method: "POST", authority: "api.example.test", path: "/" });
  const envelope = Buffer.concat([protectedRequest.envelope, Buffer.alloc(100_000 - protectedRequest.envelope.byteLength)]);
  const preparsed = server.preparse(envelope);
  assert.deepEqual(preparsed.pskId, PSK_ID);
  preparsed.close();
  protectedRequest.close();
  client.close();
  server.close();
});

test("replay rejection stays a stable protocol error", async () => {
  await initialize();
  const keys = generateKeyPair();
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID);
  const server = new Server(keys.privateKey, KEY_ID);
  const protectedRequest = client.protect({
    method: "GET",
    authority: "api.example.test",
    path: "/health",
  });
  const authenticated = server.preparse(protectedRequest.envelope).authenticate(PSK);
  assert.throws(
    () => authenticated.admit({ accepted: false }),
    (error) => error instanceof ProtocolError && error.code === "replay_rejected",
  );
  protectedRequest.close();
  client.close();
  server.close();
});

test("closing a server revokes pending pre-authentication stages", async () => {
  await initialize();
  const keys = generateKeyPair();
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID);
  const server = new Server(keys.privateKey, KEY_ID);
  const protectedRequest = client.protect({
    method: "GET",
    authority: "api.example.test",
    path: "/pending",
  });
  const preparsed = server.preparse(protectedRequest.envelope);
  assert.equal(preparsed.consumed, false);

  server.close();

  assert.equal(preparsed.consumed, true);
  assert.throws(() => preparsed.authenticate(PSK), StateError);
  preparsed.close();
  protectedRequest.close();
  client.close();
});

test("replay admission rechecks the authenticated deadline", async () => {
  await initialize();
  const keys = generateKeyPair();
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID);
  const server = new Server(keys.privateKey, KEY_ID);
  const protectedRequest = client.protect({
    method: "GET",
    authority: "api.example.test",
    path: "/delayed",
  });
  const authenticated = server.preparse(protectedRequest.envelope).authenticate(PSK);
  const originalDateNow = Date.now;
  Date.now = () => authenticated.retainUntilExclusive * 1000;
  try {
    assert.throws(
      () => authenticated.admit({ accepted: true }),
      (error) => error instanceof ProtocolError && error.code === "invalid_request_time",
    );
  } finally {
    Date.now = originalDateNow;
  }
  protectedRequest.close();
  client.close();
  server.close();
});

test("response statuses are validated before WebAssembly coercion", async () => {
  await initialize();
  const keys = generateKeyPair();
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID);
  const server = new Server(keys.privateKey, KEY_ID);
  const protectedRequest = client.protect({
    method: "GET",
    authority: "api.example.test",
    path: "/status",
  });
  const opened = server
    .preparse(protectedRequest.envelope)
    .authenticate(PSK)
    .admit({ accepted: true });

  for (const status of [Number.NaN, Number.POSITIVE_INFINITY, 199, 600, 201.5, 65_736]) {
    assert.throws(
      () => opened.protectResponse({ status }),
      (error) => error instanceof ProtocolError && error.code === "invalid_configuration",
    );
  }
  const protectedResponse = opened.protectResponse({ status: 204 });
  assert.equal(protectedRequest.openResponse(protectedResponse).status, 204);

  client.close();
  server.close();
});

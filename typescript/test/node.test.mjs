import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  BINDING_ABI_VERSION,
  PACKAGE_VERSION,
  PROTOCOL_ID,
  Client,
  AuthenticatedRequest,
  OpenedRequest,
  PreparsedRequest,
  ProtectedRequest,
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

test("continuation constructors are not exposed as runtime factories", () => {
  for (const continuation of [
    ProtectedRequest,
    PreparsedRequest,
    AuthenticatedRequest,
    OpenedRequest,
  ]) {
    assert.equal(Object.hasOwn(continuation, "fromNative"), false);
  }
});

test("Node loader and complete hpke-http transaction", async () => {
  await initialize();
  assert.equal(isInitialized(), true);
  assert.equal(PACKAGE_VERSION, packageMetadata.version);
  assert.equal(PROTOCOL_ID, "hpke-http/1");
  assert.equal(BINDING_ABI_VERSION, 1);

  const keys = generateKeyPair();
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID);
  const server = new Server(keys.privateKey, KEY_ID);
  const request = {
    method: "POST",
    authority: "api.example.test",
    path: "/v1/items?limit=2",
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

for (const coding of ["gzip", "zstd"]) {
  test(`opt-in ${coding} body compression preserves logical HTTP`, async () => {
    await initialize();
    const keys = generateKeyPair();
    const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID, {}, coding);
    const server = new Server(keys.privateKey, KEY_ID, {}, true);
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
}

test("identity-only server rejects a client compression extension", async () => {
  await initialize();
  const keys = generateKeyPair();
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID, {}, "gzip");
  const server = new Server(keys.privateKey, KEY_ID);
  const protectedRequest = client.protect({ method: "GET", authority: "api.example.test", path: "/" });
  assert.throws(
    () => server.preparse(protectedRequest.envelope).authenticate(PSK),
    (error) => error instanceof ProtocolError && error.code === "malformed_envelope",
  );
  protectedRequest.close();
  client.close();
  server.close();
});

test("binding size guards reject before copying and preserve one-shot consumption", async () => {
  await initialize();
  const keys = generateKeyPair();
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID, { maxBodyLength: 1 });
  const server = new Server(keys.privateKey, KEY_ID, { maxBodyLength: 1 });
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
  assert.throws(
    () => protectedRequest.openResponse(oversizedEnvelope),
    (error) => error instanceof ProtocolError && error.code === "limit_exceeded",
  );
  assert.equal(protectedRequest.consumed, true);
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

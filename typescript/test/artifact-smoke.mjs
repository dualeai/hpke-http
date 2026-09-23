import assert from "node:assert/strict";

const binding = await import("@dualeai/hpke-http/node");
await binding.initialize();
assert.equal(binding.isInitialized(), true);
assert.equal(binding.PROTOCOL_ID, "hpke-http/1");
assert.equal(binding.BINDING_ABI_VERSION, 1);
if (process.env.EXPECTED_VERSION !== undefined) {
  assert.equal(binding.PACKAGE_VERSION, process.env.EXPECTED_VERSION);
}

const text = new TextEncoder();
const keyId = text.encode("artifact-key");
const psk = text.encode("a 32-byte minimum artifact credential");
const pskId = text.encode("artifact-tenant");
const keys = binding.generateKeyPair();
const client = new binding.Client(keys.publicKey, keyId, psk, pskId);
const server = new binding.Server(keys.privateKey, keyId);
try {
  const protectedRequest = client.protect({
    method: "GET",
    authority: "artifact.example.test",
    path: "/smoke",
  });
  const opened = server
    .preparse(protectedRequest.envelope)
    .authenticate(psk)
    .admit({ accepted: true });
  const expectedBody = text.encode("artifact-ok");
  const protectedResponse = opened.protectResponse({ status: 200, body: expectedBody });
  const response = protectedRequest.openResponse(protectedResponse);
  assert.equal(response.status, 200);
  assert.deepEqual(response.body, expectedBody);
} finally {
  client.close();
  server.close();
}

const body = text.encode("artifact-compression-".repeat(1024));
const compressedClient = new binding.Client(keys.publicKey, keyId, psk, pskId, {}, "zstd");
const compressedServer = new binding.Server(keys.privateKey, keyId, {}, true);
try {
  const protectedRequest = compressedClient.protect({
    method: "POST",
    authority: "artifact.example.test",
    path: "/compressed",
    body,
  });
  assert.ok(protectedRequest.envelope.byteLength < body.byteLength);
  const opened = compressedServer
    .preparse(protectedRequest.envelope)
    .authenticate(psk)
    .admit({ accepted: true });
  assert.deepEqual(opened.request.body, body);
  const encrypted = opened.protectResponse({ status: 200, body });
  assert.ok(encrypted.byteLength < body.byteLength);
  assert.deepEqual(protectedRequest.openResponse(encrypted).body, body);
} finally {
  compressedClient.close();
  compressedServer.close();
}

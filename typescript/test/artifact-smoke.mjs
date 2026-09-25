import assert from "node:assert/strict";

const binding = await import("@dualeai/hpke-http/node");
const browser = await import("@dualeai/hpke-http/browser");
await binding.initialize();
assert.equal(binding.isInitialized(), true);
assert.equal(binding.PROTOCOL_ID, "hpke-http/3");
assert.equal(binding.BINDING_ABI_VERSION, 8);
assert.equal(typeof browser.createHpkeFetch, "function");
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
const compressedClient = new binding.Client(keys.publicKey, keyId, psk, pskId);
const compressedServer = new binding.Server(keys.privateKey, keyId);
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

const liveClient = new binding.Client(keys.publicKey, keyId, psk, pskId);
const liveServer = new binding.Server(keys.privateKey, keyId);
try {
  const protectedRequest = liveClient.protect({
    method: "GET", authority: "artifact.example.test", path: "/events",
  });
  const opened = liveServer.preparse(protectedRequest.envelope).authenticate(psk).admit({ accepted: true });
  const writer = opened.startResponse(200, [{ name: "content-type", value: "text/event-stream" }]);
  const reader = protectedRequest.intoOpener();
  try {
    const start = reader.feed(writer.start);
    assert.equal(start.consumed, writer.start.byteLength);
    assert.equal(start.record?.kind, "start");
    assert.equal(start.record?.mode, "sse");
    const block = text.encode(": ready\n\n");
    const first = writer.sealSseBlock(block);
    const checked = reader.feed(first);
    assert.equal(checked.consumed, first.byteLength);
    assert.equal(checked.record?.kind, "sse_data");
    assert.deepEqual(checked.record?.block, block);
    // The writer has not made END, but the clear block is already checked.
    const end = writer.finish();
    assert.equal(reader.feed(end).record?.kind, "end");
    assert.equal(reader.finishEof(), undefined);
  } finally {
    writer.close();
    reader.close();
  }
} finally {
  liveClient.close();
  liveServer.close();
}

const discoveryServer = new binding.Server(keys.privateKey, keyId);
const calls = [];
const endpoint = "https://artifact.example.test/protected";
const source = new binding.DiscoveredEndpoint(endpoint, {
  fetch: async (input, init) => {
    const request = new Request(input, init);
    calls.push(request.method);
    if (request.method === "GET") {
      const record = new Uint8Array([0x48, 0x48, 0x4b, 0x44, 2, keyId.byteLength, ...keyId, ...keys.publicKey, 0, 0, 0, 60]);
      return new Response(record, { status: 200, headers: { "content-type": "application/octet-stream" } });
    }
    const opened = discoveryServer.preparse(new Uint8Array(await request.arrayBuffer()))
      .authenticate(psk).admit({ accepted: true });
    const body = opened.protectResponse({ status: 200, body: text.encode("artifact-discovery-ok") });
    return new Response(body, { status: 200, headers: { "content-type": binding.RESPONSE_MEDIA_TYPE } });
  },
});
try {
  for (let i = 0; i < 2; i += 1) {
    const discovery = binding.createHpkeFetch({ key: source, psk, pskId });
    try {
      const response = await discovery("https://artifact.example.test/items");
      assert.equal(await response.text(), "artifact-discovery-ok");
    } finally { discovery.close(); }
  }
  assert.deepEqual(calls, ["GET", "POST", "POST"]);
} finally {
  source.close();
  discoveryServer.close();
}

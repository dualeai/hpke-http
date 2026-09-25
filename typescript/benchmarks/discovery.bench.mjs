// Measure the Fetch adapter with an in-memory transport and no network delay.

import assert from "node:assert/strict";
import { afterAll, beforeAll, bench, describe } from "vitest";

import { DiscoveredEndpoint, Server, createHpkeFetch, generateKeyPair, initialize, RESPONSE_MEDIA_TYPE } from "../dist/node.js";

const encoder = new TextEncoder();
const keyId = encoder.encode("benchmark-key");
const psk = encoder.encode("a 32-byte minimum benchmark credential");
const pskId = encoder.encode("benchmark-tenant");
const endpoint = "https://api.example.test/protected";
const target = "https://api.example.test/items";
let server;
let pinned;
let discovered;
let source;

beforeAll(async () => {
  await initialize();
  const keys = generateKeyPair();
  server = new Server(keys.privateKey, keyId);
  const record = new Uint8Array([0x48, 0x48, 0x4b, 0x44, 2, keyId.length, ...keyId, ...keys.publicKey, 0, 0, 0, 60]);
  const transport = async (input, init) => {
    const request = new Request(input, init);
    if (request.method === "GET") {
      return new Response(record, { headers: { "content-type": "application/octet-stream" } });
    }
    const opened = server.preparse(new Uint8Array(await request.arrayBuffer()))
      .authenticate(psk).admit({ accepted: true });
    return new Response(opened.protectResponse({ status: 200, body: encoder.encode("ok") }), {
      headers: { "content-type": RESPONSE_MEDIA_TYPE },
    });
  };
  pinned = createHpkeFetch({ endpoint, key: { kind: "pin", publicKey: keys.publicKey, keyId }, psk, pskId, fetch: transport });
  source = new DiscoveredEndpoint(endpoint, { fetch: transport });
  discovered = createHpkeFetch({ key: source, psk, pskId });
});

afterAll(() => {
  pinned.close();
  discovered.close();
  source.close();
  server.close();
});

for (const mode of ["pin", "discover"]) {
  describe(`fetch-adapter/${mode}`, () => {
    bench("roundtrip", async () => {
      const response = await (mode === "pin" ? pinned : discovered)(target);
      assert.equal(await response.text(), "ok");
    });
  });
}

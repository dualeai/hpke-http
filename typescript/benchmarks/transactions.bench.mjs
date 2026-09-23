// CodSpeed measures public Node/WASM transactions without transport or setup.

import assert from "node:assert/strict";

import { afterAll, beforeAll, bench, describe } from "vitest";

import { Client, Server, generateKeyPair, initialize } from "../dist/node.js";

const encoder = new TextEncoder();
const keyId = encoder.encode("benchmark-key");
const psk = encoder.encode("a 32-byte minimum benchmark credential");
const pskId = encoder.encode("benchmark-tenant");
let client;
let server;

beforeAll(async () => {
  await initialize();
  const keys = generateKeyPair();
  client = new Client(keys.publicKey, keyId, psk, pskId);
  server = new Server(keys.privateKey, keyId);
});

afterAll(() => {
  client.close();
  server.close();
});

for (const [label, size] of [["empty", 0], ["1KiB", 1024], ["1MiB", 1024 * 1024], ["8MiB", 8 * 1024 * 1024]]) {
  describe(`roundtrip/${label}`, () => {
    const request = {
      method: "POST",
      authority: "api.example.test",
      path: "/benchmark",
      body: new Uint8Array(size).fill(0x42),
    };
    const response = { status: 200, body: new Uint8Array(size).fill(0x43) };

    bench("protect-authenticate-response-open", () => {
      const protectedRequest = client.protect(request);
      const authenticated = server.preparse(protectedRequest.envelope).authenticate(psk);
      const opened = authenticated.admit({ accepted: true });
      const openedResponse = protectedRequest.openResponse(opened.protectResponse(response));
      assert.equal(openedResponse.body.byteLength, size);
    });
  });
}

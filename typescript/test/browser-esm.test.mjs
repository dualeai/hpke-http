import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  Client,
  Server,
  generateKeyPair,
  initialize,
} from "../dist/browser.js";

const KEY_ID = new TextEncoder().encode("primary-2026-09");
const PSK = new TextEncoder().encode("a 32-byte minimum test credential!");
const PSK_ID = new TextEncoder().encode("tenant-42");

test("browser ESM glue accepts explicit WASM bytes under Node", async () => {
  const wasm = await readFile(
    new URL("../_wasm/browser/hpke_http_wasm_bg.wasm", import.meta.url),
  );
  await initialize(wasm);

  const keys = generateKeyPair();
  const client = new Client(keys.publicKey, KEY_ID, PSK, PSK_ID);
  const server = new Server(keys.privateKey, KEY_ID);
  const protectedRequest = client.protect({
    method: "GET",
    authority: "api.example.test",
    path: "/health",
  });
  const opened = server
    .preparse(protectedRequest.envelope)
    .authenticate(PSK)
    .admit({ accepted: true });
  assert.equal(opened.request.path, "/health");

  const protectedResponse = opened.protectResponse({ status: 204 });
  const response = protectedRequest.openResponse(protectedResponse);
  assert.equal(response.status, 204);
  assert.equal(response.body?.byteLength, 0);

  client.close();
  server.close();
});

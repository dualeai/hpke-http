// CodSpeed measures public Node/WASM transactions without transport or setup.

import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";

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

for (const [label, size, random] of [
  ["empty", 0, false], ["1KiB", 1024, false], ["1MiB", 1024 * 1024, false],
  ["8MiB", 8 * 1024 * 1024, false], ["1MiB-random", 1024 * 1024, true],
  ["8MiB-random", 8 * 1024 * 1024, true],
]) {
  describe(`roundtrip/${label}`, () => {
    const request = {
      method: "POST",
      authority: "api.example.test",
      path: "/benchmark",
      body: random ? randomBytes(size) : new Uint8Array(size).fill(0x42),
    };
    const response = { status: 200, body: random ? randomBytes(size) : new Uint8Array(size).fill(0x43) };

    bench("protect-authenticate-response-open", () => {
      const protectedRequest = client.protect(request);
      const authenticated = server.preparse(protectedRequest.envelope).authenticate(psk);
      const opened = authenticated.admit({ accepted: true });
      const openedResponse = protectedRequest.openResponse(opened.protectResponse(response));
      assert.equal(openedResponse.body.byteLength, size);
    });
  });
}

for (const [label, random] of [["repeated", false], ["random", true]]) {
  describe(`upload-stream/1MiB/${label}`, () => {
    const head = { method: "POST", authority: "api.example.test", path: "/upload" };
    const body = random ? randomBytes(1024 * 1024) : new Uint8Array(1024 * 1024).fill(0x42);

    bench("seal", () => {
      const writer = client.beginStream(head);
      let wireBytes = writer.start.byteLength;
      for (let offset = 0; offset < body.byteLength;) {
        const { consumed, record } = writer.push(body.subarray(offset, offset + 64 * 1024));
        offset += consumed;
        wireBytes += record?.byteLength ?? 0;
      }
      const finished = writer.finish();
      wireBytes += finished.end.byteLength;
      finished.close();
      assert.ok(wireBytes > 0);
    });
  });
}

for (const [label, payloadSize, count] of [["100-small-blocks", 1, 100], ["one-large-block", 1024 * 1024, 1]]) {
  describe(`sse/${label}`, () => {
    const request = { method: "GET", authority: "api.example.test", path: "/events" };
    const block = encoder.encode(`data:${"x".repeat(payloadSize)}\n\n`);
    const headers = [{ name: "content-type", value: "text/event-stream" }];

    bench("protect-authenticate-seal-open", () => {
      const protectedRequest = client.protect(request);
      const opened = server.preparse(protectedRequest.envelope).authenticate(psk).admit({ accepted: true });
      const writer = opened.startResponse(200, headers);
      const reader = protectedRequest.intoOpener();
      try {
        assert.equal(reader.feed(writer.start).record?.kind, "start");
        let total = 0;
        for (let index = 0; index < count; index += 1) {
          const record = reader.feed(writer.sealSseBlock(block)).record;
          assert.equal(record?.kind, "sse_data");
          total += record.block.byteLength;
        }
        assert.equal(reader.feed(writer.finish()).record?.kind, "end");
        assert.equal(reader.finishEof(), undefined);
        assert.equal(total, block.byteLength * count);
      } finally {
        writer.close();
        reader.close();
      }
    });
  });
}

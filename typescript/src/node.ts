import { createRequire } from "node:module";

import { InitializationError, installNative } from "./index.js";
import type { NativeModule } from "./native.js";

export {
  BINDING_ABI_VERSION,
  PACKAGE_VERSION,
  PROTOCOL_ID,
  AuthenticatedRequest,
  AuthenticatedStreamRequest,
  Client,
  InitializationError,
  OpenedRequest,
  OpenedStreamRequest,
  PreparsedRequest,
  PreparsedStreamRequest,
  ProtectedRequest,
  ProtocolError,
  ResponseOpener,
  ResponseSealer,
  Server,
  StateError,
  StreamFinishedRequest,
  StreamRequestSealer,
  StreamResponseRight,
  generateKeyPair,
  isInitialized,
} from "./index.js";
export type { CheckedRecord, CheckedRequestRecord, Header, KeyPair, Limits, Method, Request, RequestHead, Response } from "./index.js";
export {
  REQUEST_MEDIA_TYPE,
  RESPONSE_MEDIA_TYPE,
  DiscoveredEndpoint,
  FetchTransportError,
  createHpkeFetch,
} from "./fetch.js";
export type {
  DiscoveredKeyLease,
  FetchTransport,
  FetchTransportErrorCode,
  HpkeFetch,
  HpkeFetchConfiguration,
  HpkeKeySource,
} from "./fetch.js";

const require = createRequire(import.meta.url);
let initialization: Promise<void> | undefined;

/**
 * Load and verify the Node 24 native WASM glue exactly once.
 *
 * A failed attempt can be retried after its returned promise rejects.
 */
export function initialize(): Promise<void> {
  initialization ??= Promise.resolve().then(() => {
    try {
      const bindings = require("../_wasm/node/hpke_http_wasm.cjs") as NativeModule;
      installNative(bindings);
    } catch (error: unknown) {
      initialization = undefined;
      if (error instanceof InitializationError) {
        throw error;
      }
      throw new InitializationError(
        `could not initialize hpke-http Node WASM: ${String(error)}`,
      );
    }
  });
  return initialization;
}

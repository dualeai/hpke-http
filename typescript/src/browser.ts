import { InitializationError, installNative } from "./index.js";
import type { NativeModule } from "./native.js";

export {
  BINDING_ABI_VERSION,
  PACKAGE_VERSION,
  PROTOCOL_ID,
  AuthenticatedRequest,
  Client,
  InitializationError,
  OpenedRequest,
  PreparsedRequest,
  ProtectedRequest,
  ProtocolError,
  Server,
  StateError,
  generateKeyPair,
  isInitialized,
} from "./index.js";
export type { CompressionCoding, Header, KeyPair, Limits, Method, Request, Response } from "./index.js";
export {
  REQUEST_MEDIA_TYPE,
  RESPONSE_MEDIA_TYPE,
  FetchTransportError,
  createHpkeFetch,
} from "./fetch.js";
export type {
  FetchTransport,
  FetchTransportErrorCode,
  HpkeFetch,
  HpkeFetchConfiguration,
} from "./fetch.js";

/** Explicit browser WASM input accepted by `initialize`. */
export type BrowserWasmInput =
  | RequestInfo
  | URL
  | Response
  | BufferSource
  | WebAssembly.Module;

let initialization: Promise<void> | undefined;
const wasmModulePath = "../_wasm/browser/hpke_http_wasm.js";

/**
 * Load and verify the browser ESM WASM module exactly once.
 *
 * Pass an explicit URL, response, byte buffer, or compiled module when the
 * default adjacent-WASM fetch does not fit the host's asset or CSP policy.
 * The runtime must provide WebAssembly, Fetch, Web Crypto, and Web Streams.
 * A failed attempt can be retried after its returned promise rejects.
 */
export function initialize(input?: BrowserWasmInput): Promise<void> {
  initialization ??= initializeOnce(input).catch((error: unknown) => {
    initialization = undefined;
    if (error instanceof InitializationError) {
      throw error;
    }
    throw new InitializationError(
      `could not initialize hpke-http browser WASM; check the WASM asset URL and CSP: ${String(error)}`,
    );
  });
  return initialization;
}

async function initializeOnce(input?: BrowserWasmInput): Promise<void> {
  const bindings = (await import(wasmModulePath)) as unknown as NativeModule & {
    default(options?: { module_or_path: BrowserWasmInput }): Promise<unknown>;
  };
  if (input === undefined) {
    await bindings.default();
  } else {
    await bindings.default({ module_or_path: input });
  }
  installNative(bindings);
}

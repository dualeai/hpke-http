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

/** Explicit browser WASM input accepted by `initialize`. */
export type BrowserWasmInput =
  | RequestInfo
  | URL
  | Response
  | BufferSource
  | WebAssembly.Module;

let initialization: Promise<void> | undefined;

/**
 * Load and verify the browser ESM WASM module exactly once.
 *
 * The first call selects the WASM input. Later calls share its result. After
 * a failed call, a new call can try again. The loader imports its JS glue only
 * when called, before it uses the input. A site CSP must allow that module,
 * any WASM fetch, and WASM instantiation (`'wasm-unsafe-eval'` in `script-src`).
 * The runtime must provide WebAssembly, Fetch, Web Crypto, and Web Streams.
 */
export function initialize(input?: BrowserWasmInput): Promise<void> {
  initialization ??= initializeOnce(input).catch((error: unknown) => {
    initialization = undefined;
    if (error instanceof InitializationError) {
      throw error;
    }
    throw new InitializationError(
      `could not initialize hpke-http browser WASM; check the JS glue, WASM asset URL, and CSP: ${String(error)}`,
    );
  });
  return initialization;
}

async function initializeOnce(input?: BrowserWasmInput): Promise<void> {
  const bindings = (await import("../_wasm/browser/hpke_http_wasm.js")) as unknown as NativeModule & {
    default(options?: { module_or_path: BrowserWasmInput }): Promise<unknown>;
  };
  if (input === undefined) {
    await bindings.default();
  } else {
    await bindings.default({ module_or_path: input });
  }
  installNative(bindings);
}

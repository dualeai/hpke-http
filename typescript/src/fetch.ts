import { Client, ProtocolError, ResponseOpener, StateError, normalizeLimits } from "./index.js";
import type { CheckedRecord, Header, Limits, Method, ProtectedRequest, Request as PlaintextRequest, RequestHead, Response as PlaintextResponse, StreamFinishedRequest, StreamRequestSealer } from "./index.js";
import { isMethod } from "./method.js";

/** Media type of an outer protected request envelope. */
export const REQUEST_MEDIA_TYPE = "message/hpke-http-request";

/** Media type of an outer protected response envelope. */
export const RESPONSE_MEDIA_TYPE = "message/hpke-http-response";

const DEFAULT_MAX_BODY_LENGTH = 8 * 1024 * 1024;
const MAX_BUFFER_CHUNK = 64 * 1024;
const MAX_KEY_RECORD = 297;
const KEY_MEDIA_TYPE = "application/octet-stream";

const NON_FORWARDABLE_REQUEST_HEADERS = new Set([
  "accept-encoding",
  "connection",
  "content-length",
  "expect",
  "host",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authentication-info",
  "proxy-authorization",
  "proxy-connection",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

const FETCH_HIDDEN_RESPONSE_HEADERS = new Set(["set-cookie", "set-cookie2"]);

const BODYLESS_RESPONSE_STATUSES = new Set([204, 205, 304]);

/** Fetch-compatible outer transport accepted for dependency injection. */
export type FetchTransport = typeof globalThis.fetch;

/**
 * Stable failures produced by the Fetch boundary.
 *
 * Target and request content-coding failures occur before an outer request.
 * `request_too_large` comes from the bounded byte-body path. A streamed source
 * can reach the native request limit during POST and raise `limit_exceeded`.
 * Network and `outer_*` failures occur during the envelope
 * exchange. `discovery_expired` means the key lease ended before POST START.
 * `inner_content_encoding` can also describe an authenticated
 * response; `invalid_inner_response` describes a response that Web Fetch
 * cannot represent.
 */
export type FetchTransportErrorCode =
  | "invalid_target"
  | "request_too_large"
  | "network_error"
  | "outer_status"
  | "outer_content_type"
  | "outer_content_encoding"
  | "inner_content_encoding"
  | "invalid_inner_response"
  | "discovery_network"
  | "discovery_status"
  | "discovery_response"
  | "discovery_expired";

/** A transport or logical-message error from the Fetch adapter. */
export class FetchTransportError extends Error {
  public readonly code: FetchTransportErrorCode;
  public readonly statusCode?: number;

  public constructor(code: FetchTransportErrorCode, message: string, options?: ErrorOptions, statusCode?: number) {
    super(message, options);
    this.name = "FetchTransportError";
    this.code = code;
    if (statusCode !== undefined) { this.statusCode = statusCode; }
  }
}

/** One discovered key with a service-set lease that starts before GET. */
export interface DiscoveredKeyLease {
  /** A copy of the public key ID. */
  readonly keyId: Uint8Array;
  /** A copy of the X25519 public key. */
  readonly publicKey: Uint8Array;
  /** Whether the service-set lease has time left at this call. */
  valid(): boolean;
}

class KeyLease implements DiscoveredKeyLease {
  readonly #keyId: Uint8Array;
  readonly #publicKey: Uint8Array;
  readonly #useForS: number;
  readonly #startedMonotonic: number;
  readonly #startedWall: number;
  #ageHighWater = 0;

  public constructor(
    record: { readonly keyId: Uint8Array; readonly publicKey: Uint8Array; readonly useForS: number },
    startedMonotonic: number,
    startedWall: number,
  ) {
    this.#keyId = Uint8Array.from(record.keyId);
    this.#publicKey = Uint8Array.from(record.publicKey);
    this.#useForS = record.useForS;
    this.#startedMonotonic = startedMonotonic;
    this.#startedWall = startedWall;
  }

  /** Return a copy of the public key ID. */
  public get keyId(): Uint8Array { return Uint8Array.from(this.#keyId); }
  /** Return a copy of the X25519 public key. */
  public get publicKey(): Uint8Array { return Uint8Array.from(this.#publicKey); }

  /** Test whether the lease has time left at this call. */
  public valid(): boolean {
    this.#ageHighWater = Math.max(
      this.#ageHighWater,
      Math.max(0, performance.now() - this.#startedMonotonic) / 1000,
      Math.max(0, Date.now() - this.#startedWall) / 1000,
    );
    return this.#ageHighWater < this.#useForS;
  }
}

/** One HTTPS endpoint, trusted Fetch function, and shared key lease. */
export class DiscoveredEndpoint {
  private readonly endpointUrl: URL;
  private readonly fetchTransport: FetchTransport;
  private readonly getTimeoutMs: number;
  private key: KeyLease | undefined;
  private pending: Promise<KeyLease> | undefined;
  private controller: AbortController | undefined;
  private closed = false;

  public constructor(endpoint: string, options: { readonly fetch?: FetchTransport; readonly getTimeoutMs?: number } = {}) {
    this.endpointUrl = parseKeyEndpoint(endpoint);
    this.fetchTransport = options.fetch ?? globalThis.fetch;
    if (typeof this.fetchTransport !== "function") {
      throw new FetchTransportError("network_error", "native Fetch is unavailable; provide options.fetch");
    }
    this.getTimeoutMs = options.getTimeoutMs ?? 10_000;
    if (!Number.isFinite(this.getTimeoutMs) || this.getTimeoutMs <= 0) {
      throw new RangeError("key GET timeout must be positive");
    }
  }

  /** Return a copy of the full HTTPS endpoint URL. */
  public get endpoint(): URL { return new URL(this.endpointUrl.href); }

  /** Return the trusted Fetch function bound to this source. */
  public get transport(): FetchTransport { return this.fetchTransport; }

  /** Get a lease and share a needed GET with other callers.
   *
   * Aborting `signal` stops this caller's wait. Other callers can still use
   * the GET. The lease can end before this caller starts a POST.
   */
  public async getKey(signal: AbortSignal): Promise<DiscoveredKeyLease> {
    if (this.closed) { throw new StateError(); }
    throwIfAborted(signal);
    if (this.key !== undefined && this.key.valid()) { return this.key; }
    if (this.pending === undefined) {
      const controller = new AbortController();
      this.controller = controller;
      const startedMonotonic = performance.now();
      const startedWall = Date.now();
      const timeout = setTimeout(() => controller.abort(new Error("key GET timed out")), this.getTimeoutMs);
      const pending = fetchKey(this.fetchTransport, this.endpointUrl, controller.signal).then((record): KeyLease => {
        if (this.closed) { throw new StateError(); }
        const key = new KeyLease(record, startedMonotonic, startedWall);
        if (!key.valid()) {
          throw new FetchTransportError("discovery_response", "key GET consumed its lifetime");
        }
        this.key = key;
        return key;
      }).finally(() => {
        clearTimeout(timeout);
        if (this.pending === pending) {
          this.pending = undefined;
          this.controller = undefined;
        }
      });
      this.pending = pending;
      void pending.catch(() => undefined);
    }
    return await waitForCaller(this.pending, signal);
  }

  /**
   * Stop new key lookups, clear the stored key, and abort a pending GET.
   *
   * This call is idempotent. Close clients that use this source separately.
   */
  public close(): void {
    if (this.closed) { return; }
    this.closed = true;
    this.key = undefined;
    this.controller?.abort(new StateError());
  }
}

function waitForCaller<T>(pending: Promise<T>, signal: AbortSignal): Promise<T> {
  throwIfAborted(signal);
  return new Promise<T>((resolve, reject) => {
    const onAbort = (): void => { signal.removeEventListener("abort", onAbort); reject(abortReason(signal)); };
    signal.addEventListener("abort", onAbort, { once: true });
    void pending.then(
      (value) => { signal.removeEventListener("abort", onAbort); resolve(value); },
      (error: unknown) => { signal.removeEventListener("abort", onAbort); reject(error); },
    );
  });
}

/** Share one discovered key or use one fixed key with no GET. */
export type HpkeKeySource =
  | DiscoveredEndpoint
  | { readonly kind: "pin"; readonly publicKey: Uint8Array; readonly keyId: Uint8Array };

interface HpkeFetchCommonConfiguration {
  /**
   * Logical HTTPS origin; defaults to the endpoint origin.
   * No nonroot path, query, fragment, or credentials.
   */
  readonly targetOrigin?: string;

  /**
   * PSK of at least 32 bytes with at least 32 bytes of entropy.
   * The adapter copies but cannot clear this array.
   */
  readonly psk: Uint8Array;

  /** Non-empty public PSK identifier, at most 255 bytes and not equal to `psk`. */
  readonly pskId: Uint8Array;

  /** Optional message limits. Omitted fields use the shared engine defaults. */
  readonly limits?: Limits;

}

/** Use a shared endpoint source or one fixed key with its full HTTPS endpoint. */
export type HpkeFetchConfiguration = HpkeFetchCommonConfiguration & (
  | { readonly key: DiscoveredEndpoint; readonly endpoint?: never; readonly fetch?: never }
  | {
      readonly key: Exclude<HpkeKeySource, DiscoveredEndpoint>;
      readonly endpoint: string;
      /** Inject a Fetch-compatible transport. The default is `globalThis.fetch`. */
      readonly fetch?: FetchTransport;
    }
);

/** A Fetch-shaped client with checked live SSE and finite replies. */
export interface HpkeFetch {
  /**
   * Protect one logical request. Finite replies pass END and outer EOF before
   * return. SSE returns after checked START. Each body block passes its own
   * check before delivery. A bad record, transport failure, or EOF before END
   * makes a later body read fail. Normal completion needs END and outer EOF.
   *
   * `FetchTransportError` reports adapter failures, including request body
   * size. `ProtocolError` reports protocol failures, including response record
   * limits. Before return these reject the call; after return they fail a body
   * read.
   */
  (input: RequestInfo | URL, init?: RequestInit): Promise<globalThis.Response>;

  /** Release native credential copies. This operation is idempotent. */
  close(): void;
}

/**
 * Create an HPKE-aware function backed by the runtime's native Fetch
 * and the shared Rust/WASM protocol engine.
 *
 * The returned function uses the URL, method, headers, body, and signal from a
 * Fetch request input. A shared source gets one public key for its lease;
 * a pin sends no GET. It sends one protected POST with a
 * byte body or a streamed body, based on the runtime and request size. It
 * checks START and returns a live `Response` for SSE. The platform's
 * `Headers` implementation may combine repeated authenticated fields;
 * cookie-setting fields are omitted.
 * Use the low-level client when exact response field-list preservation matters.
 *
 * From a Fetch request input, the adapter uses only the URL, method, fields,
 * body, and abort signal. The adapter owns outer credentials, redirects,
 * cache, referrer, and method. It performs no retry. A retry must call the
 * returned function again so the Rust engine creates a fresh envelope.
 */
export function createHpkeFetch(configuration: HpkeFetchConfiguration): HpkeFetch {
  const limits = normalizeLimits(configuration.limits ?? {});
  const source = configuration.key instanceof DiscoveredEndpoint ? configuration.key : undefined;
  if (source !== undefined && configuration.fetch !== undefined) {
    throw new TypeError("the shared endpoint owns Fetch");
  }
  const transport = source?.transport ?? configuration.fetch ?? globalThis.fetch;
  if (typeof transport !== "function") {
    throw new FetchTransportError(
      "network_error",
      "native Fetch is unavailable; provide configuration.fetch",
    );
  }
  if (source !== undefined && configuration.endpoint !== undefined) {
    throw new TypeError("the shared source owns the endpoint");
  }
  const endpoint = source?.endpoint ?? parseKeyEndpoint(configuration.endpoint ?? "");
  const targetOrigin = configuration.targetOrigin === undefined
    ? endpoint.origin
    : parseTargetOrigin(configuration.targetOrigin);
  const psk = Uint8Array.from(configuration.psk);
  const pskId = Uint8Array.from(configuration.pskId);
  if (psk.byteLength < 32 || pskId.byteLength < 1 || pskId.byteLength > 255 || bytesEqual(psk, pskId)) {
    throw new ProtocolError("invalid_configuration", "invalid PSK or PSK identity");
  }
  let pinnedClient: Client | undefined;
  if (source === undefined) {
    const key = configuration.key;
    if (!("kind" in key) || key.kind !== "pin") {
      throw new TypeError("key must be a shared source or a tagged pin");
    }
    pinnedClient = new Client(key.publicKey, key.keyId, psk, pskId, limits);
  }
  const maxRequestLength = limits.maxBodyLength ?? DEFAULT_MAX_BODY_LENGTH;
  const active = new Set<RecordPump>();
  const activeUploads = new Set<(reason?: unknown) => void>();
  let closed = false;

  const hpkeFetch = async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<globalThis.Response> => {
    if (closed) {
      throw new StateError();
    }

    const request = input instanceof globalThis.Request && init === undefined
      ? input
      : new globalThis.Request(input, init);
    const target = parseHttpsUrl(request.url);
    if (target.origin !== targetOrigin) {
      throw new FetchTransportError("invalid_target", "logical target has the wrong HTTPS origin");
    }
    const method = parseMethod(request.method);
    const headers = copyEndToEndHeaders(request.headers);
    const head: RequestHead = {
      method,
      authority: target.host,
      path: `${target.pathname}${target.search}`,
      headers,
    };
    const discovered = source !== undefined;
    let lease = source === undefined ? undefined : await source.getKey(request.signal);
    let requestClient = pinnedClient;
    if (lease !== undefined) {
      if (closed) { throw new StateError(); }
      try {
        requestClient = new Client(lease.publicKey, lease.keyId, psk, pskId, limits);
      } catch (error: unknown) {
        if (error instanceof ProtocolError) {
          throw new FetchTransportError("discovery_response", "key endpoint returned an unusable public key");
        }
        throw error;
      }
    }
    if (requestClient === undefined) { throw new StateError(); }
    let envelope: RequestEnvelope;
    let finiteBody: Uint8Array | undefined;
    try {
      if (request.body !== null && isNodeRuntime() && supportsRequestStreaming()) {
        envelope = makeStreamEnvelope(requestClient.beginStream(head), request.body, request.signal, lease);
      } else if (request.body !== null && !isNodeRuntime()) {
        const inspected = await inspectBody(request.body, maxRequestLength, request.signal);
        if (inspected.kind === "complete") {
          finiteBody = inspected.body;
          envelope = makeFiniteEnvelope(requestClient.protect({ ...head, body: inspected.body }));
        } else if (supportsRequestStreaming()) {
          try {
            envelope = makeStreamEnvelope(requestClient.beginStream(head), inspected.body, request.signal, lease);
          } catch (error: unknown) {
            requestCancel(inspected.body, error);
            throw error;
          }
        } else {
          requestCancel(inspected.body, new FetchTransportError("request_too_large", "body exceeds the byte POST limit"));
          throw new FetchTransportError("request_too_large", "body exceeds the byte POST limit");
        }
      } else {
        const body = await readBodyBounded(request.body, maxRequestLength, request.signal);
        finiteBody = body;
        envelope = makeFiniteEnvelope(requestClient.protect({ ...head, body }));
      }
    } catch (error: unknown) {
      if (discovered && error instanceof ProtocolError &&
          (error.code === "invalid_configuration" || error.code === "crypto_failure")) {
        throw new FetchTransportError("discovery_response", "key endpoint returned an unusable public key");
      }
      throw error;
    } finally {
      if (discovered) { requestClient.close(); }
    }

    if (lease !== undefined && !lease.valid()) {
      const expired = new FetchTransportError("discovery_expired", "key lifetime ended before protected POST");
      envelope.close(expired);
      if (finiteBody === undefined || source === undefined) { throw expired; }
      lease = await source.getKey(request.signal);
      const fresh = new Client(lease.publicKey, lease.keyId, psk, pskId, limits);
      try {
        envelope = makeFiniteEnvelope(fresh.protect({ ...head, body: finiteBody }));
      } finally {
        fresh.close();
      }
      if (!lease.valid()) {
        envelope.close(expired);
        throw expired;
      }
    }

    activeUploads.add(envelope.close);
    try {
      if (closed) { throw new StateError(); }
      let outerResponse: globalThis.Response;
      try {
        outerResponse = await sendEnvelope(
          transport,
          endpoint,
          envelope.body,
          request.signal,
        );
      } catch (error: unknown) {
        throw envelope.failure() ?? error;
      }
      activeUploads.delete(envelope.close);
      validateOuterResponse(outerResponse);
      if (outerResponse.body === null) {
        throw new ProtocolError("malformed_envelope", "response START is missing");
      }
      const pump = new RecordPump(outerResponse.body, envelope.takeRight().intoOpener(), request.signal, active);
      active.add(pump);
      try {
        const first = await pump.next();
        if (first?.kind !== "start") {
          throw new ProtocolError("malformed_envelope", "response START is missing");
        }
        const webHeaders = headersToWeb(first.headers);
        if (first.mode === "finite") {
          while (await pump.next() !== undefined) {
            // The native opener keeps the finite body until checked END and outer EOF.
          }
          const finite = pump.finiteResponse;
          if (finite === undefined) {
            throw new ProtocolError("malformed_envelope", "finite response body is missing");
          }
          pump.close();
          return new globalThis.Response(
            responseHasBody(method, finite.status) ? toArrayBuffer(finite.body ?? new Uint8Array()) : null,
            { status: finite.status, headers: webHeaders },
          );
        }
        let controllerRef: ReadableStreamDefaultController<Uint8Array> | undefined;
        let settled = false;
        const fail = (reason: unknown): void => {
          if (!settled) {
            settled = true;
            controllerRef?.error(reason);
          }
        };
        const clear = new ReadableStream<Uint8Array>({
          start(controller) {
            controllerRef = controller;
            pump.onAbort = fail;
          },
          async pull(controller) {
            try {
              while (true) {
                const record = await pump.next();
                if (record === undefined) {
                  if (!settled) {
                    settled = true;
                    controller.close();
                  }
                  pump.close();
                  return;
                }
                if (record.kind === "sse_data") {
                  controller.enqueue(record.block);
                  return;
                }
                if (record.kind === "start") {
                  throw new ProtocolError("malformed_envelope", "invalid SSE record order");
                }
              }
            } catch (error: unknown) {
              fail(error);
              pump.close(error);
            }
          },
          cancel(reason) {
            pump.close(reason);
          },
        });
        try {
          return new globalThis.Response(clear, { status: first.status, headers: webHeaders });
        } catch (error: unknown) {
          fail(error);
          throw error;
        }
      } catch (error: unknown) {
        pump.close(error);
        throw error;
      }
    } finally {
      activeUploads.delete(envelope.close);
      envelope.close();
    }
  };

  hpkeFetch.close = (): void => {
    if (!closed) {
      closed = true;
      for (const pump of active) {
        pump.close(new StateError());
      }
      for (const stop of activeUploads) {
        stop(new StateError());
      }
      pinnedClient?.close();
      psk.fill(0);
    }
  };
  return hpkeFetch;
}

function parseKeyEndpoint(input: string): URL {
  if (typeof input !== "string" || input.includes("?") || input.includes("#")) {
    throw new FetchTransportError("invalid_target", "key endpoint must be one HTTPS URL without query or fragment");
  }
  return parseHttpsUrl(input);
}

function parseTargetOrigin(input: string): string {
  if (typeof input !== "string" || input.includes("?") || input.includes("#")) {
    throw new FetchTransportError("invalid_target", "logical origin must be one HTTPS origin");
  }
  const url = parseHttpsUrl(input);
  if (url.pathname !== "/") {
    throw new FetchTransportError("invalid_target", "logical origin must have no path");
  }
  return url.origin;
}

async function fetchKey(
  transport: FetchTransport,
  endpoint: URL,
  signal: AbortSignal,
): Promise<{ keyId: Uint8Array; publicKey: Uint8Array; useForS: number }> {
  throwIfAborted(signal);
  let response: globalThis.Response;
  try {
    response = await transport(endpoint, {
      method: "GET",
      headers: { accept: KEY_MEDIA_TYPE },
      cache: "no-store",
      credentials: "omit",
      redirect: "error",
      referrerPolicy: "no-referrer",
      signal,
    });
  } catch {
    if (signal.aborted) { throw abortReason(signal); }
    throw new FetchTransportError("discovery_network", "key GET failed");
  }
  if (response.status !== 200) {
    const error = new FetchTransportError(
      "discovery_status", "key endpoint returned an invalid status", undefined, response.status,
    );
    requestCancel(response.body, error);
    throw error;
  }
  const type = response.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
  const coding = response.headers.get("content-encoding");
  if (type !== KEY_MEDIA_TYPE || (coding !== null && coding.trim().toLowerCase() !== "identity")) {
    const error = new FetchTransportError("discovery_response", "key endpoint returned invalid key bytes");
    requestCancel(response.body, error);
    throw error;
  }
  let record: Uint8Array;
  try {
    record = await readBodyBounded(response.body, MAX_KEY_RECORD, signal);
  } catch (error: unknown) {
    if (signal.aborted) { throw abortReason(signal); }
    if (error instanceof FetchTransportError && error.code === "request_too_large") {
      throw new FetchTransportError("discovery_response", "key record exceeds 297 bytes");
    }
    throw new FetchTransportError("discovery_network", "key GET body failed");
  }
  return parseKeyRecord(record);
}

function parseKeyRecord(record: Uint8Array): { keyId: Uint8Array; publicKey: Uint8Array; useForS: number } {
  const length = record[5];
  if (record.byteLength < 43 || record.byteLength > MAX_KEY_RECORD ||
      record[0] !== 0x48 || record[1] !== 0x48 || record[2] !== 0x4b || record[3] !== 0x44 ||
      record[4] !== 2 || length === undefined || length === 0 || record.byteLength !== 42 + length) {
    throw new FetchTransportError("discovery_response", "key endpoint returned an invalid key record");
  }
  const useForS = new DataView(record.buffer, record.byteOffset + record.byteLength - 4, 4).getUint32(0);
  if (useForS === 0) {
    throw new FetchTransportError("discovery_response", "key endpoint returned an invalid key lifetime");
  }
  return {
    keyId: Uint8Array.from(record.subarray(6, 6 + length)),
    publicKey: Uint8Array.from(record.subarray(6 + length, -4)),
    useForS,
  };
}

function bytesEqual(left: Uint8Array, right: Uint8Array): boolean {
  if (left.byteLength !== right.byteLength) { return false; }
  for (let index = 0; index < left.byteLength; index += 1) {
    if (left[index] !== right[index]) { return false; }
  }
  return true;
}

interface ResponseRight {
  intoOpener(): ResponseOpener;
  close(): void;
}

interface RequestEnvelope {
  readonly body: Uint8Array | ReadableStream<Uint8Array>;
  takeRight(): ResponseRight;
  failure(): unknown;
  close(reason?: unknown): void;
}

function makeFiniteEnvelope(protectedRequest: ProtectedRequest): RequestEnvelope {
  return {
    body: protectedRequest.envelope,
    takeRight: () => protectedRequest,
    failure: () => undefined,
    close: () => protectedRequest.close(),
  };
}

function makeStreamEnvelope(
  writer: StreamRequestSealer,
  source: ReadableStream<Uint8Array> | null,
  signal: AbortSignal,
  lease?: DiscoveredKeyLease,
): RequestEnvelope {
  const reader = source?.getReader();
  let part: Uint8Array<ArrayBufferLike> = new Uint8Array();
  let offset = 0;
  let sourceEnded = reader === undefined;
  let finished: StreamFinishedRequest | undefined;
  let closed = false;
  let streamEnded = false;
  let startQueued = false;
  let failed: unknown;
  let controllerRef: ReadableStreamDefaultController<Uint8Array> | undefined;
  const close = (reason?: unknown): void => {
    if (closed) { return; }
    closed = true;
    signal.removeEventListener("abort", onAbort);
    writer.close();
    finished?.close();
    if (reader !== undefined && !sourceEnded) {
      requestCancel(reader, reason);
    }
    if (!streamEnded && reason !== undefined) {
      controllerRef?.error(reason);
    }
  };
  const onAbort = (): void => close(abortReason(signal));
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controllerRef = controller;
    },
    async pull(controller) {
      try {
        if (!startQueued) {
          if (lease !== undefined && !lease.valid()) {
            throw new FetchTransportError("discovery_expired", "key lifetime ended before protected POST START");
          }
          startQueued = true;
          controller.enqueue(writer.start);
          return;
        }
        while (true) {
          if (closed) { throw new StateError(); }
          throwIfAborted(signal);
          if (offset < part.byteLength) {
            const remaining = part.subarray(offset);
            const result = writer.push(remaining);
            if (result.consumed > remaining.byteLength ||
                (result.consumed === 0 && result.record === undefined)) {
              throw new ProtocolError("malformed_envelope", "request writer made no progress");
            }
            offset += result.consumed;
            if (result.record !== undefined) {
              controller.enqueue(result.record);
              return;
            }
            continue;
          }
          if (sourceEnded) {
            finished = writer.finish();
            controller.enqueue(finished.end);
            streamEnded = true;
            controller.close();
            reader?.releaseLock();
            return;
          }
          const next = await reader?.read();
          if (next === undefined || next.done) {
            sourceEnded = true;
          } else {
            part = next.value;
            offset = 0;
          }
        }
      } catch (error: unknown) {
        failed = error;
        close(error);
      }
    },
    cancel: close,
  }, { highWaterMark: 0 });
  signal.addEventListener("abort", onAbort, { once: true });
  if (signal.aborted) { onAbort(); }
  return {
    body,
    failure: () => failed,
    takeRight() {
      if (finished === undefined) {
        throw new FetchTransportError("network_error", "outer transport ended before the request body");
      }
      return finished;
    },
    close,
  };
}

function supportsRequestStreaming(): boolean {
  try {
    const body = new ReadableStream<Uint8Array>({ start(controller) { controller.close(); } });
    const options: RequestInit & { duplex: "half" } = { method: "POST", body, duplex: "half" };
    new globalThis.Request("https://hpke-http.invalid/", options);
    return true;
  } catch {
    return false;
  }
}

function isNodeRuntime(): boolean {
  return typeof process !== "undefined" && typeof process.versions?.node === "string";
}

async function inspectBody(
  body: ReadableStream<Uint8Array>,
  maximum: number,
  signal: AbortSignal,
): Promise<{ readonly kind: "complete"; readonly body: Uint8Array } |
           { readonly kind: "stream"; readonly body: ReadableStream<Uint8Array> }> {
  throwIfAborted(signal);
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let partial: Uint8Array | undefined;
  let partialLength = 0;
  let nextChunkSize = 1024;
  let length = 0;
  let transferred = false;
  let rejectAbort: ((reason?: unknown) => void) | undefined;
  const aborted = new Promise<never>((_resolve, reject) => { rejectAbort = reject; });
  const onAbort = (): void => rejectAbort?.(abortReason(signal));
  signal.addEventListener("abort", onAbort, { once: true });
  try {
    while (true) {
      const next = await Promise.race([reader.read(), aborted]);
      if (next.done) {
        const bytes = new Uint8Array(length);
        let offset = 0;
        for (const chunk of chunks) {
          bytes.set(chunk, offset);
          offset += chunk.byteLength;
        }
        if (partial !== undefined) {
          bytes.set(partial.subarray(0, partialLength), offset);
        }
        return { kind: "complete", body: bytes };
      }
      if (next.value.byteLength > maximum - length) {
        const pending = partial === undefined
          ? [...chunks, next.value]
          : [...chunks, partial.subarray(0, partialLength), next.value];
        let pendingIndex = 0;
        let done = false;
        transferred = true;
        const stream = new ReadableStream<Uint8Array>({
          async pull(controller) {
            if (pendingIndex < pending.length) {
              const queued = pending[pendingIndex++];
              if (queued === undefined) { throw new StateError(); }
              controller.enqueue(queued);
              return;
            }
            try {
              const part = await reader.read();
              if (part.done) {
                done = true;
                controller.close();
                reader.releaseLock();
              } else {
                controller.enqueue(part.value);
              }
            } catch (error: unknown) {
              controller.error(error);
              reader.releaseLock();
            }
          },
          cancel(reason) {
            if (!done) { requestCancel(reader, reason); }
            reader.releaseLock();
          },
        }, { highWaterMark: 0 });
        return { kind: "stream", body: stream };
      }
      const chunk = next.value;
      length += chunk.byteLength;
      let offset = 0;
      while (offset < chunk.byteLength) {
        if (partial === undefined && chunk.byteLength - offset >= MAX_BUFFER_CHUNK) {
          chunks.push(Uint8Array.from(chunk.subarray(offset)));
          break;
        }
        if (partial === undefined) {
          partial = new Uint8Array(nextChunkSize);
          nextChunkSize = Math.min(nextChunkSize * 2, MAX_BUFFER_CHUNK);
        }
        const count = Math.min(partial.byteLength - partialLength, chunk.byteLength - offset);
        partial.set(chunk.subarray(offset, offset + count), partialLength);
        partialLength += count;
        offset += count;
        if (partialLength === partial.byteLength) {
          chunks.push(partial);
          partial = undefined;
          partialLength = 0;
        }
      }
    }
  } catch (error: unknown) {
    requestCancel(reader, error);
    throw error;
  } finally {
    signal.removeEventListener("abort", onAbort);
    if (!transferred) { reader.releaseLock(); }
  }
}

async function sendEnvelope(
  transport: FetchTransport,
  endpoint: URL,
  envelope: Uint8Array | ReadableStream<Uint8Array>,
  signal: AbortSignal,
): Promise<globalThis.Response> {
  throwIfAborted(signal);
  try {
    const options: RequestInit & { duplex?: "half" } = {
      method: "POST",
      headers: {
        accept: RESPONSE_MEDIA_TYPE,
        "cache-control": "no-store",
        "content-type": REQUEST_MEDIA_TYPE,
      },
      body: envelope instanceof ReadableStream ? envelope : toArrayBuffer(envelope),
      cache: "no-store",
      credentials: "omit",
      redirect: "error",
      referrerPolicy: "no-referrer",
      signal,
    };
    if (envelope instanceof ReadableStream) { options.duplex = "half"; }
    return await transport(endpoint, options);
  } catch (error: unknown) {
    if (signal.aborted) {
      throw abortReason(signal);
    }
    throw new FetchTransportError("network_error", "protected Fetch request failed", {
      cause: error,
    });
  }
}

function validateOuterResponse(response: globalThis.Response): void {
  if (response.status !== 200) {
    const error = new FetchTransportError(
      "outer_status",
      `protected endpoint returned outer status ${String(response.status)}`,
    );
    requestCancel(response.body, error);
    throw error;
  }
  const contentType = response.headers.get("content-type")?.trim().toLowerCase();
  if (contentType !== RESPONSE_MEDIA_TYPE) {
    const error = new FetchTransportError(
      "outer_content_type",
      `protected endpoint must return ${RESPONSE_MEDIA_TYPE}`,
    );
    requestCancel(response.body, error);
    throw error;
  }
  const contentEncoding = response.headers.get("content-encoding");
  if (contentEncoding !== null && contentEncoding.trim().toLowerCase() !== "identity") {
    const error = new FetchTransportError("outer_content_encoding", "protected envelope must not use content encoding");
    requestCancel(response.body, error);
    throw error;
  }
}

class RecordPump {
  readonly #reader: ReadableStreamDefaultReader<Uint8Array>;
  readonly #opener: ResponseOpener;
  readonly #signal: AbortSignal;
  readonly #registry: Set<RecordPump>;
  #pending: Uint8Array = new Uint8Array();
  #offset = 0;
  #closed = false;
  #eof = false;
  public finiteResponse: PlaintextResponse | undefined;
  public onAbort: ((reason: unknown) => void) | undefined;

  public constructor(body: ReadableStream<Uint8Array>, opener: ResponseOpener, signal: AbortSignal, registry: Set<RecordPump>) {
    this.#reader = body.getReader();
    this.#opener = opener;
    this.#signal = signal;
    this.#registry = registry;
    signal.addEventListener("abort", this.#abort, { once: true });
    if (signal.aborted) { this.#abort(); }
  }

  readonly #abort = (): void => {
    const reason = abortReason(this.#signal);
    this.close(reason);
  };

  public async next(): Promise<CheckedRecord | undefined> {
    if (this.#closed) {
      throw this.#signal.aborted ? abortReason(this.#signal) : new StateError();
    }
    if (this.#eof) {
      return undefined;
    }
    while (true) {
      if (this.#offset < this.#pending.byteLength) {
        const end = Math.min(this.#offset + MAX_BUFFER_CHUNK, this.#pending.byteLength);
        const { consumed, record } = this.#opener.feed(this.#pending.subarray(this.#offset, end));
        this.#offset += consumed;
        if (record !== undefined) {
          return record;
        }
        if (consumed === 0) {
          throw new ProtocolError("malformed_envelope", "record reader made no progress");
        }
        continue;
      }
      this.#pending = new Uint8Array();
      this.#offset = 0;
      let result: ReadableStreamReadResult<Uint8Array>;
      try {
        result = await this.#reader.read();
      } catch (error: unknown) {
        if (this.#signal.aborted) { throw abortReason(this.#signal); }
        throw new FetchTransportError("network_error", "protected Fetch response body failed", { cause: error });
      }
      if (this.#signal.aborted) { throw abortReason(this.#signal); }
      if (this.#closed) { throw new StateError(); }
      if (result.done) {
        this.finiteResponse = this.#opener.finishEof();
        this.#eof = true;
        return undefined;
      }
      this.#pending = result.value;
    }
  }

  public close(reason?: unknown): void {
    if (this.#closed) { return; }
    this.#closed = true;
    this.#registry.delete(this);
    this.#signal.removeEventListener("abort", this.#abort);
    if (reason !== undefined) { this.onAbort?.(reason); }
    this.#opener.close();
    requestCancel(this.#reader, reason);
    this.#reader.releaseLock();
  }
}

function responseHasBody(method: Method, status: number): boolean {
  return method !== "HEAD" && !BODYLESS_RESPONSE_STATUSES.has(status);
}

function copyEndToEndHeaders(headers: Headers): readonly Header[] {
  requireIdentityContentCoding(headers.get("content-encoding"), "logical request");
  const excluded = new Set(NON_FORWARDABLE_REQUEST_HEADERS);
  for (const option of headers.get("connection")?.split(",") ?? []) {
    const name = option.trim().toLowerCase();
    if (name !== "") {
      excluded.add(name);
    }
  }

  const output: Header[] = [];
  for (const [name, value] of headers) {
    if (!excluded.has(name)) {
      output.push({ name, value });
    }
  }
  return output;
}

function headersToWeb(fields: readonly Header[]): Headers {
  for (const field of fields) {
    if (field.name === "content-encoding") {
      requireIdentityContentCoding(field.value, "authenticated response");
    }
  }
  try {
    const headers = new globalThis.Headers();
    for (const field of fields) {
      if (!FETCH_HIDDEN_RESPONSE_HEADERS.has(field.name)) {
        headers.append(field.name, field.value);
      }
    }
    return headers;
  } catch (error: unknown) {
    throw new FetchTransportError(
      "invalid_inner_response",
      "authenticated response contains a field that Fetch cannot represent",
      { cause: error },
    );
  }
}

function requireIdentityContentCoding(value: string | null, message: string): void {
  if (
    value !== null &&
    value.split(",").some((coding) => coding.trim().toLowerCase() !== "identity")
  ) {
    throw new FetchTransportError(
      "inner_content_encoding",
      `${message} supports only identity content coding`,
    );
  }
}

function parseMethod(value: string): Method {
  if (isMethod(value)) {
    return value;
  }
  throw new ProtocolError("unsupported_method", "request method is not supported by hpke-http");
}

function parseHttpsUrl(input: string | URL): URL {
  let url: URL;
  try {
    url = new URL(input);
  } catch (error: unknown) {
    throw new FetchTransportError("invalid_target", "protected requests require an absolute URL", {
      cause: error,
    });
  }
  if (url.protocol !== "https:" || url.username !== "" || url.password !== "") {
    throw new FetchTransportError(
      "invalid_target",
      "protected requests require an HTTPS URL without embedded credentials",
    );
  }
  url.hash = "";
  return url;
}

async function readBodyBounded(
  body: ReadableStream<Uint8Array> | null,
  maximum: number,
  signal: AbortSignal,
): Promise<Uint8Array> {
  throwIfAborted(signal);
  if (body === null) {
    return new Uint8Array();
  }
  const inspected = await inspectBody(body, maximum, signal);
  if (inspected.kind === "complete") {
    return inspected.body;
  }
  const error = new FetchTransportError("request_too_large", "buffered Fetch body exceeds the configured limit");
  requestCancel(inspected.body, error);
  throw error;
}

function requestCancel(
  body: ReadableStream<Uint8Array> | ReadableStreamDefaultReader<Uint8Array> | null,
  reason: unknown,
): void {
  if (body !== null) {
    // A custom stream can leave its cancel promise pending; cleanup must not wait.
    void body.cancel(reason).catch(() => undefined);
  }
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) {
    throw abortReason(signal);
  }
}

function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("The operation was aborted", "AbortError");
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  if (bytes.buffer instanceof ArrayBuffer && bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength) {
    return bytes.buffer;
  }
  return Uint8Array.from(bytes).buffer;
}

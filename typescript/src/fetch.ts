import { Client, ProtocolError, ResponseOpener, StateError, normalizeLimits } from "./index.js";
import type { CheckedRecord, CompressionCoding, Header, Limits, Method, Request as PlaintextRequest, Response as PlaintextResponse } from "./index.js";

/** Media type of an outer protected request envelope. */
export const REQUEST_MEDIA_TYPE = "message/hpke-http-request";

/** Media type of an outer protected response envelope. */
export const RESPONSE_MEDIA_TYPE = "message/hpke-http-response";

const DEFAULT_MAX_BODY_LENGTH = 8 * 1024 * 1024;
const MAX_BUFFER_CHUNK = 64 * 1024;

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
 * Target, request-body-limit, and request content-coding failures occur before
 * an outer request. Network and `outer_*` failures occur during the envelope
 * exchange. `inner_content_encoding` can also describe an authenticated
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
  | "invalid_inner_response";

/** A transport or logical-message error from the bounded Fetch adapter. */
export class FetchTransportError extends Error {
  public readonly code: FetchTransportErrorCode;

  public constructor(code: FetchTransportErrorCode, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "FetchTransportError";
    this.code = code;
  }
}

export interface HpkeFetchConfiguration {
  /** Encoded 32-byte X25519 recipient public key. */
  readonly recipientPublicKey: Uint8Array;

  /** Non-empty public recipient-key identifier, at most 255 bytes. */
  readonly recipientKeyId: Uint8Array;

  /** PSK of at least 32 bytes. The adapter copies but cannot clear this array. */
  readonly psk: Uint8Array;

  /** Non-empty public PSK identifier, at most 255 bytes and not equal to `psk`. */
  readonly pskId: Uint8Array;

  /** Optional message limits. Omitted fields use the shared engine defaults. */
  readonly limits?: Limits;

  /** Opt-in Rust-owned protocol body coding, separate from HTTP representation coding. */
  readonly compression?: CompressionCoding;

  /**
   * Fixed HTTPS endpoint that receives protected envelopes. When omitted, the
   * original request URL is also the transport endpoint.
   */
  readonly transportEndpoint?: string | URL;

  /** Inject a Fetch-compatible transport. The default is `globalThis.fetch`. */
  readonly fetch?: FetchTransport;
}

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
 * Fetch request input. It protects the complete logical request, sends one POST
 * envelope, checks START, and returns a live `Response` for SSE. The platform's
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
  const transport = configuration.fetch ?? globalThis.fetch;
  if (typeof transport !== "function") {
    throw new FetchTransportError(
      "network_error",
      "native Fetch is unavailable; provide configuration.fetch",
    );
  }
  const fixedEndpoint =
    configuration.transportEndpoint === undefined
      ? undefined
      : parseHttpsUrl(configuration.transportEndpoint);
  const client = new Client(
    configuration.recipientPublicKey,
    configuration.recipientKeyId,
    configuration.psk,
    configuration.pskId,
    limits,
    configuration.compression,
  );
  const maxRequestLength = limits.maxBodyLength ?? DEFAULT_MAX_BODY_LENGTH;
  const active = new Set<RecordPump>();
  let closed = false;

  const hpkeFetch = async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<globalThis.Response> => {
    if (closed) {
      throw new StateError();
    }

    const request = new globalThis.Request(input, init);
    const target = parseHttpsUrl(request.url);
    const method = parseMethod(request.method);
    const headers = copyEndToEndHeaders(request.headers);
    const body = await readBodyBounded(request.body, maxRequestLength, request.signal);
    const plaintext: PlaintextRequest = {
      method,
      authority: target.host,
      path: `${target.pathname}${target.search}`,
      headers,
      body,
    };
    const protectedRequest = client.protect(plaintext);

    try {
      const outerResponse = await sendEnvelope(
        transport,
        fixedEndpoint ?? target,
        protectedRequest.envelope,
        request.signal,
      );
      await validateOuterResponse(outerResponse);
      if (outerResponse.body === null) {
        throw new ProtocolError("malformed_envelope", "response START is missing");
      }
      const pump = new RecordPump(outerResponse.body, protectedRequest.intoOpener(), request.signal, active);
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
          await pump.close();
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
                  await pump.close();
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
              await pump.close(error);
            }
          },
          async cancel(reason) {
            await pump.close(reason);
          },
        });
        try {
          return new globalThis.Response(clear, { status: first.status, headers: webHeaders });
        } catch (error: unknown) {
          fail(error);
          throw error;
        }
      } catch (error: unknown) {
        await pump.close(error);
        throw error;
      }
    } finally {
      protectedRequest.close();
    }
  };

  hpkeFetch.close = (): void => {
    if (!closed) {
      closed = true;
      for (const pump of active) {
        void pump.close(new StateError());
      }
      client.close();
    }
  };
  return hpkeFetch;
}

async function sendEnvelope(
  transport: FetchTransport,
  endpoint: URL,
  envelope: Uint8Array,
  signal: AbortSignal,
): Promise<globalThis.Response> {
  throwIfAborted(signal);
  try {
    return await transport(endpoint, {
      method: "POST",
      headers: {
        accept: RESPONSE_MEDIA_TYPE,
        "cache-control": "no-store",
        "content-type": REQUEST_MEDIA_TYPE,
      },
      body: toArrayBuffer(envelope),
      cache: "no-store",
      credentials: "omit",
      redirect: "error",
      referrerPolicy: "no-referrer",
      signal,
    });
  } catch (error: unknown) {
    if (signal.aborted) {
      throw abortReason(signal);
    }
    throw new FetchTransportError("network_error", "protected Fetch request failed", {
      cause: error,
    });
  }
}

async function validateOuterResponse(response: globalThis.Response): Promise<void> {
  if (response.status !== 200) {
    const error = new FetchTransportError(
      "outer_status",
      `protected endpoint returned outer status ${String(response.status)}`,
    );
    await cancelBody(response.body, error);
    throw error;
  }
  const contentType = response.headers.get("content-type")?.trim().toLowerCase();
  if (contentType !== RESPONSE_MEDIA_TYPE) {
    const error = new FetchTransportError(
      "outer_content_type",
      `protected endpoint must return ${RESPONSE_MEDIA_TYPE}`,
    );
    await cancelBody(response.body, error);
    throw error;
  }
  const contentEncoding = response.headers.get("content-encoding");
  if (contentEncoding !== null && contentEncoding.trim().toLowerCase() !== "identity") {
    const error = new FetchTransportError("outer_content_encoding", "protected envelope must not use content encoding");
    await cancelBody(response.body, error);
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
    void this.close(reason);
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

  public async close(reason?: unknown): Promise<void> {
    if (this.#closed) { return; }
    this.#closed = true;
    this.#registry.delete(this);
    this.#signal.removeEventListener("abort", this.#abort);
    if (reason !== undefined) { this.onAbort?.(reason); }
    this.#opener.close();
    try { await this.#reader.cancel(reason); }
    catch { /* The outer body can already be closed. */ }
    finally { this.#reader.releaseLock(); }
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
  switch (value) {
    case "GET":
    case "POST":
    case "PUT":
    case "PATCH":
    case "DELETE":
    case "HEAD":
    case "OPTIONS":
      return value;
    default:
      throw new ProtocolError("unsupported_method", "request method is not supported by hpke-http");
  }
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

  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let partial: Uint8Array | undefined;
  let partialLength = 0;
  let nextChunkSize = 1024;
  let length = 0;
  let rejectAbort: ((reason?: unknown) => void) | undefined;
  const aborted = new Promise<never>((_resolve, reject) => {
    rejectAbort = reject;
  });
  const onAbort = (): void => {
    rejectAbort?.(abortReason(signal));
  };
  signal.addEventListener("abort", onAbort, { once: true });

  try {
    while (true) {
      const result = await Promise.race([reader.read(), aborted]);
      if (result.done) {
        break;
      }
      if (result.value.byteLength > maximum - length) {
        throw new FetchTransportError("request_too_large", "buffered Fetch body exceeds the configured limit");
      }
      const chunk = result.value;
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
    await reader.cancel(error).catch(() => undefined);
    throw error;
  } finally {
    signal.removeEventListener("abort", onAbort);
    reader.releaseLock();
  }

  const output = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  if (partial !== undefined) {
    output.set(partial.subarray(0, partialLength), offset);
  }
  return output;
}

async function cancelBody(
  body: ReadableStream<Uint8Array> | null,
  reason: unknown,
): Promise<void> {
  await body?.cancel(reason).catch(() => undefined);
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
  return Uint8Array.from(bytes).buffer;
}

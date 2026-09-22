import { Client, ProtocolError, StateError, normalizeLimits } from "./index.js";
import type { CompressionCoding, Header, Limits, Method, Request as PlaintextRequest } from "./index.js";

/** Media type of an outer protected request envelope. */
export const REQUEST_MEDIA_TYPE = "message/hpke-http-request";

/** Media type of an outer protected response envelope. */
export const RESPONSE_MEDIA_TYPE = "message/hpke-http-response";

const DEFAULT_MAX_BODY_LENGTH = 8 * 1024 * 1024;
const DEFAULT_MAX_HEADER_BYTES = 16 * 1024;
const DEFAULT_MAX_HEADER_COUNT = 64;
const BHTTP_FIXED_ALLOWANCE = 256;
const BHTTP_PER_HEADER_ALLOWANCE = 16;
const RESPONSE_ENVELOPE_OVERHEAD = 48;
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
 * an outer request. Network, `outer_*`, and response-body-limit failures occur
 * during the envelope exchange. `inner_content_encoding` can also describe an
 * authenticated response; `invalid_inner_response` describes a response that
 * Web Fetch cannot represent.
 */
export type FetchTransportErrorCode =
  | "invalid_target"
  | "request_too_large"
  | "network_error"
  | "outer_status"
  | "outer_content_type"
  | "response_too_large"
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

/** A bounded, buffered Fetch-shaped client with an explicit lifecycle. */
export interface HpkeFetch {
  /** Protect one logical request and authenticate its complete response. */
  (input: RequestInfo | URL, init?: RequestInit): Promise<globalThis.Response>;

  /** Release native credential copies. This operation is idempotent. */
  close(): void;
}

/**
 * Create a buffered HPKE-aware function backed by the runtime's native Fetch
 * and the shared Rust/WASM protocol engine.
 *
 * The returned function uses the URL, method, headers, body, and signal from a
 * Fetch request input. It protects the complete logical request, sends one POST
 * envelope, authenticates the complete response envelope, and only then creates
 * the returned Web `Response`. The platform's `Headers` implementation may
 * combine repeated authenticated fields; cookie-setting fields are omitted.
 * Use the low-level client when exact response field-list preservation matters.
 *
 * Only the URL, method, fields, body, and abort signal are logical request
 * inputs. The adapter owns outer credentials, redirects, cache, referrer, and
 * method. It performs no retry. A retry must call the returned function again
 * so the Rust engine creates a fresh envelope.
 *
 * @throws {@link FetchTransportError} for target, transport, representation, or
 * bounded-buffer failures.
 * @throws {@link ProtocolError} for invalid or unauthenticated protocol data.
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
  const maxResponseLength = calculateMaxResponseEnvelopeLength(limits);
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
    const body = await readBodyBounded(
      request.body,
      maxRequestLength,
      request.signal,
      "request_too_large",
    );
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
      let responseEnvelope: Uint8Array;
      try {
        responseEnvelope = await readBodyBounded(
          outerResponse.body,
          maxResponseLength,
          request.signal,
          "response_too_large",
        );
      } catch (error: unknown) {
        if (error instanceof FetchTransportError || request.signal.aborted) {
          throw error;
        }
        throw new FetchTransportError("network_error", "protected Fetch response body failed", {
          cause: error,
        });
      }
      const authenticated = protectedRequest.openResponse(responseEnvelope);
      return new globalThis.Response(
        responseHasBody(method, authenticated.status)
          ? toArrayBuffer(authenticated.body ?? new Uint8Array())
          : null,
        {
          status: authenticated.status,
          headers: headersToWeb(authenticated.headers ?? []),
        },
      );
    } finally {
      protectedRequest.close();
    }
  };

  hpkeFetch.close = (): void => {
    if (!closed) {
      closed = true;
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
  const contentType = response.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
  if (contentType !== RESPONSE_MEDIA_TYPE) {
    const error = new FetchTransportError(
      "outer_content_type",
      `protected endpoint must return ${RESPONSE_MEDIA_TYPE}`,
    );
    await cancelBody(response.body, error);
    throw error;
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

function calculateMaxResponseEnvelopeLength(limits: Limits): number {
  const body = limits.maxBodyLength ?? DEFAULT_MAX_BODY_LENGTH;
  const headerBytes = limits.maxHeaderBytes ?? DEFAULT_MAX_HEADER_BYTES;
  const headerCount = limits.maxHeaderCount ?? DEFAULT_MAX_HEADER_COUNT;
  return (
    body +
    headerBytes +
    headerCount * BHTTP_PER_HEADER_ALLOWANCE +
    BHTTP_FIXED_ALLOWANCE +
    RESPONSE_ENVELOPE_OVERHEAD
  );
}

async function readBodyBounded(
  body: ReadableStream<Uint8Array> | null,
  maximum: number,
  signal: AbortSignal,
  errorCode: "request_too_large" | "response_too_large",
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
        throw new FetchTransportError(errorCode, "buffered Fetch body exceeds the configured limit");
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

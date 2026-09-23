import type {
  NativeAuthenticatedRequest,
  NativeClient,
  NativeModule,
  NativeOpenedRequest,
  NativePreparsedRequest,
  NativeProtectedRequest,
  NativeResponse,
  NativeResponseOpener,
  NativeResponseSealer,
  NativeServer,
} from "./native.js";
import { PACKAGE_VERSION } from "./_package-version.js";

/** Stable language-neutral wire-protocol identifier. */
export const PROTOCOL_ID = "hpke-http/2";

/** ABI version shared by the TypeScript facade and its private WASM module. */
export const BINDING_ABI_VERSION = 2;

/** npm package version used to reject a mismatched private WASM module. */
export { PACKAGE_VERSION };

/** HTTP methods accepted by protocol version 2. */
export type Method = "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "HEAD" | "OPTIONS";

/** Opt-in protocol body coding, independent of HTTP `Content-Encoding`. */
export type CompressionCoding = "gzip" | "zstd";

/** One ordered end-to-end HTTP field. Repeated names remain separate entries. */
export interface Header {
  /** Lower-case HTTP token name. */
  readonly name: string;

  /** Canonical ASCII value without leading or trailing optional whitespace. */
  readonly value: string;
}

/** One complete bounded HTTPS request before protection or after authentication. */
export interface Request {
  /** Supported HTTP method. */
  readonly method: Method;

  /** RFC 3986 host and optional decimal port, without user information. */
  readonly authority: string;

  /** Absolute path plus optional query, or `*` for `OPTIONS`. */
  readonly path: string;

  /** Ordered end-to-end fields. The default is an empty field section. */
  readonly headers?: readonly Header[];

  /** Complete request body. The default is an empty body. */
  readonly body?: Uint8Array;
}

/** One complete bounded HTTP response before protection or after authentication. */
export interface Response {
  /** Final status from 200 through 599. */
  readonly status: number;

  /** Ordered end-to-end fields. The default is an empty field section. */
  readonly headers?: readonly Header[];

  /** Complete response body. The default is an empty body. */
  readonly body?: Uint8Array;
}

/** Optional per-engine limits. An omitted value selects the documented default. */
export interface Limits {
  /**
   * Request or finite response body bytes, or one SSE block.
   * An SSE stream has no total body limit. Default: 8 MiB. Hard maximum: 64 MiB.
   */
  readonly maxBodyLength?: number;

  /** Combined field-name and field-value bytes. Default: 16 KiB. Hard maximum: 64 KiB. */
  readonly maxHeaderBytes?: number;

  /** Number of fields per message. Default: 64. Hard maximum: 256. */
  readonly maxHeaderCount?: number;

  /** Combined authority and path bytes. Default and hard maximum: 8 KiB. */
  readonly maxTargetLength?: number;
}

/**
 * Encoded X25519 recipient key pair.
 *
 * The arrays belong to the caller. Clear `privateKey` when it is no longer
 * needed; closing a client or server cannot clear caller-owned arrays.
 */
export interface KeyPair {
  /** Encoded 32-byte private key. */
  readonly privateKey: Uint8Array;

  /** Encoded 32-byte public key. */
  readonly publicKey: Uint8Array;
}

/**
 * Stable protocol failure reported by the shared Rust engine.
 *
 * `code` is language-neutral. It can be used for control flow. The message is
 * intentionally coarse and contains no plaintext, credential, or parser offset.
 */
export class ProtocolError extends Error {
  /** Stable language-neutral failure code. */
  public readonly code: string;

  public constructor(code: string, message: string) {
    super(message);
    this.name = "ProtocolError";
    this.code = code;
  }
}

/** A client, server, or one-shot continuation was closed or consumed. */
export class StateError extends ProtocolError {
  public constructor() {
    super("state_consumed", "continuation already consumed");
    this.name = "StateError";
  }
}

/** The private WASM module could not load or did not match this package. */
export class InitializationError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = "InitializationError";
  }
}

let nativeModule: NativeModule | undefined;
const serverHandles = new WeakMap<Server, NativeServer>();
const serverLimits = new WeakMap<Server, { body: number; envelope: number }>();
let createProtectedRequest: (native: NativeProtectedRequest) => ProtectedRequest;
let createPreparsedRequest: (native: NativePreparsedRequest, server: Server) => PreparsedRequest;
let createAuthenticatedRequest: (native: NativeAuthenticatedRequest, maximumBody: number) => AuthenticatedRequest;
let createOpenedRequest: (native: NativeOpenedRequest, maximumBody: number) => OpenedRequest;

/** @internal */
export function installNative(module: NativeModule): void {
  const protocol = module.protocol_id();
  const abi = module.binding_abi();
  const version = module.engine_version();
  if (protocol !== PROTOCOL_ID || abi !== BINDING_ABI_VERSION || version !== PACKAGE_VERSION) {
    throw new InitializationError(
      `hpke-http WASM mismatch: expected version=${PACKAGE_VERSION}, protocol=${PROTOCOL_ID}, ` +
        `abi=${String(BINDING_ABI_VERSION)}; got version=${version}, protocol=${protocol}, ` +
        `abi=${String(abi)}`,
    );
  }
  if (nativeModule !== undefined && nativeModule !== module) {
    throw new InitializationError("hpke-http WASM was already initialized with another module");
  }
  nativeModule = module;
}

/** Return whether a verified private WASM module is ready for use. */
export function isInitialized(): boolean {
  return nativeModule !== undefined;
}

/**
 * Generate an X25519 recipient key pair with Web Crypto or operating-system entropy.
 *
 * Clear the returned `privateKey` after it has been copied into its long-term
 * protected storage. The facade cannot clear caller-owned arrays.
 *
 * @throws {@link ProtocolError} with `entropy_unavailable` when secure entropy fails.
 */
export function generateKeyPair(): KeyPair {
  const pair = callNative(() => requireNative().generateKeyPair());
  try {
    return {
      privateKey: pair.private_key,
      publicKey: pair.public_key,
    };
  } finally {
    pair.free();
  }
}

/** Reusable client configuration for one recipient key and one explicit PSK identity. */
export class Client {
  #native: NativeClient | undefined;
  readonly #maximumBody: number;

  /**
   * Validate and copy one client credential configuration.
   *
   * @param recipientPublicKey - Encoded 32-byte X25519 public key.
   * @param recipientKeyId - Non-empty public key identifier, at most 255 bytes.
   * @param psk - Secret of at least 32 bytes.
   * @param pskId - Non-empty public opaque identifier, at most 255 bytes and not equal to `psk`.
   * @param limits - Optional limits applied to every request and response.
   * @param compression - Optional authenticated body coding. Ciphertext size can reveal
   * compression ratio, so do not mix attacker input and secrets in one body.
   * @throws {@link ProtocolError} with `invalid_configuration` for invalid input.
   */
  public constructor(
    recipientPublicKey: Uint8Array,
    recipientKeyId: Uint8Array,
    psk: Uint8Array,
    pskId: Uint8Array,
    limits: Limits = {},
    compression?: CompressionCoding,
  ) {
    const module = requireNative();
    const normalized = normalizeLimits(limits);
    const nativeLimits = makeLimits(module, normalized);
    this.#maximumBody = maximumBodyLength(normalized);
    try {
      this.#native = callNative(
        () =>
          new module.Client(
            ownedBytes(recipientPublicKey),
            ownedBytes(recipientKeyId),
            ownedBytes(psk),
            ownedBytes(pskId),
            nativeLimits,
            compression === undefined ? 0 : compression === "gzip" ? 1 : compression === "zstd" ? 2 : 255,
          ),
      );
    } finally {
      nativeLimits.free();
    }
  }

  /**
   * Protect one complete request and return its one-shot response transaction.
   *
   * Each call creates a fresh envelope. Retries must call `protect` again.
   */
  public protect(request: Request): ProtectedRequest {
    const native = requireHandle(this.#native);
    checkLength((request.body ?? EMPTY_BYTES).byteLength, this.#maximumBody);
    const protectedRequest = callNative(() =>
      native.protect(
        request.method,
        request.authority,
        request.path,
        encodeHeaders(request.headers),
        ownedBytes(request.body ?? EMPTY_BYTES),
        currentUnixSeconds(),
      ),
    );
    return createProtectedRequest(protectedRequest);
  }

  /**
   * Release native credential copies. This method is idempotent.
   *
   * It cannot clear arrays that the caller passed to the constructor.
   */
  public close(): void {
    this.#native?.free();
    this.#native = undefined;
  }
}

/** Protected request bytes plus the one-shot capability that opens its response. */
export class ProtectedRequest {
  #native: NativeProtectedRequest | undefined;
  /** Owned copy of the complete protected request envelope. */
  public readonly envelope: Uint8Array;

  static {
    createProtectedRequest = (native) => new ProtectedRequest(native);
  }

  private constructor(native: NativeProtectedRequest) {
    this.#native = native;
    this.envelope = native.take_envelope();
  }

  /** Return whether the response capability was used or discarded. */
  public get consumed(): boolean {
    return this.#native === undefined || this.#native.consumed;
  }

  /** Check one complete finite response through the v2 record reader. */
  public openResponse(envelope: Uint8Array): Response {
    const opener = this.intoOpener();
    try {
      let offset = 0;
      while (offset < envelope.byteLength) {
        const { consumed, record } = opener.feed(envelope.subarray(offset, Math.min(offset + 64 * 1024, envelope.byteLength)));
        if (record?.kind === "start" && record.mode === "sse") {
          throw new StateError();
        }
        if (consumed === 0) {
          throw new ProtocolError("malformed_envelope", "response reader made no progress");
        }
        offset += consumed;
      }
      const response = opener.finishEof();
      if (response === undefined) {
        throw new ProtocolError("malformed_envelope", "finite response body is missing");
      }
      return response;
    } finally {
      opener.close();
    }
  }

  /** Transfer the one response right to a live checked record reader. */
  public intoOpener(): ResponseOpener {
    const native = requireHandle(this.#native);
    try {
      return new ResponseOpener(callNative(() => native.into_opener()));
    } finally {
      this.close();
    }
  }

  /** Discard the response capability. This method is idempotent. */
  public close(): void {
    if (this.#native !== undefined) {
      this.#native.discard();
      this.#native.free();
      this.#native = undefined;
    }
  }
}

/** One checked response record. SSE DATA holds one clear complete block. */
export type CheckedRecord =
  | { readonly kind: "start"; readonly status: number; readonly headers: readonly Header[]; readonly mode: "finite" | "sse" }
  | { readonly kind: "sse_data"; readonly block: Uint8Array }
  | { readonly kind: "end" };

/** One response reader that accepts any byte cuts. */
export class ResponseOpener {
  #native: NativeResponseOpener | undefined;

  public constructor(native: NativeResponseOpener) { this.#native = native; }

  /** Consume at most one record; finite DATA returns no public record. */
  public feed(input: Uint8Array): { readonly consumed: number; readonly record?: CheckedRecord } {
    const native = requireHandle(this.#native);
    const result = callNative(() => native.feed(input));
    try {
      let record: CheckedRecord | undefined;
      switch (result.kind) {
        case 1:
          record = { kind: "start", status: result.status, headers: decodeHeaders(result.headers_json), mode: result.mode === 2 ? "sse" : "finite" };
          break;
        case 2:
          record = { kind: "sse_data", block: result.block };
          break;
        case 3:
          record = { kind: "end" };
          break;
      }
      return record === undefined ? { consumed: result.consumed } : { consumed: result.consumed, record };
    } finally {
      result.free();
    }
  }

  /** Confirm true outer body EOF after END. */
  public finishEof(): Response | undefined {
    const native = requireHandle(this.#native);
    const response = callNative(() => native.finish_eof());
    this.close();
    return response === undefined ? undefined : responseFromNative(response);
  }

  /** Stop without a complete-response claim. */
  public close(): void {
    if (this.#native !== undefined) {
      this.#native.close();
      this.#native.free();
      this.#native = undefined;
    }
  }
}

/** Reusable server configuration for one static recipient key. */
export class Server {
  /**
   * Validate and copy one server key configuration.
   *
   * @param recipientPrivateKey - Encoded 32-byte X25519 private key.
   * @param recipientKeyId - Non-empty public key identifier, at most 255 bytes.
   * @param limits - Optional limits applied to every request and response.
   * @param compression - Accept gzip/zstd protocol body coding when advertised by a client.
   */
  public constructor(
    recipientPrivateKey: Uint8Array,
    recipientKeyId: Uint8Array,
    limits: Limits = {},
    compression = false,
  ) {
    const module = requireNative();
    const normalized = normalizeLimits(limits);
    const nativeLimits = makeLimits(module, normalized);
    try {
      const native = callNative(
        () =>
          new module.Server(
            ownedBytes(recipientPrivateKey),
            ownedBytes(recipientKeyId),
            nativeLimits,
            compression,
          ),
      );
      serverHandles.set(this, native);
      serverLimits.set(this, {
        body: maximumBodyLength(normalized),
        envelope: maximumEnvelopeLength(normalized),
      });
    } finally {
      nativeLimits.free();
    }
  }

  /** Parse bounded public fields without accepting credentials or releasing plaintext. */
  public preparse(envelope: Uint8Array): PreparsedRequest {
    const native = requireServerHandle(this);
    checkLength(envelope.byteLength, requireServerLimits(this).envelope);
    return createPreparsedRequest(
      callNative(() => native.preparse(ownedBytes(envelope))),
      this,
    );
  }

  /**
   * Release the native private-key copy and revoke pending pre-authentication stages.
   * This method is idempotent and cannot clear the caller-owned key array.
   */
  public close(): void {
    const native = serverHandles.get(this);
    native?.free();
    serverHandles.delete(this);
    serverLimits.delete(this);
  }
}

/** One-shot server stage waiting for host PSK resolution. */
export class PreparsedRequest {
  #native: NativePreparsedRequest | undefined;
  readonly #server: Server;
  /** Owned copy of the public opaque PSK identifier. It is not the PSK. */
  public readonly pskId: Uint8Array;

  static {
    createPreparsedRequest = (native, server) => new PreparsedRequest(native, server);
  }

  private constructor(native: NativePreparsedRequest, server: Server) {
    this.#native = native;
    this.#server = server;
    this.pskId = ownedBytes(native.psk_id);
  }

  /** Return whether this stage was used, discarded, or revoked with its server. */
  public get consumed(): boolean {
    return (
      this.#native === undefined || this.#native.consumed || !serverHandles.has(this.#server)
    );
  }

  /**
   * Consume this stage, authenticate the request, and retain plaintext until replay admission.
   */
  public authenticate(psk: Uint8Array): AuthenticatedRequest {
    const native = requireHandle(this.#native);
    try {
      return createAuthenticatedRequest(
        callNative(() =>
          native.authenticate(
            requireServerHandle(this.#server),
            ownedBytes(psk),
            currentUnixSeconds(),
          ),
        ),
        requireServerLimits(this.#server).body,
      );
    } finally {
      this.close();
    }
  }

  /** Discard this stage without authenticating. This method is idempotent. */
  public close(): void {
    if (this.#native !== undefined) {
      this.#native.discard();
      this.#native.free();
      this.#native = undefined;
    }
  }
}

/** Authenticated request paused before atomic replay admission and plaintext release. */
export class AuthenticatedRequest {
  #native: NativeAuthenticatedRequest | undefined;
  readonly #maximumBody: number;
  /** Stable 32-byte key for one atomic reserve-if-absent operation. */
  public readonly replayId: Uint8Array;

  /** Exclusive Unix-second deadline through which the replay reservation must remain. */
  public readonly retainUntilExclusive: number;

  static {
    createAuthenticatedRequest = (native, maximumBody) => new AuthenticatedRequest(native, maximumBody);
  }

  private constructor(native: NativeAuthenticatedRequest, maximumBody: number) {
    this.#native = native;
    this.#maximumBody = maximumBody;
    this.replayId = ownedBytes(native.replay_id);
    this.retainUntilExclusive = native.retain_until_exclusive;
  }

  /** Return whether this admission stage was used or discarded. */
  public get consumed(): boolean {
    return this.#native === undefined || this.#native.consumed;
  }

  /**
   * Consume this stage and apply the result of one atomic replay-store operation.
   *
   * Plaintext is released only when `accepted` is true and the authenticated
   * deadline has not passed. Uncertain store results must be reported as false.
   */
  public admit(options: { readonly accepted: boolean }): OpenedRequest {
    const native = requireHandle(this.#native);
    try {
      return createOpenedRequest(
        callNative(() => native.admit(options.accepted, currentUnixSeconds())),
        this.#maximumBody,
      );
    } finally {
      this.close();
    }
  }

  /** Discard the authenticated request without releasing plaintext. */
  public close(): void {
    if (this.#native !== undefined) {
      this.#native.discard();
      this.#native.free();
      this.#native = undefined;
    }
  }
}

/** Verified plaintext request plus its one-shot response protector. */
export class OpenedRequest {
  #native: NativeOpenedRequest | undefined;
  readonly #maximumBody: number;
  /** Owned authenticated request data that is safe for application dispatch. */
  public readonly request: Required<Request>;

  static {
    createOpenedRequest = (native, maximumBody) => new OpenedRequest(native, maximumBody);
  }

  private constructor(native: NativeOpenedRequest, maximumBody: number) {
    this.#native = native;
    this.#maximumBody = maximumBody;
    this.request = {
      method: parseMethod(native.method),
      authority: native.authority,
      path: native.path,
      headers: decodeHeaders(native.headers_json),
      body: native.take_body(),
    };
  }

  /** Return whether the response capability was used or discarded. */
  public get responseConsumed(): boolean {
    return this.#native === undefined || this.#native.response_consumed;
  }

  /** Consume the capability and protect one complete response. */
  public protectResponse(response: Response): Uint8Array {
    const native = requireHandle(this.#native);
    if (!Number.isSafeInteger(response.status) || response.status < 200 || response.status > 599) {
      throw new ProtocolError(
        "invalid_configuration",
        "response status must be an integer from 200 through 599",
      );
    }
    try {
      checkLength((response.body ?? EMPTY_BYTES).byteLength, this.#maximumBody);
      return callNative(() =>
        native.protect_response(
          response.status,
          encodeHeaders(response.headers),
          ownedBytes(response.body ?? EMPTY_BYTES),
        ),
      );
    } finally {
      this.close();
    }
  }

  /** Start a checked response stream and transfer its one-use writer right. */
  public startResponse(status: number, headers: readonly Header[]): ResponseSealer {
    const native = requireHandle(this.#native);
    if (!Number.isSafeInteger(status) || status < 200 || status > 599) {
      throw new ProtocolError("invalid_configuration", "response status must be 200 through 599");
    }
    try {
      return new ResponseSealer(callNative(() => native.start_response(status, encodeHeaders(headers))));
    } finally {
      this.close();
    }
  }

  /** Discard the response capability. This method is idempotent. */
  public close(): void {
    if (this.#native !== undefined) {
      this.#native.discard_response();
      this.#native.free();
      this.#native = undefined;
    }
  }
}

/** One checked response writer for a finite body or complete LF-normalized SSE blocks. */
export class ResponseSealer {
  #native: NativeResponseSealer | undefined;
  public readonly start: Uint8Array;

  public constructor(native: NativeResponseSealer) {
    this.#native = native;
    this.start = native.take_start();
  }

  /** Protect a finite body once. An empty body emits no DATA record. */
  public sealFiniteBody(body: Uint8Array): Uint8Array | undefined {
    return callNative(() => requireHandle(this.#native).seal_finite_body(ownedBytes(body)));
  }

  /** Protect one complete LF-normalized SSE block, including its final blank line. */
  public sealSseBlock(block: Uint8Array): Uint8Array {
    return callNative(() => requireHandle(this.#native).seal_sse_block(ownedBytes(block)));
  }

  /** Protect END. The caller must then end the outer HTTP body. */
  public finish(): Uint8Array {
    const native = requireHandle(this.#native);
    try { return callNative(() => native.finish()); }
    finally { this.close(); }
  }

  /** Discard the writer without END. This method is idempotent. */
  public close(): void {
    if (this.#native !== undefined) {
      this.#native.close();
      this.#native.free();
      this.#native = undefined;
    }
  }
}

const EMPTY_BYTES = new Uint8Array();

function requireNative(): NativeModule {
  if (nativeModule === undefined) {
    throw new InitializationError(
      "hpke-http is not initialized; await initialize() from the browser or node entry point",
    );
  }
  return nativeModule;
}

function makeLimits(module: NativeModule, limits: Limits): InstanceType<NativeModule["Limits"]> {
  return callNative(
    () =>
      new module.Limits(
        limits.maxBodyLength,
        limits.maxHeaderBytes,
        limits.maxHeaderCount,
        limits.maxTargetLength,
      ),
  );
}

/** @internal */
export function normalizeLimits(limits: Limits): Limits {
  const normalized: {
    maxBodyLength?: number;
    maxHeaderBytes?: number;
    maxHeaderCount?: number;
    maxTargetLength?: number;
  } = {};
  const entries = [
    ["maxBodyLength", limits.maxBodyLength],
    ["maxHeaderBytes", limits.maxHeaderBytes],
    ["maxHeaderCount", limits.maxHeaderCount],
    ["maxTargetLength", limits.maxTargetLength],
  ] as const;
  for (const [name, value] of entries) {
    if (value !== undefined) {
      normalized[name] = validateLimit(name, value);
    }
  }
  return normalized;
}

function validateLimit(name: keyof Limits, value: number): number {
  const maximums: Readonly<Record<keyof Limits, number>> = {
    maxBodyLength: 64 * 1024 * 1024,
    maxHeaderBytes: 64 * 1024,
    maxHeaderCount: 256,
    maxTargetLength: 8 * 1024,
  };
  const maximum = maximums[name];
  if (!Number.isSafeInteger(value) || value < 0 || value > maximum) {
    throw new ProtocolError(
      "invalid_configuration",
      `${name} must be an integer from 0 through ${String(maximum)}`,
    );
  }
  return value;
}

function currentUnixSeconds(): number {
  const seconds = Math.floor(Date.now() / 1000);
  if (!Number.isSafeInteger(seconds) || seconds < 0) {
    throw new ProtocolError("clock_unavailable", "system clock cannot provide a Unix timestamp");
  }
  return seconds;
}

function requireHandle<T>(handle: T | undefined): T {
  if (handle === undefined) {
    throw new StateError();
  }
  return handle;
}

function requireServerHandle(server: Server): NativeServer {
  return requireHandle(serverHandles.get(server));
}

function requireServerLimits(server: Server): { body: number; envelope: number } {
  return requireHandle(serverLimits.get(server));
}

function maximumBodyLength(limits: Limits): number {
  return limits.maxBodyLength ?? 8 * 1024 * 1024;
}

function maximumEnvelopeLength(limits: Limits): number {
  // Covers the bounded BHTTP fields, request target, outer IDs, and crypto framing.
  return maximumBodyLength(limits) + (limits.maxHeaderBytes ?? 16 * 1024) + 64 * 1024;
}

function checkLength(length: number, maximum: number): void {
  if (length > maximum) {
    throw new ProtocolError("limit_exceeded", "limit exceeded");
  }
}

function callNative<T>(operation: () => T): T {
  try {
    return operation();
  } catch (error: unknown) {
    throw mapNativeError(error);
  }
}

function mapNativeError(error: unknown): Error {
  const raw = typeof error === "string" ? error : undefined;
  if (raw !== undefined) {
    const separator = raw.indexOf("\n");
    const code = separator === -1 ? "native_error" : raw.slice(0, separator);
    const message = separator === -1 ? raw : raw.slice(separator + 1);
    return code === "state_consumed" ? new StateError() : new ProtocolError(code, message);
  }
  return error instanceof Error ? error : new ProtocolError("native_error", String(error));
}

function ownedBytes(value: Uint8Array): Uint8Array {
  return Uint8Array.from(value);
}

function encodeHeaders(headers: readonly Header[] | undefined): string {
  return JSON.stringify((headers ?? []).map((field) => [field.name, field.value]));
}

function decodeHeaders(value: string): readonly Header[] {
  const parsed: unknown = JSON.parse(value);
  if (
    !Array.isArray(parsed) ||
    !parsed.every(
      (field: unknown) =>
        Array.isArray(field) &&
        field.length === 2 &&
        typeof field[0] === "string" &&
        typeof field[1] === "string",
    )
  ) {
    throw new ProtocolError("malformed_envelope", "native module returned invalid headers");
  }
  return Object.freeze(
    parsed.map((field: [string, string]) => Object.freeze({ name: field[0], value: field[1] })),
  );
}

function responseFromNative(native: NativeResponse): Response {
  try {
    return {
      status: native.status,
      headers: decodeHeaders(native.headers_json),
      body: native.body,
    };
  } finally {
    native.free();
  }
}

function parseMethod(value: string): Method {
  if (
    value === "GET" ||
    value === "POST" ||
    value === "PUT" ||
    value === "PATCH" ||
    value === "DELETE" ||
    value === "HEAD" ||
    value === "OPTIONS"
  ) {
    return value;
  }
  throw new ProtocolError("malformed_envelope", "native module returned an invalid method");
}

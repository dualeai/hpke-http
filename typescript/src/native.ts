/** Private structural types for wasm-bindgen output. */

export interface NativeDisposable {
  free(): void;
}

export interface NativeLimits extends NativeDisposable {}

export interface NativeKeyPair extends NativeDisposable {
  readonly private_key: Uint8Array;
  readonly public_key: Uint8Array;
}

export interface NativeResponse extends NativeDisposable {
  readonly status: number;
  readonly headers_json: string;
  readonly body: Uint8Array;
}

export interface NativeProtectedRequest extends NativeDisposable {
  take_envelope(): Uint8Array;
  readonly consumed: boolean;
  open_finite_response(envelope: Uint8Array): NativeResponse;
  into_opener(): NativeResponseOpener;
  discard(): void;
}

export interface NativeStreamPush extends NativeDisposable {
  readonly consumed: number;
  take_record(): Uint8Array | undefined;
}

export interface NativeStreamRequestSealer extends NativeDisposable {
  take_start(): Uint8Array;
  push(input: Uint8Array): NativeStreamPush;
  finish(): NativeProtectedRequest;
  close(): void;
}

export interface NativeFeedResult extends NativeDisposable {
  readonly consumed: number;
  readonly kind: number;
  readonly status: number;
  readonly headers_json: string;
  readonly mode: number;
  readonly block: Uint8Array;
}

export interface NativeResponseOpener extends NativeDisposable {
  feed(input: Uint8Array): NativeFeedResult;
  finish_eof(): NativeResponse | undefined;
  close(): void;
}

export interface NativeResponseSealer extends NativeDisposable {
  take_start(): Uint8Array;
  seal_finite_body(body: Uint8Array): Uint8Array | undefined;
  seal_sse_block(block: Uint8Array): Uint8Array;
  finish(): Uint8Array;
  close(): void;
}

export interface NativeClient extends NativeDisposable {
  begin_stream(method: string, authority: string, path: string, headersJson: string, nowUnixSeconds: number): NativeStreamRequestSealer;
  protect(
    method: string,
    authority: string,
    path: string,
    headersJson: string,
    body: Uint8Array,
    nowUnixSeconds: number,
  ): NativeProtectedRequest;
}

export interface NativePreparsedRequest extends NativeDisposable {
  readonly psk_id: Uint8Array;
  readonly consumed: boolean;
  authenticate(
    server: NativeServer,
    psk: Uint8Array,
    nowUnixSeconds: number,
  ): NativeAuthenticatedRequest;
  discard(): void;
}

export interface NativePreparsedStreamRequest extends NativeDisposable {
  readonly psk_id: Uint8Array;
  readonly consumed: boolean;
  authenticate(server: NativeServer, psk: Uint8Array, nowUnixSeconds: number): NativeAuthenticatedStreamRequest;
  discard(): void;
}

export interface NativeAuthenticatedStreamRequest extends NativeDisposable {
  readonly replay_id: Uint8Array;
  readonly retain_until_exclusive: number;
  readonly consumed: boolean;
  admit(accepted: boolean, nowUnixSeconds: number): NativeOpenedStreamRequest;
  discard(): void;
}

export interface NativeRequestFeed extends NativeDisposable {
  readonly consumed: number;
  readonly kind: number;
  readonly block: Uint8Array;
}

export interface NativeOpenedStreamRequest extends NativeDisposable {
  readonly method: string;
  readonly authority: string;
  readonly path: string;
  readonly headers_json: string;
  feed(input: Uint8Array): NativeRequestFeed;
  finish_eof(): NativeOpenedRequest;
  close(): void;
}

export interface NativeAuthenticatedRequest extends NativeDisposable {
  readonly replay_id: Uint8Array;
  readonly retain_until_exclusive: number;
  readonly consumed: boolean;
  admit(accepted: boolean, nowUnixSeconds: number): NativeOpenedRequest;
  discard(): void;
}

export interface NativeOpenedRequest extends NativeDisposable {
  readonly method: string;
  readonly authority: string;
  readonly path: string;
  readonly headers_json: string;
  take_body(): Uint8Array;
  readonly response_consumed: boolean;
  protect_response(status: number, headersJson: string, body: Uint8Array): Uint8Array;
  start_response(status: number, headersJson: string): NativeResponseSealer;
  discard_response(): void;
}

export interface NativeServer extends NativeDisposable {
  preparse(envelope: Uint8Array): NativePreparsedRequest;
  max_complete_envelope_len(): number;
  stream_start_length(input: Uint8Array): number | undefined;
  preparse_stream(first: Uint8Array): NativePreparsedStreamRequest;
}

interface NativeConstructor<T, Arguments extends readonly unknown[]> {
  new (...args: Arguments): T;
}

export interface NativeModule {
  engine_version(): string;
  protocol_id(): string;
  binding_abi(): number;
  generateKeyPair(): NativeKeyPair;
  Limits: NativeConstructor<
    NativeLimits,
    [number | undefined, number | undefined, number | undefined, number | undefined, number | undefined]
  >;
  Client: NativeConstructor<
    NativeClient,
    [Uint8Array, Uint8Array, Uint8Array, Uint8Array, NativeLimits]
  >;
  Server: NativeConstructor<NativeServer, [Uint8Array, Uint8Array, NativeLimits]>;
}

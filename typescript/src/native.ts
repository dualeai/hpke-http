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
  open_response(envelope: Uint8Array): NativeResponse;
  discard(): void;
}

export interface NativeClient extends NativeDisposable {
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
  discard_response(): void;
}

export interface NativeServer extends NativeDisposable {
  preparse(envelope: Uint8Array): NativePreparsedRequest;
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
    [number | undefined, number | undefined, number | undefined, number | undefined]
  >;
  Client: NativeConstructor<
    NativeClient,
    [Uint8Array, Uint8Array, Uint8Array, Uint8Array, NativeLimits, number]
  >;
  Server: NativeConstructor<NativeServer, [Uint8Array, Uint8Array, NativeLimits, boolean]>;
}

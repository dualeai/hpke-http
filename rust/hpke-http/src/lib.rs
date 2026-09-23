//! Safe, synchronous, bounded whole-message HPKE transactions for HTTP.
//!
//! This crate is the self-documenting source contract for stable protocol
//! [`PROTOCOL_ID`]. It performs no networking and exposes no transport or
//! framework objects. Protocol version 1 composes:
//!
//! - RFC 9180 HPKE in PSK mode with DHKEM(X25519, HKDF-SHA256), HKDF-SHA256,
//!   and ChaCha20-Poly1305;
//! - canonical RFC 9292 known-length Binary HTTP Messages as plaintext, with
//!   an optional authenticated body-coding extension; and
//! - the response-secret, recipient-nonce, and `enc || response_nonce`
//!   derivation pattern from RFC 9458 section 4.4.
//!
//! The protocol is not wire-compatible with RFC 9458 Oblivious HTTP. Its narrow PSK
//! request envelope is project-specific. HTTPS remains required because the
//! protocol does not hide its endpoint, public identifiers, sizes, or timing.
//!
//! # Request bytes
//!
//! Multi-byte integers use network byte order. The authenticated HPKE `info`
//! is `"message/hpke-http request" || 0x00 || "v1" || 0x00 || header`, where `header`
//! ends after the PSK ID. HPKE seals the request plaintext with empty AAD.
//!
//! | Field | Size | Value |
//! | --- | ---: | --- |
//! | magic | 4 | ASCII `HHRQ` |
//! | version | 1 | `0x01` |
//! | flags | 1 | `0x00` |
//! | recipient key-ID length | 1 | `1..=255` |
//! | PSK-ID length | 1 | `1..=255` |
//! | KEM ID | 2 | `0x0020` |
//! | KDF ID | 2 | `0x0001` |
//! | AEAD ID | 2 | `0x0003` |
//! | issued-at Unix seconds | 8 | unsigned network-order client time |
//! | recipient key ID | variable | opaque public bytes |
//! | PSK ID | variable | opaque public bytes, never the PSK |
//! | `enc` | 32 | encoded X25519 encapsulated key |
//! | ciphertext | variable | request plaintext plus 16-byte tag |
//!
//! The PSK is at least 32 bytes. Both IDs are non-empty, at most 255 bytes,
//! and the PSK ID must not equal the PSK.
//!
//! # Optional authenticated body coding
//!
//! Identity requests seal the original canonical BHTTP bytes, unchanged. A
//! client that explicitly selects [`CompressionCoding`] instead seals
//! `0x04 || request_coding || response_coding || BHTTP_request`, where coding
//! IDs are 0 for identity, 1 for gzip, and 2 for zstd. `response_coding` is
//! the client's advertised preference. `0x04` is not a BHTTP request framing
//! indicator, so a peer without this extension rejects the request. The
//! server must explicitly enable the extension; it never compresses a response
//! for an identity-only client. When compression saves bytes, the server seals
//! `0x04 || response_coding || BHTTP_response`; otherwise it seals the original
//! canonical BHTTP response. Unadvertised response coding is rejected.
//!
//! Only the BHTTP body vector is coded. Its fields remain logical HTTP fields;
//! in particular, `content-length` names the decoded logical body and is
//! checked after bounded decompression. This private, authenticated transform
//! is independent of HTTP `Content-Encoding`, which remains ordinary opaque
//! representation metadata at the low-level API. Gzip accepts concatenated
//! members; zstd accepts up to 16 concatenated or skippable frames without
//! dictionaries, with history windows capped at 8 MiB. Both coded and decoded
//! bodies have the configured body limit. The request is decompressed only after replay
//! admission. Compression is off by default because ciphertext length can
//! reveal information about a body containing both secrets and attacker input.
//!
//! # Authenticated HTTP subset
//!
//! Requests use the fixed `https` scheme and one of [`Method`]'s seven values.
//! The authority and path are bounded ASCII. Authorities follow the RFC 3986
//! host plus optional decimal-port grammar, without userinfo or normalization;
//! IPv6 and `IPvFuture` literals require brackets. Request targets reject
//! malformed percent escapes, backslashes, fragments, and dot segments; `*` is
//! limited to `OPTIONS`. Headers are ordered fields with lower-case HTTP token names and
//! canonical ASCII values without leading or trailing optional whitespace;
//! repeated names remain ordered. Connection-specific framing, unsupported
//! proxy semantics, and `Expect` are rejected: `connection`, `expect`, `host`,
//! `keep-alive`, `proxy-authenticate`, `proxy-authentication-info`,
//! `proxy-authorization`, `proxy-connection`, `te`, `trailer`,
//! `transfer-encoding`, and `upgrade`. A single decimal `content-length` is
//! optional. It must equal the authenticated body length on requests and normal
//! responses, is metadata on `HEAD` and 304 responses, and is forbidden on 204.
//! Responses contain one status from 200 through 599. `HEAD` responses and
//! status 204, 205, or 304 responses cannot contain a body. Empty bodies remain
//! part of an authenticated canonical message.
//!
//! # Replay pause and one-shot response
//!
//! After complete request authentication and canonical parsing, [`Server`]
//! checks the authenticated issued-at time against [`REQUEST_LIFETIME_SECS`]
//! and [`CLOCK_SKEW_SECS`]. It then returns
//! `SHA-256("hpke-http/replay" || 0x00 || "v1" || 0x00 || header || enc)` and an
//! exclusive retention deadline through [`ReplayRequest`]. The verified
//! plaintext stays in an opaque [`ReplayToken`] until the host reports one
//! matching atomic replay-admission decision. A provider must reserve the ID
//! until that deadline and fail closed on an uncertain result. This crate does
//! not provide or claim a distributed replay store. [`ReplayToken::admit`]
//! reads the trusted clock again and refuses to release plaintext at or after
//! the authenticated deadline, including when a provider call was delayed.
//!
//! Both HPKE contexts export 32 bytes with context
//! `"message/hpke-http response" || 0x00 || "v1"`. A server samples a fresh 32-byte
//! `response_nonce`, uses `enc || response_nonce` as HKDF-SHA256 salt, and
//! expands the exported secret with raw info `"key"` (32 bytes) and
//! `"nonce"` (12 bytes). The response bytes are:
//!
//! ```text
//! response_nonce[32] || ChaCha20Poly1305(response_plaintext, empty_aad)
//! ```
//!
//! [`ResponseToken`] and [`ResponseCapability`] consume ownership, so each
//! request can open and create at most one response.
//!
//! # Complete transaction
//!
//! ```
//! use hpke_http::{
//!     Client, Limits, Method, Request, Response, Server, generate_key_pair,
//! };
//!
//! let key_pair = generate_key_pair()?;
//! let (recipient_private_key, recipient_public_key) = key_pair.into_parts();
//! let recipient_key_id = b"primary-2026-09".to_vec();
//! let psk = vec![0x42; 32];
//! let psk_id = b"tenant-42".to_vec();
//! let limits = Limits::default();
//!
//! let client = Client::new(
//!     &recipient_public_key,
//!     recipient_key_id.clone(),
//!     psk.clone(),
//!     psk_id.clone(),
//!     limits,
//! )?;
//! let server = Server::new(&recipient_private_key, recipient_key_id, limits)?;
//! let request = Request {
//!     method: Method::Post,
//!     authority: b"api.example.test".to_vec(),
//!     path: b"/items".to_vec(),
//!     headers: Vec::new(),
//!     body: b"payload".to_vec(),
//! };
//!
//! let protected = client.protect(&request)?;
//! let (request_envelope, response_token) = protected.into_parts();
//! let preparsed = server.preparse(&request_envelope)?;
//! assert_eq!(preparsed.credential.psk_id, psk_id);
//! let authenticated = server.authenticate(preparsed.token, &psk)?;
//! let replay_decision = authenticated.replay.decision(true);
//! let opened = authenticated.token.admit(replay_decision)?;
//! assert_eq!(opened.request, request);
//!
//! let expected = Response {
//!     status: 200,
//!     headers: Vec::new(),
//!     body: b"ok".to_vec(),
//! };
//! let response_envelope = opened.response.protect(&expected)?;
//! assert_eq!(response_token.open(&response_envelope)?, expected);
//! # Ok::<(), hpke_http::Error>(())
//! ```
//!
//! # Limits and errors
//!
//! [`Limits`] defaults to an 8 MiB body, 16 KiB of header names/values, 64
//! fields, and an 8 KiB authority-plus-path. Its documented hard ceilings are
//! enforced before attacker-controlled lengths drive allocations. [`Error`]
//! provides stable coarse categories and contains no credentials, plaintext,
//! ciphertext, or parser offsets.
//!
//! The crate version, [`PROTOCOL_ID`], and [`BINDING_ABI_VERSION`] are distinct
//! identities. First-party bindings call [`build_info`] before accepting
//! credentials. Incompatible wire changes require a new protocol identifier and
//! envelope version rather than reinterpretation of request version `0x01`.

#![forbid(unsafe_code)]

mod codec;
mod compression;
mod engine;
mod entropy;
mod error;
mod limits;
mod message;
mod method;

pub use compression::CompressionCoding;
pub use engine::{
    AuthenticatedRequest, Client, CredentialRequest, KeyPair, OpenedRequest, PreparsedRequest,
    ProtectedRequest, ReplayDecision, ReplayRequest, ReplayToken, ResponseCapability,
    ResponseToken, Server, StartToken, generate_key_pair, generate_key_pair_with_entropy,
};
pub use entropy::{EntropySource, SystemEntropy};
pub use error::Error;
pub use limits::{
    HARD_MAX_BODY_LEN, HARD_MAX_HEADER_BYTES, HARD_MAX_HEADER_COUNT, HARD_MAX_ID_LEN,
    HARD_MAX_TARGET_LEN, Limits,
};
pub use message::{HeaderField, Request, Response};
pub use method::Method;

/// Language-neutral protocol identifier.
pub const PROTOCOL_ID: &str = "hpke-http/1";
/// Boundary ABI version used by the first-party bindings.
pub const BINDING_ABI_VERSION: u32 = 1;
/// Time for which a newly created request can be accepted (five minutes).
pub const REQUEST_LIFETIME_SECS: u64 = 300;
/// Maximum accepted client/server clock difference (30 seconds).
pub const CLOCK_SKEW_SECS: u64 = 30;

/// Immutable engine identity returned before a binding accepts secrets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BuildInfo {
    /// Rust crate version.
    pub version: &'static str,
    /// Language-neutral protocol identifier.
    pub protocol: &'static str,
    /// Boundary ABI version.
    pub binding_abi: u32,
}

/// Return the engine identity used for binding skew checks.
#[must_use]
pub const fn build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        protocol: PROTOCOL_ID,
        binding_abi: BINDING_ABI_VERSION,
    }
}

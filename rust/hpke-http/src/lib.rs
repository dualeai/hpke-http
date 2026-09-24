//! Bounded HPKE requests and checked response records for HTTP.
//!
//! This crate defines the sole wire contract, [`PROTOCOL_ID`] `hpke-http/2`.
//! It does no network I/O. HTTP hosts must use HTTPS because this protocol
//! does not hide endpoints, public IDs, record sizes, counts, or timing.
//! The suite is RFC 9180 HPKE PSK mode with X25519, HKDF-SHA256, and
//! ChaCha20-Poly1305. The request plaintext uses canonical known-length
//! RFC 9292 Binary HTTP. This protocol is not RFC 9458 Oblivious HTTP.
//!
//! # Request bytes
//!
//! All fixed-width integers are big-endian. The request is the following
//! byte string, with no trailing fields:
//!
//! | Field | Bytes | Value |
//! | --- | ---: | --- |
//! | magic | 4 | ASCII `HHRQ` |
//! | version, flags | 1 each | `0x02`, `0x00` |
//! | key ID length, PSK ID length | 1 each | `1..=255` |
//! | KEM, KDF, AEAD IDs | 2 each | `0x0020`, `0x0001`, `0x0003` |
//! | issued-at Unix seconds | 8 | client time |
//! | key ID, PSK ID | stated lengths | opaque public IDs |
//! | `enc` | 32 | X25519 HPKE encapsulated key |
//! | ciphertext | variable | request plaintext and 16-byte tag |
//!
//! `header` ends after the PSK ID. HPKE `info` is
//! `"message/hpke-http request\0v2\0" || header`; request AEAD AAD is empty.
//! The PSK must be at least 32 bytes long, contain at least 32 bytes of
//! entropy, and differ from its public ID.
//! Normal request plaintext is a canonical known-length Binary HTTP request.
//! If the client opts into private body coding, plaintext is
//! `0x04 || request_coding:u8 || response_coding:u8 || BHTTP_request`.
//! Coding IDs are 0 (identity), 1 (gzip), and 2 (zstd). Only the body bytes
//! are coded; the logical headers and `Content-Length` name clear bytes.
//! Coding is off by default because compressed length can leak information.
//!
//! # Public key discovery
//!
//! The crate does no HTTP I/O. A host can call [`Server::public_key`] to
//! publish its current X25519 public key. The Python ASGI host serves one
//! binary record through GET at the same HTTPS URL that accepts protected
//! POST: `"HHKD" || 0x01 || id_len:u8 || id || public_key[32]`. `id_len` is
//! 1 through 255, and the complete record is 39 through 293 bytes. The key
//! record version is separate from [`PROTOCOL_ID`]. Clients bind this GET and
//! the later POST to one configured HTTPS endpoint and one logical origin.
//! The host must keep the one Server key equal across its workers.
//!
//! After request authentication and parsing, the server checks the client
//! time against [`REQUEST_LIFETIME_SECS`] and [`CLOCK_SKEW_SECS`]. The replay
//! ID is SHA-256 of `"hpke-http/replay\0v2\0" || header || enc`.
//! [`ReplayToken::admit`] needs one host replay decision and checks the
//! deadline again. The host must reserve the ID through that deadline.
//!
//! # Response bytes
//!
//! Each HPKE context exports 32 bytes with context
//! `"message/hpke-http response\0v2"`. The server samples a fresh 32-byte
//! `server_nonce`. HKDF-SHA256 uses `enc || server_nonce` as salt and the
//! exported secret as input key material. It expands 32 bytes with info
//! `"hpke-http/2 response key"` and 12 bytes with info
//! `"hpke-http/2 response nonce"`. A response has this exact form:
//!
//! ```text
//! "HHRP" || 0x02 || server_nonce[32] || records...
//! record = ciphertext_len:u32be || ciphertext[ciphertext_len]
//! ```
//!
//! The 37-byte prefix stays fixed for all records. Each record plaintext
//! starts with a one-byte kind. `START` is `0x01 || status:u16be ||
//! coding:u8 || fields`. `fields` is the ordered sequence of name/value
//! pairs, each encoded as an RFC 9292 QUIC variable-length integer and that
//! many raw bytes. There is no field count or end marker: the START record
//! end terminates the sequence. `DATA` is `0x02 || body`; `END` is only
//! `0x03`. The record length includes the 16-byte tag and must be at least
//! 17. The stream has one START at sequence 0, zero or more DATA records,
//! then one END; END must be the last record. Each sequence is a `u64`.
//! The writer reserves `u64::MAX` for END and rejects DATA that would use
//! it. The reader rejects a sequence overflow.
//!
//! For record number `n`, the 12-byte nonce is the derived base nonce with
//! its last 8 bytes combined with `n:u64be` by XOR. Its `AAD` is the exact byte string
//! `"hpke-http/2 response record\0" || prefix[37] || n:u64be ||
//! ciphertext_len:u32be`. ChaCha20-Poly1305 seals the kind and body under
//! this nonce and AAD. The client checks the full tag before using any
//! START fields or yielding a DATA block. Any bad tag, form, order, or limit
//! closes the state. A checked END is not success until the caller reports
//! actual outer HTTP body EOF through [`ResponseOpener::finish_eof`]. Extra
//! bytes after END, including bytes in the same transport chunk, fail.
//!
//! A single checked `Content-Type` whose media type is `text/event-stream`
//! selects SSE. SSE requires status 200, a request other than HEAD, identity
//! private coding, and no logical `Content-Length` or `Content-Encoding`.
//! Each SSE DATA holds exactly one complete clear SSE block: line endings
//! are LF, and the block ends with one blank line. This includes comment
//! and control blocks. [`SseSplitter`] maps CR, LF, and CRLF to LF as the
//! server reads app bytes. It drops an unfinished tail at a clean end.
//! The caller parses SSE fields; this crate returns checked block bytes.
//! Each block has its own [`Limits::max_body_len`] bound; an SSE stream has
//! no total-body bound. A finite response has at most one DATA and becomes
//! public only after END and true EOF. Its body may use the one advertised
//! private gzip or zstd coding, with bounded clear and coded lengths.
//! HEAD and status 204, 205, or 304 have no body.
//!
//! # Complete finite transaction
//!
//! ```
//! use hpke_http::{
//!     Client, EntropySource, Limits, Method, Request, Response, Server, SystemEntropy,
//!     generate_key_pair,
//! };
//!
//! let keys = generate_key_pair()?;
//! let (private, public) = keys.into_parts();
//! let key_id = b"primary".to_vec();
//! let mut psk = vec![0; 32];
//! let mut entropy = SystemEntropy;
//! entropy.fill(&mut psk)?;
//! let psk_id = b"tenant".to_vec();
//! let client = Client::new(&public, key_id.clone(), psk.clone(), psk_id.clone(), Limits::default())?;
//! let server = Server::new(&private, key_id, Limits::default())?;
//! let request = Request {
//!     method: Method::Post, authority: b"api.example.test".to_vec(),
//!     path: b"/items".to_vec(), headers: Vec::new(), body: b"payload".to_vec(),
//! };
//! let (envelope, token) = client.protect(&request)?.into_parts();
//! let preparsed = server.preparse(&envelope)?;
//! assert_eq!(preparsed.credential.psk_id, psk_id);
//! let authenticated = server.authenticate(preparsed.token, &psk)?;
//! let opened = authenticated.token.admit(authenticated.replay.decision(true))?;
//! assert_eq!(opened.request, request);
//! let expected = Response { status: 200, headers: Vec::new(), body: b"ok".to_vec() };
//! let response = opened.response.protect_finite(&expected)?;
//! assert_eq!(token.open_finite(&response)?, expected);
//! # Ok::<(), hpke_http::Error>(())
//! ```
//!
//! [`Limits`] defaults to an 8 MiB body, 16 KiB of header bytes, 64
//! fields, and an 8 KiB authority plus path. Bounds apply before lengths
//! drive allocations. [`Error`] has coarse codes and no clear or secret
//! bytes. The crate version, [`PROTOCOL_ID`], and [`BINDING_ABI_VERSION`]
//! are separate values; bindings check all three before accepting secrets.

#![forbid(unsafe_code)]

mod codec;
mod compression;
mod engine;
mod entropy;
mod error;
mod limits;
mod message;
mod method;
mod response;
mod sse;

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
pub use response::{ResponseHead, ResponseMode, ResponseOpener, ResponseRecord, ResponseSealer};
pub use sse::SseSplitter;

/// Language-neutral protocol identifier.
pub const PROTOCOL_ID: &str = "hpke-http/2";
/// Boundary ABI version used by the first-party bindings.
pub const BINDING_ABI_VERSION: u32 = 3;
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

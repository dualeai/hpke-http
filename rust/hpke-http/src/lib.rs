//! HPKE requests and checked response records for HTTP.
//!
//! This crate defines one wire contract, [`PROTOCOL_ID`] `hpke-http/3`.
//! It does no network I/O. Hosts must use HTTPS. The protocol does not hide
//! endpoints, public IDs, payload size, record counts, or timing, and it does
//! not add padding. It uses RFC 9180 HPKE PSK mode with X25519,
//! HKDF-SHA256, and ChaCha20-Poly1305. It is not Oblivious HTTP.
//!
//! # Request bytes
//!
//! All fixed-width integers use big-endian order. Every request, including
//! an empty one, uses this one START/DATA/END form:
//!
//! ```text
//! "HHRQ" || 0x03 || key_id_len:u8 || psk_id_len:u8
//! || 0x0020 || 0x0001 || 0x0003 || issued_at_unix_s:u64be
//! || key_id || psk_id || enc[32] || records...
//! record = ciphertext_len:u32be || ciphertext[ciphertext_len]
//! ```
//!
//! Both IDs have 1 to 255 bytes. The public header ends after the PSK ID.
//! Each ciphertext includes a 16-byte HPKE tag. The first clear record is
//! `0x01 || method:vector || authority:vector || path:vector || fields:vector`.
//! These four fields use shortest RFC 9292 QUIC variable-length integers.
//! `fields` contains ordered name/value vector pairs. The scheme is HTTPS.
//! Each DATA is `0x02 || coding:u8 || coded_body`, with a nonempty clear
//! body of at most 64 KiB. Coding 0 means raw; 1 means one zstd frame. Rust
//! picks zstd only when it makes that DATA record shorter. It compresses
//! clear body bytes before HPKE encryption. A reader accepts a valid zstd
//! frame within record limits even if it is longer than its clear bytes.
//! END is exactly `0x03`.
//! There is one START, zero or more DATA records, one END, and then true
//! outer HTTP body EOF. Extra bytes, missing END, and partial frames fail.
//!
//! One HPKE PSK context seals all request records. Its `info` is
//! `"message/hpke-http request\0v3\0" || public_header`. Record number
//! `n` starts at zero. Its AAD is
//! `"hpke-http/3 request record\0" || public_header || n:u64be ||
//! ciphertext_len:u32be`. HPKE also advances its own nonce sequence per
//! record. A host must check END and true outer EOF before app dispatch.
//! A logical `Content-Length`, if present, names the sum of clear DATA bytes.
//! [`Limits::max_request_bytes`] defaults to 1 GiB and has a 4 GiB hard cap.
//! There can be at most 1,048,576 DATA records.
//! Complete in-memory request helpers also use [`Limits::max_body_len`],
//! which defaults to 8 MiB. Incremental requests use the total request cap.
//!
//! # Public key discovery
//!
//! The crate does no HTTP I/O. [`Server::public_key`] returns the advertised
//! X25519 public key. The host key GET record is `"HHKD" || 0x02 ||
//! id_len:u8 || id || public_key[32] || use_for_s:u32be`. The ID has 1 to
//! 255 bytes. The service sets a positive `use_for_s` lease in seconds. The
//! record has 43 to 297 bytes. Its format version is separate from
//! [`PROTOCOL_ID`]. There is no HHKD v1 discovery path.
//!
//! A host serves the record on GET at the protected HTTPS endpoint. A valid
//! GET reply has status 200, `Content-Type: application/octet-stream`,
//! `Cache-Control: no-store`, and an unencoded body that holds only the record.
//! The same URL accepts a protected POST with
//! `Content-Type: message/hpke-http-request`. A successful POST has outer
//! status 200, `Content-Type: message/hpke-http-response`, and an unencoded
//! body. See the [HTTP discovery guide](https://github.com/dualeai/hpke-http#key-discovery)
//! for client checks and host key-switch rules.
//!
//! [`Server::with_accepted_keys`] advertises one key and accepts other keys.
//! To switch from A to B, first make all workers advertise A and accept B.
//! Then make all workers advertise B and accept A. After the last A lease,
//! the bound to deliver and parse POST START, and a worker clock margin end,
//! make all workers advertise B alone. Each accepted pair holds a private key
//! followed by its public key ID. Do not reuse a KID while keys overlap. The
//! host must not replay a POST after a failed or lost reply.
//!
//! # Replay check
//!
//! The replay ID is SHA-256 of
//! `"hpke-http/replay\0v3\0" || public_header || enc`. The server checks
//! the request time against [`REQUEST_LIFETIME_SECS`] and [`CLOCK_SKEW_SECS`].
//! The host must make one atomic replay decision and keep it through the
//! authenticated deadline. The app must not see clear bytes before the host
//! checks the full request, including END and EOF.
//!
//! # Response bytes
//!
//! The HPKE context exports 32 bytes with context
//! `"message/hpke-http response\0v3"`. The server samples a fresh 32-byte
//! nonce. HKDF-SHA256 uses `enc || server_nonce` as salt and that exported
//! secret as input. It expands the key with `"hpke-http/3 response key"`
//! and nonce with `"hpke-http/3 response nonce"`. The response is:
//!
//! ```text
//! "HHRP" || 0x03 || server_nonce[32] || records...
//! record = ciphertext_len:u32be || ciphertext[ciphertext_len]
//! ```
//!
//! START is `0x01 || status:u16be || fields`, with ordered name/value
//! vectors. DATA is `0x02 || coding:u8 || coded_body`; coding 0 is raw and
//! 1 is one zstd frame. Rust picks zstd only when it saves bytes. Each SSE
//! DATA holds one complete SSE block after decoding.
//! END is exactly `0x03`. The response reader needs true outer EOF after
//! END. The record AAD is `"hpke-http/3 response record\0" ||
//! prefix[37] || n:u64be || ciphertext_len:u32be`. The record nonce is the
//! derived base nonce with its last eight bytes combined with `n:u64be` by XOR.
//!
//! [`Limits`] also bounds each finite reply, SSE block, head, and target.
//! Errors do not include clear or secret bytes. The crate version,
//! [`PROTOCOL_ID`], and [`BINDING_ABI_VERSION`] are separate values.
//!
//! # Complete transaction
//!
//! The host must make a shared atomic replay decision before it admits a
//! request. This example accepts one request to show the API sequence.
//!
//! ```
//! use hpke_http::{Client, Limits, Method, Request, Response, Server, generate_key_pair};
//!
//! fn main() -> Result<(), hpke_http::Error> {
//!     let keys = generate_key_pair()?;
//!     let (private_key, public_key) = keys.into_parts();
//!     let psk = b"example pre-shared key with 32 bytes!!";
//!     let client = Client::new(
//!         &public_key, b"key-1".to_vec(), psk.to_vec(), b"tenant-1".to_vec(),
//!         Limits::default(),
//!     )?;
//!     let server = Server::new(&private_key, b"key-1".to_vec(), Limits::default())?;
//!     let request = Request {
//!         method: Method::Post,
//!         authority: b"api.example.test".to_vec(),
//!         path: b"/items".to_vec(),
//!         headers: Vec::new(),
//!         body: b"hello".to_vec(),
//!     };
//!     let (request_bytes, response_right) = client.protect(&request)?.into_parts();
//!     let parsed = server.preparse(&request_bytes)?;
//!     let authenticated = server.authenticate(parsed.token, psk)?;
//!     let opened = authenticated.token.admit(authenticated.replay.decision(true))?;
//!     assert_eq!(opened.request, request);
//!     let reply = Response { status: 200, headers: Vec::new(), body: b"ok".to_vec() };
//!     let reply_bytes = opened.response.protect_finite(&reply)?;
//!     assert_eq!(response_right.open_finite(&reply_bytes)?, reply);
//!     Ok(())
//! }
//! ```
//!
//! # Streamed upload
//!
//! Send the `begin_stream` bytes first, then each record from `push`, then the
//! bytes from `finish`. End the outer HTTP body after `finish`. On the server,
//! hold checked DATA until `finish_eof` confirms END and real outer EOF. This
//! example accepts one replay decision and passes bytes directly between the
//! client and server in place of an HTTP transport. A host must use a shared
//! atomic replay store:
//!
//! ```
//! use hpke_http::{
//!     Client, Error, Limits, Method, RequestHead, Server, StreamRequestOpener,
//!     StreamRequestRecord, generate_key_pair,
//! };
//!
//! fn feed(reader: &mut StreamRequestOpener, bytes: &[u8], body: &mut Vec<u8>) -> Result<(), Error> {
//!     let mut remaining = bytes;
//!     while !remaining.is_empty() {
//!         let (used, record) = reader.feed(remaining)?;
//!         remaining = &remaining[used..];
//!         if let Some(StreamRequestRecord::Data(part)) = record {
//!             body.extend_from_slice(&part);
//!         }
//!     }
//!     Ok(())
//! }
//!
//! fn main() -> Result<(), Error> {
//!     let keys = generate_key_pair()?;
//!     let (private_key, public_key) = keys.into_parts();
//!     let psk = b"example pre-shared key with 32 bytes!!";
//!     let client = Client::new(&public_key, b"key-1".to_vec(), psk.to_vec(), b"tenant-1".to_vec(), Limits::default())?;
//!     let server = Server::new(&private_key, b"key-1".to_vec(), Limits::default())?;
//!     let head = RequestHead {
//!         method: Method::Post,
//!         authority: b"api.example.test".to_vec(),
//!         path: b"/upload".to_vec(),
//!         headers: Vec::new(),
//!     };
//!     let (mut writer, first) = client.begin_stream(&head)?;
//!     let parsed = server.preparse_stream(&first)?;
//!     let authenticated = server.authenticate_stream(parsed.token, psk)?;
//!     let opened = authenticated.token.admit(authenticated.replay.decision(true))?;
//!     let mut reader = opened.reader;
//!     let mut body = Vec::new();
//!     for part in [b"hello".as_slice(), b" world".as_slice()] {
//!         let mut remaining = part;
//!         while !remaining.is_empty() {
//!             let (used, record) = writer.push(remaining)?;
//!             remaining = &remaining[used..];
//!             if let Some(record) = record {
//!                 feed(&mut reader, &record, &mut body)?;
//!             }
//!         }
//!     }
//!     let (end, _response_right) = writer.finish()?;
//!     feed(&mut reader, &end, &mut body)?;
//!     let _server_response_right = reader.finish_eof()?;
//!     assert_eq!(body.as_slice(), b"hello world");
//!     Ok(())
//! }
//! ```
//!
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

pub use engine::{
    AuthenticatedRequest, AuthenticatedStreamRequest, Client, CredentialRequest, KeyPair,
    OpenedRequest, OpenedStreamRequest, PreparsedRequest, PreparsedStreamRequest, ProtectedRequest,
    ReplayDecision, ReplayRequest, ReplayToken, ResponseCapability, ResponseToken, STREAM_DATA_LEN,
    Server, StartToken, StreamReplayToken, StreamRequestOpener, StreamRequestRecord,
    StreamRequestSealer, StreamStartToken, generate_key_pair, generate_key_pair_with_entropy,
};
pub use entropy::{EntropySource, SystemEntropy};
pub use error::Error;
pub use limits::{
    HARD_MAX_BODY_LEN, HARD_MAX_HEADER_BYTES, HARD_MAX_HEADER_COUNT, HARD_MAX_ID_LEN,
    HARD_MAX_REQUEST_BYTES, HARD_MAX_TARGET_LEN, Limits,
};
pub use message::{HeaderField, Request, RequestHead, Response};
pub use method::Method;
pub use response::{ResponseHead, ResponseMode, ResponseOpener, ResponseRecord, ResponseSealer};
pub use sse::SseSplitter;

/// Language-neutral protocol identifier.
pub const PROTOCOL_ID: &str = "hpke-http/3";
/// Boundary ABI version used by the first-party bindings.
pub const BINDING_ABI_VERSION: u32 = 8;
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

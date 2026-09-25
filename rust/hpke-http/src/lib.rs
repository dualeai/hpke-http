//! HPKE requests and checked response records for HTTP.
//!
//! This crate implements `PROTOCOL_ID` hpke-http/3. The
//! [central protocol specification](https://github.com/dualeai/hpke-http/blob/main/PROTOCOL.md)
//! gives the exact request, response, and HHKD v2 bytes; cryptographic inputs;
//! limits; checks; and client and server steps. It links the frozen wire vectors.
//! Package versions and `BINDING_ABI_VERSION` do not change the wire version.
//!
//! The crate does no network I/O. A host uses HTTPS, resolves PSKs, and makes
//! one atomic replay decision shared by its workers. The public PSK ID is an
//! untrusted lookup hint until START authentication passes. An admitted
//! request gives one response right. The host checks the full request, END,
//! real outer body EOF, and logical target policy before it sends clear bytes
//! to the app.
//!
//! `Client::protect` and `Server::preparse` serve complete bounded calls.
//! `Client::begin_stream` and `Server::preparse_stream` serve uploads. The
//! stream path checks replay after START; the complete path checks all
//! records before replay. Both paths withhold clear request data until the
//! host admits replay. The host reports real outer EOF to `ResponseOpener`
//! after END; the opener then gives a finite response. It gives each checked
//! SSE block after that block passes its own checks.
//!
//! The examples below show API use. Use the linked specification to build
//! another language implementation or to read the exact wire contract.
//!
//! # Complete transaction
//!
//! The host must make a shared atomic replay decision before it admits a
//! request. This example passes complete request and response bytes directly,
//! in place of HTTP bodies, to show the API sequence.
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
//! hold checked DATA until the host observes real outer EOF and `finish_eof`
//! checks END. This example accepts one replay decision and passes bytes
//! directly between the client and server in place of an HTTP transport.
//! A host must use a shared atomic replay store:
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

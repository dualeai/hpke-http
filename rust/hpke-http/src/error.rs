//! Stable public error categories.

use thiserror::Error;

/// Errors returned by the protocol engine.
///
/// Messages contain no credentials, plaintext, ciphertext, or parser offsets.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Configuration is missing, inconsistent, or outside the protocol limits.
    #[error("invalid configuration")]
    InvalidConfiguration,
    /// A caller-provided value exceeds a configured or hard protocol limit.
    #[error("limit exceeded")]
    LimitExceeded,
    /// The envelope is truncated, has trailing data, or has an invalid field.
    #[error("malformed envelope")]
    MalformedEnvelope,
    /// The envelope uses a protocol version this implementation does not support.
    #[error("unsupported protocol version")]
    UnsupportedVersion,
    /// The envelope names a cryptographic suite outside protocol version 1.
    #[error("unsupported cryptographic suite")]
    UnsupportedSuite,
    /// The request uses an HTTP method outside protocol version 1.
    #[error("unsupported HTTP method")]
    UnsupportedMethod,
    /// The request names a recipient key that is not configured on this server.
    #[error("unknown recipient key")]
    UnknownRecipientKey,
    /// A supplied PSK violates the credential length or public-ID rules.
    #[error("invalid credential")]
    InvalidCredential,
    /// An authenticated operation failed.
    #[error("authentication failed")]
    AuthenticationFailed,
    /// The replay-admission provider rejected this authenticated request.
    #[error("request replay rejected")]
    ReplayRejected,
    /// The authenticated request timestamp is expired or too far in the future.
    #[error("invalid request time")]
    InvalidRequestTime,
    /// A replay decision was created for another authenticated request.
    #[error("replay decision does not match request")]
    ReplayDecisionMismatch,
    /// The platform could not provide a usable Unix time.
    #[error("system clock unavailable")]
    ClockUnavailable,
    /// The platform could not provide cryptographically secure entropy.
    #[error("secure entropy unavailable")]
    EntropyUnavailable,
    /// A cryptographic primitive rejected otherwise bounded input.
    #[error("cryptographic operation failed")]
    CryptoFailure,
    /// A local body compressor failed before an envelope was produced.
    #[error("compression failed")]
    CompressionFailure,
}

impl Error {
    /// Return the stable language-neutral error code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::InvalidConfiguration => "invalid_configuration",
            Self::LimitExceeded => "limit_exceeded",
            Self::MalformedEnvelope => "malformed_envelope",
            Self::UnsupportedVersion => "unsupported_version",
            Self::UnsupportedSuite => "unsupported_suite",
            Self::UnsupportedMethod => "unsupported_method",
            Self::UnknownRecipientKey => "unknown_recipient_key",
            Self::InvalidCredential => "invalid_credential",
            Self::AuthenticationFailed => "authentication_failed",
            Self::ReplayRejected => "replay_rejected",
            Self::InvalidRequestTime => "invalid_request_time",
            Self::ReplayDecisionMismatch => "replay_decision_mismatch",
            Self::ClockUnavailable => "clock_unavailable",
            Self::EntropyUnavailable => "entropy_unavailable",
            Self::CryptoFailure => "crypto_failure",
            Self::CompressionFailure => "compression_failure",
        }
    }
}

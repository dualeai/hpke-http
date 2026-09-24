//! Bounded-input policy for protocol version 3.

use crate::Error;

/// Absolute body limit accepted by the implementation (64 MiB).
pub const HARD_MAX_BODY_LEN: usize = 64 * 1024 * 1024;
/// Absolute combined header-name and header-value limit (64 KiB).
pub const HARD_MAX_HEADER_BYTES: usize = 64 * 1024;
/// Absolute number of header fields accepted in one message.
pub const HARD_MAX_HEADER_COUNT: usize = 256;
/// Absolute combined authority and path limit (8 KiB).
pub const HARD_MAX_TARGET_LEN: usize = 8 * 1024;
/// Absolute public identifier limit accepted by the implementation.
pub const HARD_MAX_ID_LEN: usize = 255;
/// Absolute clear request body limit (4 GiB).
pub const HARD_MAX_REQUEST_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Per-engine limits. Set values within the hard limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    /// Maximum body for complete request helpers, finite replies, and SSE blocks.
    pub max_body_len: usize,
    /// Maximum combined header-name and header-value bytes per message.
    pub max_header_bytes: usize,
    /// Maximum number of header fields per message.
    pub max_header_count: usize,
    /// Maximum combined request authority and path size.
    pub max_target_len: usize,
    /// Maximum clear request body across all DATA records.
    pub max_request_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_body_len: 8 * 1024 * 1024,
            max_header_bytes: 16 * 1024,
            max_header_count: 64,
            max_target_len: HARD_MAX_TARGET_LEN,
            max_request_bytes: 1024 * 1024 * 1024,
        }
    }
}

impl Limits {
    /// Validate limits before an engine accepts credentials or messages.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] if a value exceeds its hard
    /// implementation limit.
    pub fn validate(self) -> Result<Self, Error> {
        if self.max_body_len > HARD_MAX_BODY_LEN
            || self.max_header_bytes > HARD_MAX_HEADER_BYTES
            || self.max_header_count > HARD_MAX_HEADER_COUNT
            || self.max_target_len > HARD_MAX_TARGET_LEN
            || self.max_request_bytes == 0
            || self.max_request_bytes > HARD_MAX_REQUEST_BYTES
        {
            return Err(Error::InvalidConfiguration);
        }
        Ok(self)
    }
}

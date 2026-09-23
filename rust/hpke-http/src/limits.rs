//! Bounded-input policy for protocol version 2.

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

const BHTTP_FIXED_ALLOWANCE: usize = 256;
const BHTTP_PER_HEADER_ALLOWANCE: usize = 16;

/// Per-engine limits. Values can become stricter but never exceed hard limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    /// Maximum request or finite response body, and maximum one SSE block.
    pub max_body_len: usize,
    /// Maximum combined header-name and header-value bytes per message.
    pub max_header_bytes: usize,
    /// Maximum number of header fields per message.
    pub max_header_count: usize,
    /// Maximum combined request authority and path size.
    pub max_target_len: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_body_len: 8 * 1024 * 1024,
            max_header_bytes: 16 * 1024,
            max_header_count: 64,
            max_target_len: HARD_MAX_TARGET_LEN,
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
        {
            return Err(Error::InvalidConfiguration);
        }
        Ok(self)
    }

    pub(crate) fn max_encoded_message_len(self) -> usize {
        self.max_body_len
            .saturating_add(self.max_header_bytes)
            .saturating_add(self.max_target_len)
            .saturating_add(
                self.max_header_count
                    .saturating_mul(BHTTP_PER_HEADER_ALLOWANCE),
            )
            .saturating_add(BHTTP_FIXED_ALLOWANCE)
    }
}

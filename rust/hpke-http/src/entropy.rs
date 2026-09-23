//! Fallible entropy injection.

use crate::Error;

/// A source of cryptographically secure random bytes.
pub trait EntropySource {
    /// Fill the destination or return before any protocol output is produced.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EntropyUnavailable`] when the source cannot fill the
    /// entire destination with cryptographically secure random bytes.
    fn fill(&mut self, destination: &mut [u8]) -> Result<(), Error>;
}

/// The operating system or Web Crypto entropy source selected by `getrandom`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemEntropy;

impl EntropySource for SystemEntropy {
    fn fill(&mut self, destination: &mut [u8]) -> Result<(), Error> {
        getrandom::fill(destination).map_err(|_| Error::EntropyUnavailable)
    }
}

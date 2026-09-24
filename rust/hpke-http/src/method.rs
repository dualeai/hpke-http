//! Supported HTTP request methods.

use crate::Error;

/// Supported HTTP request methods.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Method {
    /// GET.
    Get,
    /// POST.
    Post,
    /// PUT.
    Put,
    /// PATCH.
    Patch,
    /// DELETE.
    Delete,
    /// HEAD.
    Head,
    /// OPTIONS.
    Options,
}

impl Method {
    /// Return the method name used at the HTTP boundary.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Put => "PUT",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
        }
    }

    pub(crate) const fn as_bytes(self) -> &'static [u8] {
        self.as_str().as_bytes()
    }

    /// Read a supported HTTP method name.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedMethod`] for any other name.
    pub fn from_bytes(value: &[u8]) -> Result<Self, Error> {
        match value {
            b"GET" => Ok(Self::Get),
            b"POST" => Ok(Self::Post),
            b"PUT" => Ok(Self::Put),
            b"PATCH" => Ok(Self::Patch),
            b"DELETE" => Ok(Self::Delete),
            b"HEAD" => Ok(Self::Head),
            b"OPTIONS" => Ok(Self::Options),
            _ => Err(Error::UnsupportedMethod),
        }
    }
}

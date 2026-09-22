//! HTTP methods included in protocol version 1.

use crate::Error;

/// HTTP request methods supported by protocol version 1.
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
    pub(crate) const fn as_bytes(self) -> &'static [u8] {
        match self {
            Self::Get => b"GET",
            Self::Post => b"POST",
            Self::Put => b"PUT",
            Self::Patch => b"PATCH",
            Self::Delete => b"DELETE",
            Self::Head => b"HEAD",
            Self::Options => b"OPTIONS",
        }
    }

    pub(crate) fn from_bytes(value: &[u8]) -> Result<Self, Error> {
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

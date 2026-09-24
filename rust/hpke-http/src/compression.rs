//! Per-record zstd coding for clear payload bytes.

use std::{borrow::Cow, io::Read};

use ruzstd::{
    decoding::StreamingDecoder,
    encoding::{CompressionLevel, compress_to_vec},
};

use crate::Error;

/// Raw DATA bytes.
pub(crate) const RAW: u8 = 0;
/// Zstd DATA bytes.
pub(crate) const ZSTD: u8 = 1;

/// Compress only when the coded record is shorter than its clear data.
pub(crate) fn encode(body: &[u8]) -> (u8, Cow<'_, [u8]>) {
    if body.len() < 64 {
        return (RAW, Cow::Borrowed(body));
    }
    let coded = compress_to_vec(body, CompressionLevel::Fastest);
    if coded.len() < body.len() {
        (ZSTD, Cow::Owned(coded))
    } else {
        (RAW, Cow::Borrowed(body))
    }
}

/// Decode one bounded record. The caller checks the total clear-body limit.
pub(crate) fn decode(kind: u8, body: &[u8], max_clear: usize) -> Result<Vec<u8>, Error> {
    match kind {
        RAW => {
            if body.is_empty() || body.len() > max_clear {
                return Err(if body.is_empty() {
                    Error::MalformedEnvelope
                } else {
                    Error::LimitExceeded
                });
            }
            Ok(body.to_vec())
        }
        ZSTD => {
            if body.is_empty() {
                return Err(Error::MalformedEnvelope);
            }
            let mut decoder =
                StreamingDecoder::new_with_max_window_size(body, max_clear.max(1024 * 1024) as u64)
                    .map_err(|_| Error::MalformedEnvelope)?;
            let declared_size = decoder.decoder.content_size();
            if declared_size > max_clear as u64 {
                return Err(Error::LimitExceeded);
            }
            let mut output = Vec::new();
            let mut buffer = [0_u8; 8192];
            loop {
                let count = decoder
                    .read(&mut buffer)
                    .map_err(|_| Error::MalformedEnvelope)?;
                if count == 0 {
                    break;
                }
                if count > max_clear.saturating_sub(output.len()) {
                    return Err(Error::LimitExceeded);
                }
                output.extend_from_slice(&buffer[..count]);
            }
            if declared_size != 0 && declared_size != output.len() as u64 {
                return Err(Error::MalformedEnvelope);
            }
            if decoder.decoder.get_checksum_from_data().is_some()
                && decoder.decoder.get_checksum_from_data()
                    != decoder.decoder.get_calculated_checksum()
            {
                return Err(Error::MalformedEnvelope);
            }
            if output.is_empty() || !decoder.into_inner().is_empty() {
                return Err(Error::MalformedEnvelope);
            }
            Ok(output)
        }
        _ => Err(Error::MalformedEnvelope),
    }
}

#[cfg(test)]
mod tests {
    use super::{RAW, ZSTD, decode, encode};
    use crate::Error;
    use ruzstd::encoding::{CompressionLevel, compress_to_vec};

    #[test]
    fn chooses_raw_or_zstd_and_bounds_expansion() {
        let small = b"abc";
        assert_eq!(encode(small).0, RAW);
        let large = vec![b'a'; 4096];
        let (kind, coded) = encode(&large);
        assert_eq!(kind, ZSTD);
        assert_eq!(decode(kind, &coded, large.len()), Ok(large.clone()));
        assert_eq!(decode(kind, &coded, 128), Err(Error::LimitExceeded));
        assert_eq!(decode(9, &coded, 4096), Err(Error::MalformedEnvelope));
        assert_eq!(
            decode(ZSTD, &coded[..coded.len() - 1], 4096),
            Err(Error::MalformedEnvelope)
        );
        let mut with_extra = coded.clone().into_owned();
        with_extra.push(0);
        assert_eq!(
            decode(ZSTD, &with_extra, 4096),
            Err(Error::MalformedEnvelope)
        );
        let mut corrupt = coded.into_owned();
        corrupt[0] ^= 1;
        assert_eq!(decode(ZSTD, &corrupt, 4096), Err(Error::MalformedEnvelope));
    }

    #[test]
    fn accepts_valid_zstd_even_when_it_does_not_save_bytes() {
        let clear = b"abc";
        let coded = compress_to_vec(clear.as_slice(), CompressionLevel::Fastest);
        assert!(coded.len() >= clear.len());
        assert_eq!(decode(ZSTD, &coded, clear.len()), Ok(clear.to_vec()));
    }
}

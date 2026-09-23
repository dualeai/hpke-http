//! Authenticated protocol body coding, separate from HTTP `Content-Encoding`.
//!
//! Request compression uses a marker. Response records name their body coding.

use std::io::{Read, Write};

use flate2::{Compression, bufread::MultiGzDecoder, write::GzEncoder};
use ruzstd::{
    decoding::StreamingDecoder,
    encoding::{CompressionLevel, compress_to_vec},
};

use crate::{
    Error, Limits,
    message::{self, Request},
};

pub(crate) const MARKER: u8 = 0x04;
const MIN_BODY_LEN: usize = 64;
// Standard encoders can declare a larger history window than the decoded
// payload. Keep a 1 MiB interoperability floor, still far below ruzstd's
// 100 MiB default, and cap even the 64 MiB body configuration at 8 MiB.
const MIN_ZSTD_WINDOW: usize = 1024 * 1024;
const MAX_ZSTD_WINDOW: usize = 8 * 1024 * 1024;
const MAX_ZSTD_FRAMES: usize = 16;

/// Opt-in, authenticated protocol body coding. This is not HTTP `Content-Encoding`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompressionCoding {
    /// RFC 1952 gzip members.
    Gzip,
    /// RFC 8878 Zstandard frames without a dictionary.
    Zstd,
}

impl CompressionCoding {
    pub(crate) const fn wire_id(self) -> u8 {
        match self {
            Self::Gzip => 1,
            Self::Zstd => 2,
        }
    }

    pub(crate) const fn from_wire_id(id: u8) -> Result<Option<Self>, Error> {
        match id {
            0 => Ok(None),
            1 => Ok(Some(Self::Gzip)),
            2 => Ok(Some(Self::Zstd)),
            _ => Err(Error::MalformedEnvelope),
        }
    }
}

pub(crate) fn encode_request(
    request: &Request,
    limits: Limits,
    preferred: Option<CompressionCoding>,
) -> Result<Vec<u8>, Error> {
    let Some(preferred) = preferred else {
        return message::encode_request(request, limits);
    };
    let coded = maybe_compress(&request.body, preferred)?;
    let body_coding = coded.as_ref().map(|_| preferred);
    let mut output = vec![
        MARKER,
        body_coding.map_or(0, CompressionCoding::wire_id),
        preferred.wire_id(),
    ];
    let bhttp = message::encode_request_with_body(
        request,
        coded.as_deref().unwrap_or(&request.body),
        limits,
    )?;
    output.extend_from_slice(&bhttp);
    Ok(output)
}

pub(crate) fn decode_request(
    input: &[u8],
    limits: Limits,
    enabled: bool,
) -> Result<
    (
        Request,
        Option<CompressionCoding>,
        Option<CompressionCoding>,
    ),
    Error,
> {
    if !input.starts_with(&[MARKER]) {
        return Ok((message::decode_request(input, limits)?, None, None));
    }
    if !enabled || input.len() < 4 {
        return Err(Error::MalformedEnvelope);
    }
    let body_coding = CompressionCoding::from_wire_id(input[1])?;
    let response_coding = CompressionCoding::from_wire_id(input[2])?;
    if body_coding.is_none() && response_coding.is_none() {
        return Err(Error::MalformedEnvelope);
    }
    let request = if body_coding.is_some() {
        message::decode_coded_request(&input[3..], limits)?
    } else {
        message::decode_request(&input[3..], limits)?
    };
    Ok((request, body_coding, response_coding))
}

pub(crate) fn finish_request(
    mut request: Request,
    coding: Option<CompressionCoding>,
    limits: Limits,
) -> Result<Request, Error> {
    if let Some(coding) = coding {
        request.body = decompress(&request.body, coding, limits.max_body_len)?;
        message::validate_decoded_request_body(&request)?;
    }
    Ok(request)
}

pub(crate) fn maybe_compress(
    body: &[u8],
    coding: CompressionCoding,
) -> Result<Option<Vec<u8>>, Error> {
    if body.len() < MIN_BODY_LEN {
        return Ok(None);
    }
    let coded = compress(body, coding)?;
    Ok((coded.len() < body.len()).then_some(coded))
}

pub(crate) fn compress(body: &[u8], coding: CompressionCoding) -> Result<Vec<u8>, Error> {
    let coded = match coding {
        CompressionCoding::Gzip => {
            let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
            encoder
                .write_all(body)
                .map_err(|_| Error::CompressionFailure)?;
            encoder.finish().map_err(|_| Error::CompressionFailure)?
        }
        CompressionCoding::Zstd => compress_to_vec(body, CompressionLevel::Fastest),
    };
    Ok(coded)
}

pub(crate) fn decompress(
    coded: &[u8],
    coding: CompressionCoding,
    maximum_body_len: usize,
) -> Result<Vec<u8>, Error> {
    if coded.is_empty() {
        return Err(Error::MalformedEnvelope);
    }
    match coding {
        CompressionCoding::Gzip => {
            let mut decoder = MultiGzDecoder::new(coded);
            let body = read_bounded(&mut decoder, maximum_body_len)?;
            if !decoder.into_inner().is_empty() {
                return Err(Error::MalformedEnvelope);
            }
            Ok(body)
        }
        CompressionCoding::Zstd => {
            let mut remaining = coded;
            let mut body = Vec::new();
            let mut decoded_frames = 0_usize;
            let mut seen_frames = 0_usize;
            while !remaining.is_empty() {
                seen_frames += 1;
                if seen_frames > MAX_ZSTD_FRAMES {
                    return Err(Error::LimitExceeded);
                }
                // RFC 8878 skippable frames can precede or separate data frames.
                if let Some(skip_len) = skippable_frame_len(remaining)? {
                    remaining = &remaining[skip_len..];
                    continue;
                }
                let mut decoder = StreamingDecoder::new_with_max_window_size(
                    remaining,
                    maximum_body_len.clamp(MIN_ZSTD_WINDOW, MAX_ZSTD_WINDOW) as u64,
                )
                .map_err(|_| Error::MalformedEnvelope)?;
                let declared_size = decoder.decoder.content_size();
                if declared_size > maximum_body_len.saturating_sub(body.len()) as u64 {
                    return Err(Error::LimitExceeded);
                }
                let decoded_before = body.len();
                read_bounded_into(&mut decoder, &mut body, maximum_body_len)?;
                if declared_size != 0 && declared_size != (body.len() - decoded_before) as u64 {
                    return Err(Error::MalformedEnvelope);
                }
                if decoder.decoder.get_checksum_from_data().is_some()
                    && decoder.decoder.get_checksum_from_data()
                        != decoder.decoder.get_calculated_checksum()
                {
                    return Err(Error::MalformedEnvelope);
                }
                remaining = decoder.into_inner();
                decoded_frames += 1;
            }
            if decoded_frames == 0 {
                return Err(Error::MalformedEnvelope);
            }
            Ok(body)
        }
    }
}

fn skippable_frame_len(input: &[u8]) -> Result<Option<usize>, Error> {
    if input.len() < 4 {
        return Err(Error::MalformedEnvelope);
    }
    let magic = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
    if !(0x184d_2a50..=0x184d_2a5f).contains(&magic) {
        return Ok(None);
    }
    if input.len() < 8 {
        return Err(Error::MalformedEnvelope);
    }
    let length = u32::from_le_bytes([input[4], input[5], input[6], input[7]]) as usize;
    let total = length.checked_add(8).ok_or(Error::MalformedEnvelope)?;
    if total > input.len() {
        return Err(Error::MalformedEnvelope);
    }
    Ok(Some(total))
}

fn read_bounded(reader: &mut impl Read, maximum_body_len: usize) -> Result<Vec<u8>, Error> {
    let mut output = Vec::new();
    read_bounded_into(reader, &mut output, maximum_body_len)?;
    Ok(output)
}

fn read_bounded_into(
    reader: &mut impl Read,
    output: &mut Vec<u8>,
    maximum_body_len: usize,
) -> Result<(), Error> {
    let mut buffer = [0_u8; 8192];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|_| Error::MalformedEnvelope)?;
        if count == 0 {
            return Ok(());
        }
        if count > maximum_body_len.saturating_sub(output.len()) {
            return Err(Error::LimitExceeded);
        }
        output.extend_from_slice(&buffer[..count]);
    }
}

#[cfg(test)]
mod tests {
    use super::{CompressionCoding, decompress, encode_request, maybe_compress};
    use crate::{Error, Limits, Method, Request};

    // Produced independently with `printf ... | gzip -n -c` and
    // `printf ... | zstd -q -1 -c`, not by this crate's encoders.
    const GZIP: &[u8] = &[
        0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xcb, 0x48, 0xcd, 0xc9, 0xc9,
        0x57, 0x48, 0x2b, 0xca, 0xcf, 0x55, 0x48, 0xcc, 0x53, 0xc8, 0xcc, 0x4b, 0x49, 0x2d, 0x48,
        0x05, 0x12, 0x79, 0x25, 0x0a, 0xe9, 0x55, 0x99, 0x05, 0x0a, 0xa9, 0x79, 0xc9, 0xf9, 0x29,
        0xa9, 0x45, 0x0a, 0x19, 0x44, 0x29, 0x03, 0x00, 0x61, 0xc4, 0x34, 0x58, 0x4d, 0x00, 0x00,
        0x00,
    ];
    const ZSTD: &[u8] = &[
        0x28, 0xb5, 0x2f, 0xfd, 0x04, 0x48, 0x7d, 0x01, 0x00, 0x74, 0x02, 0x68, 0x65, 0x6c, 0x6c,
        0x6f, 0x20, 0x66, 0x72, 0x6f, 0x6d, 0x20, 0x61, 0x6e, 0x20, 0x69, 0x6e, 0x64, 0x65, 0x70,
        0x65, 0x6e, 0x64, 0x65, 0x6e, 0x74, 0x20, 0x7a, 0x73, 0x74, 0x64, 0x20, 0x65, 0x6e, 0x63,
        0x6f, 0x64, 0x65, 0x72, 0x20, 0x01, 0x00, 0xaf, 0x1e, 0xf5, 0x04, 0x2a, 0x5c, 0xa5, 0xa6,
    ];

    #[test]
    fn independent_gzip_members_and_strict_trailer() {
        let expected =
            b"hello from an independent gzip encoder hello from an independent gzip encoder";
        assert_eq!(
            decompress(GZIP, CompressionCoding::Gzip, 1024),
            Ok(expected.to_vec())
        );
        let mut joined = GZIP.to_vec();
        joined.extend_from_slice(GZIP);
        assert_eq!(
            decompress(&joined, CompressionCoding::Gzip, 1024),
            Ok(expected.repeat(2))
        );
        assert_eq!(
            decompress(&GZIP[..GZIP.len() - 1], CompressionCoding::Gzip, 1024),
            Err(Error::MalformedEnvelope)
        );
        let mut bad_checksum = GZIP.to_vec();
        bad_checksum[GZIP.len() - 8] ^= 1;
        assert_eq!(
            decompress(&bad_checksum, CompressionCoding::Gzip, 1024),
            Err(Error::MalformedEnvelope)
        );
        let mut garbage = GZIP.to_vec();
        garbage.push(0xff);
        assert_eq!(
            decompress(&garbage, CompressionCoding::Gzip, 1024),
            Err(Error::MalformedEnvelope)
        );
    }

    #[test]
    fn independent_zstd_frames_and_skippable_frame() {
        let expected =
            b"hello from an independent zstd encoder hello from an independent zstd encoder";
        assert_eq!(
            decompress(ZSTD, CompressionCoding::Zstd, 1024),
            Ok(expected.to_vec())
        );
        let mut joined = ZSTD.to_vec();
        joined.extend_from_slice(ZSTD);
        assert_eq!(
            decompress(&joined, CompressionCoding::Zstd, 1024),
            Ok(expected.repeat(2))
        );
        let mut skipped = vec![0x50, 0x2a, 0x4d, 0x18, 3, 0, 0, 0, b'x', b'y', b'z'];
        skipped.extend_from_slice(ZSTD);
        assert_eq!(
            decompress(&skipped, CompressionCoding::Zstd, 1024),
            Ok(expected.to_vec())
        );
        assert_eq!(
            decompress(&ZSTD[..ZSTD.len() - 1], CompressionCoding::Zstd, 1024),
            Err(Error::MalformedEnvelope)
        );
        let mut bad_checksum = ZSTD.to_vec();
        bad_checksum[ZSTD.len() - 1] ^= 1;
        assert_eq!(
            decompress(&bad_checksum, CompressionCoding::Zstd, 1024),
            Err(Error::MalformedEnvelope)
        );
        let mut garbage = ZSTD.to_vec();
        garbage.push(0xff);
        assert_eq!(
            decompress(&garbage, CompressionCoding::Zstd, 1024),
            Err(Error::MalformedEnvelope)
        );
        assert_eq!(
            decompress(&ZSTD.repeat(17), CompressionCoding::Zstd, 4096),
            Err(Error::LimitExceeded)
        );
        let mut oversized_window = ZSTD.to_vec();
        oversized_window[5] = 0x90;
        assert_eq!(
            decompress(&oversized_window, CompressionCoding::Zstd, 64 * 1024 * 1024),
            Err(Error::MalformedEnvelope)
        );
        // `zstd --stream-size=77` emits the same block with a one-byte frame
        // content size. Mutating that authenticated-format field must fail.
        let mut sized = ZSTD.to_vec();
        sized[4] = 0x24;
        sized[5] = 77;
        assert_eq!(
            decompress(&sized, CompressionCoding::Zstd, 1024),
            Ok(expected.to_vec())
        );
        sized[5] = 76;
        assert_eq!(
            decompress(&sized, CompressionCoding::Zstd, 1024),
            Err(Error::MalformedEnvelope)
        );
        sized[5] = 200;
        assert_eq!(
            decompress(&sized, CompressionCoding::Zstd, 100),
            Err(Error::LimitExceeded)
        );
    }

    #[test]
    fn decoded_length_is_bounded_during_expansion() -> Result<(), Error> {
        for coding in [CompressionCoding::Gzip, CompressionCoding::Zstd] {
            let encoded =
                maybe_compress(&vec![b'a'; 4096], coding)?.ok_or(Error::CompressionFailure)?;
            assert_eq!(decompress(&encoded, coding, 128), Err(Error::LimitExceeded));
        }
        Ok(())
    }

    #[test]
    fn body_coding_extension_has_literal_wire_prefixes() -> Result<(), Error> {
        let limits = Limits::default();
        let request = Request {
            method: Method::Post,
            authority: b"api.example.test".to_vec(),
            path: b"/compressed".to_vec(),
            headers: Vec::new(),
            body: vec![b'a'; 4096],
        };
        for (coding, request_prefix) in [
            (CompressionCoding::Gzip, [0x04, 0x01, 0x01]),
            (CompressionCoding::Zstd, [0x04, 0x02, 0x02]),
        ] {
            assert!(encode_request(&request, limits, Some(coding))?.starts_with(&request_prefix));

            let mut small_request = request.clone();
            small_request.body.clear();
            assert!(
                encode_request(&small_request, limits, Some(coding))?.starts_with(&[
                    0x04,
                    0x00,
                    request_prefix[2]
                ])
            );
        }
        Ok(())
    }
}

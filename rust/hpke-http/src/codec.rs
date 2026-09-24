//! Version 3 request record framing.

use crate::Error;

pub(crate) const REQUEST_MAGIC: &[u8; 4] = b"HHRQ";
pub(crate) const VERSION: u8 = 3;
pub(crate) const KEM_ID: u16 = 0x0020;
pub(crate) const KDF_ID: u16 = 0x0001;
pub(crate) const AEAD_ID: u16 = 0x0003;
pub(crate) const ENC_LEN: usize = 32;
pub(crate) const TAG_LEN: usize = 16;
pub(crate) const RESPONSE_NONCE_LEN: usize = 32;

pub(crate) const REQUEST_FIXED_LEN: usize = 21;
const ISSUED_AT_OFFSET: usize = 13;
const IDS_OFFSET: usize = 21;

pub(crate) fn encode_request_header(
    key_id: &[u8],
    psk_id: &[u8],
    issued_at_unix_s: u64,
) -> Result<Vec<u8>, Error> {
    let key_id_len = u8::try_from(key_id.len()).map_err(|_| Error::LimitExceeded)?;
    let psk_id_len = u8::try_from(psk_id.len()).map_err(|_| Error::LimitExceeded)?;
    if key_id_len == 0 || psk_id_len == 0 {
        return Err(Error::InvalidConfiguration);
    }

    let capacity = REQUEST_FIXED_LEN
        .checked_add(key_id.len())
        .and_then(|length| length.checked_add(psk_id.len()))
        .ok_or(Error::LimitExceeded)?;
    let mut output = Vec::with_capacity(capacity);
    output.extend_from_slice(REQUEST_MAGIC);
    output.push(VERSION);
    output.push(key_id_len);
    output.push(psk_id_len);
    output.extend_from_slice(&KEM_ID.to_be_bytes());
    output.extend_from_slice(&KDF_ID.to_be_bytes());
    output.extend_from_slice(&AEAD_ID.to_be_bytes());
    output.extend_from_slice(&issued_at_unix_s.to_be_bytes());
    output.extend_from_slice(key_id);
    output.extend_from_slice(psk_id);
    Ok(output)
}

#[derive(Debug)]
pub(crate) struct ParsedStreamStart<'a> {
    pub header: &'a [u8],
    pub key_id: &'a [u8],
    pub psk_id: &'a [u8],
    pub issued_at_unix_s: u64,
    pub enc: &'a [u8],
    pub ciphertext: &'a [u8],
}

/// Return the exact prefix plus START length once its four-byte frame size is present.
pub(crate) fn stream_start_length(
    input: &[u8],
    max_start_ciphertext: usize,
) -> Result<Option<usize>, Error> {
    if input.len() >= 4 && input.get(..4) != Some(REQUEST_MAGIC.as_slice()) {
        return Err(Error::MalformedEnvelope);
    }
    if input.len() >= 5 && input[4] != VERSION {
        return Err(Error::UnsupportedVersion);
    }
    if input.len() < REQUEST_FIXED_LEN {
        return Ok(None);
    }
    if read_u16(input, 7)? != KEM_ID
        || read_u16(input, 9)? != KDF_ID
        || read_u16(input, 11)? != AEAD_ID
    {
        return Err(Error::UnsupportedSuite);
    }
    let key_id_len = usize::from(input[5]);
    let psk_id_len = usize::from(input[6]);
    if key_id_len == 0 || psk_id_len == 0 {
        return Err(Error::MalformedEnvelope);
    }
    let frame_offset = REQUEST_FIXED_LEN + key_id_len + psk_id_len + ENC_LEN;
    let Some(frame_len_bytes) = input.get(frame_offset..frame_offset + 4) else {
        return Ok(None);
    };
    let frame_len = u32::from_be_bytes(
        frame_len_bytes
            .try_into()
            .map_err(|_| Error::MalformedEnvelope)?,
    ) as usize;
    if frame_len < TAG_LEN + 1 {
        return Err(Error::MalformedEnvelope);
    }
    if frame_len > max_start_ciphertext {
        return Err(Error::LimitExceeded);
    }
    Ok(Some(frame_offset + 4 + frame_len))
}

pub(crate) fn parse_stream_start(
    input: &[u8],
    max_start_ciphertext: usize,
) -> Result<ParsedStreamStart<'_>, Error> {
    if stream_start_length(input, max_start_ciphertext)? != Some(input.len()) {
        return Err(Error::MalformedEnvelope);
    }
    let key_id_end = REQUEST_FIXED_LEN + usize::from(input[5]);
    let header_len = key_id_end + usize::from(input[6]);
    let enc_end = header_len + ENC_LEN;
    let ciphertext_start = enc_end + 4;
    Ok(ParsedStreamStart {
        header: &input[..header_len],
        key_id: &input[REQUEST_FIXED_LEN..key_id_end],
        psk_id: &input[key_id_end..header_len],
        issued_at_unix_s: u64::from_be_bytes(
            input[ISSUED_AT_OFFSET..IDS_OFFSET]
                .try_into()
                .map_err(|_| Error::MalformedEnvelope)?,
        ),
        enc: &input[header_len..enc_end],
        ciphertext: &input[ciphertext_start..],
    })
}

fn read_u16(input: &[u8], offset: usize) -> Result<u16, Error> {
    let bytes: [u8; 2] = input
        .get(offset..offset + 2)
        .ok_or(Error::MalformedEnvelope)?
        .try_into()
        .map_err(|_| Error::MalformedEnvelope)?;
    Ok(u16::from_be_bytes(bytes))
}

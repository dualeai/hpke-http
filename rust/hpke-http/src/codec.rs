//! Version 2 request envelope and response record framing.

use crate::{Error, HARD_MAX_ID_LEN, Limits};

pub(crate) const REQUEST_MAGIC: &[u8; 4] = b"HHRQ";
pub(crate) const VERSION: u8 = 2;
pub(crate) const FLAGS: u8 = 0;
pub(crate) const KEM_ID: u16 = 0x0020;
pub(crate) const KDF_ID: u16 = 0x0001;
pub(crate) const AEAD_ID: u16 = 0x0003;
pub(crate) const ENC_LEN: usize = 32;
pub(crate) const TAG_LEN: usize = 16;
pub(crate) const RESPONSE_NONCE_LEN: usize = 32;

const REQUEST_FIXED_LEN: usize = 22;
const ISSUED_AT_OFFSET: usize = 14;
const IDS_OFFSET: usize = 22;

#[derive(Debug)]
pub(crate) struct ParsedRequest<'a> {
    pub header: &'a [u8],
    pub key_id: &'a [u8],
    pub psk_id: &'a [u8],
    pub issued_at_unix_s: u64,
    pub enc: &'a [u8],
    pub ciphertext: &'a [u8],
}

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
    output.push(FLAGS);
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

pub(crate) fn encode_request(
    header: &[u8],
    enc: &[u8],
    ciphertext: &[u8],
) -> Result<Vec<u8>, Error> {
    if enc.len() != ENC_LEN || ciphertext.len() < TAG_LEN {
        return Err(Error::CryptoFailure);
    }
    let capacity = header
        .len()
        .checked_add(enc.len())
        .and_then(|length| length.checked_add(ciphertext.len()))
        .ok_or(Error::LimitExceeded)?;
    let mut output = Vec::with_capacity(capacity);
    output.extend_from_slice(header);
    output.extend_from_slice(enc);
    output.extend_from_slice(ciphertext);
    Ok(output)
}

pub(crate) fn parse_request(input: &[u8], limits: Limits) -> Result<ParsedRequest<'_>, Error> {
    let minimum_len = REQUEST_FIXED_LEN
        .checked_add(2)
        .and_then(|length| length.checked_add(ENC_LEN + TAG_LEN))
        .ok_or(Error::MalformedEnvelope)?;
    if input.len() < minimum_len || input.get(..4) != Some(REQUEST_MAGIC.as_slice()) {
        return Err(Error::MalformedEnvelope);
    }
    if input[4] != VERSION {
        return Err(Error::UnsupportedVersion);
    }
    if input[5] != FLAGS {
        return Err(Error::MalformedEnvelope);
    }
    if read_u16(input, 8)? != KEM_ID
        || read_u16(input, 10)? != KDF_ID
        || read_u16(input, 12)? != AEAD_ID
    {
        return Err(Error::UnsupportedSuite);
    }

    let key_id_len = usize::from(input[6]);
    let psk_id_len = usize::from(input[7]);
    if key_id_len == 0
        || psk_id_len == 0
        || key_id_len > HARD_MAX_ID_LEN
        || psk_id_len > HARD_MAX_ID_LEN
    {
        return Err(Error::MalformedEnvelope);
    }
    let header_len = REQUEST_FIXED_LEN
        .checked_add(key_id_len)
        .and_then(|length| length.checked_add(psk_id_len))
        .ok_or(Error::MalformedEnvelope)?;
    let ciphertext_start = header_len
        .checked_add(ENC_LEN)
        .ok_or(Error::MalformedEnvelope)?;
    if input.len() < ciphertext_start.saturating_add(TAG_LEN) {
        return Err(Error::MalformedEnvelope);
    }
    let ciphertext_len = input.len() - ciphertext_start;
    if ciphertext_len - TAG_LEN > limits.max_encoded_message_len() {
        return Err(Error::LimitExceeded);
    }

    let issued_at_unix_s = u64::from_be_bytes(
        input[ISSUED_AT_OFFSET..IDS_OFFSET]
            .try_into()
            .map_err(|_| Error::MalformedEnvelope)?,
    );
    let key_id_start = REQUEST_FIXED_LEN;
    let psk_id_start = key_id_start + key_id_len;
    Ok(ParsedRequest {
        header: &input[..header_len],
        key_id: &input[key_id_start..psk_id_start],
        psk_id: &input[psk_id_start..header_len],
        issued_at_unix_s,
        enc: &input[header_len..ciphertext_start],
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

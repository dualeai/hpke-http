//! Checked v3 response records. A caller supplies true transport EOF separately.

use chacha20poly1305::{
    ChaCha20Poly1305,
    aead::{AeadInOut, KeyInit, array::Array},
};

use crate::{
    EntropySource, Error, HeaderField, Method, Response,
    codec::{RESPONSE_NONCE_LEN, TAG_LEN, VERSION},
    compression,
    engine::{ResponseMaterial, derive_response_key_nonce},
    message,
    sse::validate_block,
};

const MAGIC: &[u8; 4] = b"HHRP";
const PREFIX_LEN: usize = 4 + 1 + RESPONSE_NONCE_LEN;
const AAD_DOMAIN: &[u8] = b"hpke-http/3 response record\0";
const AAD_LEN: usize = AAD_DOMAIN.len() + PREFIX_LEN + 8 + 4;
const START: u8 = 1;
const DATA: u8 = 2;
const END: u8 = 3;
const MIN_CIPHERTEXT_LEN: usize = 1 + TAG_LEN;
const RETAIN_FRAME_CAPACITY: usize = 64 * 1024;

/// The checked response kind, based on its single Content-Type field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponseMode {
    /// A bounded complete body follows, in zero or one DATA record.
    Finite,
    /// Each DATA record contains one complete SSE block.
    Sse,
}

/// Fields released after a valid START tag.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResponseHead {
    /// Inner HTTP status.
    pub status: u16,
    /// Ordered checked fields.
    pub headers: Vec<HeaderField>,
    /// Response body kind.
    pub mode: ResponseMode,
}

/// One checked record. Finite body bytes remain in the opener until END.
#[derive(Debug, Eq, PartialEq)]
pub enum ResponseRecord {
    /// The first checked record.
    Start(ResponseHead),
    /// One checked complete SSE block.
    SseData(Vec<u8>),
    /// The checked terminal record. Transport EOF must still follow.
    End,
}

struct Crypto {
    key: zeroize::Zeroizing<[u8; 32]>,
    base_nonce: zeroize::Zeroizing<[u8; 12]>,
    prefix: [u8; PREFIX_LEN],
    sequence: u64,
}

impl Crypto {
    fn new(material: &ResponseMaterial, prefix: [u8; PREFIX_LEN]) -> Result<Self, Error> {
        let (key, base_nonce) = derive_response_key_nonce(material, &prefix[5..])?;
        Ok(Self {
            key,
            base_nonce,
            prefix,
            sequence: 0,
        })
    }

    fn nonce(&self) -> [u8; 12] {
        let mut nonce = *self.base_nonce;
        for (target, byte) in nonce[4..].iter_mut().zip(self.sequence.to_be_bytes()) {
            *target ^= byte;
        }
        nonce
    }

    fn aad(&self, ciphertext_len: u32) -> [u8; AAD_LEN] {
        let mut aad = [0; AAD_LEN];
        let prefix_end = AAD_DOMAIN.len() + PREFIX_LEN;
        aad[..AAD_DOMAIN.len()].copy_from_slice(AAD_DOMAIN);
        aad[AAD_DOMAIN.len()..prefix_end].copy_from_slice(&self.prefix);
        aad[prefix_end..prefix_end + 8].copy_from_slice(&self.sequence.to_be_bytes());
        aad[prefix_end + 8..].copy_from_slice(&ciphertext_len.to_be_bytes());
        aad
    }

    fn seal_record(
        &mut self,
        kind: u8,
        coding: Option<u8>,
        body: &[u8],
        terminal: bool,
    ) -> Result<Vec<u8>, Error> {
        if self.sequence == u64::MAX && !terminal {
            return Err(Error::LimitExceeded);
        }
        let length = body
            .len()
            .checked_add(1 + usize::from(coding.is_some()))
            .and_then(|size| size.checked_add(TAG_LEN))
            .ok_or(Error::LimitExceeded)?;
        let length = u32::try_from(length).map_err(|_| Error::LimitExceeded)?;
        let frame_len = 4_usize
            .checked_add(length as usize)
            .ok_or(Error::LimitExceeded)?;
        let mut framed = Vec::with_capacity(frame_len);
        framed.extend_from_slice(&length.to_be_bytes());
        framed.push(kind);
        if let Some(coding) = coding {
            framed.push(coding);
        }
        framed.extend_from_slice(body);
        let cipher = ChaCha20Poly1305::new(&Array(*self.key));
        let tag = cipher
            .encrypt_inout_detached(
                &Array(self.nonce()),
                &self.aad(length),
                (&mut framed[4..]).into(),
            )
            .map_err(|_| Error::CryptoFailure)?;
        framed.extend_from_slice(&tag);
        if !terminal {
            self.sequence += 1;
        }
        Ok(framed)
    }

    fn open(&mut self, ciphertext: &mut Vec<u8>, length: u32) -> Result<(), Error> {
        let cipher = ChaCha20Poly1305::new(&Array(*self.key));
        cipher
            .decrypt_in_place(&Array(self.nonce()), &self.aad(length), ciphertext)
            .map_err(|_| Error::AuthenticationFailed)
    }

    fn advance(&mut self) -> Result<(), Error> {
        self.sequence = self.sequence.checked_add(1).ok_or(Error::LimitExceeded)?;
        Ok(())
    }
}

fn classify_head(
    status: u16,
    headers: &[HeaderField],
    material: &ResponseMaterial,
) -> Result<ResponseMode, Error> {
    message::validate_response_head(status, headers, material.limits)?;
    let mut content_type = None;
    for field in headers {
        if field.name == b"content-type" && content_type.replace(field.value.as_slice()).is_some() {
            return Err(Error::InvalidConfiguration);
        }
    }
    let is_sse = content_type.is_some_and(|value| {
        value
            .split(|byte| *byte == b';')
            .next()
            .unwrap_or_default()
            .trim_ascii()
            .eq_ignore_ascii_case(b"text/event-stream")
    });
    if is_sse {
        if status != 200
            || material.method == Method::Head
            || headers
                .iter()
                .any(|field| field.name == b"content-length" || field.name == b"content-encoding")
        {
            return Err(Error::InvalidConfiguration);
        }
        Ok(ResponseMode::Sse)
    } else {
        Ok(ResponseMode::Finite)
    }
}

/// One response writer. Its key and sequence cannot be reused after failure.
pub struct ResponseSealer {
    crypto: Option<Crypto>,
    head: ResponseHead,
    method: Method,
    limits: crate::Limits,
    finite_data_sent: bool,
    done: bool,
}

impl ResponseSealer {
    /// Return the checked response mode.
    #[must_use]
    pub const fn head_mode(&self) -> ResponseMode {
        self.head.mode
    }

    pub(crate) fn new(
        material: &ResponseMaterial,
        status: u16,
        headers: Vec<HeaderField>,
        entropy: &mut impl EntropySource,
    ) -> Result<(Self, Vec<u8>), Error> {
        let mode = classify_head(status, &headers, material)?;
        let mut prefix = [0_u8; PREFIX_LEN];
        prefix[..4].copy_from_slice(MAGIC);
        prefix[4] = VERSION;
        entropy.fill(&mut prefix[5..])?;
        let mut crypto = Crypto::new(material, prefix)?;
        let fields = message::encode_fields(&headers)?;
        let mut start = Vec::with_capacity(2 + fields.len());
        start.extend_from_slice(&status.to_be_bytes());
        start.extend_from_slice(&fields);
        let frame = crypto.seal_record(START, None, &start, false)?;
        let mut output = Vec::with_capacity(PREFIX_LEN + frame.len());
        output.extend_from_slice(&prefix);
        output.extend_from_slice(&frame);
        Ok((
            Self {
                crypto: Some(crypto),
                head: ResponseHead {
                    status,
                    headers,
                    mode,
                },
                method: material.method,
                limits: material.limits,
                finite_data_sent: false,
                done: false,
            },
            output,
        ))
    }

    /// Protect one complete SSE block, including its final blank line.
    ///
    /// # Errors
    /// Returns a shape, limit, or cryptographic error and closes the writer.
    pub fn seal_sse_block(&mut self, block: &[u8]) -> Result<Vec<u8>, Error> {
        let result = self.seal_sse_block_inner(block);
        if result.is_err() {
            self.poison();
        }
        result
    }

    fn seal_sse_block_inner(&mut self, block: &[u8]) -> Result<Vec<u8>, Error> {
        if self.done || self.head.mode != ResponseMode::Sse {
            return Err(Error::InvalidConfiguration);
        }
        validate_block(block, self.limits.max_body_len)?;
        let (coding, coded) = compression::encode(block);
        self.crypto
            .as_mut()
            .ok_or(Error::InvalidConfiguration)?
            .seal_record(DATA, Some(coding), &coded, false)
    }

    /// Protect the complete logical finite body. An empty body emits no DATA.
    ///
    /// # Errors
    /// Returns a validation, limit, or cryptographic error and closes the writer.
    pub fn seal_finite_body(&mut self, body: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        let result = self.seal_finite_body_inner(body);
        if result.is_err() {
            self.poison();
        }
        result
    }

    fn seal_finite_body_inner(&mut self, body: &[u8]) -> Result<Option<Vec<u8>>, Error> {
        if self.done || self.head.mode != ResponseMode::Finite || self.finite_data_sent {
            return Err(Error::InvalidConfiguration);
        }
        if body.len() > self.limits.max_body_len {
            return Err(Error::LimitExceeded);
        }
        message::validate_response_content_length(
            self.head.status,
            &self.head.headers,
            body.len(),
            self.method,
        )?;
        if !body.is_empty()
            && (self.method == Method::Head || matches!(self.head.status, 204 | 205 | 304))
        {
            return Err(Error::InvalidConfiguration);
        }
        if body.is_empty() {
            self.finite_data_sent = true;
            return Ok(None);
        }
        let (coding, encoded) = compression::encode(body);
        let frame = self
            .crypto
            .as_mut()
            .ok_or(Error::InvalidConfiguration)?
            .seal_record(DATA, Some(coding), &encoded, false)?;
        self.finite_data_sent = true;
        Ok(Some(frame))
    }

    /// Protect END. In finite mode, call `seal_finite_body` once first, even
    /// for an empty body. The caller then ends the outer HTTP body.
    ///
    /// # Errors
    /// Returns an order or cryptographic error and closes the writer.
    pub fn finish(&mut self) -> Result<Vec<u8>, Error> {
        if self.done || (self.head.mode == ResponseMode::Finite && !self.finite_data_sent) {
            self.poison();
            return Err(Error::InvalidConfiguration);
        }
        let result = self
            .crypto
            .as_mut()
            .ok_or(Error::InvalidConfiguration)?
            .seal_record(END, None, &[], true);
        self.poison();
        result
    }

    /// Close this writer without a terminal record.
    pub fn close(&mut self) {
        self.poison();
    }

    fn poison(&mut self) {
        self.crypto = None;
        self.done = true;
    }
}

/// A checked record reader that accepts any byte cuts.
pub struct ResponseOpener {
    material: Option<ResponseMaterial>,
    crypto: Option<Crypto>,
    prefix: Vec<u8>,
    length: Vec<u8>,
    frame: Vec<u8>,
    expected: Option<usize>,
    head: Option<ResponseHead>,
    finite_body: Option<Vec<u8>>,
    finite_response: Option<Response>,
    done: bool,
    failed: bool,
    eof_checked: bool,
}

impl ResponseOpener {
    pub(crate) fn new(material: ResponseMaterial) -> Self {
        Self {
            material: Some(material),
            crypto: None,
            prefix: Vec::with_capacity(PREFIX_LEN),
            length: Vec::with_capacity(4),
            frame: Vec::new(),
            expected: None,
            head: None,
            finite_body: None,
            finite_response: None,
            done: false,
            failed: false,
            eof_checked: false,
        }
    }

    /// Read at most one checked record and return how many input bytes were used.
    /// Pass the unread suffix to the next call. Finite DATA stays inside the
    /// reader and returns no public record.
    ///
    /// # Errors
    /// Any parse, limit, order, or tag error closes this reader.
    pub fn feed(&mut self, input: &[u8]) -> Result<(usize, Option<ResponseRecord>), Error> {
        let result = self.feed_inner(input);
        if result.is_err() {
            self.poison();
        }
        result
    }

    fn feed_inner(&mut self, input: &[u8]) -> Result<(usize, Option<ResponseRecord>), Error> {
        if self.failed || self.eof_checked || (self.done && !input.is_empty()) {
            return Err(Error::MalformedEnvelope);
        }
        let mut used = 0;
        if used < input.len() {
            if self.crypto.is_none() {
                let take = (PREFIX_LEN - self.prefix.len()).min(input.len() - used);
                self.prefix.extend_from_slice(&input[used..used + take]);
                used += take;
                if self.prefix.len() < PREFIX_LEN {
                    return Ok((used, None));
                }
                if self.prefix[..4] != *MAGIC || self.prefix[4] != VERSION {
                    return Err(Error::MalformedEnvelope);
                }
                let prefix: [u8; PREFIX_LEN] = self
                    .prefix
                    .as_slice()
                    .try_into()
                    .map_err(|_| Error::MalformedEnvelope)?;
                let material = self.material.as_ref().ok_or(Error::MalformedEnvelope)?;
                self.crypto = Some(Crypto::new(material, prefix)?);
            }
            if self.expected.is_none() {
                let take = (4 - self.length.len()).min(input.len() - used);
                self.length.extend_from_slice(&input[used..used + take]);
                used += take;
                if self.length.len() < 4 {
                    return Ok((used, None));
                }
                let length = u32::from_be_bytes(
                    self.length
                        .as_slice()
                        .try_into()
                        .map_err(|_| Error::MalformedEnvelope)?,
                ) as usize;
                let material = self.material.as_ref().ok_or(Error::MalformedEnvelope)?;
                let maximum = if self.head.is_none() {
                    4 + material.limits.max_header_bytes
                        + material.limits.max_header_count * 16
                        + TAG_LEN
                } else {
                    2 + material.limits.max_body_len + TAG_LEN
                };
                if length < MIN_CIPHERTEXT_LEN {
                    return Err(Error::MalformedEnvelope);
                }
                if length > maximum {
                    return Err(Error::LimitExceeded);
                }
                self.expected = Some(length);
            }
            let expected = self.expected.ok_or(Error::MalformedEnvelope)?;
            let take = (expected - self.frame.len()).min(input.len() - used);
            self.frame.extend_from_slice(&input[used..used + take]);
            used += take;
            if self.frame.len() < expected {
                return Ok((used, None));
            }
            let record = self.open_frame()?;
            if self.frame.capacity() > RETAIN_FRAME_CAPACITY {
                self.frame = Vec::new();
            } else {
                self.frame.clear();
            }
            self.length.clear();
            self.expected = None;
            if self.done && used < input.len() {
                return Err(Error::MalformedEnvelope);
            }
            return Ok((used, record));
        }
        Ok((used, None))
    }

    fn open_frame(&mut self) -> Result<Option<ResponseRecord>, Error> {
        let length = u32::try_from(self.frame.len()).map_err(|_| Error::LimitExceeded)?;
        self.crypto
            .as_mut()
            .ok_or(Error::MalformedEnvelope)?
            .open(&mut self.frame, length)?;
        let (&kind, body) = self.frame.split_first().ok_or(Error::MalformedEnvelope)?;
        if self.head.is_none() {
            if kind != START || body.len() < 2 {
                return Err(Error::MalformedEnvelope);
            }
            let status = u16::from_be_bytes([body[0], body[1]]);
            let material = self.material.as_ref().ok_or(Error::MalformedEnvelope)?;
            let headers =
                message::decode_fields(&body[2..], material.limits).map_err(decode_error)?;
            let mode = classify_head(status, &headers, material).map_err(decode_error)?;
            let head = ResponseHead {
                status,
                headers,
                mode,
            };
            self.head = Some(head.clone());
            self.crypto
                .as_mut()
                .ok_or(Error::MalformedEnvelope)?
                .advance()?;
            return Ok(Some(ResponseRecord::Start(head)));
        }
        let head = self.head.as_ref().ok_or(Error::MalformedEnvelope)?;
        match kind {
            DATA => {
                if body.len() < 2 {
                    return Err(Error::MalformedEnvelope);
                }
                if head.mode == ResponseMode::Sse {
                    let material = self.material.as_ref().ok_or(Error::MalformedEnvelope)?;
                    let block =
                        compression::decode(body[0], &body[1..], material.limits.max_body_len)?;
                    validate_block(&block, material.limits.max_body_len)?;
                    self.crypto
                        .as_mut()
                        .ok_or(Error::MalformedEnvelope)?
                        .advance()?;
                    Ok(Some(ResponseRecord::SseData(block)))
                } else {
                    if self.finite_body.is_some() {
                        return Err(Error::MalformedEnvelope);
                    }
                    self.crypto
                        .as_mut()
                        .ok_or(Error::MalformedEnvelope)?
                        .advance()?;
                    let material = self.material.as_ref().ok_or(Error::MalformedEnvelope)?;
                    self.finite_body = Some(compression::decode(
                        body[0],
                        &body[1..],
                        material.limits.max_body_len,
                    )?);
                    Ok(None)
                }
            }
            END if body.is_empty() && length as usize == MIN_CIPHERTEXT_LEN => {
                if head.mode == ResponseMode::Finite {
                    self.finish_finite()?;
                }
                self.done = true;
                self.crypto = None;
                Ok(Some(ResponseRecord::End))
            }
            _ => Err(Error::MalformedEnvelope),
        }
    }

    fn finish_finite(&mut self) -> Result<(), Error> {
        let head = self.head.as_ref().ok_or(Error::MalformedEnvelope)?;
        let material = self.material.as_ref().ok_or(Error::MalformedEnvelope)?;
        let body = self.finite_body.take().unwrap_or_default();
        let response = Response {
            status: head.status,
            headers: head.headers.clone(),
            body,
        };
        message::validate_decoded_response_body(&response, material.method)?;
        if !response.body.is_empty()
            && (material.method == Method::Head || matches!(response.status, 204 | 205 | 304))
        {
            return Err(Error::MalformedEnvelope);
        }
        self.finite_response = Some(response);
        Ok(())
    }

    /// Call after the host observes real outer body EOF. Check END and release
    /// a finite reply, if any. This method cannot observe transport EOF itself.
    ///
    /// # Errors
    /// Returns an error if END is absent, the reader failed, or EOF was checked twice.
    pub fn finish_eof(&mut self) -> Result<Option<Response>, Error> {
        if self.failed || !self.done || self.eof_checked {
            self.poison();
            return Err(Error::MalformedEnvelope);
        }
        self.eof_checked = true;
        self.material = None;
        Ok(self.finite_response.take())
    }

    /// Close this reader without claiming a complete response.
    pub fn close(&mut self) {
        self.poison();
    }

    fn poison(&mut self) {
        self.crypto = None;
        self.material = None;
        self.frame = Vec::new();
        self.finite_body = None;
        self.finite_response = None;
        self.failed = true;
    }
}

const fn decode_error(error: Error) -> Error {
    match error {
        Error::InvalidConfiguration => Error::MalformedEnvelope,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_counter_value_is_reserved_for_end() -> Result<(), Error> {
        let material = ResponseMaterial {
            secret: zeroize::Zeroizing::new([7; 32]),
            enc: [8; 32],
            method: Method::Get,
            limits: crate::Limits::default(),
        };
        let mut prefix = [0; PREFIX_LEN];
        prefix[..4].copy_from_slice(MAGIC);
        prefix[4] = VERSION;
        let mut writer = Crypto::new(&material, prefix)?;
        writer.sequence = u64::MAX - 1;
        let data = writer.seal_record(DATA, None, b"x", false)?;
        assert_eq!(writer.sequence, u64::MAX);
        assert_eq!(
            writer.seal_record(DATA, None, b"y", false),
            Err(Error::LimitExceeded)
        );
        let end = writer.seal_record(END, None, &[], true)?;
        let mut reader = Crypto::new(&material, prefix)?;
        reader.sequence = u64::MAX - 1;
        let mut data_body = data[4..].to_vec();
        reader.open(
            &mut data_body,
            u32::from_be_bytes(data[..4].try_into().map_err(|_| Error::MalformedEnvelope)?),
        )?;
        assert_eq!(data_body, [DATA, b'x']);
        reader.advance()?;
        let mut end_body = end[4..].to_vec();
        reader.open(
            &mut end_body,
            u32::from_be_bytes(end[..4].try_into().map_err(|_| Error::MalformedEnvelope)?),
        )?;
        assert_eq!(end_body, [END]);
        assert_eq!(reader.advance(), Err(Error::LimitExceeded));
        Ok(())
    }
}

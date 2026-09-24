//! Typed one-shot client and staged server transactions.

use std::{
    fmt,
    time::{SystemTime, UNIX_EPOCH},
};

use hkdf::Hkdf;
use hpke::{
    Deserializable, Kem as KemTrait, OpModeR, OpModeS, PskBundle, Serializable,
    aead::{AeadCtxR, AeadCtxS, AeadTag, ChaCha20Poly1305 as HpkeChaCha20Poly1305},
    inout::InOutBuf,
    kdf::HkdfSha256,
    kem::X25519HkdfSha256,
    setup_receiver, setup_sender_with_rng,
};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::{
    CLOCK_SKEW_SECS, EntropySource, Error, HARD_MAX_ID_LEN, Limits, Method, REQUEST_LIFETIME_SECS,
    SystemEntropy,
    codec::{
        ENC_LEN, REQUEST_FIXED_LEN, RESPONSE_NONCE_LEN, TAG_LEN, encode_request_header,
        parse_stream_start, stream_start_length,
    },
    compression,
    message::{self, HeaderField, Request, RequestHead, Response},
    response::{ResponseMode, ResponseOpener, ResponseRecord, ResponseSealer},
};

type Kem = X25519HkdfSha256;
type Kdf = HkdfSha256;
type HpkeAead = HpkeChaCha20Poly1305;
type RecipientContext = AeadCtxR<HpkeAead, Kdf, Kem>;
type SenderContext = AeadCtxS<HpkeAead, Kdf, Kem>;

const STREAM_REQUEST_INFO_LABEL: &[u8] = b"message/hpke-http request\0v3\0";
const STREAM_REQUEST_AAD_LABEL: &[u8] = b"hpke-http/3 request record\0";
const STREAM_START: u8 = 1;
const STREAM_DATA: u8 = 2;
const STREAM_END: u8 = 3;
/// Largest clear DATA part in a protected upload.
pub const STREAM_DATA_LEN: usize = 64 * 1024;
/// Maximum DATA records, including small writes.
const STREAM_MAX_DATA_RECORDS: u64 = 1_048_576;
const RESPONSE_EXPORT_LABEL: &[u8] = b"message/hpke-http response\0v3";
const REPLAY_LABEL: &[u8] = b"hpke-http/replay\0v3\0";
const RESPONSE_KEY_INFO: &[u8] = b"hpke-http/3 response key";
const RESPONSE_NONCE_INFO: &[u8] = b"hpke-http/3 response nonce";
const RESPONSE_SECRET_LEN: usize = 32;
const RESPONSE_KEY_LEN: usize = 32;
const AEAD_NONCE_LEN: usize = 12;
const MIN_PSK_LEN: usize = 32;
pub(crate) type ResponseKey = Zeroizing<[u8; RESPONSE_KEY_LEN]>;
pub(crate) type ResponseNonce = Zeroizing<[u8; AEAD_NONCE_LEN]>;

/// A generated X25519 recipient key pair.
pub struct KeyPair {
    private_key: Zeroizing<Vec<u8>>,
    public_key: Vec<u8>,
}

impl fmt::Debug for KeyPair {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KeyPair")
            .field("private_key", &"[REDACTED]")
            .field("public_key_len", &self.public_key.len())
            .finish()
    }
}

impl KeyPair {
    /// Return the encoded private and public keys.
    #[must_use]
    pub fn into_parts(self) -> (Zeroizing<Vec<u8>>, Vec<u8>) {
        (self.private_key, self.public_key)
    }

    /// Borrow the encoded public key.
    #[must_use]
    pub fn public_key(&self) -> &[u8] {
        &self.public_key
    }
}

/// Generate a recipient key pair with the platform entropy source.
///
/// # Errors
///
/// Returns [`Error::EntropyUnavailable`] if the platform source fails.
pub fn generate_key_pair() -> Result<KeyPair, Error> {
    generate_key_pair_with_entropy(&mut SystemEntropy)
}

/// Generate a recipient key pair with an injected fallible entropy source.
///
/// # Errors
///
/// Returns the error reported by `entropy` before producing a key pair.
pub fn generate_key_pair_with_entropy(entropy: &mut impl EntropySource) -> Result<KeyPair, Error> {
    let mut input_key_material = Zeroizing::new([0_u8; 32]);
    entropy.fill(input_key_material.as_mut())?;
    let (private_key, public_key) = Kem::derive_keypair(input_key_material.as_ref());
    Ok(KeyPair {
        private_key: Zeroizing::new(private_key.to_bytes().to_vec()),
        public_key: public_key.to_bytes().to_vec(),
    })
}

/// A request envelope and the one-use capability needed to open its response.
#[must_use]
pub struct ProtectedRequest {
    envelope: Vec<u8>,
    response_token: ResponseToken,
}

impl fmt::Debug for ProtectedRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProtectedRequest")
            .field("envelope_len", &self.envelope.len())
            .field("response_token", &self.response_token)
            .finish()
    }
}

impl ProtectedRequest {
    /// Borrow the complete protected request envelope.
    #[must_use]
    pub fn envelope(&self) -> &[u8] {
        &self.envelope
    }

    /// Split the envelope from its one-use response token.
    pub fn into_parts(self) -> (Vec<u8>, ResponseToken) {
        (self.envelope, self.response_token)
    }
}

/// A client configured for one recipient key and one explicit PSK identity.
pub struct Client {
    recipient_public_key: <Kem as KemTrait>::PublicKey,
    recipient_key_id: Vec<u8>,
    psk: Zeroizing<Vec<u8>>,
    psk_id: Vec<u8>,
    limits: Limits,
}

impl fmt::Debug for Client {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Client")
            .field("recipient_key_id_len", &self.recipient_key_id.len())
            .field("psk", &"[REDACTED]")
            .field("psk_id_len", &self.psk_id.len())
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

impl Client {
    /// Validate and create a client. The public PSK ID cannot equal the PSK.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] for invalid keys, identifiers,
    /// credentials, or limits.
    pub fn new(
        recipient_public_key: &[u8],
        recipient_key_id: Vec<u8>,
        psk: Vec<u8>,
        psk_id: Vec<u8>,
        limits: Limits,
    ) -> Result<Self, Error> {
        let psk = Zeroizing::new(psk);
        validate_id(&recipient_key_id)?;
        validate_credential(psk.as_slice(), &psk_id)?;
        let limits = limits.validate()?;
        let recipient_public_key = <Kem as KemTrait>::PublicKey::from_bytes(recipient_public_key)
            .map_err(|_| Error::InvalidConfiguration)?;
        Ok(Self {
            recipient_public_key,
            recipient_key_id,
            psk,
            psk_id,
            limits,
        })
    }

    /// Protect one complete request with the platform entropy source.
    ///
    /// # Errors
    ///
    /// Returns a validation, clock, entropy, or cryptographic
    /// error before exposing a partially protected envelope.
    pub fn protect(&self, request: &Request) -> Result<ProtectedRequest, Error> {
        self.protect_at_with_entropy(request, unix_time_seconds()?, &mut SystemEntropy)
    }

    /// Protect one complete request at an injected Unix time.
    ///
    /// This entry point exists for hosts with their own trusted clock and for
    /// deterministic conformance tests. Normal native callers use [`Self::protect`].
    ///
    /// # Errors
    ///
    /// Returns a validation, entropy, or cryptographic error.
    pub fn protect_at(
        &self,
        request: &Request,
        now_unix_s: u64,
    ) -> Result<ProtectedRequest, Error> {
        self.protect_at_with_entropy(request, now_unix_s, &mut SystemEntropy)
    }

    /// Protect one complete request with an injected fallible entropy source.
    ///
    /// # Errors
    ///
    /// Returns a validation, clock, entropy, or cryptographic
    /// error before exposing a partially protected envelope.
    pub fn protect_with_entropy(
        &self,
        request: &Request,
        entropy: &mut impl EntropySource,
    ) -> Result<ProtectedRequest, Error> {
        self.protect_at_with_entropy(request, unix_time_seconds()?, entropy)
    }

    /// Protect a request with injected time and entropy.
    ///
    /// # Errors
    ///
    /// Returns a validation, entropy, or cryptographic error
    /// before exposing a partial envelope.
    pub fn protect_at_with_entropy(
        &self,
        request: &Request,
        now_unix_s: u64,
        entropy: &mut impl EntropySource,
    ) -> Result<ProtectedRequest, Error> {
        message::validate_request(request, self.limits)?;
        let head = RequestHead {
            method: request.method,
            authority: request.authority.clone(),
            path: request.path.clone(),
            headers: request.headers.clone(),
        };
        let (mut writer, mut envelope) =
            self.begin_stream_at_with_entropy(&head, now_unix_s, entropy)?;
        let mut remaining = request.body.as_slice();
        while !remaining.is_empty() {
            let (used, record) = writer.push(remaining)?;
            remaining = &remaining[used..];
            if let Some(record) = record {
                envelope.extend_from_slice(&record);
            }
        }
        let (end, response_token) = writer.finish()?;
        envelope.extend_from_slice(&end);
        Ok(ProtectedRequest {
            envelope,
            response_token,
        })
    }
}

/// Public information needed by a host credential resolver.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CredentialRequest {
    /// Opaque public PSK identifier authenticated with the request.
    pub psk_id: Vec<u8>,
}

/// Result of the bounded public preparse stage.
#[must_use]
pub struct PreparsedRequest {
    /// Input for the host credential resolver.
    pub credential: CredentialRequest,
    /// One-use input to [`Server::authenticate`].
    pub token: StartToken,
}

/// Opaque one-use continuation after public preparse.
#[must_use]
pub struct StartToken {
    envelope: Vec<u8>,
}

impl fmt::Debug for StartToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StartToken")
            .field("envelope_len", &self.envelope.len())
            .finish()
    }
}

/// Stable request identifier passed to an atomic replay-admission provider.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ReplayRequest {
    /// SHA-256 replay key for this authenticated attempt.
    pub id: [u8; 32],
    /// Unix second at which this reservation can be evicted.
    pub retain_until_exclusive: u64,
}

impl ReplayRequest {
    /// Bind one provider result to this exact replay request.
    #[must_use]
    pub const fn decision(self, admitted: bool) -> ReplayDecision {
        ReplayDecision {
            id: self.id,
            retain_until_exclusive: self.retain_until_exclusive,
            admitted,
        }
    }
}

/// One atomic replay-provider result bound to its request ID.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayDecision {
    id: [u8; 32],
    retain_until_exclusive: u64,
    admitted: bool,
}

fn check_replay_decision(
    replay_id: [u8; 32],
    retain_until_exclusive: u64,
    decision: ReplayDecision,
    now_unix_s: u64,
) -> Result<(), Error> {
    if replay_id != decision.id || retain_until_exclusive != decision.retain_until_exclusive {
        return Err(Error::ReplayDecisionMismatch);
    }
    if !decision.admitted {
        return Err(Error::ReplayRejected);
    }
    if now_unix_s >= retain_until_exclusive {
        return Err(Error::InvalidRequestTime);
    }
    Ok(())
}

/// Result of authentication, paused before replay admission and plaintext release.
#[must_use]
pub struct AuthenticatedRequest {
    /// Input for an atomic replay-admission provider.
    pub replay: ReplayRequest,
    /// One-use input to [`ReplayToken::admit`].
    pub token: ReplayToken,
}

/// Opaque one-use continuation after request authentication.
#[must_use]
pub struct ReplayToken {
    replay_id: [u8; 32],
    retain_until_exclusive: u64,
    request: Request,
    response: ResponseCapability,
}

impl fmt::Debug for ReplayToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReplayToken")
            .field("replay_id", &self.replay_id)
            .field("retain_until_exclusive", &self.retain_until_exclusive)
            .field("request", &"[REDACTED]")
            .finish_non_exhaustive()
    }
}

impl ReplayToken {
    /// Apply one atomic replay decision before its deadline, then release the
    /// authenticated request.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ReplayDecisionMismatch`] when the decision belongs to a
    /// different request, [`Error::ReplayRejected`] when the provider denies
    /// admission, [`Error::InvalidRequestTime`] when admission completes after
    /// the authenticated deadline, or [`Error::ClockUnavailable`] when the
    /// platform clock cannot be read.
    pub fn admit(self, decision: ReplayDecision) -> Result<OpenedRequest, Error> {
        self.admit_at(decision, unix_time_seconds()?)
    }

    /// Apply one replay decision at a caller-supplied trusted Unix time.
    ///
    /// This entry point exists for hosts with their own trusted clock and for
    /// deterministic conformance tests. Normal native callers use [`Self::admit`].
    ///
    /// # Errors
    ///
    /// Returns the same decision, freshness, and clock-independent errors as
    /// [`Self::admit`].
    pub fn admit_at(
        self,
        decision: ReplayDecision,
        now_unix_s: u64,
    ) -> Result<OpenedRequest, Error> {
        check_replay_decision(
            self.replay_id,
            self.retain_until_exclusive,
            decision,
            now_unix_s,
        )?;
        Ok(OpenedRequest {
            request: self.request,
            response: self.response,
        })
    }
}

/// A fully authenticated request and its one-use response capability.
#[must_use]
pub struct OpenedRequest {
    /// Verified request data safe for application dispatch.
    pub request: Request,
    /// One-use capability bound to this request.
    pub response: ResponseCapability,
}

/// A server configured for one static recipient key.
pub struct Server {
    recipient_private_key: <Kem as KemTrait>::PrivateKey,
    recipient_key_id: Vec<u8>,
    limits: Limits,
}

impl fmt::Debug for Server {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Server")
            .field("recipient_private_key", &"[REDACTED]")
            .field("recipient_key_id_len", &self.recipient_key_id.len())
            .field("limits", &self.limits)
            .finish()
    }
}

impl Server {
    /// Validate and create a server for one static recipient key.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfiguration`] for invalid key material,
    /// identifiers, or limits.
    pub fn new(
        recipient_private_key: &[u8],
        recipient_key_id: Vec<u8>,
        limits: Limits,
    ) -> Result<Self, Error> {
        validate_id(&recipient_key_id)?;
        let limits = limits.validate()?;
        let recipient_private_key =
            <Kem as KemTrait>::PrivateKey::from_bytes(recipient_private_key)
                .map_err(|_| Error::InvalidConfiguration)?;
        Ok(Self {
            recipient_private_key,
            recipient_key_id,
            limits,
        })
    }

    /// Return the encoded public key for this server's private key.
    #[must_use]
    pub fn public_key(&self) -> Vec<u8> {
        Kem::sk_to_pk(&self.recipient_private_key)
            .to_bytes()
            .to_vec()
    }

    /// Return the largest complete request envelope accepted by `preparse`.
    ///
    /// Bindings use this limit before they copy a complete envelope.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LimitExceeded`] if the size cannot fit in `usize`.
    pub fn max_complete_envelope_len(&self) -> Result<usize, Error> {
        max_complete_envelope_len(self.limits)
    }

    /// Parse bounded public fields without accepting credentials or plaintext.
    ///
    /// # Errors
    ///
    /// Returns a stable parse, limit, version, suite, or recipient-key error.
    pub fn preparse(&self, envelope: &[u8]) -> Result<PreparsedRequest, Error> {
        let credential = self.preparse_credential(envelope)?;
        Ok(PreparsedRequest {
            credential,
            token: StartToken {
                envelope: envelope.to_vec(),
            },
        })
    }

    /// Parse an owned envelope without cloning its complete ciphertext.
    ///
    /// Bindings which already own their input can move it into the one-shot
    /// credential-resolution token. The same bounded public checks run before
    /// the token is created.
    ///
    /// # Errors
    ///
    /// Returns a stable parse, limit, version, suite, or recipient-key error.
    pub fn preparse_owned(&self, envelope: Vec<u8>) -> Result<PreparsedRequest, Error> {
        let credential = self.preparse_credential(&envelope)?;
        Ok(PreparsedRequest {
            credential,
            token: StartToken { envelope },
        })
    }

    fn preparse_credential(&self, envelope: &[u8]) -> Result<CredentialRequest, Error> {
        if envelope.len() > self.max_complete_envelope_len()? {
            return Err(Error::LimitExceeded);
        }
        let first_len = self
            .stream_start_length(envelope)?
            .ok_or(Error::MalformedEnvelope)?;
        if first_len > envelope.len() {
            return Err(Error::MalformedEnvelope);
        }
        let parsed = parse_stream_start(
            &envelope[..first_len],
            max_stream_start_ciphertext(self.limits),
        )?;
        if parsed.key_id != self.recipient_key_id {
            return Err(Error::UnknownRecipientKey);
        }
        Ok(CredentialRequest {
            psk_id: parsed.psk_id.to_vec(),
        })
    }

    /// Authenticate and parse the full request, then pause before replay admission.
    ///
    /// Plaintext stays inside the returned opaque token until admission.
    ///
    /// # Errors
    ///
    /// Returns a credential, recipient-key, parse, authentication,
    /// request-time, or clock error.
    pub fn authenticate(
        &self,
        token: StartToken,
        psk: &[u8],
    ) -> Result<AuthenticatedRequest, Error> {
        self.authenticate_at(token, psk, unix_time_seconds()?)
    }

    /// Authenticate at an injected Unix time, then pause before replay admission.
    ///
    /// # Errors
    ///
    /// Returns a credential, recipient-key, parse, authentication, or request-time error.
    pub fn authenticate_at(
        &self,
        token: StartToken,
        psk: &[u8],
        now_unix_s: u64,
    ) -> Result<AuthenticatedRequest, Error> {
        let StartToken { envelope } = token;
        let first_len = self
            .stream_start_length(&envelope)?
            .ok_or(Error::MalformedEnvelope)?;
        if first_len > envelope.len() {
            return Err(Error::MalformedEnvelope);
        }
        let authenticated = self.authenticate_stream_at(
            StreamStartToken {
                first: envelope[..first_len].to_vec(),
            },
            psk,
            now_unix_s,
        )?;
        let replay = authenticated.replay;
        let StreamReplayToken {
            head, mut reader, ..
        } = authenticated.token;
        let mut body = Vec::new();
        let mut remaining = &envelope[first_len..];
        while !remaining.is_empty() {
            let (used, record) = reader.feed(remaining)?;
            remaining = &remaining[used..];
            if let Some(StreamRequestRecord::Data(part)) = record {
                if part.len() > self.limits.max_body_len.saturating_sub(body.len()) {
                    return Err(Error::LimitExceeded);
                }
                body.extend_from_slice(&part);
            }
        }
        let response = reader.finish_eof()?;
        let request = Request {
            method: head.method,
            authority: head.authority,
            path: head.path,
            headers: head.headers,
            body,
        };
        Ok(AuthenticatedRequest {
            replay,
            token: ReplayToken {
                replay_id: replay.id,
                retain_until_exclusive: replay.retain_until_exclusive,
                request,
                response,
            },
        })
    }
}

#[cfg(test)]
mod stream_tests {
    use super::STREAM_MAX_DATA_RECORDS;
    use crate::{Client, Error, Limits, Method, RequestHead, Server, generate_key_pair};

    #[test]
    fn data_record_count_guard_closes_both_sides() -> Result<(), Error> {
        let keys = generate_key_pair()?;
        let psk = b"a 32-byte minimum upload credential";
        let client = Client::new(
            keys.public_key(),
            b"key".to_vec(),
            psk.to_vec(),
            b"psk".to_vec(),
            Limits::default(),
        )?;
        let server = Server::new(&keys.into_parts().0, b"key".to_vec(), Limits::default())?;
        let head = RequestHead {
            method: Method::Post,
            authority: b"api.example.test".to_vec(),
            path: b"/upload".to_vec(),
            headers: Vec::new(),
        };
        let (mut writer, _) = client.begin_stream(&head)?;
        writer.data_records = STREAM_MAX_DATA_RECORDS;
        assert_eq!(writer.push(b"x").err(), Some(Error::LimitExceeded));
        assert!(writer.context.is_none());

        let (mut writer, first) = client.begin_stream(&head)?;
        let (used, record) = writer.push(b"x")?;
        assert_eq!(used, 1);
        assert!(record.is_none());
        let (data, _) = writer.finish()?;
        let preparsed = server.preparse_stream(&first)?;
        let authenticated = server.authenticate_stream(preparsed.token, psk)?;
        let mut reader = authenticated
            .token
            .admit(authenticated.replay.decision(true))?
            .reader;
        reader.data_records = STREAM_MAX_DATA_RECORDS;
        assert_eq!(reader.feed(&data).err(), Some(Error::LimitExceeded));
        assert!(reader.context.is_none());
        assert!(reader.finish_eof().is_err());
        Ok(())
    }
}

pub(crate) struct ResponseMaterial {
    pub(crate) secret: Zeroizing<[u8; RESPONSE_SECRET_LEN]>,
    pub(crate) enc: [u8; ENC_LEN],
    pub(crate) method: Method,
    pub(crate) limits: Limits,
}

impl fmt::Debug for ResponseMaterial {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResponseMaterial")
            .field("secret", &"[REDACTED]")
            .field("enc_len", &self.enc.len())
            .field("method", &self.method)
            .field("limits", &self.limits)
            .finish()
    }
}

/// One-use client capability for opening the response to one request.
#[must_use]
pub struct ResponseToken(ResponseMaterial);

impl fmt::Debug for ResponseToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ResponseToken")
            .field(&self.0)
            .finish()
    }
}

impl ResponseToken {
    /// Consume this token to check response records as they arrive.
    #[must_use]
    pub fn into_opener(self) -> ResponseOpener {
        ResponseOpener::new(self.0)
    }

    /// Open one complete finite response with the v3 record reader.
    ///
    /// # Errors
    /// Returns a parse, limit, authentication, or response-mode error.
    pub fn open_finite(self, envelope: &[u8]) -> Result<Response, Error> {
        let mut opener = self.into_opener();
        let mut offset = 0;
        while offset < envelope.len() {
            let (used, record) = opener.feed(&envelope[offset..])?;
            offset += used;
            if matches!(record, Some(ResponseRecord::Start(ref head)) if head.mode == ResponseMode::Sse)
            {
                return Err(Error::InvalidConfiguration);
            }
            if used == 0 {
                return Err(Error::MalformedEnvelope);
            }
        }
        opener.finish_eof()?.ok_or(Error::MalformedEnvelope)
    }
}

/// One-use server capability for protecting the response to one opened request.
#[must_use]
pub struct ResponseCapability(ResponseMaterial);

impl fmt::Debug for ResponseCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ResponseCapability")
            .field(&self.0)
            .finish()
    }
}

impl ResponseCapability {
    /// Start one checked response. The returned bytes hold the prefix and START.
    ///
    /// # Errors
    /// Returns a validation, entropy, or cryptographic error.
    pub fn into_sealer(
        self,
        status: u16,
        headers: Vec<HeaderField>,
    ) -> Result<(ResponseSealer, Vec<u8>), Error> {
        self.into_sealer_with_entropy(status, headers, &mut SystemEntropy)
    }

    /// Start one checked response with an injected entropy source.
    ///
    /// # Errors
    /// Returns a validation, entropy, or cryptographic error.
    pub fn into_sealer_with_entropy(
        self,
        status: u16,
        headers: Vec<HeaderField>,
        entropy: &mut impl EntropySource,
    ) -> Result<(ResponseSealer, Vec<u8>), Error> {
        ResponseSealer::new(&self.0, status, headers, entropy)
    }

    /// Protect one complete finite response with the v3 record writer.
    ///
    /// # Errors
    /// Returns a validation, limit, entropy, or cryptographic error.
    pub fn protect_finite(self, response: &Response) -> Result<Vec<u8>, Error> {
        self.protect_finite_with_entropy(response, &mut SystemEntropy)
    }

    /// Protect one complete finite response with injected entropy.
    ///
    /// # Errors
    /// Returns a validation, limit, entropy, or cryptographic error.
    pub fn protect_finite_with_entropy(
        self,
        response: &Response,
        entropy: &mut impl EntropySource,
    ) -> Result<Vec<u8>, Error> {
        self.protect_finite_parts_with_entropy(
            response.status,
            response.headers.clone(),
            &response.body,
            entropy,
        )
    }

    /// Protect a finite response from a borrowed body without copying it first.
    ///
    /// # Errors
    /// Returns a validation, limit, entropy, or cryptographic error.
    pub fn protect_finite_parts(
        self,
        status: u16,
        headers: Vec<HeaderField>,
        body: &[u8],
    ) -> Result<Vec<u8>, Error> {
        self.protect_finite_parts_with_entropy(status, headers, body, &mut SystemEntropy)
    }

    fn protect_finite_parts_with_entropy(
        self,
        status: u16,
        headers: Vec<HeaderField>,
        body: &[u8],
        entropy: &mut impl EntropySource,
    ) -> Result<Vec<u8>, Error> {
        let (mut sealer, mut output) = self.into_sealer_with_entropy(status, headers, entropy)?;
        if sealer.head_mode() != ResponseMode::Finite {
            return Err(Error::InvalidConfiguration);
        }
        let data = sealer.seal_finite_body(body)?;
        let end = sealer.finish()?;
        output.reserve(data.as_ref().map_or(0, Vec::len) + end.len());
        if let Some(data) = data {
            output.extend_from_slice(&data);
        }
        output.extend_from_slice(&end);
        Ok(output)
    }
}

fn validate_id(identifier: &[u8]) -> Result<(), Error> {
    if identifier.is_empty() || identifier.len() > HARD_MAX_ID_LEN {
        return Err(Error::InvalidConfiguration);
    }
    Ok(())
}

fn validate_credential(psk: &[u8], psk_id: &[u8]) -> Result<(), Error> {
    validate_id(psk_id)?;
    if psk.len() < MIN_PSK_LEN || psk == psk_id {
        return Err(Error::InvalidConfiguration);
    }
    Ok(())
}

fn unix_time_seconds() -> Result<u64, Error> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| Error::ClockUnavailable)
        .map(|duration| duration.as_secs())
}

fn replay_retention_deadline(issued_at_unix_s: u64, now_unix_s: u64) -> Result<u64, Error> {
    let latest_issued_at = now_unix_s
        .checked_add(CLOCK_SKEW_SECS)
        .ok_or(Error::InvalidRequestTime)?;
    let retain_until_exclusive = issued_at_unix_s
        .checked_add(REQUEST_LIFETIME_SECS)
        .and_then(|value| value.checked_add(CLOCK_SKEW_SECS))
        .ok_or(Error::InvalidRequestTime)?;
    if issued_at_unix_s > latest_issued_at || now_unix_s >= retain_until_exclusive {
        return Err(Error::InvalidRequestTime);
    }
    Ok(retain_until_exclusive)
}

fn replay_id(header: &[u8], enc: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(REPLAY_LABEL);
    digest.update(header);
    digest.update(enc);
    digest.finalize().into()
}

fn export_response_material<C>(
    context: &C,
    enc: &[u8],
    method: Method,
    limits: Limits,
) -> Result<ResponseMaterial, Error>
where
    C: Exporter,
{
    let enc: [u8; ENC_LEN] = enc.try_into().map_err(|_| Error::CryptoFailure)?;
    let mut secret = Zeroizing::new([0_u8; RESPONSE_SECRET_LEN]);
    context.export_secret(RESPONSE_EXPORT_LABEL, secret.as_mut())?;
    Ok(ResponseMaterial {
        secret,
        enc,
        method,
        limits,
    })
}

pub(crate) fn derive_response_key_nonce(
    material: &ResponseMaterial,
    response_nonce: &[u8],
) -> Result<(ResponseKey, ResponseNonce), Error> {
    if response_nonce.len() != RESPONSE_NONCE_LEN {
        return Err(Error::MalformedEnvelope);
    }
    let mut salt = Zeroizing::new(Vec::with_capacity(ENC_LEN + RESPONSE_NONCE_LEN));
    salt.extend_from_slice(&material.enc);
    salt.extend_from_slice(response_nonce);
    let hkdf = Hkdf::<Sha256>::new(Some(&salt), material.secret.as_ref());
    let mut key = Zeroizing::new([0_u8; RESPONSE_KEY_LEN]);
    hkdf.expand(RESPONSE_KEY_INFO, key.as_mut())
        .map_err(|_| Error::CryptoFailure)?;
    let mut nonce = Zeroizing::new([0_u8; AEAD_NONCE_LEN]);
    hkdf.expand(RESPONSE_NONCE_INFO, nonce.as_mut())
        .map_err(|_| Error::CryptoFailure)?;
    Ok((key, nonce))
}

trait Exporter {
    fn export_secret(&self, context: &[u8], output: &mut [u8]) -> Result<(), Error>;
}

impl Exporter for hpke::aead::AeadCtxS<HpkeAead, Kdf, Kem> {
    fn export_secret(&self, context: &[u8], output: &mut [u8]) -> Result<(), Error> {
        self.export(context, output)
            .map_err(|_| Error::CryptoFailure)
    }
}

impl Exporter for RecipientContext {
    fn export_secret(&self, context: &[u8], output: &mut [u8]) -> Result<(), Error> {
        self.export(context, output)
            .map_err(|_| Error::CryptoFailure)
    }
}

fn max_stream_start_ciphertext(limits: Limits) -> usize {
    limits.max_header_bytes + limits.max_target_len + limits.max_header_count * 16 + 256 + TAG_LEN
}

fn max_complete_envelope_len(limits: Limits) -> Result<usize, Error> {
    let records = limits
        .max_body_len
        .min(usize::try_from(STREAM_MAX_DATA_RECORDS).map_err(|_| Error::LimitExceeded)?);
    let data_overhead = records
        .checked_mul(4 + 2 + TAG_LEN)
        .ok_or(Error::LimitExceeded)?;
    REQUEST_FIXED_LEN
        .checked_add(2 * HARD_MAX_ID_LEN)
        .and_then(|length| length.checked_add(ENC_LEN + 4))
        .and_then(|length| length.checked_add(max_stream_start_ciphertext(limits)))
        .and_then(|length| length.checked_add(limits.max_body_len))
        .and_then(|length| length.checked_add(data_overhead))
        .and_then(|length| length.checked_add(4 + 1 + TAG_LEN))
        .ok_or(Error::LimitExceeded)
}

fn stream_request_info(header: &[u8]) -> Vec<u8> {
    let mut info = Vec::with_capacity(STREAM_REQUEST_INFO_LABEL.len() + header.len());
    info.extend_from_slice(STREAM_REQUEST_INFO_LABEL);
    info.extend_from_slice(header);
    info
}

fn stream_record_aad(header: &[u8], sequence: u64, ciphertext_len: u32) -> Vec<u8> {
    let mut aad = Vec::with_capacity(STREAM_REQUEST_AAD_LABEL.len() + header.len() + 12);
    aad.extend_from_slice(STREAM_REQUEST_AAD_LABEL);
    aad.extend_from_slice(header);
    aad.extend_from_slice(&sequence.to_be_bytes());
    aad.extend_from_slice(&ciphertext_len.to_be_bytes());
    aad
}

fn seal_stream_record(
    context: &mut SenderContext,
    header: &[u8],
    sequence: u64,
    kind: u8,
    coding: Option<u8>,
    body: &[u8],
) -> Result<Vec<u8>, Error> {
    let ciphertext_len = body
        .len()
        .checked_add(1 + usize::from(coding.is_some()) + TAG_LEN)
        .ok_or(Error::LimitExceeded)?;
    let ciphertext_len = u32::try_from(ciphertext_len).map_err(|_| Error::LimitExceeded)?;
    let mut frame = Vec::with_capacity(4 + ciphertext_len as usize);
    frame.extend_from_slice(&ciphertext_len.to_be_bytes());
    frame.push(kind);
    if let Some(coding) = coding {
        frame.push(coding);
    }
    frame.extend_from_slice(body);
    let aad = stream_record_aad(header, sequence, ciphertext_len);
    let tag = context
        .seal_inout_detached(InOutBuf::from(&mut frame[4..]), &aad)
        .map_err(|_| Error::CryptoFailure)?;
    frame.extend_from_slice(tag.to_bytes().as_ref());
    Ok(frame)
}

/// One request writer. A failed call closes it and prevents a valid END.
pub struct StreamRequestSealer {
    context: Option<SenderContext>,
    header: Vec<u8>,
    sequence: u64,
    data_records: u64,
    total_bytes: u64,
    content_length: Option<u64>,
    max_upload_bytes: u64,
    pending: Vec<u8>,
    response_token: Option<ResponseToken>,
}

impl fmt::Debug for StreamRequestSealer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamRequestSealer")
            .field("sequence", &self.sequence)
            .field("total_bytes", &self.total_bytes)
            .finish_non_exhaustive()
    }
}

impl StreamRequestSealer {
    /// Add clear bytes. A full 64 KiB DATA record is returned when ready.
    /// The returned count can be less than `part.len()`. Pass `&part[used..]`
    /// to the next call until all input bytes are used.
    ///
    /// # Errors
    /// Returns a state, length, count, or cryptographic error and closes the writer.
    pub fn push(&mut self, part: &[u8]) -> Result<(usize, Option<Vec<u8>>), Error> {
        let result = self.push_inner(part);
        if result.is_err() {
            self.context = None;
            self.response_token = None;
        }
        result
    }

    fn push_inner(&mut self, part: &[u8]) -> Result<(usize, Option<Vec<u8>>), Error> {
        if self.context.is_none() {
            return Err(Error::InvalidConfiguration);
        }
        if part.is_empty() {
            return Ok((0, None));
        }
        if self.data_records >= STREAM_MAX_DATA_RECORDS {
            return Err(Error::LimitExceeded);
        }
        let consumed = part.len().min(STREAM_DATA_LEN - self.pending.len());
        let next_total = self
            .total_bytes
            .checked_add(consumed as u64)
            .ok_or(Error::LimitExceeded)?;
        if next_total > self.max_upload_bytes
            || self
                .content_length
                .is_some_and(|length| next_total > length)
        {
            return Err(Error::LimitExceeded);
        }
        self.pending.extend_from_slice(&part[..consumed]);
        self.total_bytes = next_total;
        if self.pending.len() == STREAM_DATA_LEN {
            let frame = self.seal_pending()?;
            Ok((consumed, Some(frame)))
        } else {
            Ok((consumed, None))
        }
    }

    fn seal_pending(&mut self) -> Result<Vec<u8>, Error> {
        if self.pending.is_empty() || self.data_records >= STREAM_MAX_DATA_RECORDS {
            return Err(Error::LimitExceeded);
        }
        let (coding, coded) = compression::encode(&self.pending);
        let context = self.context.as_mut().ok_or(Error::InvalidConfiguration)?;
        let frame = seal_stream_record(
            context,
            &self.header,
            self.sequence,
            STREAM_DATA,
            Some(coding),
            &coded,
        )?;
        self.pending.clear();
        self.sequence += 1;
        self.data_records += 1;
        Ok(frame)
    }

    /// Protect END and release the one-use response right.
    ///
    /// # Errors
    /// Returns a state, logical length, or cryptographic error.
    pub fn finish(mut self) -> Result<(Vec<u8>, ResponseToken), Error> {
        if self
            .content_length
            .is_some_and(|length| self.total_bytes != length)
        {
            return Err(Error::InvalidConfiguration);
        }
        let mut frame = if self.pending.is_empty() {
            Vec::new()
        } else {
            self.seal_pending()?
        };
        let context = self.context.as_mut().ok_or(Error::InvalidConfiguration)?;
        frame.extend_from_slice(&seal_stream_record(
            context,
            &self.header,
            self.sequence,
            STREAM_END,
            None,
            &[],
        )?);
        let token = self
            .response_token
            .take()
            .ok_or(Error::InvalidConfiguration)?;
        Ok((frame, token))
    }
}

impl Client {
    /// Start a protected request and return its bounded prefix and checked START.
    ///
    /// # Errors
    /// Returns a validation, entropy, or cryptographic error before source bytes are read.
    pub fn begin_stream(
        &self,
        head: &RequestHead,
    ) -> Result<(StreamRequestSealer, Vec<u8>), Error> {
        self.begin_stream_at_with_entropy(head, unix_time_seconds()?, &mut SystemEntropy)
    }

    /// Start a stream with caller-supplied time and entropy for wire tests.
    ///
    /// # Errors
    /// Returns a validation, entropy, or cryptographic error.
    pub fn begin_stream_at_with_entropy(
        &self,
        head: &RequestHead,
        now_unix_s: u64,
        entropy: &mut impl EntropySource,
    ) -> Result<(StreamRequestSealer, Vec<u8>), Error> {
        let max_upload_bytes = self.limits.max_request_bytes;
        let content_length = message::stream_content_length(&head.headers)?;
        if content_length.is_some_and(|length| length > max_upload_bytes) {
            return Err(Error::LimitExceeded);
        }
        let clear_head = message::encode_stream_head(head, self.limits)?;
        let header = encode_request_header(&self.recipient_key_id, &self.psk_id, now_unix_s)?;
        let psk_bundle = PskBundle::new(self.psk.as_slice(), &self.psk_id)
            .map_err(|_| Error::InvalidConfiguration)?;
        let mut random_seed = Zeroizing::new([0_u8; 32]);
        entropy.fill(random_seed.as_mut())?;
        let mut random = ChaCha20Rng::from_seed(*random_seed);
        let (encapped_key, mut context) = setup_sender_with_rng::<HpkeAead, Kdf, Kem>(
            &OpModeS::Psk(psk_bundle),
            &self.recipient_public_key,
            &stream_request_info(&header),
            &mut random,
        )
        .map_err(|_| Error::CryptoFailure)?;
        let enc = encapped_key.to_bytes();
        let response_material =
            export_response_material(&context, enc.as_ref(), head.method, self.limits)?;
        let start = seal_stream_record(&mut context, &header, 0, STREAM_START, None, &clear_head)?;
        let mut first = Vec::with_capacity(header.len() + ENC_LEN + start.len());
        first.extend_from_slice(&header);
        first.extend_from_slice(enc.as_ref());
        first.extend_from_slice(&start);
        Ok((
            StreamRequestSealer {
                context: Some(context),
                header,
                sequence: 1,
                data_records: 0,
                total_bytes: 0,
                content_length,
                max_upload_bytes,
                pending: Vec::with_capacity(STREAM_DATA_LEN),
                response_token: Some(ResponseToken(response_material)),
            },
            first,
        ))
    }
}

/// Public credential ID and a one-use bounded START token.
#[must_use]
pub struct PreparsedStreamRequest {
    /// Input for the host credential resolver.
    pub credential: CredentialRequest,
    /// One-use input for START authentication.
    pub token: StreamStartToken,
}

/// Opaque START bytes before credential resolution.
#[must_use]
pub struct StreamStartToken {
    first: Vec<u8>,
}

/// Authenticated START, with plaintext withheld until replay admission.
#[must_use]
pub struct AuthenticatedStreamRequest {
    /// Input for an atomic replay-admission provider.
    pub replay: ReplayRequest,
    /// One-use input for replay admission.
    pub token: StreamReplayToken,
}

/// One-use continuation after START authentication.
#[must_use]
pub struct StreamReplayToken {
    replay_id: [u8; 32],
    retain_until_exclusive: u64,
    head: RequestHead,
    reader: StreamRequestOpener,
}

impl StreamReplayToken {
    /// Apply the replay result before plaintext or DATA is released.
    ///
    /// # Errors
    /// Returns a replay or clock error.
    pub fn admit(self, decision: ReplayDecision) -> Result<OpenedStreamRequest, Error> {
        self.admit_at(decision, unix_time_seconds()?)
    }

    /// Apply the replay result at a caller-supplied trusted Unix time.
    ///
    /// # Errors
    /// Returns a replay or freshness error.
    pub fn admit_at(
        self,
        decision: ReplayDecision,
        now_unix_s: u64,
    ) -> Result<OpenedStreamRequest, Error> {
        check_replay_decision(
            self.replay_id,
            self.retain_until_exclusive,
            decision,
            now_unix_s,
        )?;
        Ok(OpenedStreamRequest {
            head: self.head,
            reader: self.reader,
        })
    }
}

/// Checked request head and a live body reader. The response right waits for END and EOF.
#[must_use]
pub struct OpenedStreamRequest {
    /// Authenticated request fields.
    pub head: RequestHead,
    /// Checks each DATA part before it is released.
    pub reader: StreamRequestOpener,
}

/// One checked upload record.
#[derive(Debug, Eq, PartialEq)]
pub enum StreamRequestRecord {
    /// One checked nonempty clear DATA part.
    Data(Vec<u8>),
    /// Checked END. True outer EOF must still follow.
    End,
}

/// Incremental request reader with bounded frame storage.
pub struct StreamRequestOpener {
    context: Option<RecipientContext>,
    header: Vec<u8>,
    sequence: u64,
    data_records: u64,
    total_bytes: u64,
    content_length: Option<u64>,
    max_upload_bytes: u64,
    frame: Vec<u8>,
    ended: bool,
    response_material: Option<ResponseMaterial>,
}

impl fmt::Debug for StreamRequestOpener {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamRequestOpener")
            .field("sequence", &self.sequence)
            .field("total_bytes", &self.total_bytes)
            .field("ended", &self.ended)
            .finish_non_exhaustive()
    }
}

impl StreamRequestOpener {
    /// Read at most one record from any byte cut.
    /// The returned count is the number of input bytes used. Pass the unread
    /// suffix to the next call.
    ///
    /// # Errors
    /// Returns a parse, order, limit, or authentication error and closes the reader.
    pub fn feed(&mut self, input: &[u8]) -> Result<(usize, Option<StreamRequestRecord>), Error> {
        let result = self.feed_inner(input);
        if result.is_err() {
            self.context = None;
            self.response_material = None;
            self.frame.clear();
        }
        result
    }

    fn feed_inner(&mut self, input: &[u8]) -> Result<(usize, Option<StreamRequestRecord>), Error> {
        if self.context.is_none() || (self.ended && !input.is_empty()) {
            return Err(Error::MalformedEnvelope);
        }
        let mut used = 0;
        if self.frame.len() < 4 {
            let take = (4 - self.frame.len()).min(input.len());
            self.frame.extend_from_slice(&input[..take]);
            used += take;
            if self.frame.len() < 4 {
                return Ok((used, None));
            }
        }
        let ciphertext_len = u32::from_be_bytes(
            self.frame[..4]
                .try_into()
                .map_err(|_| Error::MalformedEnvelope)?,
        ) as usize;
        if ciphertext_len < 1 + TAG_LEN {
            return Err(Error::MalformedEnvelope);
        }
        if ciphertext_len > STREAM_DATA_LEN + 2 + TAG_LEN {
            return Err(Error::LimitExceeded);
        }
        let frame_len = 4 + ciphertext_len;
        let take = (frame_len - self.frame.len()).min(input.len() - used);
        self.frame.extend_from_slice(&input[used..used + take]);
        used += take;
        if self.frame.len() < frame_len {
            return Ok((used, None));
        }
        let plain_end = frame_len - TAG_LEN;
        let tag = AeadTag::<HpkeAead>::from_bytes(&self.frame[plain_end..])
            .map_err(|_| Error::AuthenticationFailed)?;
        let aad = stream_record_aad(
            &self.header,
            self.sequence,
            u32::try_from(ciphertext_len).map_err(|_| Error::LimitExceeded)?,
        );
        self.context
            .as_mut()
            .ok_or(Error::MalformedEnvelope)?
            .open_inout_detached(InOutBuf::from(&mut self.frame[4..plain_end]), &aad, &tag)
            .map_err(|_| Error::AuthenticationFailed)?;
        let kind = self.frame[4];
        let body = &self.frame[5..plain_end];
        let record = match kind {
            STREAM_DATA => {
                if body.len() < 2 || self.data_records >= STREAM_MAX_DATA_RECORDS {
                    return Err(if body.len() < 2 {
                        Error::MalformedEnvelope
                    } else {
                        Error::LimitExceeded
                    });
                }
                let clear = compression::decode(body[0], &body[1..], STREAM_DATA_LEN)?;
                let next_total = self
                    .total_bytes
                    .checked_add(clear.len() as u64)
                    .ok_or(Error::LimitExceeded)?;
                if next_total > self.max_upload_bytes
                    || self
                        .content_length
                        .is_some_and(|length| next_total > length)
                {
                    return Err(Error::LimitExceeded);
                }
                self.total_bytes = next_total;
                self.data_records += 1;
                self.sequence += 1;
                StreamRequestRecord::Data(clear)
            }
            STREAM_END => {
                if !body.is_empty()
                    || self
                        .content_length
                        .is_some_and(|length| self.total_bytes != length)
                {
                    return Err(Error::MalformedEnvelope);
                }
                self.ended = true;
                StreamRequestRecord::End
            }
            _ => return Err(Error::MalformedEnvelope),
        };
        self.frame.clear();
        Ok((used, Some(record)))
    }

    /// Check true outer EOF after END and release the response right.
    ///
    /// # Errors
    /// Returns an error for a missing END or partial frame.
    pub fn finish_eof(mut self) -> Result<ResponseCapability, Error> {
        if !self.ended || !self.frame.is_empty() || self.context.is_none() {
            return Err(Error::MalformedEnvelope);
        }
        Ok(ResponseCapability(
            self.response_material
                .take()
                .ok_or(Error::MalformedEnvelope)?,
        ))
    }
}

impl Server {
    /// Return the START boundary once its bounded frame length is known.
    ///
    /// # Errors
    /// Returns a parse, version, suite, or size error.
    pub fn stream_start_length(&self, input: &[u8]) -> Result<Option<usize>, Error> {
        stream_start_length(input, max_stream_start_ciphertext(self.limits))
    }

    /// Read the exact public prefix and START frame before credential lookup.
    ///
    /// # Errors
    /// Returns a parse, limit, or recipient-key error.
    pub fn preparse_stream(&self, first: &[u8]) -> Result<PreparsedStreamRequest, Error> {
        let parsed = parse_stream_start(first, max_stream_start_ciphertext(self.limits))?;
        if parsed.key_id != self.recipient_key_id {
            return Err(Error::UnknownRecipientKey);
        }
        Ok(PreparsedStreamRequest {
            credential: CredentialRequest {
                psk_id: parsed.psk_id.to_vec(),
            },
            token: StreamStartToken {
                first: first.to_vec(),
            },
        })
    }

    /// Authenticate START, then pause before replay admission.
    ///
    /// # Errors
    /// Returns a credential, time, authentication, or parse error.
    pub fn authenticate_stream(
        &self,
        token: StreamStartToken,
        psk: &[u8],
    ) -> Result<AuthenticatedStreamRequest, Error> {
        self.authenticate_stream_at(token, psk, unix_time_seconds()?)
    }

    /// Authenticate START at a caller-supplied trusted Unix time.
    ///
    /// # Errors
    /// Returns a credential, time, authentication, or parse error.
    pub fn authenticate_stream_at(
        &self,
        token: StreamStartToken,
        psk: &[u8],
        now_unix_s: u64,
    ) -> Result<AuthenticatedStreamRequest, Error> {
        let StreamStartToken { first } = token;
        let max_upload_bytes = self.limits.max_request_bytes;
        let parsed = parse_stream_start(&first, max_stream_start_ciphertext(self.limits))?;
        if parsed.key_id != self.recipient_key_id {
            return Err(Error::UnknownRecipientKey);
        }
        validate_credential(psk, parsed.psk_id).map_err(|_| Error::InvalidCredential)?;
        let encapped_key = <Kem as KemTrait>::EncappedKey::from_bytes(parsed.enc)
            .map_err(|_| Error::AuthenticationFailed)?;
        let psk_bundle =
            PskBundle::new(psk, parsed.psk_id).map_err(|_| Error::InvalidCredential)?;
        let mut context = setup_receiver::<HpkeAead, Kdf, Kem>(
            &OpModeR::Psk(psk_bundle),
            &self.recipient_private_key,
            &encapped_key,
            &stream_request_info(parsed.header),
        )
        .map_err(|_| Error::AuthenticationFailed)?;
        let mut clear = parsed.ciphertext.to_vec();
        let plain_end = clear.len() - TAG_LEN;
        let tag = AeadTag::<HpkeAead>::from_bytes(&clear[plain_end..])
            .map_err(|_| Error::AuthenticationFailed)?;
        let aad = stream_record_aad(
            parsed.header,
            0,
            u32::try_from(clear.len()).map_err(|_| Error::LimitExceeded)?,
        );
        context
            .open_inout_detached(InOutBuf::from(&mut clear[..plain_end]), &aad, &tag)
            .map_err(|_| Error::AuthenticationFailed)?;
        if clear.first() != Some(&STREAM_START) {
            return Err(Error::MalformedEnvelope);
        }
        let head = message::decode_stream_head(&clear[1..plain_end], self.limits)?;
        let content_length = message::stream_content_length(&head.headers)?;
        if content_length.is_some_and(|length| length > max_upload_bytes) {
            return Err(Error::LimitExceeded);
        }
        let retain_until_exclusive =
            replay_retention_deadline(parsed.issued_at_unix_s, now_unix_s)?;
        let replay_id = replay_id(parsed.header, parsed.enc);
        let response_material =
            export_response_material(&context, parsed.enc, head.method, self.limits)?;
        Ok(AuthenticatedStreamRequest {
            replay: ReplayRequest {
                id: replay_id,
                retain_until_exclusive,
            },
            token: StreamReplayToken {
                replay_id,
                retain_until_exclusive,
                head,
                reader: StreamRequestOpener {
                    context: Some(context),
                    header: parsed.header.to_vec(),
                    sequence: 1,
                    data_records: 0,
                    total_bytes: 0,
                    content_length,
                    max_upload_bytes,
                    frame: Vec::new(),
                    ended: false,
                    response_material: Some(response_material),
                },
            },
        })
    }
}

//! Typed one-shot client and staged server transactions.

use std::{
    fmt,
    time::{SystemTime, UNIX_EPOCH},
};

use hkdf::Hkdf;
use hpke::{
    Deserializable, Kem as KemTrait, OpModeR, OpModeS, PskBundle, Serializable,
    aead::{AeadCtxR, AeadTag, ChaCha20Poly1305 as HpkeChaCha20Poly1305},
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
    CLOCK_SKEW_SECS, CompressionCoding, EntropySource, Error, HARD_MAX_ID_LEN, Limits, Method,
    REQUEST_LIFETIME_SECS, SystemEntropy,
    codec::{
        ENC_LEN, RESPONSE_NONCE_LEN, TAG_LEN, encode_request, encode_request_header, parse_request,
    },
    compression,
    message::{HeaderField, Request, Response},
    response::{ResponseMode, ResponseOpener, ResponseRecord, ResponseSealer},
};

type Kem = X25519HkdfSha256;
type Kdf = HkdfSha256;
type HpkeAead = HpkeChaCha20Poly1305;
type RecipientContext = AeadCtxR<HpkeAead, Kdf, Kem>;

const REQUEST_INFO_LABEL: &[u8] = b"message/hpke-http request\0v2";
const RESPONSE_EXPORT_LABEL: &[u8] = b"message/hpke-http response\0v2";
const REPLAY_LABEL: &[u8] = b"hpke-http/replay\0v2\0";
const RESPONSE_KEY_INFO: &[u8] = b"hpke-http/2 response key";
const RESPONSE_NONCE_INFO: &[u8] = b"hpke-http/2 response nonce";
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
    compression: Option<CompressionCoding>,
}

impl fmt::Debug for Client {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Client")
            .field("recipient_key_id_len", &self.recipient_key_id.len())
            .field("psk", &"[REDACTED]")
            .field("psk_id_len", &self.psk_id.len())
            .field("limits", &self.limits)
            .field("compression", &self.compression)
            .finish_non_exhaustive()
    }
}

impl Client {
    /// Validate and create a client. A secret-derived public ID is rejected.
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
            compression: None,
        })
    }

    /// Opt in to authenticated request-body compression and advertise the
    /// same coding for responses. Even an identity-sized request carries the
    /// extension marker; servers without compression support reject it.
    /// Do not mix attacker-controlled text with secrets in a body being coded:
    /// ciphertext length can reveal information about compression ratio.
    #[must_use]
    pub fn with_compression(mut self, coding: CompressionCoding) -> Self {
        self.compression = Some(coding);
        self
    }

    /// Protect one complete request with the platform entropy source.
    ///
    /// # Errors
    ///
    /// Returns a validation, clock, entropy, compression, or cryptographic
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
    /// Returns a validation, entropy, compression, or cryptographic error.
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
    /// Returns a validation, clock, entropy, compression, or cryptographic
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
    /// Returns a validation, entropy, compression, or cryptographic error
    /// before exposing a partial envelope.
    pub fn protect_at_with_entropy(
        &self,
        request: &Request,
        now_unix_s: u64,
        entropy: &mut impl EntropySource,
    ) -> Result<ProtectedRequest, Error> {
        let mut ciphertext = compression::encode_request(request, self.limits, self.compression)?;
        let header = encode_request_header(&self.recipient_key_id, &self.psk_id, now_unix_s)?;
        let info = request_info(&header);
        let psk_bundle = PskBundle::new(self.psk.as_slice(), &self.psk_id)
            .map_err(|_| Error::InvalidConfiguration)?;
        let mut random_seed = Zeroizing::new([0_u8; 32]);
        entropy.fill(random_seed.as_mut())?;
        let mut random = ChaCha20Rng::from_seed(*random_seed);
        let (encapped_key, mut context) = setup_sender_with_rng::<HpkeAead, Kdf, Kem>(
            &OpModeS::Psk(psk_bundle),
            &self.recipient_public_key,
            &info,
            &mut random,
        )
        .map_err(|_| Error::CryptoFailure)?;
        let enc = encapped_key.to_bytes();
        let tag = context
            .seal_inout_detached(InOutBuf::from(ciphertext.as_mut_slice()), &[])
            .map_err(|_| Error::CryptoFailure)?;
        ciphertext.extend_from_slice(tag.to_bytes().as_ref());
        let mut response_material =
            export_response_material(&context, enc.as_ref(), request.method, self.limits)?;
        response_material.compression = self.compression;
        let envelope = encode_request(&header, enc.as_ref(), &ciphertext)?;
        Ok(ProtectedRequest {
            envelope,
            response_token: ResponseToken(response_material),
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
    request_coding: Option<CompressionCoding>,
    response_material: ResponseMaterial,
}

impl fmt::Debug for ReplayToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReplayToken")
            .field("replay_id", &self.replay_id)
            .field("retain_until_exclusive", &self.retain_until_exclusive)
            .field("request", &"[REDACTED]")
            .field("request_coding", &self.request_coding)
            .field("response_material", &self.response_material)
            .finish()
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
        if self.replay_id != decision.id
            || self.retain_until_exclusive != decision.retain_until_exclusive
        {
            return Err(Error::ReplayDecisionMismatch);
        }
        if !decision.admitted {
            return Err(Error::ReplayRejected);
        }
        if now_unix_s >= self.retain_until_exclusive {
            return Err(Error::InvalidRequestTime);
        }
        let request = compression::finish_request(
            self.request,
            self.request_coding,
            self.response_material.limits,
        )?;
        Ok(OpenedRequest {
            request,
            response: ResponseCapability(self.response_material),
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
    compression_enabled: bool,
}

impl fmt::Debug for Server {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Server")
            .field("recipient_private_key", &"[REDACTED]")
            .field("recipient_key_id_len", &self.recipient_key_id.len())
            .field("limits", &self.limits)
            .field("compression_enabled", &self.compression_enabled)
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
            compression_enabled: false,
        })
    }

    /// Accept the authenticated gzip/zstd body-coding extension and compress
    /// a response only when its client advertised a supported coding.
    #[must_use]
    pub fn with_compression(mut self) -> Self {
        self.compression_enabled = true;
        self
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
        let parsed = parse_request(envelope, self.limits)?;
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
        let StartToken { mut envelope } = token;
        let parsed = parse_request(&envelope, self.limits)?;
        if parsed.key_id != self.recipient_key_id {
            return Err(Error::UnknownRecipientKey);
        }
        validate_credential(psk, parsed.psk_id).map_err(|_| Error::InvalidCredential)?;
        let encapped_key = <Kem as KemTrait>::EncappedKey::from_bytes(parsed.enc)
            .map_err(|_| Error::AuthenticationFailed)?;
        let psk_bundle =
            PskBundle::new(psk, parsed.psk_id).map_err(|_| Error::InvalidCredential)?;
        let info = request_info(parsed.header);
        let mut context = setup_receiver::<HpkeAead, Kdf, Kem>(
            &OpModeR::Psk(psk_bundle),
            &self.recipient_private_key,
            &encapped_key,
            &info,
        )
        .map_err(|_| Error::AuthenticationFailed)?;
        let ciphertext_start = envelope.len() - parsed.ciphertext.len();
        let plaintext_end = envelope.len() - TAG_LEN;
        let tag = AeadTag::<HpkeAead>::from_bytes(&envelope[plaintext_end..])
            .map_err(|_| Error::AuthenticationFailed)?;
        let replay_id = replay_id(parsed.header, parsed.enc);
        let issued_at_unix_s = parsed.issued_at_unix_s;
        let enc = parsed.enc.to_vec();
        context
            .open_inout_detached(
                InOutBuf::from(&mut envelope[ciphertext_start..plaintext_end]),
                &[],
                &tag,
            )
            .map_err(|_| Error::AuthenticationFailed)?;
        let (request, request_coding, response_coding) = compression::decode_request(
            &envelope[ciphertext_start..plaintext_end],
            self.limits,
            self.compression_enabled,
        )?;
        let retain_until_exclusive = replay_retention_deadline(issued_at_unix_s, now_unix_s)?;
        let mut response_material =
            export_response_material(&context, &enc, request.method, self.limits)?;
        response_material.compression = response_coding;
        Ok(AuthenticatedRequest {
            replay: ReplayRequest {
                id: replay_id,
                retain_until_exclusive,
            },
            token: ReplayToken {
                replay_id,
                retain_until_exclusive,
                request,
                request_coding,
                response_material,
            },
        })
    }
}

pub(crate) struct ResponseMaterial {
    pub(crate) secret: Zeroizing<[u8; RESPONSE_SECRET_LEN]>,
    pub(crate) enc: [u8; ENC_LEN],
    pub(crate) method: Method,
    pub(crate) limits: Limits,
    pub(crate) compression: Option<CompressionCoding>,
}

impl fmt::Debug for ResponseMaterial {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResponseMaterial")
            .field("secret", &"[REDACTED]")
            .field("enc_len", &self.enc.len())
            .field("method", &self.method)
            .field("limits", &self.limits)
            .field("compression", &self.compression)
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

    /// Open one complete finite response with the v2 record reader.
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
        coding: Option<CompressionCoding>,
    ) -> Result<(ResponseSealer, Vec<u8>), Error> {
        self.into_sealer_with_entropy(status, headers, coding, &mut SystemEntropy)
    }

    /// Start one checked response with an injected entropy source.
    ///
    /// # Errors
    /// Returns a validation, entropy, or cryptographic error.
    pub fn into_sealer_with_entropy(
        self,
        status: u16,
        headers: Vec<HeaderField>,
        coding: Option<CompressionCoding>,
        entropy: &mut impl EntropySource,
    ) -> Result<(ResponseSealer, Vec<u8>), Error> {
        ResponseSealer::new(&self.0, status, headers, coding, entropy)
    }

    /// Protect one complete finite response with the v2 record writer.
    ///
    /// # Errors
    /// Returns a validation, limit, entropy, compression, or cryptographic error.
    pub fn protect_finite(self, response: &Response) -> Result<Vec<u8>, Error> {
        self.protect_finite_with_entropy(response, &mut SystemEntropy)
    }

    /// Protect one complete finite response with injected entropy.
    ///
    /// # Errors
    /// Returns a validation, limit, entropy, compression, or cryptographic error.
    pub fn protect_finite_with_entropy(
        self,
        response: &Response,
        entropy: &mut impl EntropySource,
    ) -> Result<Vec<u8>, Error> {
        let (coding, compressed) = match self.0.compression {
            Some(coding) if !response.body.is_empty() => {
                let compressed = compression::maybe_compress(&response.body, coding)?;
                (compressed.as_ref().map(|_| coding), compressed)
            }
            _ => (None, None),
        };
        let (mut sealer, mut output) = self.into_sealer_with_entropy(
            response.status,
            response.headers.clone(),
            coding,
            entropy,
        )?;
        if sealer.head_mode() != ResponseMode::Finite {
            return Err(Error::InvalidConfiguration);
        }
        let data =
            sealer.seal_finite_body_with_precompressed(&response.body, compressed.as_deref())?;
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

fn request_info(header: &[u8]) -> Vec<u8> {
    let mut info = Vec::with_capacity(REQUEST_INFO_LABEL.len() + 1 + header.len());
    info.extend_from_slice(REQUEST_INFO_LABEL);
    info.push(0);
    info.extend_from_slice(header);
    info
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
        compression: None,
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

//! Private wasm-bindgen boundary. The TypeScript facade owns public API names.

#![forbid(unsafe_code)]

use hpke_http::{
    Client as CoreClient, Error, HeaderField, Limits, Method, ReplayRequest, ReplayToken, Request,
    RequestHead, Response, ResponseCapability, ResponseMode, ResponseOpener, ResponseRecord,
    ResponseSealer, ResponseToken, Server as CoreServer, StartToken, StreamReplayToken,
    StreamRequestOpener, StreamRequestRecord, StreamRequestSealer, StreamStartToken, SystemEntropy,
    generate_key_pair,
};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use zeroize::Zeroizing;

type NativeHeaders = Vec<(String, String)>;

/// Return the native engine package version before accepting secrets.
#[wasm_bindgen]
#[must_use]
pub fn engine_version() -> String {
    hpke_http::build_info().version.to_owned()
}

/// Return the language-neutral protocol identifier.
#[wasm_bindgen]
#[must_use]
pub fn protocol_id() -> String {
    hpke_http::build_info().protocol.to_owned()
}

/// Return the binding ABI used for skew checks.
#[wasm_bindgen]
#[must_use]
pub fn binding_abi() -> u32 {
    hpke_http::build_info().binding_abi
}

/// Native limits object shared by client and server constructors.
#[wasm_bindgen(js_name = Limits)]
pub struct WasmLimits(Limits);

#[wasm_bindgen(js_class = Limits)]
impl WasmLimits {
    /// Build validated limits. Missing values use the native defaults.
    ///
    /// # Errors
    ///
    /// Returns a stable JavaScript error value when a limit is invalid.
    #[wasm_bindgen(constructor)]
    pub fn new(
        max_body_len: Option<u32>,
        max_header_bytes: Option<u32>,
        max_header_count: Option<u32>,
        max_target_len: Option<u32>,
        max_request_bytes: Option<f64>,
    ) -> Result<WasmLimits, JsValue> {
        let defaults = Limits::default();
        let limits = Limits {
            max_body_len: usize_from_option(max_body_len, defaults.max_body_len)?,
            max_header_bytes: usize_from_option(max_header_bytes, defaults.max_header_bytes)?,
            max_header_count: usize_from_option(max_header_count, defaults.max_header_count)?,
            max_target_len: usize_from_option(max_target_len, defaults.max_target_len)?,
            max_request_bytes: request_limit_from_js(
                max_request_bytes,
                defaults.max_request_bytes,
            )?,
        }
        .validate()
        .map_err(js_error)?;
        Ok(Self(limits))
    }
}

/// Encoded X25519 recipient key pair.
#[wasm_bindgen(js_name = KeyPair)]
pub struct WasmKeyPair {
    private_key: Zeroizing<Vec<u8>>,
    public_key: Vec<u8>,
}

#[wasm_bindgen(js_class = KeyPair)]
impl WasmKeyPair {
    /// Return an owned copy of the private key.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn private_key(&self) -> Vec<u8> {
        self.private_key.as_slice().to_vec()
    }

    /// Return an owned copy of the public key.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn public_key(&self) -> Vec<u8> {
        self.public_key.clone()
    }
}

/// Generate an X25519 recipient key pair.
///
/// # Errors
///
/// Returns a stable JavaScript error value if secure entropy is unavailable.
#[wasm_bindgen(js_name = generateKeyPair)]
pub fn generate_wasm_key_pair() -> Result<WasmKeyPair, JsValue> {
    let pair = generate_key_pair().map_err(js_error)?;
    let (private_key, public_key) = pair.into_parts();
    Ok(WasmKeyPair {
        private_key: Zeroizing::new(private_key.to_vec()),
        public_key,
    })
}

/// Native protocol client configuration.
#[wasm_bindgen(js_name = Client)]
pub struct WasmClient {
    inner: CoreClient,
}

#[wasm_bindgen(js_class = Client)]
impl WasmClient {
    /// Create a client for one recipient and explicit PSK identity.
    ///
    /// # Errors
    ///
    /// Returns a stable JavaScript error value for invalid configuration.
    #[wasm_bindgen(constructor)]
    pub fn new(
        recipient_public_key: &[u8],
        recipient_key_id: &[u8],
        psk: &[u8],
        psk_id: &[u8],
        limits: &WasmLimits,
    ) -> Result<WasmClient, JsValue> {
        let inner = CoreClient::new(
            recipient_public_key,
            recipient_key_id.to_vec(),
            psk.to_vec(),
            psk_id.to_vec(),
            limits.0,
        )
        .map_err(js_error)?;
        Ok(Self { inner })
    }

    /// Start a protected request without reading body bytes.
    ///
    /// # Errors
    /// Returns a validation, entropy, or cryptographic error.
    pub fn begin_stream(
        &self,
        method: &str,
        authority: &str,
        path: &str,
        headers_json: &str,
        now_unix_s: f64,
    ) -> Result<WasmStreamRequestSealer, JsValue> {
        let head = RequestHead {
            method: parse_method(method)?,
            authority: authority.as_bytes().to_vec(),
            path: path.as_bytes().to_vec(),
            headers: parse_headers(headers_json)?,
        };
        let (inner, first) = self
            .inner
            .begin_stream_at_with_entropy(
                &head,
                unix_seconds_from_js(now_unix_s)?,
                &mut SystemEntropy,
            )
            .map_err(js_error)?;
        Ok(WasmStreamRequestSealer {
            inner: Some(inner),
            first,
        })
    }

    /// Protect one complete bounded request.
    ///
    /// # Errors
    ///
    /// Returns a stable JavaScript error value for validation, entropy, or
    /// cryptographic failure.
    pub fn protect(
        &self,
        method: &str,
        authority: &str,
        path: &str,
        headers_json: &str,
        body: &[u8],
        now_unix_s: f64,
    ) -> Result<WasmProtectedRequest, JsValue> {
        let request = Request {
            method: parse_method(method)?,
            authority: authority.as_bytes().to_vec(),
            path: path.as_bytes().to_vec(),
            headers: parse_headers(headers_json)?,
            body: body.to_vec(),
        };
        let protected = self
            .inner
            .protect_at(&request, unix_seconds_from_js(now_unix_s)?)
            .map_err(js_error)?;
        let (envelope, response_token) = protected.into_parts();
        Ok(WasmProtectedRequest {
            envelope,
            response_token: Some(response_token),
        })
    }
}

/// Native request writer. START, DATA, and END stay in the Rust engine.
#[wasm_bindgen(js_name = StreamRequestSealer)]
pub struct WasmStreamRequestSealer {
    inner: Option<StreamRequestSealer>,
    first: Vec<u8>,
}

#[wasm_bindgen(js_class = StreamRequestSealer)]
impl WasmStreamRequestSealer {
    /// Move the protected START into one owned JavaScript byte array.
    #[must_use]
    pub fn take_start(&mut self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(std::mem::take(&mut self.first).as_slice())
    }

    /// Accept some clear bytes and return at most one protected DATA record.
    ///
    /// # Errors
    /// Returns a state, length, compression, or cryptographic error.
    pub fn push(&mut self, input: &[u8]) -> Result<WasmStreamPush, JsValue> {
        let (consumed, record) = self
            .inner
            .as_mut()
            .ok_or_else(consumed_error)?
            .push(input)
            .map_err(js_error)?;
        Ok(WasmStreamPush { consumed, record })
    }

    /// Flush the last DATA part, seal END, and transfer the response right.
    ///
    /// # Errors
    /// Returns a state, length, compression, or cryptographic error.
    pub fn finish(&mut self) -> Result<WasmProtectedRequest, JsValue> {
        let inner = self.inner.take().ok_or_else(consumed_error)?;
        let (envelope, response_token) = inner.finish().map_err(js_error)?;
        Ok(WasmProtectedRequest {
            envelope,
            response_token: Some(response_token),
        })
    }

    /// Discard the writer without END.
    pub fn close(&mut self) {
        self.inner = None;
        self.first.clear();
    }
}

/// One incremental request write result.
#[wasm_bindgen(js_name = StreamPush)]
pub struct WasmStreamPush {
    consumed: usize,
    record: Option<Vec<u8>>,
}

#[wasm_bindgen(js_class = StreamPush)]
impl WasmStreamPush {
    /// Number of clear source bytes accepted by this call.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn consumed(&self) -> usize {
        self.consumed
    }

    /// Move the protected DATA record, if any.
    #[must_use]
    pub fn take_record(&mut self) -> Option<js_sys::Uint8Array> {
        self.record
            .take()
            .map(|record| js_sys::Uint8Array::from(record.as_slice()))
    }
}

/// Protected request plus its one-shot response token.
#[wasm_bindgen(js_name = ProtectedRequest)]
pub struct WasmProtectedRequest {
    envelope: Vec<u8>,
    response_token: Option<ResponseToken>,
}

#[wasm_bindgen(js_class = ProtectedRequest)]
impl WasmProtectedRequest {
    /// Move the request envelope into one owned JavaScript byte array.
    #[must_use]
    pub fn take_envelope(&mut self) -> js_sys::Uint8Array {
        let envelope = std::mem::take(&mut self.envelope);
        js_sys::Uint8Array::from(envelope.as_slice())
    }

    /// Return whether the response token is no longer usable.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn consumed(&self) -> bool {
        self.response_token.is_none()
    }

    /// Consume the response token and authenticate one response.
    ///
    /// # Errors
    ///
    /// Returns a stable JavaScript error value for consumed state or invalid
    /// response bytes.
    pub fn open_finite_response(&mut self, envelope: &[u8]) -> Result<WasmResponse, JsValue> {
        let token = self.response_token.take().ok_or_else(consumed_error)?;
        let response = token.open_finite(envelope).map_err(js_error)?;
        WasmResponse::from_core(response)
    }

    /// Transfer the response token to a checked record reader.
    ///
    /// # Errors
    /// Returns a state error if this token was used.
    pub fn into_opener(&mut self) -> Result<WasmResponseOpener, JsValue> {
        let token = self.response_token.take().ok_or_else(consumed_error)?;
        Ok(WasmResponseOpener {
            inner: Some(token.into_opener()),
        })
    }

    /// Discard the response token immediately.
    pub fn discard(&mut self) {
        self.response_token = None;
    }
}

/// One bounded checked response reader.
#[wasm_bindgen(js_name = ResponseOpener)]
pub struct WasmResponseOpener {
    inner: Option<ResponseOpener>,
}

#[wasm_bindgen(js_class = ResponseOpener)]
impl WasmResponseOpener {
    /// Consume input through at most one checked record.
    ///
    /// # Errors
    /// Returns a parse, limit, order, or authentication error.
    pub fn feed(&mut self, input: &[u8]) -> Result<WasmFeed, JsValue> {
        let inner = self.inner.as_mut().ok_or_else(consumed_error)?;
        let (consumed, record) = inner.feed(input).map_err(js_error)?;
        WasmFeed::new(consumed, record)
    }

    /// Check true outer-body EOF after END.
    ///
    /// # Errors
    /// Returns an error when END is missing or the reader failed.
    pub fn finish_eof(&mut self) -> Result<Option<WasmResponse>, JsValue> {
        let inner = self.inner.as_mut().ok_or_else(consumed_error)?;
        let response = inner.finish_eof().map_err(js_error)?;
        self.inner = None;
        response.map(WasmResponse::from_core).transpose()
    }

    /// Discard the state without a complete-response claim.
    pub fn close(&mut self) {
        self.inner = None;
    }
}

/// One result from the bounded response record reader.
#[wasm_bindgen(js_name = FeedResult)]
pub struct WasmFeed {
    consumed: usize,
    kind: u8,
    status: u16,
    headers_json: String,
    mode: u8,
    block: Vec<u8>,
}

impl WasmFeed {
    fn new(consumed: usize, record: Option<ResponseRecord>) -> Result<Self, JsValue> {
        let mut result = Self {
            consumed,
            kind: 0,
            status: 0,
            headers_json: String::new(),
            mode: 0,
            block: Vec::new(),
        };
        match record {
            Some(ResponseRecord::Start(head)) => {
                result.kind = 1;
                result.status = head.status;
                result.headers_json = serialize_headers(&head.headers)?;
                result.mode = match head.mode {
                    ResponseMode::Finite => 1,
                    ResponseMode::Sse => 2,
                };
            }
            Some(ResponseRecord::SseData(block)) => {
                result.kind = 2;
                result.block = block;
            }
            Some(ResponseRecord::End) => result.kind = 3,
            None => {}
        }
        Ok(result)
    }
}

#[wasm_bindgen(js_class = FeedResult)]
impl WasmFeed {
    /// Number of input bytes consumed.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn consumed(&self) -> usize {
        self.consumed
    }
    /// 0 means partial; 1 START, 2 SSE DATA, 3 END.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn kind(&self) -> u8 {
        self.kind
    }
    /// Checked START status.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn status(&self) -> u16 {
        self.status
    }
    /// Checked START field pairs in private JSON form.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn headers_json(&self) -> String {
        self.headers_json.clone()
    }
    /// Checked START mode: 1 finite, 2 SSE.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn mode(&self) -> u8 {
        self.mode
    }
    /// Clear SSE block bytes, only for kind 2.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn block(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(self.block.as_slice())
    }
}

/// Authenticated plaintext response returned by the native engine.
///
/// Its private JavaScript name must not shadow the platform `Response` used by
/// wasm-bindgen's browser loader.
#[wasm_bindgen(js_name = NativeResponse)]
pub struct WasmResponse {
    status: u16,
    headers_json: String,
    body: Vec<u8>,
}

#[wasm_bindgen(js_class = NativeResponse)]
impl WasmResponse {
    fn from_core(response: Response) -> Result<Self, JsValue> {
        Ok(Self {
            status: response.status,
            headers_json: serialize_headers(&response.headers)?,
            body: response.body,
        })
    }

    /// Return the final HTTP status code.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn status(&self) -> u16 {
        self.status
    }

    /// Return headers as a private JSON boundary value.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn headers_json(&self) -> String {
        self.headers_json.clone()
    }

    /// Return an owned copy of the response body.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn body(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(self.body.as_slice())
    }
}

/// Native protocol server configuration.
#[wasm_bindgen(js_name = Server)]
pub struct WasmServer {
    inner: CoreServer,
}

#[wasm_bindgen(js_class = Server)]
impl WasmServer {
    /// Create a server with one advertised key and optional older keys.
    ///
    /// # Errors
    ///
    /// Returns a stable JavaScript error value for invalid configuration.
    #[wasm_bindgen(constructor)]
    pub fn new(
        recipient_private_key: &[u8],
        recipient_key_id: &[u8],
        limits: &WasmLimits,
        accepted_keys: &js_sys::Array,
    ) -> Result<WasmServer, JsValue> {
        let mut accepted = Vec::with_capacity(accepted_keys.length() as usize);
        for value in accepted_keys.iter() {
            let pair: js_sys::Array = value
                .dyn_into()
                .map_err(|_| js_error(Error::InvalidConfiguration))?;
            if pair.length() != 2 {
                return Err(js_error(Error::InvalidConfiguration));
            }
            let private_key: js_sys::Uint8Array = pair
                .get(0)
                .dyn_into()
                .map_err(|_| js_error(Error::InvalidConfiguration))?;
            let key_id: js_sys::Uint8Array = pair
                .get(1)
                .dyn_into()
                .map_err(|_| js_error(Error::InvalidConfiguration))?;
            accepted.push((private_key.to_vec(), key_id.to_vec()));
        }
        let inner = CoreServer::with_accepted_keys(
            recipient_private_key,
            recipient_key_id.to_vec(),
            accepted,
            limits.0,
        )
        .map_err(js_error)?;
        Ok(Self { inner })
    }

    /// Parse bounded public fields before credential resolution.
    ///
    /// # Errors
    ///
    /// Returns a stable JavaScript error value for an invalid request envelope.
    pub fn preparse(&self, envelope: Vec<u8>) -> Result<WasmPreparsedRequest, JsValue> {
        let preparsed = self.inner.preparse_owned(envelope).map_err(js_error)?;
        Ok(WasmPreparsedRequest {
            psk_id: preparsed.credential.psk_id,
            token: Some(preparsed.token),
        })
    }

    /// Return the largest complete envelope accepted by `preparse`.
    ///
    /// # Errors
    ///
    /// Returns a size error if the limit cannot fit in memory.
    pub fn max_complete_envelope_len(&self) -> Result<usize, JsValue> {
        self.inner.max_complete_envelope_len().map_err(js_error)
    }

    /// Return the total public prefix and START length when enough bytes arrived.
    ///
    /// # Errors
    /// Returns a parse, version, suite, or size error.
    pub fn stream_start_length(&self, input: &[u8]) -> Result<Option<usize>, JsValue> {
        self.inner.stream_start_length(input).map_err(js_error)
    }

    /// Parse one exact public prefix and START before credential lookup.
    ///
    /// # Errors
    /// Returns a parse, limit, or recipient-key error.
    pub fn preparse_stream(&self, first: &[u8]) -> Result<WasmPreparsedStreamRequest, JsValue> {
        let preparsed = self.inner.preparse_stream(first).map_err(js_error)?;
        Ok(WasmPreparsedStreamRequest {
            psk_id: preparsed.credential.psk_id,
            token: Some(preparsed.token),
        })
    }
}

/// A checked public START waiting for one PSK lookup.
#[wasm_bindgen(js_name = PreparsedStreamRequest)]
pub struct WasmPreparsedStreamRequest {
    psk_id: Vec<u8>,
    token: Option<StreamStartToken>,
}

#[wasm_bindgen(js_class = PreparsedStreamRequest)]
impl WasmPreparsedStreamRequest {
    /// Public PSK identifier.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn psk_id(&self) -> Vec<u8> {
        self.psk_id.clone()
    }

    /// Whether this one-use stage was consumed.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn consumed(&self) -> bool {
        self.token.is_none()
    }

    /// Authenticate START with the host-resolved PSK.
    ///
    /// # Errors
    /// Returns a credential, time, authentication, or parse error.
    pub fn authenticate(
        &mut self,
        server: &WasmServer,
        psk: &[u8],
        now_unix_s: f64,
    ) -> Result<WasmAuthenticatedStreamRequest, JsValue> {
        let token = self.token.take().ok_or_else(consumed_error)?;
        let authenticated = server
            .inner
            .authenticate_stream_at(token, psk, unix_seconds_from_js(now_unix_s)?)
            .map_err(js_error)?;
        Ok(WasmAuthenticatedStreamRequest {
            replay: authenticated.replay,
            token: Some(authenticated.token),
        })
    }

    /// Discard this stage.
    pub fn discard(&mut self) {
        self.token = None;
    }
}

/// Authenticated START waiting for one atomic replay decision.
#[wasm_bindgen(js_name = AuthenticatedStreamRequest)]
pub struct WasmAuthenticatedStreamRequest {
    replay: ReplayRequest,
    token: Option<StreamReplayToken>,
}

#[wasm_bindgen(js_class = AuthenticatedStreamRequest)]
impl WasmAuthenticatedStreamRequest {
    /// Stable replay identifier.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn replay_id(&self) -> Vec<u8> {
        self.replay.id.to_vec()
    }

    /// Exclusive replay reservation deadline.
    #[wasm_bindgen(getter)]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn retain_until_exclusive(&self) -> f64 {
        self.replay.retain_until_exclusive as f64
    }

    /// Whether this one-use stage was consumed.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn consumed(&self) -> bool {
        self.token.is_none()
    }

    /// Apply the host's atomic replay decision.
    ///
    /// # Errors
    /// Returns a replay or time error.
    pub fn admit(
        &mut self,
        accepted: bool,
        now_unix_s: f64,
    ) -> Result<WasmOpenedStreamRequest, JsValue> {
        let token = self.token.take().ok_or_else(consumed_error)?;
        let opened = token
            .admit_at(
                self.replay.decision(accepted),
                unix_seconds_from_js(now_unix_s)?,
            )
            .map_err(js_error)?;
        WasmOpenedStreamRequest::from_core(opened.head, opened.reader)
    }

    /// Discard this stage.
    pub fn discard(&mut self) {
        self.token = None;
    }
}

/// Checked request head and incremental DATA reader.
#[wasm_bindgen(js_name = OpenedStreamRequest)]
pub struct WasmOpenedStreamRequest {
    head: Option<RequestHead>,
    method: &'static str,
    authority: String,
    path: String,
    headers_json: String,
    reader: Option<StreamRequestOpener>,
}

impl WasmOpenedStreamRequest {
    fn from_core(head: RequestHead, reader: StreamRequestOpener) -> Result<Self, JsValue> {
        let method = head.method.as_str();
        let authority = String::from_utf8(head.authority.clone())
            .map_err(|_| malformed_text_error("authority"))?;
        let path =
            String::from_utf8(head.path.clone()).map_err(|_| malformed_text_error("path"))?;
        let headers_json = serialize_headers(&head.headers)?;
        Ok(Self {
            head: Some(head),
            method,
            authority,
            path,
            headers_json,
            reader: Some(reader),
        })
    }
}

#[wasm_bindgen(js_class = OpenedStreamRequest)]
impl WasmOpenedStreamRequest {
    /// Authenticated method.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn method(&self) -> String {
        self.method.to_owned()
    }

    /// Authenticated authority.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn authority(&self) -> String {
        self.authority.clone()
    }

    /// Authenticated path.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn path(&self) -> String {
        self.path.clone()
    }

    /// Authenticated headers in private JSON boundary form.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn headers_json(&self) -> String {
        self.headers_json.clone()
    }

    /// Feed any byte cut and release at most one checked DATA part or END.
    ///
    /// # Errors
    /// Returns a parse, order, limit, or authentication error.
    pub fn feed(&mut self, input: &[u8]) -> Result<WasmRequestFeed, JsValue> {
        let reader = self.reader.as_mut().ok_or_else(consumed_error)?;
        let (consumed, record) = reader.feed(input).map_err(js_error)?;
        Ok(WasmRequestFeed::new(consumed, record))
    }

    /// Check true outer EOF after END, then grant the response right.
    ///
    /// # Errors
    /// Returns an error when END is absent or a record is partial.
    pub fn finish_eof(&mut self) -> Result<WasmOpenedRequest, JsValue> {
        let reader = self.reader.take().ok_or_else(consumed_error)?;
        let capability = reader.finish_eof().map_err(js_error)?;
        let head = self.head.take().ok_or_else(consumed_error)?;
        WasmOpenedRequest::from_core(
            Request {
                method: head.method,
                authority: head.authority,
                path: head.path,
                headers: head.headers,
                body: Vec::new(),
            },
            capability,
        )
    }

    /// Discard the reader before EOF.
    pub fn close(&mut self) {
        self.reader = None;
        self.head = None;
    }
}

/// One result from the checked DATA reader.
#[wasm_bindgen(js_name = RequestFeed)]
pub struct WasmRequestFeed {
    consumed: usize,
    kind: u8,
    block: Vec<u8>,
}

impl WasmRequestFeed {
    fn new(consumed: usize, record: Option<StreamRequestRecord>) -> Self {
        match record {
            None => Self {
                consumed,
                kind: 0,
                block: Vec::new(),
            },
            Some(StreamRequestRecord::Data(block)) => Self {
                consumed,
                kind: 1,
                block,
            },
            Some(StreamRequestRecord::End) => Self {
                consumed,
                kind: 2,
                block: Vec::new(),
            },
        }
    }
}

#[wasm_bindgen(js_class = RequestFeed)]
impl WasmRequestFeed {
    /// Number of protected source bytes accepted.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn consumed(&self) -> usize {
        self.consumed
    }

    /// 0 means partial, 1 means DATA, and 2 means END.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn kind(&self) -> u8 {
        self.kind
    }

    /// Checked clear DATA part, present for kind 1.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn block(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(self.block.as_slice())
    }
}

/// One-shot server continuation waiting for credential resolution.
#[wasm_bindgen(js_name = PreparsedRequest)]
pub struct WasmPreparsedRequest {
    psk_id: Vec<u8>,
    token: Option<StartToken>,
}

#[wasm_bindgen(js_class = PreparsedRequest)]
impl WasmPreparsedRequest {
    /// Return the public PSK identity.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn psk_id(&self) -> Vec<u8> {
        self.psk_id.clone()
    }

    /// Return whether this continuation is no longer usable.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn consumed(&self) -> bool {
        self.token.is_none()
    }

    /// Consume this stage and authenticate the complete request.
    ///
    /// # Errors
    ///
    /// Returns a stable JavaScript error value for consumed state, credentials,
    /// or authentication failure.
    pub fn authenticate(
        &mut self,
        server: &WasmServer,
        psk: &[u8],
        now_unix_s: f64,
    ) -> Result<WasmAuthenticatedRequest, JsValue> {
        let token = self.token.take().ok_or_else(consumed_error)?;
        let authenticated = server
            .inner
            .authenticate_at(token, psk, unix_seconds_from_js(now_unix_s)?)
            .map_err(js_error)?;
        Ok(WasmAuthenticatedRequest {
            replay: authenticated.replay,
            token: Some(authenticated.token),
        })
    }

    /// Discard this continuation immediately.
    pub fn discard(&mut self) {
        self.token = None;
    }
}

/// Authenticated request paused for atomic replay admission.
#[wasm_bindgen(js_name = AuthenticatedRequest)]
pub struct WasmAuthenticatedRequest {
    replay: ReplayRequest,
    token: Option<ReplayToken>,
}

#[wasm_bindgen(js_class = AuthenticatedRequest)]
impl WasmAuthenticatedRequest {
    /// Return the stable replay-admission identifier.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn replay_id(&self) -> Vec<u8> {
        self.replay.id.to_vec()
    }

    /// Return the exclusive Unix-second replay-retention deadline.
    #[wasm_bindgen(getter)]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn retain_until_exclusive(&self) -> f64 {
        self.replay.retain_until_exclusive as f64
    }

    /// Return whether this continuation is no longer usable.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn consumed(&self) -> bool {
        self.token.is_none()
    }

    /// Consume this stage and apply one replay-admission decision.
    ///
    /// # Errors
    ///
    /// Returns a stable JavaScript error value for consumed, rejected, or
    /// expired state.
    pub fn admit(&mut self, accepted: bool, now_unix_s: f64) -> Result<WasmOpenedRequest, JsValue> {
        let token = self.token.take().ok_or_else(consumed_error)?;
        let opened = token
            .admit_at(
                self.replay.decision(accepted),
                unix_seconds_from_js(now_unix_s)?,
            )
            .map_err(js_error)?;
        WasmOpenedRequest::from_core(opened.request, opened.response)
    }

    /// Discard this continuation immediately.
    pub fn discard(&mut self) {
        self.token = None;
    }
}

/// Verified plaintext request and its one-shot response capability.
#[wasm_bindgen(js_name = OpenedRequest)]
pub struct WasmOpenedRequest {
    method: &'static str,
    authority: String,
    path: String,
    headers_json: String,
    body: Vec<u8>,
    response_capability: Option<ResponseCapability>,
}

#[wasm_bindgen(js_class = OpenedRequest)]
impl WasmOpenedRequest {
    fn from_core(request: Request, response: ResponseCapability) -> Result<Self, JsValue> {
        let authority =
            String::from_utf8(request.authority).map_err(|_| malformed_text_error("authority"))?;
        let path = String::from_utf8(request.path).map_err(|_| malformed_text_error("path"))?;
        Ok(Self {
            method: request.method.as_str(),
            authority,
            path,
            headers_json: serialize_headers(&request.headers)?,
            body: request.body,
            response_capability: Some(response),
        })
    }

    /// Return the request method.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn method(&self) -> String {
        self.method.to_owned()
    }

    /// Return the request authority.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn authority(&self) -> String {
        self.authority.clone()
    }

    /// Return the request path and optional query.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn path(&self) -> String {
        self.path.clone()
    }

    /// Return headers as a private JSON boundary value.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn headers_json(&self) -> String {
        self.headers_json.clone()
    }

    /// Move the authenticated body into one owned JavaScript byte array.
    #[must_use]
    pub fn take_body(&mut self) -> js_sys::Uint8Array {
        let body = std::mem::take(&mut self.body);
        js_sys::Uint8Array::from(body.as_slice())
    }

    /// Return whether the response capability is no longer usable.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn response_consumed(&self) -> bool {
        self.response_capability.is_none()
    }

    /// Consume the response capability and protect one response.
    ///
    /// # Errors
    ///
    /// Returns a stable JavaScript error value for consumed state, invalid
    /// response data, entropy failure, or cryptographic failure.
    pub fn protect_response(
        &mut self,
        status: u16,
        headers_json: &str,
        body: &[u8],
    ) -> Result<Vec<u8>, JsValue> {
        let capability = self.response_capability.take().ok_or_else(consumed_error)?;
        capability
            .protect_finite_parts(status, parse_headers(headers_json)?, body)
            .map_err(js_error)
    }

    /// Start a response record stream with checked status and fields.
    ///
    /// # Errors
    /// Returns a state, validation, entropy, or cryptographic error.
    pub fn start_response(
        &mut self,
        status: u16,
        headers_json: &str,
    ) -> Result<WasmResponseSealer, JsValue> {
        let capability = self.response_capability.take().ok_or_else(consumed_error)?;
        let (inner, first) = capability
            .into_sealer(status, parse_headers(headers_json)?)
            .map_err(js_error)?;
        Ok(WasmResponseSealer {
            inner: Some(inner),
            first,
        })
    }

    /// Discard the response capability immediately.
    pub fn discard_response(&mut self) {
        self.response_capability = None;
    }
}

/// One checked response writer for a finite body or complete SSE blocks.
#[wasm_bindgen(js_name = ResponseSealer)]
pub struct WasmResponseSealer {
    inner: Option<ResponseSealer>,
    first: Vec<u8>,
}

#[wasm_bindgen(js_class = ResponseSealer)]
impl WasmResponseSealer {
    /// Move the response prefix and START record into JavaScript.
    pub fn take_start(&mut self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(std::mem::take(&mut self.first).as_slice())
    }

    /// Protect a finite body once. An empty body emits no DATA record.
    ///
    /// # Errors
    /// Returns a validation, limit, compression, or cryptographic error.
    pub fn seal_finite_body(&mut self, body: &[u8]) -> Result<Option<Vec<u8>>, JsValue> {
        self.inner
            .as_mut()
            .ok_or_else(consumed_error)?
            .seal_finite_body(body)
            .map_err(js_error)
    }

    /// Protect one complete LF-normalized SSE block.
    ///
    /// # Errors
    /// Returns a shape, limit, or cryptographic error.
    pub fn seal_sse_block(&mut self, block: &[u8]) -> Result<Vec<u8>, JsValue> {
        self.inner
            .as_mut()
            .ok_or_else(consumed_error)?
            .seal_sse_block(block)
            .map_err(js_error)
    }

    /// Protect END. In finite mode, call `seal_finite_body` once first, even
    /// for an empty body. The caller then ends the outer HTTP body.
    ///
    /// # Errors
    /// Returns a state or cryptographic error.
    pub fn finish(&mut self) -> Result<Vec<u8>, JsValue> {
        let frame = self
            .inner
            .as_mut()
            .ok_or_else(consumed_error)?
            .finish()
            .map_err(js_error)?;
        self.inner = None;
        Ok(frame)
    }

    /// Discard without END.
    pub fn close(&mut self) {
        self.inner = None;
        self.first.clear();
    }
}

fn usize_from_option(value: Option<u32>, default: usize) -> Result<usize, JsValue> {
    value.map_or(Ok(default), |value| {
        usize::try_from(value)
            .map_err(|_| js_message("invalid_configuration", "limit conversion failed"))
    })
}

fn request_limit_from_js(value: Option<f64>, default: u64) -> Result<u64, JsValue> {
    match value {
        None => Ok(default),
        Some(value)
            if value.is_finite()
                && value.fract() == 0.0
                && (1.0..=4_294_967_296.0).contains(&value) =>
        {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            Ok(value as u64)
        }
        Some(_) => Err(js_message(
            "invalid_configuration",
            "invalid request byte limit",
        )),
    }
}

fn unix_seconds_from_js(value: f64) -> Result<u64, JsValue> {
    const MAX_SAFE_WITH_DEADLINE: f64 = 9_007_199_254_740_631.0;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > MAX_SAFE_WITH_DEADLINE {
        return Err(js_message(
            "clock_unavailable",
            "Unix time must be a non-negative safe integer",
        ));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let seconds = value as u64;
    Ok(seconds)
}

fn parse_method(value: &str) -> Result<Method, JsValue> {
    Method::from_bytes(value.as_bytes())
        .map_err(|_| js_message("unsupported_method", "unsupported HTTP method"))
}

fn parse_headers(input: &str) -> Result<Vec<HeaderField>, JsValue> {
    let headers: NativeHeaders = serde_json::from_str(input)
        .map_err(|_| js_message("invalid_configuration", "invalid header boundary data"))?;
    Ok(headers
        .into_iter()
        .map(|(name, value)| HeaderField {
            name: name.into_bytes(),
            value: value.into_bytes(),
        })
        .collect())
}

fn serialize_headers(headers: &[HeaderField]) -> Result<String, JsValue> {
    let headers: Result<NativeHeaders, JsValue> = headers
        .iter()
        .map(|field| {
            let name = std::str::from_utf8(&field.name)
                .map_err(|_| malformed_text_error("header name"))?;
            let value = std::str::from_utf8(&field.value)
                .map_err(|_| malformed_text_error("header value"))?;
            Ok((name.to_owned(), value.to_owned()))
        })
        .collect();
    serde_json::to_string(&headers?)
        .map_err(|_| js_message("crypto_failure", "could not serialize native result"))
}

fn js_error(error: Error) -> JsValue {
    js_message(error.code(), &error.to_string())
}

fn consumed_error() -> JsValue {
    js_message("state_consumed", "continuation already consumed")
}

fn malformed_text_error(field: &str) -> JsValue {
    js_message(
        "malformed_envelope",
        &format!("invalid UTF-8 {field} in authenticated message"),
    )
}

fn js_message(code: &str, message: &str) -> JsValue {
    JsValue::from_str(&format!("{code}\n{message}"))
}

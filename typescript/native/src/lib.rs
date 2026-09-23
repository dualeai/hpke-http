//! Private wasm-bindgen boundary. The TypeScript facade owns public API names.

#![forbid(unsafe_code)]

use hpke_http::{
    Client as CoreClient, CompressionCoding, Error, HeaderField, Limits, Method, ReplayRequest,
    ReplayToken, Request, Response, ResponseCapability, ResponseMode, ResponseOpener,
    ResponseRecord, ResponseSealer, ResponseToken, Server as CoreServer, StartToken,
    generate_key_pair,
};
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
    ) -> Result<WasmLimits, JsValue> {
        let defaults = Limits::default();
        let limits = Limits {
            max_body_len: usize_from_option(max_body_len, defaults.max_body_len)?,
            max_header_bytes: usize_from_option(max_header_bytes, defaults.max_header_bytes)?,
            max_header_count: usize_from_option(max_header_count, defaults.max_header_count)?,
            max_target_len: usize_from_option(max_target_len, defaults.max_target_len)?,
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
        compression: u8,
    ) -> Result<WasmClient, JsValue> {
        let mut inner = CoreClient::new(
            recipient_public_key,
            recipient_key_id.to_vec(),
            psk.to_vec(),
            psk_id.to_vec(),
            limits.0,
        )
        .map_err(js_error)?;
        inner = match compression {
            0 => inner,
            1 => inner.with_compression(CompressionCoding::Gzip),
            2 => inner.with_compression(CompressionCoding::Zstd),
            _ => return Err(js_error(Error::InvalidConfiguration)),
        };
        Ok(Self { inner })
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
    /// Create a server for one static recipient key.
    ///
    /// # Errors
    ///
    /// Returns a stable JavaScript error value for invalid configuration.
    #[wasm_bindgen(constructor)]
    pub fn new(
        recipient_private_key: &[u8],
        recipient_key_id: &[u8],
        limits: &WasmLimits,
        compression: bool,
    ) -> Result<WasmServer, JsValue> {
        let mut inner = CoreServer::new(recipient_private_key, recipient_key_id.to_vec(), limits.0)
            .map_err(js_error)?;
        if compression {
            inner = inner.with_compression();
        }
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
            method: method_name(request.method),
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
            .protect_finite(&Response {
                status,
                headers: parse_headers(headers_json)?,
                body: body.to_vec(),
            })
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
            .into_sealer(status, parse_headers(headers_json)?, None)
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

    /// Protect END. The caller then ends the outer HTTP body.
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
    match value {
        "GET" => Ok(Method::Get),
        "POST" => Ok(Method::Post),
        "PUT" => Ok(Method::Put),
        "PATCH" => Ok(Method::Patch),
        "DELETE" => Ok(Method::Delete),
        "HEAD" => Ok(Method::Head),
        "OPTIONS" => Ok(Method::Options),
        _ => Err(js_message("unsupported_method", "unsupported HTTP method")),
    }
}

const fn method_name(method: Method) -> &'static str {
    match method {
        Method::Get => "GET",
        Method::Post => "POST",
        Method::Put => "PUT",
        Method::Patch => "PATCH",
        Method::Delete => "DELETE",
        Method::Head => "HEAD",
        Method::Options => "OPTIONS",
    }
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

//! Private `PyO3` boundary. The Python facade owns all public API names.

#![forbid(unsafe_code)]

use hpke_http::{
    Client, Error, HeaderField, Limits, Method, ReplayRequest, ReplayToken, Request, RequestHead,
    Response, ResponseCapability, ResponseMode, ResponseOpener, ResponseRecord, ResponseSealer,
    ResponseToken, Server, SseSplitter, StartToken, StreamReplayToken, StreamRequestOpener,
    StreamRequestRecord, StreamRequestSealer, StreamStartToken, build_info, generate_key_pair,
};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyRuntimeError, PyValueError},
    prelude::*,
    pybacked::PyBackedBytes,
    types::{PyBytes, PyModule},
};

create_exception!(
    _native,
    NativeError,
    PyException,
    "Stable error reported by the native hpke-http engine."
);

type NativeLimitTuple = (
    Option<usize>,
    Option<usize>,
    Option<usize>,
    Option<usize>,
    Option<u64>,
);
type NativeHeaders = Vec<(String, String)>;
type NativeResponse = (u16, NativeHeaders, Vec<u8>);
type NativeRecord = (String, u16, NativeHeaders, String, Vec<u8>);
type NativeStreamRecord = (String, Vec<u8>);

#[pyfunction]
fn native_build_info() -> (&'static str, &'static str, u32) {
    let info = build_info();
    (info.version, info.protocol, info.binding_abi)
}

#[pyfunction]
fn native_generate_key_pair() -> PyResult<(Vec<u8>, Vec<u8>)> {
    let pair = generate_key_pair().map_err(native_error)?;
    let (private_key, public_key) = pair.into_parts();
    Ok((private_key.to_vec(), public_key))
}

#[pyclass(name = "Client", module = "hpke_http._native")]
struct NativeClient {
    inner: Client,
}

#[pymethods]
impl NativeClient {
    #[new]
    #[pyo3(signature = (recipient_public_key, recipient_key_id, psk, psk_id, limits))]
    fn new(
        recipient_public_key: &[u8],
        recipient_key_id: Vec<u8>,
        psk: Vec<u8>,
        psk_id: Vec<u8>,
        limits: NativeLimitTuple,
    ) -> PyResult<Self> {
        let limits = make_limits(limits)?;
        let inner = Client::new(recipient_public_key, recipient_key_id, psk, psk_id, limits)
            .map_err(native_error)?;
        Ok(Self { inner })
    }

    fn protect(
        &self,
        py: Python<'_>,
        method: &str,
        authority: Vec<u8>,
        path: Vec<u8>,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) -> PyResult<NativeProtectedRequest> {
        let request = Request {
            method: parse_method(method)?,
            authority,
            path,
            headers: import_headers(headers),
            body,
        };
        let protected = py
            .detach(|| self.inner.protect(&request))
            .map_err(native_error)?;
        let (envelope, response_token) = protected.into_parts();
        Ok(NativeProtectedRequest {
            envelope,
            response_token: Some(response_token),
        })
    }

    fn begin_stream(
        &self,
        py: Python<'_>,
        method: &str,
        authority: Vec<u8>,
        path: Vec<u8>,
        headers: NativeHeaders,
    ) -> PyResult<(NativeStreamRequestSealer, Vec<u8>)> {
        let head = RequestHead {
            method: parse_method(method)?,
            authority,
            path,
            headers: import_headers(headers),
        };
        let (inner, first) = py
            .detach(|| self.inner.begin_stream(&head))
            .map_err(native_error)?;
        Ok((NativeStreamRequestSealer { inner: Some(inner) }, first))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(name = "StreamRequestSealer", module = "hpke_http._native")]
struct NativeStreamRequestSealer {
    inner: Option<StreamRequestSealer>,
}

#[pymethods]
impl NativeStreamRequestSealer {
    #[allow(clippy::needless_pass_by_value)]
    fn push(&mut self, py: Python<'_>, part: PyBackedBytes) -> PyResult<(usize, Option<Vec<u8>>)> {
        let inner = self.inner.as_mut().ok_or_else(continuation_consumed)?;
        let result = py.detach(|| inner.push(&part)).map_err(native_error);
        if result.is_err() {
            self.inner = None;
        }
        result
    }

    fn finish(&mut self, py: Python<'_>) -> PyResult<(Vec<u8>, NativeProtectedRequest)> {
        let inner = self.inner.take().ok_or_else(continuation_consumed)?;
        let (end, response_token) = py.detach(|| inner.finish()).map_err(native_error)?;
        Ok((
            end,
            NativeProtectedRequest {
                envelope: Vec::new(),
                response_token: Some(response_token),
            },
        ))
    }

    fn close(&mut self) {
        self.inner = None;
    }
}

#[pyclass(name = "ProtectedRequest", module = "hpke_http._native")]
struct NativeProtectedRequest {
    envelope: Vec<u8>,
    response_token: Option<ResponseToken>,
}

#[pymethods]
impl NativeProtectedRequest {
    fn take_envelope<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let envelope = std::mem::take(&mut self.envelope);
        PyBytes::new(py, &envelope)
    }

    fn open_finite_response(
        &mut self,
        py: Python<'_>,
        envelope: PyBackedBytes,
    ) -> PyResult<NativeResponse> {
        let token = self
            .response_token
            .take()
            .ok_or_else(continuation_consumed)?;
        let response = py
            .detach(move || token.open_finite(&envelope))
            .map_err(native_error)?;
        export_response(response)
    }

    fn take_opener(&mut self) -> PyResult<NativeResponseOpener> {
        let token = self
            .response_token
            .take()
            .ok_or_else(continuation_consumed)?;
        Ok(NativeResponseOpener {
            inner: Some(token.into_opener()),
        })
    }

    #[getter]
    fn consumed(&self) -> bool {
        self.response_token.is_none()
    }

    fn discard(&mut self) {
        self.response_token = None;
    }

    fn __repr__(&self) -> String {
        format!(
            "ProtectedRequest(consumed={})",
            self.response_token.is_none()
        )
    }
}

#[pyclass(name = "Server", module = "hpke_http._native")]
struct NativeServer {
    inner: Server,
}

#[pymethods]
impl NativeServer {
    #[new]
    #[pyo3(signature = (recipient_private_key, recipient_key_id, limits))]
    fn new(
        recipient_private_key: &[u8],
        recipient_key_id: Vec<u8>,
        limits: NativeLimitTuple,
    ) -> PyResult<Self> {
        let limits = make_limits(limits)?;
        let inner =
            Server::new(recipient_private_key, recipient_key_id, limits).map_err(native_error)?;
        Ok(Self { inner })
    }

    fn preparse(
        &self,
        py: Python<'_>,
        envelope: PyBackedBytes,
    ) -> PyResult<NativePreparsedRequest> {
        let preparsed = py
            .detach(move || self.inner.preparse(&envelope))
            .map_err(native_error)?;
        Ok(NativePreparsedRequest {
            psk_id: preparsed.credential.psk_id,
            token: Some(preparsed.token),
        })
    }

    fn max_complete_envelope_len(&self) -> PyResult<usize> {
        self.inner.max_complete_envelope_len().map_err(native_error)
    }

    fn stream_start_length(&self, input: &[u8]) -> PyResult<Option<usize>> {
        self.inner.stream_start_length(input).map_err(native_error)
    }

    fn preparse_stream(&self, first: &[u8]) -> PyResult<NativePreparsedStreamRequest> {
        let preparsed = self.inner.preparse_stream(first).map_err(native_error)?;
        Ok(NativePreparsedStreamRequest {
            psk_id: preparsed.credential.psk_id,
            token: Some(preparsed.token),
        })
    }

    #[getter]
    fn public_key(&self) -> Vec<u8> {
        self.inner.public_key()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(name = "PreparsedStreamRequest", module = "hpke_http._native")]
struct NativePreparsedStreamRequest {
    psk_id: Vec<u8>,
    token: Option<StreamStartToken>,
}

#[pymethods]
impl NativePreparsedStreamRequest {
    #[getter]
    fn psk_id(&self) -> Vec<u8> {
        self.psk_id.clone()
    }

    #[allow(clippy::needless_pass_by_value)]
    fn authenticate(
        &mut self,
        py: Python<'_>,
        server: &NativeServer,
        psk: PyBackedBytes,
    ) -> PyResult<NativeAuthenticatedStreamRequest> {
        let token = self.token.take().ok_or_else(continuation_consumed)?;
        let authenticated = py
            .detach(|| server.inner.authenticate_stream(token, &psk))
            .map_err(native_error)?;
        Ok(NativeAuthenticatedStreamRequest {
            replay: authenticated.replay,
            token: Some(authenticated.token),
        })
    }

    fn discard(&mut self) {
        self.token = None;
    }
}

#[pyclass(name = "AuthenticatedStreamRequest", module = "hpke_http._native")]
struct NativeAuthenticatedStreamRequest {
    replay: ReplayRequest,
    token: Option<StreamReplayToken>,
}

#[pymethods]
impl NativeAuthenticatedStreamRequest {
    #[getter]
    fn replay_id(&self) -> Vec<u8> {
        self.replay.id.to_vec()
    }

    #[getter]
    fn retain_until_exclusive(&self) -> u64 {
        self.replay.retain_until_exclusive
    }

    fn admit(&mut self, admitted: bool) -> PyResult<NativeOpenedStreamRequest> {
        let token = self.token.take().ok_or_else(continuation_consumed)?;
        let opened = token
            .admit(self.replay.decision(admitted))
            .map_err(native_error)?;
        Ok(NativeOpenedStreamRequest {
            method: opened.head.method.as_str(),
            authority: opened.head.authority,
            path: opened.head.path,
            headers: export_headers(opened.head.headers)?,
            reader: Some(opened.reader),
        })
    }

    fn discard(&mut self) {
        self.token = None;
    }
}

#[pyclass(name = "OpenedStreamRequest", module = "hpke_http._native")]
struct NativeOpenedStreamRequest {
    method: &'static str,
    authority: Vec<u8>,
    path: Vec<u8>,
    headers: NativeHeaders,
    reader: Option<StreamRequestOpener>,
}

#[pymethods]
impl NativeOpenedStreamRequest {
    #[getter]
    fn method(&self) -> &'static str {
        self.method
    }
    #[getter]
    fn authority(&self) -> Vec<u8> {
        self.authority.clone()
    }
    #[getter]
    fn path(&self) -> Vec<u8> {
        self.path.clone()
    }
    #[getter]
    fn headers(&self) -> NativeHeaders {
        self.headers.clone()
    }

    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (input, offset=0))]
    fn feed(
        &mut self,
        py: Python<'_>,
        input: PyBackedBytes,
        offset: usize,
    ) -> PyResult<(usize, Option<NativeStreamRecord>)> {
        let reader = self.reader.as_mut().ok_or_else(continuation_consumed)?;
        let input = input
            .get(offset..)
            .ok_or_else(|| PyValueError::new_err("offset exceeds input length"))?;
        let result = py.detach(|| reader.feed(input)).map_err(native_error);
        if result.is_err() {
            self.reader = None;
        }
        let (used, record) = result?;
        Ok((
            used,
            record.map(|record| match record {
                StreamRequestRecord::Data(part) => ("data".to_owned(), part),
                StreamRequestRecord::End => ("end".to_owned(), Vec::new()),
            }),
        ))
    }

    fn finish_eof(&mut self, py: Python<'_>) -> PyResult<NativeOpenedRequest> {
        let reader = self.reader.take().ok_or_else(continuation_consumed)?;
        let response_capability = py.detach(|| reader.finish_eof()).map_err(native_error)?;
        Ok(NativeOpenedRequest {
            method: self.method,
            authority: self.authority.clone(),
            path: self.path.clone(),
            headers: self.headers.clone(),
            body: None,
            response_capability: Some(response_capability),
        })
    }

    fn close(&mut self) {
        self.reader = None;
    }
}

#[pyclass(name = "PreparsedRequest", module = "hpke_http._native")]
struct NativePreparsedRequest {
    psk_id: Vec<u8>,
    token: Option<StartToken>,
}

#[pymethods]
impl NativePreparsedRequest {
    #[getter]
    fn psk_id(&self) -> Vec<u8> {
        self.psk_id.clone()
    }

    fn authenticate(
        &mut self,
        py: Python<'_>,
        server: &NativeServer,
        psk: PyBackedBytes,
    ) -> PyResult<NativeAuthenticatedRequest> {
        let token = self.token.take().ok_or_else(continuation_consumed)?;
        let authenticated = py
            .detach(move || server.inner.authenticate(token, &psk))
            .map_err(native_error)?;
        Ok(NativeAuthenticatedRequest {
            replay: authenticated.replay,
            token: Some(authenticated.token),
        })
    }

    #[getter]
    fn consumed(&self) -> bool {
        self.token.is_none()
    }

    fn discard(&mut self) {
        self.token = None;
    }
}

#[pyclass(name = "AuthenticatedRequest", module = "hpke_http._native")]
struct NativeAuthenticatedRequest {
    replay: ReplayRequest,
    token: Option<ReplayToken>,
}

#[pymethods]
impl NativeAuthenticatedRequest {
    #[getter]
    fn replay_id(&self) -> Vec<u8> {
        self.replay.id.to_vec()
    }

    #[getter]
    fn retain_until_exclusive(&self) -> u64 {
        self.replay.retain_until_exclusive
    }

    fn admit(&mut self, admitted: bool) -> PyResult<NativeOpenedRequest> {
        let token = self.token.take().ok_or_else(continuation_consumed)?;
        let opened = token
            .admit(self.replay.decision(admitted))
            .map_err(native_error)?;
        Ok(NativeOpenedRequest {
            method: opened.request.method.as_str(),
            authority: opened.request.authority,
            path: opened.request.path,
            headers: export_headers(opened.request.headers)?,
            body: Some(opened.request.body),
            response_capability: Some(opened.response),
        })
    }

    #[getter]
    fn consumed(&self) -> bool {
        self.token.is_none()
    }

    fn discard(&mut self) {
        self.token = None;
    }
}

#[pyclass(name = "OpenedRequest", module = "hpke_http._native")]
struct NativeOpenedRequest {
    method: &'static str,
    authority: Vec<u8>,
    path: Vec<u8>,
    headers: Vec<(String, String)>,
    body: Option<Vec<u8>>,
    response_capability: Option<ResponseCapability>,
}

#[pymethods]
impl NativeOpenedRequest {
    #[getter]
    fn method(&self) -> &'static str {
        self.method
    }

    #[getter]
    fn authority(&self) -> Vec<u8> {
        self.authority.clone()
    }

    #[getter]
    fn path(&self) -> Vec<u8> {
        self.path.clone()
    }

    #[getter]
    fn headers(&self) -> Vec<(String, String)> {
        self.headers.clone()
    }

    fn take_body<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let body = self.body.take().ok_or_else(continuation_consumed)?;
        Ok(PyBytes::new(py, &body))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn protect_response(
        &mut self,
        py: Python<'_>,
        status: u16,
        headers: Vec<(String, String)>,
        body: PyBackedBytes,
    ) -> PyResult<Vec<u8>> {
        let capability = self
            .response_capability
            .take()
            .ok_or_else(continuation_consumed)?;
        let headers = import_headers(headers);
        py.detach(|| capability.protect_finite_parts(status, headers, &body))
            .map_err(native_error)
    }

    fn take_sealer(
        &mut self,
        py: Python<'_>,
        status: u16,
        headers: NativeHeaders,
    ) -> PyResult<(NativeResponseSealer, Vec<u8>)> {
        let capability = self
            .response_capability
            .take()
            .ok_or_else(continuation_consumed)?;
        let (inner, first) = py
            .detach(|| capability.into_sealer(status, import_headers(headers)))
            .map_err(native_error)?;
        Ok((NativeResponseSealer { inner: Some(inner) }, first))
    }

    #[getter]
    fn response_consumed(&self) -> bool {
        self.response_capability.is_none()
    }

    fn discard_response(&mut self) {
        self.response_capability = None;
    }
}

#[pyclass(name = "ResponseOpener", module = "hpke_http._native")]
struct NativeResponseOpener {
    inner: Option<ResponseOpener>,
}

#[pymethods]
impl NativeResponseOpener {
    // PyO3 owns this byte view while the GIL is released.
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (input, offset=0))]
    fn feed(
        &mut self,
        py: Python<'_>,
        input: PyBackedBytes,
        offset: usize,
    ) -> PyResult<(usize, Option<NativeRecord>)> {
        let inner = self.inner.as_mut().ok_or_else(continuation_consumed)?;
        let input = input
            .get(offset..)
            .ok_or_else(|| PyValueError::new_err("offset exceeds input length"))?;
        let (used, record) = py.detach(|| inner.feed(input)).map_err(native_error)?;
        let record = match record {
            Some(ResponseRecord::Start(head)) => Some((
                "start".to_owned(),
                head.status,
                export_headers(head.headers)?,
                match head.mode {
                    ResponseMode::Finite => "finite",
                    ResponseMode::Sse => "sse",
                }
                .to_owned(),
                Vec::new(),
            )),
            Some(ResponseRecord::SseData(block)) => {
                Some(("data".to_owned(), 0, Vec::new(), String::new(), block))
            }
            Some(ResponseRecord::End) => {
                Some(("end".to_owned(), 0, Vec::new(), String::new(), Vec::new()))
            }
            None => None,
        };
        Ok((used, record))
    }

    fn finish_eof(&mut self, py: Python<'_>) -> PyResult<Option<NativeResponse>> {
        let inner = self.inner.as_mut().ok_or_else(continuation_consumed)?;
        let response = py.detach(|| inner.finish_eof()).map_err(native_error)?;
        self.inner = None;
        response.map(export_response).transpose()
    }

    fn close(&mut self) {
        self.inner = None;
    }
}

#[pyclass(name = "ResponseSealer", module = "hpke_http._native")]
struct NativeResponseSealer {
    inner: Option<ResponseSealer>,
}

#[pymethods]
impl NativeResponseSealer {
    // PyO3 owns this byte view while the GIL is released.
    #[allow(clippy::needless_pass_by_value)]
    fn seal_finite_body(
        &mut self,
        py: Python<'_>,
        body: PyBackedBytes,
    ) -> PyResult<Option<Vec<u8>>> {
        let inner = self.inner.as_mut().ok_or_else(continuation_consumed)?;
        py.detach(|| inner.seal_finite_body(&body))
            .map_err(native_error)
    }

    // PyO3 owns this byte view while the GIL is released.
    #[allow(clippy::needless_pass_by_value)]
    fn seal_sse_block(&mut self, py: Python<'_>, block: PyBackedBytes) -> PyResult<Vec<u8>> {
        let inner = self.inner.as_mut().ok_or_else(continuation_consumed)?;
        py.detach(|| inner.seal_sse_block(&block))
            .map_err(native_error)
    }

    fn finish(&mut self, py: Python<'_>) -> PyResult<Vec<u8>> {
        let inner = self.inner.as_mut().ok_or_else(continuation_consumed)?;
        let output = py.detach(|| inner.finish()).map_err(native_error)?;
        self.inner = None;
        Ok(output)
    }

    fn close(&mut self) {
        self.inner = None;
    }
}

#[pyclass(name = "SseSplitter", module = "hpke_http._native")]
struct NativeSseSplitter {
    inner: Option<SseSplitter>,
}

#[pymethods]
impl NativeSseSplitter {
    #[new]
    fn new(max_block_len: usize) -> Self {
        Self {
            inner: Some(SseSplitter::new(max_block_len)),
        }
    }

    // PyO3 owns this byte view while the GIL is released.
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (input, offset=0, max_bytes=None))]
    fn feed(
        &mut self,
        py: Python<'_>,
        input: PyBackedBytes,
        offset: usize,
        max_bytes: Option<usize>,
    ) -> PyResult<(usize, Option<Vec<u8>>)> {
        let inner = self.inner.as_mut().ok_or_else(continuation_consumed)?;
        let input = input
            .get(offset..)
            .ok_or_else(|| PyValueError::new_err("offset exceeds input length"))?;
        let input = &input[..max_bytes.unwrap_or(input.len()).min(input.len())];
        py.detach(|| inner.feed(input)).map_err(native_error)
    }

    fn finish(&mut self) {
        if let Some(inner) = self.inner.as_mut() {
            inner.finish();
        }
        self.inner = None;
    }

    fn close(&mut self) {
        self.inner = None;
    }
}

fn make_limits(values: NativeLimitTuple) -> PyResult<Limits> {
    let (max_body_len, max_header_bytes, max_header_count, max_target_len, max_request_bytes) =
        values;
    let defaults = Limits::default();
    Limits {
        max_body_len: max_body_len.unwrap_or(defaults.max_body_len),
        max_header_bytes: max_header_bytes.unwrap_or(defaults.max_header_bytes),
        max_header_count: max_header_count.unwrap_or(defaults.max_header_count),
        max_target_len: max_target_len.unwrap_or(defaults.max_target_len),
        max_request_bytes: max_request_bytes.unwrap_or(defaults.max_request_bytes),
    }
    .validate()
    .map_err(native_error)
}

fn parse_method(value: &str) -> PyResult<Method> {
    Method::from_bytes(value.as_bytes())
        .map_err(|_| PyValueError::new_err("unsupported HTTP method"))
}

fn import_headers(headers: Vec<(String, String)>) -> Vec<HeaderField> {
    headers
        .into_iter()
        .map(|(name, value)| HeaderField {
            name: name.into_bytes(),
            value: value.into_bytes(),
        })
        .collect()
}

fn export_headers(headers: Vec<HeaderField>) -> PyResult<Vec<(String, String)>> {
    headers
        .into_iter()
        .map(|field| {
            let name = String::from_utf8(field.name).map_err(|_| {
                NativeError::new_err(("malformed_envelope", "invalid UTF-8 header name"))
            })?;
            let value = String::from_utf8(field.value).map_err(|_| {
                NativeError::new_err(("malformed_envelope", "invalid UTF-8 header value"))
            })?;
            Ok((name, value))
        })
        .collect()
}

fn export_response(response: Response) -> PyResult<NativeResponse> {
    Ok((
        response.status,
        export_headers(response.headers)?,
        response.body,
    ))
}

fn native_error(error: Error) -> PyErr {
    NativeError::new_err((error.code(), error.to_string()))
}

fn continuation_consumed() -> PyErr {
    PyRuntimeError::new_err("continuation already consumed")
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("NativeError", module.py().get_type::<NativeError>())?;
    module.add_function(wrap_pyfunction!(native_build_info, module)?)?;
    module.add_function(wrap_pyfunction!(native_generate_key_pair, module)?)?;
    module.add_class::<NativeClient>()?;
    module.add_class::<NativeProtectedRequest>()?;
    module.add_class::<NativeStreamRequestSealer>()?;
    module.add_class::<NativeServer>()?;
    module.add_class::<NativePreparsedStreamRequest>()?;
    module.add_class::<NativeAuthenticatedStreamRequest>()?;
    module.add_class::<NativeOpenedStreamRequest>()?;
    module.add_class::<NativePreparsedRequest>()?;
    module.add_class::<NativeAuthenticatedRequest>()?;
    module.add_class::<NativeOpenedRequest>()?;
    module.add_class::<NativeResponseOpener>()?;
    module.add_class::<NativeResponseSealer>()?;
    module.add_class::<NativeSseSplitter>()?;
    Ok(())
}

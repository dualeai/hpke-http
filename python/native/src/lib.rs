//! Private `PyO3` boundary. The Python facade owns all public API names.

#![forbid(unsafe_code)]

use hpke_http::{
    Client, CompressionCoding, Error, HeaderField, Limits, Method, ReplayRequest, ReplayToken,
    Request, Response, ResponseCapability, ResponseToken, Server, StartToken, build_info,
    generate_key_pair,
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

type NativeLimitTuple = (Option<usize>, Option<usize>, Option<usize>, Option<usize>);
type NativeHeaders = Vec<(String, String)>;
type NativeResponse = (u16, NativeHeaders, Vec<u8>);

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
    #[pyo3(signature = (recipient_public_key, recipient_key_id, psk, psk_id, limits, compression=None))]
    fn new(
        recipient_public_key: &[u8],
        recipient_key_id: Vec<u8>,
        psk: Vec<u8>,
        psk_id: Vec<u8>,
        limits: NativeLimitTuple,
        compression: Option<&str>,
    ) -> PyResult<Self> {
        let limits = make_limits(limits)?;
        let mut inner = Client::new(recipient_public_key, recipient_key_id, psk, psk_id, limits)
            .map_err(native_error)?;
        if let Some(coding) = compression {
            inner = inner.with_compression(parse_compression(coding)?);
        }
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

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
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

    fn open_response(
        &mut self,
        py: Python<'_>,
        envelope: PyBackedBytes,
    ) -> PyResult<NativeResponse> {
        let token = self
            .response_token
            .take()
            .ok_or_else(continuation_consumed)?;
        let response = py
            .detach(move || token.open(&envelope))
            .map_err(native_error)?;
        export_response(response)
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
    #[pyo3(signature = (recipient_private_key, recipient_key_id, limits, compression=false))]
    fn new(
        recipient_private_key: &[u8],
        recipient_key_id: Vec<u8>,
        limits: NativeLimitTuple,
        compression: bool,
    ) -> PyResult<Self> {
        let limits = make_limits(limits)?;
        let mut inner =
            Server::new(recipient_private_key, recipient_key_id, limits).map_err(native_error)?;
        if compression {
            inner = inner.with_compression();
        }
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

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
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
            method: method_name(opened.request.method),
            authority: opened.request.authority,
            path: opened.request.path,
            headers: export_headers(opened.request.headers)?,
            body: opened.request.body,
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
    body: Vec<u8>,
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

    fn take_body<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let body = std::mem::take(&mut self.body);
        PyBytes::new(py, &body)
    }

    fn protect_response(
        &mut self,
        py: Python<'_>,
        status: u16,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) -> PyResult<Vec<u8>> {
        let capability = self
            .response_capability
            .take()
            .ok_or_else(continuation_consumed)?;
        let response = Response {
            status,
            headers: import_headers(headers),
            body,
        };
        py.detach(|| capability.protect(&response))
            .map_err(native_error)
    }

    #[getter]
    fn response_consumed(&self) -> bool {
        self.response_capability.is_none()
    }

    fn discard_response(&mut self) {
        self.response_capability = None;
    }
}

fn make_limits(values: NativeLimitTuple) -> PyResult<Limits> {
    let (max_body_len, max_header_bytes, max_header_count, max_target_len) = values;
    let defaults = Limits::default();
    Limits {
        max_body_len: max_body_len.unwrap_or(defaults.max_body_len),
        max_header_bytes: max_header_bytes.unwrap_or(defaults.max_header_bytes),
        max_header_count: max_header_count.unwrap_or(defaults.max_header_count),
        max_target_len: max_target_len.unwrap_or(defaults.max_target_len),
    }
    .validate()
    .map_err(native_error)
}

fn parse_compression(value: &str) -> PyResult<CompressionCoding> {
    match value {
        "gzip" => Ok(CompressionCoding::Gzip),
        "zstd" => Ok(CompressionCoding::Zstd),
        _ => Err(native_error(Error::InvalidConfiguration)),
    }
}

fn parse_method(value: &str) -> PyResult<Method> {
    match value {
        "GET" => Ok(Method::Get),
        "POST" => Ok(Method::Post),
        "PUT" => Ok(Method::Put),
        "PATCH" => Ok(Method::Patch),
        "DELETE" => Ok(Method::Delete),
        "HEAD" => Ok(Method::Head),
        "OPTIONS" => Ok(Method::Options),
        _ => Err(PyValueError::new_err("unsupported HTTP method")),
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
    module.add_class::<NativeServer>()?;
    module.add_class::<NativePreparsedRequest>()?;
    module.add_class::<NativeAuthenticatedRequest>()?;
    module.add_class::<NativeOpenedRequest>()?;
    Ok(())
}

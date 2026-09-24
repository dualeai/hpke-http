//! Canonical request heads and field pairs used by protocol version 3.
//!
//! This module implements only the small protocol subset that the engine uses.
//! It does not depend on a general OHTTP/BHTTP package, accept indeterminate
//! messages, informational responses, trailers, padding, or non-minimal integer
//! encodings. Decode limits apply before any field or body allocation. The
//! request body arrives in checked DATA records.

use crate::{Error, Limits, Method};

const FORBIDDEN_HEADERS: &[&[u8]] = &[
    b"connection",
    b"expect",
    b"host",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authentication-info",
    b"proxy-authorization",
    b"proxy-connection",
    b"te",
    b"trailer",
    b"transfer-encoding",
    b"upgrade",
];

/// One canonical end-to-end HTTP field in an ordered field section.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeaderField {
    /// Lower-case HTTP field name. Repeated names preserve their original order.
    pub name: Vec<u8>,
    /// Canonical ASCII field value without outer optional whitespace.
    pub value: Vec<u8>,
}

/// A complete plaintext request supplied to or returned by the engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Request {
    /// Supported HTTP method.
    pub method: Method,
    /// URI authority, such as `api.example.com` or `api.example.com:8443`.
    pub authority: Vec<u8>,
    /// Path and optional query, beginning with `/`, or `*` for `OPTIONS`.
    pub path: Vec<u8>,
    /// Ordered end-to-end HTTP fields.
    pub headers: Vec<HeaderField>,
    /// Complete bounded request body. Empty bodies remain authenticated.
    pub body: Vec<u8>,
}

/// Request fields without a body. Client input before protection and checked
/// fields after server admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestHead {
    /// Supported HTTP method.
    pub method: Method,
    /// URI authority.
    pub authority: Vec<u8>,
    /// Path and optional query.
    pub path: Vec<u8>,
    /// Ordered end-to-end HTTP fields.
    pub headers: Vec<HeaderField>,
}

pub(crate) fn encode_stream_head(head: &RequestHead, limits: Limits) -> Result<Vec<u8>, Error> {
    validate_target(head.method, &head.authority, &head.path, limits)?;
    validate_payload(&head.headers, &[], limits)?;
    stream_content_length(&head.headers)?;
    let fields = encode_fields(&head.headers)?;
    let mut output = Vec::with_capacity(fields.len() + head.authority.len() + head.path.len() + 64);
    // This private stream head is not an RFC 9292 Binary HTTP message.
    write_vector(&mut output, head.method.as_bytes())?;
    write_vector(&mut output, &head.authority)?;
    write_vector(&mut output, &head.path)?;
    write_vector(&mut output, &fields)?;
    Ok(output)
}

pub(crate) fn decode_stream_head(input: &[u8], limits: Limits) -> Result<RequestHead, Error> {
    let mut reader = SliceReader::new(input);
    let method = Method::from_bytes(reader.read_vector()?)?;
    let authority = reader.read_vector()?;
    let path = reader.read_vector()?;
    validate_target(method, authority, path, limits).map_err(decode_validation_error)?;
    let headers = decode_fields(reader.read_vector()?, limits).map_err(decode_validation_error)?;
    if !reader.is_finished() {
        return Err(Error::MalformedEnvelope);
    }
    stream_content_length(&headers).map_err(decode_validation_error)?;
    Ok(RequestHead {
        method,
        authority: authority.to_vec(),
        path: path.to_vec(),
        headers,
    })
}

pub(crate) fn stream_content_length(headers: &[HeaderField]) -> Result<Option<u64>, Error> {
    let mut values = headers
        .iter()
        .filter(|field| field.name == b"content-length");
    let Some(field) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some()
        || field.value.is_empty()
        || !field.value.iter().all(u8::is_ascii_digit)
    {
        return Err(Error::InvalidConfiguration);
    }
    let mut length = 0_u64;
    for digit in &field.value {
        length = length
            .checked_mul(10)
            .and_then(|value| value.checked_add(u64::from(digit - b'0')))
            .ok_or(Error::LimitExceeded)?;
    }
    Ok(Some(length))
}

/// A complete plaintext response supplied to or returned by the engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Response {
    /// Final HTTP status code in the inclusive range 200 through 599.
    pub status: u16,
    /// Ordered end-to-end HTTP fields.
    pub headers: Vec<HeaderField>,
    /// Complete bounded response body. Empty bodies remain authenticated.
    pub body: Vec<u8>,
}

pub(crate) fn validate_decoded_response_body(
    response: &Response,
    method: Method,
) -> Result<(), Error> {
    validate_response_content_length(
        response.status,
        &response.headers,
        response.body.len(),
        method,
    )
    .map_err(decode_validation_error)
}

const fn decode_validation_error(error: Error) -> Error {
    match error {
        Error::InvalidConfiguration => Error::MalformedEnvelope,
        other => other,
    }
}

pub(crate) fn encode_fields(headers: &[HeaderField]) -> Result<Vec<u8>, Error> {
    let mut output = Vec::new();
    for field in headers {
        write_vector(&mut output, &field.name)?;
        write_vector(&mut output, &field.value)?;
    }
    Ok(output)
}

pub(crate) fn decode_fields(input: &[u8], limits: Limits) -> Result<Vec<HeaderField>, Error> {
    let mut reader = SliceReader::new(input);
    let mut headers: Vec<HeaderField> = Vec::new();
    let mut header_bytes = 0_usize;
    while !reader.is_finished() {
        if headers.len() >= limits.max_header_count {
            return Err(Error::LimitExceeded);
        }
        let name = reader.read_vector()?;
        let value = reader.read_vector()?;
        header_bytes = header_bytes
            .checked_add(name.len())
            .and_then(|length| length.checked_add(value.len()))
            .ok_or(Error::LimitExceeded)?;
        if header_bytes > limits.max_header_bytes {
            return Err(Error::LimitExceeded);
        }
        validate_header_parts(name, value)?;
        headers.push(HeaderField {
            name: name.to_vec(),
            value: value.to_vec(),
        });
    }
    Ok(headers)
}

pub(crate) fn validate_request(request: &Request, limits: Limits) -> Result<(), Error> {
    validate_target(request.method, &request.authority, &request.path, limits)?;
    validate_payload(&request.headers, &request.body, limits)?;
    validate_content_length(
        &request.headers,
        request.body.len(),
        ContentLengthRule::Exact,
    )
}

pub(crate) fn validate_response_content_length(
    status: u16,
    headers: &[HeaderField],
    body_len: usize,
    method: Method,
) -> Result<(), Error> {
    validate_content_length(
        headers,
        body_len,
        response_content_length_rule(method, status),
    )
}

pub(crate) fn validate_response_head(
    status: u16,
    headers: &[HeaderField],
    limits: Limits,
) -> Result<(), Error> {
    if !(200..=599).contains(&status) {
        return Err(Error::InvalidConfiguration);
    }
    validate_payload(headers, &[], limits)
}

fn validate_target(
    method: Method,
    authority: &[u8],
    path: &[u8],
    limits: Limits,
) -> Result<(), Error> {
    let target_len = authority
        .len()
        .checked_add(path.len())
        .ok_or(Error::LimitExceeded)?;
    if target_len > limits.max_target_len {
        return Err(Error::LimitExceeded);
    }
    if !valid_authority(authority) || !valid_request_target(method, path) {
        return Err(Error::InvalidConfiguration);
    }
    Ok(())
}

fn valid_authority(authority: &[u8]) -> bool {
    if authority.is_empty() || !authority.iter().all(u8::is_ascii_graphic) {
        return false;
    }
    if authority[0] == b'[' {
        return valid_ip_literal_authority(authority);
    }

    let (host, port) = authority
        .iter()
        .rposition(|byte| *byte == b':')
        .map_or((authority, None), |separator| {
            (&authority[..separator], Some(&authority[separator + 1..]))
        });
    if host.contains(&b':') || !valid_reg_name(host) || !port.is_none_or(valid_port) {
        return false;
    }
    let numeric_ipv4_candidate = host.split(|byte| *byte == b'.').count() == 4
        && host
            .iter()
            .all(|byte| byte.is_ascii_digit() || *byte == b'.');
    !numeric_ipv4_candidate
        || std::str::from_utf8(host)
            .ok()
            .and_then(|value| value.parse::<std::net::Ipv4Addr>().ok())
            .is_some()
}

fn valid_ip_literal_authority(authority: &[u8]) -> bool {
    let Some(close) = authority.iter().position(|byte| *byte == b']') else {
        return false;
    };
    let literal = &authority[1..close];
    let suffix = &authority[close + 1..];
    let valid_suffix = suffix.is_empty() || suffix.strip_prefix(b":").is_some_and(valid_port);
    if literal.is_empty() || !valid_suffix {
        return false;
    }
    let Ok(value) = std::str::from_utf8(literal) else {
        return false;
    };
    value.parse::<std::net::Ipv6Addr>().is_ok() || valid_ipv_future(literal)
}

fn valid_ipv_future(value: &[u8]) -> bool {
    let Some(version_and_address) = value
        .strip_prefix(b"v")
        .or_else(|| value.strip_prefix(b"V"))
    else {
        return false;
    };
    let Some(separator) = version_and_address.iter().position(|byte| *byte == b'.') else {
        return false;
    };
    let version = &version_and_address[..separator];
    let address = &version_and_address[separator + 1..];
    !version.is_empty()
        && version.iter().all(u8::is_ascii_hexdigit)
        && !address.is_empty()
        && address
            .iter()
            .copied()
            .all(|byte| is_unreserved(byte) || is_sub_delim(byte) || byte == b':')
}

fn valid_reg_name(value: &[u8]) -> bool {
    if value.is_empty() {
        return false;
    }
    let mut position = 0;
    while position < value.len() {
        let byte = value[position];
        if is_unreserved(byte) || is_sub_delim(byte) {
            position += 1;
        } else if byte == b'%'
            && value
                .get(position + 1..position + 3)
                .is_some_and(|encoded| encoded.iter().all(u8::is_ascii_hexdigit))
        {
            position += 3;
        } else {
            return false;
        }
    }
    true
}

fn valid_port(value: &[u8]) -> bool {
    if value.is_empty() || !value.iter().all(u8::is_ascii_digit) {
        return false;
    }
    let significant = value
        .iter()
        .position(|byte| *byte != b'0')
        .map_or(&[][..], |position| &value[position..]);
    significant.is_empty()
        || (significant.len() <= 5
            && std::str::from_utf8(significant)
                .ok()
                .and_then(|port| port.parse::<u16>().ok())
                .is_some())
}

const fn is_unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~')
}

const fn is_sub_delim(byte: u8) -> bool {
    matches!(
        byte,
        b'!' | b'$' | b'&' | b'\'' | b'(' | b')' | b'*' | b'+' | b',' | b';' | b'='
    )
}

fn validate_payload(headers: &[HeaderField], body: &[u8], limits: Limits) -> Result<(), Error> {
    if headers.len() > limits.max_header_count || body.len() > limits.max_body_len {
        return Err(Error::LimitExceeded);
    }
    let mut header_bytes = 0_usize;
    for field in headers {
        validate_header_parts(&field.name, &field.value)?;
        header_bytes = header_bytes
            .checked_add(field.name.len())
            .and_then(|length| length.checked_add(field.value.len()))
            .ok_or(Error::LimitExceeded)?;
    }
    if header_bytes > limits.max_header_bytes {
        return Err(Error::LimitExceeded);
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ContentLengthRule {
    Exact,
    Metadata,
    Forbidden,
}

fn response_content_length_rule(method: Method, status: u16) -> ContentLengthRule {
    if status == 204 {
        ContentLengthRule::Forbidden
    } else if method == Method::Head || status == 304 {
        ContentLengthRule::Metadata
    } else {
        ContentLengthRule::Exact
    }
}

fn validate_content_length(
    headers: &[HeaderField],
    body_len: usize,
    rule: ContentLengthRule,
) -> Result<(), Error> {
    let mut values = headers
        .iter()
        .filter(|field| field.name == b"content-length")
        .map(|field| field.value.as_slice());
    let Some(value) = values.next() else {
        return Ok(());
    };
    if values.next().is_some() || value.is_empty() || !value.iter().all(u8::is_ascii_digit) {
        return Err(Error::InvalidConfiguration);
    }
    match rule {
        ContentLengthRule::Forbidden => Err(Error::InvalidConfiguration),
        ContentLengthRule::Metadata => Ok(()),
        ContentLengthRule::Exact => {
            let significant = value
                .iter()
                .position(|byte| *byte != b'0')
                .map_or(&[][..], |position| &value[position..]);
            let expected = body_len.to_string();
            let matches = if significant.is_empty() {
                body_len == 0
            } else {
                significant == expected.as_bytes()
            };
            if matches {
                Ok(())
            } else {
                Err(Error::InvalidConfiguration)
            }
        }
    }
}

fn validate_header_parts(name: &[u8], value: &[u8]) -> Result<(), Error> {
    if name.is_empty()
        || !name.iter().copied().all(is_lower_tchar)
        || FORBIDDEN_HEADERS.contains(&name)
        || !value
            .iter()
            .all(|byte| *byte == b'\t' || (b' '..=b'~').contains(byte))
        || value.first().is_some_and(|byte| is_ows(*byte))
        || value.last().is_some_and(|byte| is_ows(*byte))
    {
        return Err(Error::InvalidConfiguration);
    }
    Ok(())
}

fn valid_request_target(method: Method, path: &[u8]) -> bool {
    if path == b"*" {
        return method == Method::Options;
    }
    if !path.starts_with(b"/")
        || path.contains(&b'#')
        || path.contains(&b'\\')
        || !path.iter().all(u8::is_ascii_graphic)
        || !valid_percent_encoding(path)
    {
        return false;
    }
    let path_only = path.split(|byte| *byte == b'?').next().unwrap_or_default();
    let Some(decoded_path) = decode_path(path_only) else {
        return false;
    };
    std::str::from_utf8(&decoded_path).is_ok()
        && !decoded_path
            .iter()
            .any(|byte| byte.is_ascii_control() || *byte == b'\\')
        && !decoded_path.split(|byte| *byte == b'/').any(is_dot_segment)
}

fn decode_path(path: &[u8]) -> Option<Vec<u8>> {
    let mut decoded = Vec::with_capacity(path.len());
    let mut position = 0;
    while position < path.len() {
        if path[position] == b'%' {
            let high = hex_value(*path.get(position + 1)?)?;
            let low = hex_value(*path.get(position + 2)?)?;
            let byte = high.checked_mul(16)?.checked_add(low)?;
            if byte == b'/' {
                return None;
            }
            decoded.push(byte);
            position += 3;
        } else {
            decoded.push(path[position]);
            position += 1;
        }
    }
    Some(decoded)
}

const fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn valid_percent_encoding(value: &[u8]) -> bool {
    let mut position = 0;
    while position < value.len() {
        if value[position] == b'%' {
            let Some(encoded) = value.get(position + 1..position + 3) else {
                return false;
            };
            if !encoded.iter().all(u8::is_ascii_hexdigit) {
                return false;
            }
            position += 3;
        } else {
            position += 1;
        }
    }
    true
}

fn is_dot_segment(segment: &[u8]) -> bool {
    segment == b"." || segment == b".."
}

const fn is_ows(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t')
}

const fn is_lower_tchar(byte: u8) -> bool {
    byte.is_ascii_lowercase()
        || byte.is_ascii_digit()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

fn write_vector(output: &mut Vec<u8>, value: &[u8]) -> Result<(), Error> {
    let length = u64::try_from(value.len()).map_err(|_| Error::LimitExceeded)?;
    write_varint(output, length)?;
    output.extend_from_slice(value);
    Ok(())
}

fn write_varint(output: &mut Vec<u8>, value: u64) -> Result<(), Error> {
    match value {
        0..=0x3f => output.push(u8::try_from(value).map_err(|_| Error::LimitExceeded)?),
        0x40..=0x3fff => {
            let encoded = u16::try_from(value).map_err(|_| Error::LimitExceeded)? | 0x4000;
            output.extend_from_slice(&encoded.to_be_bytes());
        }
        0x4000..=0x3fff_ffff => {
            let encoded = u32::try_from(value).map_err(|_| Error::LimitExceeded)? | 0x8000_0000;
            output.extend_from_slice(&encoded.to_be_bytes());
        }
        0x4000_0000..=0x3fff_ffff_ffff_ffff => {
            output.extend_from_slice(&(value | 0xc000_0000_0000_0000).to_be_bytes());
        }
        _ => return Err(Error::LimitExceeded),
    }
    Ok(())
}

struct SliceReader<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> SliceReader<'a> {
    const fn new(input: &'a [u8]) -> Self {
        Self { input, position: 0 }
    }

    fn read_varint(&mut self) -> Result<u64, Error> {
        let first = *self
            .input
            .get(self.position)
            .ok_or(Error::MalformedEnvelope)?;
        let length = 1_usize << (first >> 6);
        let end = self
            .position
            .checked_add(length)
            .ok_or(Error::MalformedEnvelope)?;
        let encoded = self
            .input
            .get(self.position..end)
            .ok_or(Error::MalformedEnvelope)?;
        let mut value = u64::from(first & 0x3f);
        for byte in &encoded[1..] {
            value = (value << 8) | u64::from(*byte);
        }
        if varint_length(value) != length {
            return Err(Error::MalformedEnvelope);
        }
        self.position = end;
        Ok(value)
    }

    fn read_vector(&mut self) -> Result<&'a [u8], Error> {
        let length = usize::try_from(self.read_varint()?).map_err(|_| Error::MalformedEnvelope)?;
        let end = self
            .position
            .checked_add(length)
            .ok_or(Error::MalformedEnvelope)?;
        let value = self
            .input
            .get(self.position..end)
            .ok_or(Error::MalformedEnvelope)?;
        self.position = end;
        Ok(value)
    }

    const fn is_finished(&self) -> bool {
        self.position == self.input.len()
    }
}

const fn varint_length(value: u64) -> usize {
    match value {
        0..=0x3f => 1,
        0x40..=0x3fff => 2,
        0x4000..=0x3fff_ffff => 4,
        _ => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::{SliceReader, decode_fields, decode_stream_head, encode_stream_head, write_varint};
    use crate::{Error, HeaderField, Limits, Method, RequestHead};

    #[test]
    fn field_decoder_stops_at_the_configured_count() {
        let limits = Limits {
            max_header_count: 2,
            ..Limits::default()
        };
        let encoded = [1, b'a', 0, 1, b'b', 0, 1, b'c', 0];
        assert_eq!(decode_fields(&encoded, limits), Err(Error::LimitExceeded));
    }

    #[test]
    fn varint_round_trips_boundaries_and_rejects_non_minimal_values() -> Result<(), Error> {
        let cases: &[(u64, &[u8])] = &[
            (0, &[0x00]),
            (63, &[0x3f]),
            (64, &[0x40, 0x40]),
            (16_383, &[0x7f, 0xff]),
            (16_384, &[0x80, 0x00, 0x40, 0x00]),
            (0x3fff_ffff, &[0xbf, 0xff, 0xff, 0xff]),
            (
                0x4000_0000,
                &[0xc0, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00],
            ),
        ];
        for (value, expected) in cases {
            let mut encoded = Vec::new();
            write_varint(&mut encoded, *value)?;
            assert_eq!(&encoded, expected);
            let mut reader = SliceReader::new(expected);
            assert_eq!(reader.read_varint()?, *value);
            assert!(reader.is_finished());
        }
        let mut non_minimal = SliceReader::new(&[0x40, 0x01]);
        assert_eq!(non_minimal.read_varint(), Err(Error::MalformedEnvelope));
        Ok(())
    }

    #[test]
    fn stream_head_decoder_rechecks_authenticated_content_length() -> Result<(), Error> {
        let request = RequestHead {
            method: Method::Post,
            authority: b"api.example.test".to_vec(),
            path: b"/items".to_vec(),
            headers: vec![HeaderField {
                name: b"content-length".to_vec(),
                value: b"4".to_vec(),
            }],
        };
        let mut encoded = encode_stream_head(&request, Limits::default())?;
        let name_offset = encoded
            .windows(b"content-length".len())
            .position(|window| window == b"content-length")
            .ok_or(Error::MalformedEnvelope)?;
        let value_offset = name_offset + b"content-length".len() + 1;
        encoded[value_offset] = b'x';
        assert_eq!(
            decode_stream_head(&encoded, Limits::default()).err(),
            Some(Error::MalformedEnvelope)
        );
        Ok(())
    }

    #[test]
    fn stream_head_decoder_rechecks_authenticated_authority_syntax() -> Result<(), Error> {
        let request = RequestHead {
            method: Method::Get,
            authority: b"api.example".to_vec(),
            path: b"/items".to_vec(),
            headers: Vec::new(),
        };
        let mut encoded = encode_stream_head(&request, Limits::default())?;
        let authority_offset = encoded
            .windows(b"api.example".len())
            .position(|window| window == b"api.example")
            .ok_or(Error::MalformedEnvelope)?;
        encoded[authority_offset] = b'[';
        assert_eq!(
            decode_stream_head(&encoded, Limits::default()).err(),
            Some(Error::MalformedEnvelope)
        );
        Ok(())
    }
}

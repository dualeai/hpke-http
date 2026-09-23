//! Known-answer checks against frozen protocol data cross-checked independently.

use hpke_http::{
    Client, EntropySource, Error, HeaderField, Limits, Method, Request, Response, Server,
};
use serde_json::Value;

const CORPUS: &str = include_str!("vectors/protocol-v1.json");

struct FixedEntropy(Vec<u8>);

impl EntropySource for FixedEntropy {
    fn fill(&mut self, destination: &mut [u8]) -> Result<(), Error> {
        if destination.len() != self.0.len() {
            return Err(Error::EntropyUnavailable);
        }
        destination.copy_from_slice(&self.0);
        Ok(())
    }
}

#[test]
fn rust_server_opens_frozen_request_and_matches_response() -> Result<(), Box<dyn std::error::Error>>
{
    let corpus: Value = serde_json::from_str(CORPUS)?;
    assert_eq!(text(&corpus, "schema")?, "hpke-http-protocol-corpus/1");
    assert_eq!(text(&corpus, "protocol")?, hpke_http::PROTOCOL_ID);
    assert_eq!(text(&corpus, "generator")?, "frozen-cross-implementation/1");

    let recipient_private_key = hex_field(&corpus, "recipient_private_key")?;
    let recipient_key_id = hex_field(&corpus, "recipient_key_id")?;
    let psk = hex_field(&corpus, "psk")?;
    let psk_id = hex_field(&corpus, "psk_id")?;
    let request_envelope = hex_field(&corpus, "request_envelope")?;
    let replay_id = hex_nested(&corpus, "intermediates", "replay_id")?;

    let server = Server::new(
        &recipient_private_key,
        recipient_key_id,
        hpke_http::Limits::default(),
    )?;
    let preparsed = server.preparse(&request_envelope)?;
    assert_eq!(preparsed.credential.psk_id, psk_id);
    let issued_at_unix_s = corpus
        .get("issued_at_unix_s")
        .and_then(Value::as_u64)
        .ok_or("missing issued_at_unix_s")?;
    let authenticated = server.authenticate_at(preparsed.token, &psk, issued_at_unix_s)?;
    assert_eq!(authenticated.replay.id.as_slice(), replay_id);
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    assert_eq!(opened.request, expected_request(&corpus)?);

    let response_nonce = hex_field(&corpus, "response_nonce")?;
    let mut entropy = FixedEntropy(response_nonce);
    let response_envelope = opened
        .response
        .protect_with_entropy(&expected_response(&corpus)?, &mut entropy)?;
    assert_eq!(response_envelope, hex_field(&corpus, "response_envelope")?);
    Ok(())
}

#[test]
fn rust_client_matches_frozen_sender_envelope() -> Result<(), Box<dyn std::error::Error>> {
    let corpus: Value = serde_json::from_str(CORPUS)?;
    let recipient_public_key = hex_field(&corpus, "recipient_public_key")?;
    let recipient_key_id = hex_field(&corpus, "recipient_key_id")?;
    let psk = hex_field(&corpus, "psk")?;
    let psk_id = hex_field(&corpus, "psk_id")?;
    let issued_at_unix_s = corpus
        .get("issued_at_unix_s")
        .and_then(Value::as_u64)
        .ok_or("missing issued_at_unix_s")?;
    let mut entropy = FixedEntropy(hex_field(&corpus, "sender_entropy_seed")?);
    let client = Client::new(
        &recipient_public_key,
        recipient_key_id,
        psk,
        psk_id,
        Limits::default(),
    )?;
    let protected = client.protect_at_with_entropy(
        &expected_request(&corpus)?,
        issued_at_unix_s,
        &mut entropy,
    )?;
    assert_eq!(
        protected.envelope(),
        hex_field(&corpus, "sender_request_envelope")?
    );
    Ok(())
}

fn expected_request(corpus: &Value) -> Result<Request, Box<dyn std::error::Error>> {
    let request = object(corpus, "request")?;
    Ok(Request {
        method: match text_value(request, "method")? {
            "POST" => Method::Post,
            _ => return Err("unsupported corpus request method".into()),
        },
        authority: text_value(request, "authority")?.as_bytes().to_vec(),
        path: text_value(request, "path")?.as_bytes().to_vec(),
        headers: headers(request)?,
        body: decode_hex(text_value(request, "body")?)?,
    })
}

fn expected_response(corpus: &Value) -> Result<Response, Box<dyn std::error::Error>> {
    let response = object(corpus, "response")?;
    let status = response
        .get("status")
        .and_then(Value::as_u64)
        .ok_or("missing response status")?;
    Ok(Response {
        status: u16::try_from(status)?,
        headers: headers(response)?,
        body: decode_hex(text_value(response, "body")?)?,
    })
}

fn headers(value: &Value) -> Result<Vec<HeaderField>, Box<dyn std::error::Error>> {
    value
        .get("headers")
        .and_then(Value::as_array)
        .ok_or_else(|| -> Box<dyn std::error::Error> { "missing headers".into() })?
        .iter()
        .map(|field| {
            let pair = field.as_array().ok_or("invalid header pair")?;
            if pair.len() != 2 {
                return Err("invalid header pair length".into());
            }
            Ok(HeaderField {
                name: pair[0]
                    .as_str()
                    .ok_or("invalid header name")?
                    .as_bytes()
                    .to_vec(),
                value: pair[1]
                    .as_str()
                    .ok_or("invalid header value")?
                    .as_bytes()
                    .to_vec(),
            })
        })
        .collect()
}

fn object<'a>(value: &'a Value, field: &str) -> Result<&'a Value, Box<dyn std::error::Error>> {
    value
        .get(field)
        .filter(|nested| nested.is_object())
        .ok_or_else(|| format!("missing object {field}").into())
}

fn text<'a>(value: &'a Value, field: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    text_value(value, field)
}

fn text_value<'a>(value: &'a Value, field: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing text {field}").into())
}

fn hex_field(value: &Value, field: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    decode_hex(text(value, field)?)
}

fn hex_nested(
    value: &Value,
    parent: &str,
    field: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    decode_hex(text_value(object(value, parent)?, field)?)
}

fn decode_hex(value: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if !value.len().is_multiple_of(2) {
        return Err("odd hex input".into());
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair)?;
            Ok(u8::from_str_radix(text, 16)?)
        })
        .collect()
}

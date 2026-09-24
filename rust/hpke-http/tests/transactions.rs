//! End-to-end protocol version 3 transaction and rejection tests.

use hpke_http::{
    Client, EntropySource, Error, HeaderField, Limits, Method, Request, Response, Server,
    generate_key_pair,
};

const KEY_ID: &[u8] = b"primary-2026-09";
const PSK: &[u8] = b"a 32-byte minimum test credential!";
const PSK_ID: &[u8] = b"tenant-42";

fn engines() -> Result<(Client, Server), Error> {
    let key_pair = generate_key_pair()?;
    let public_key = key_pair.public_key().to_vec();
    let (private_key, _) = key_pair.into_parts();
    let client = Client::new(
        &public_key,
        KEY_ID.to_vec(),
        PSK.to_vec(),
        PSK_ID.to_vec(),
        Limits::default(),
    )?;
    let server = Server::new(&private_key, KEY_ID.to_vec(), Limits::default())?;
    Ok((client, server))
}

fn request(body: &[u8]) -> Request {
    Request {
        method: Method::Post,
        authority: b"api.example.test".to_vec(),
        path: b"/v3/items?limit=2".to_vec(),
        headers: vec![HeaderField {
            name: b"content-type".to_vec(),
            value: b"application/json".to_vec(),
        }],
        body: body.to_vec(),
    }
}

fn response() -> Response {
    Response {
        status: 201,
        headers: vec![HeaderField {
            name: b"content-type".to_vec(),
            value: b"application/json".to_vec(),
        }],
        body: br#"{"id":"item-1"}"#.to_vec(),
    }
}

#[test]
fn complete_request_and_response_round_trip() -> Result<(), Error> {
    let (client, server) = engines()?;
    let protected = client.protect(&request(br#"{"name":"Ada"}"#))?;
    let (envelope, response_token) = protected.into_parts();

    let preparsed = server.preparse(&envelope)?;
    assert_eq!(preparsed.credential.psk_id, PSK_ID);
    let authenticated = server.authenticate(preparsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    assert_eq!(opened.request, request(br#"{"name":"Ada"}"#));

    let protected_response = opened.response.protect_finite(&response())?;
    assert_eq!(response_token.open_finite(&protected_response)?, response());
    Ok(())
}

#[test]
fn owned_envelope_authenticates_without_changing_the_transaction() -> Result<(), Error> {
    let (client, server) = engines()?;
    let expected = request(&vec![0x42; 1024 * 1024]);
    let (envelope, response_token) = client.protect(&expected)?.into_parts();
    let preparsed = server.preparse_owned(envelope)?;
    assert_eq!(preparsed.credential.psk_id, PSK_ID);
    let authenticated = server.authenticate(preparsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    assert_eq!(opened.request, expected);
    let protected_response = opened.response.protect_finite(&response())?;
    assert_eq!(response_token.open_finite(&protected_response)?, response());
    Ok(())
}

#[test]
fn repeated_fields_and_exact_content_lengths_round_trip() -> Result<(), Error> {
    let (client, server) = engines()?;
    let mut expected_request = request(b"body");
    expected_request.headers.extend([
        HeaderField {
            name: b"x-value".to_vec(),
            value: b"one".to_vec(),
        },
        HeaderField {
            name: b"x-value".to_vec(),
            value: b"two".to_vec(),
        },
        HeaderField {
            name: b"content-length".to_vec(),
            value: b"4".to_vec(),
        },
    ]);
    let protected = client.protect(&expected_request)?;
    let (envelope, response_token) = protected.into_parts();
    let preparsed = server.preparse(&envelope)?;
    let authenticated = server.authenticate(preparsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    assert_eq!(opened.request, expected_request);

    let expected_response = Response {
        status: 200,
        headers: vec![
            HeaderField {
                name: b"set-cookie".to_vec(),
                value: b"first=1".to_vec(),
            },
            HeaderField {
                name: b"set-cookie".to_vec(),
                value: b"second=2".to_vec(),
            },
            HeaderField {
                name: b"content-length".to_vec(),
                value: b"2".to_vec(),
            },
        ],
        body: b"ok".to_vec(),
    };
    let envelope = opened.response.protect_finite(&expected_response)?;
    assert_eq!(response_token.open_finite(&envelope)?, expected_response);
    Ok(())
}

#[test]
fn head_and_not_modified_preserve_content_length_metadata() -> Result<(), Error> {
    let (client, server) = engines()?;
    for (method, status) in [(Method::Head, 200), (Method::Get, 304)] {
        let protected = client.protect(&Request {
            method,
            authority: b"api.example.test".to_vec(),
            path: b"/metadata".to_vec(),
            headers: Vec::new(),
            body: Vec::new(),
        })?;
        let (envelope, response_token) = protected.into_parts();
        let preparsed = server.preparse(&envelope)?;
        let authenticated = server.authenticate(preparsed.token, PSK)?;
        let opened = authenticated
            .token
            .admit(authenticated.replay.decision(true))?;
        let expected = Response {
            status,
            headers: vec![HeaderField {
                name: b"content-length".to_vec(),
                value: b"123".to_vec(),
            }],
            body: Vec::new(),
        };
        let response = opened.response.protect_finite(&expected)?;
        assert_eq!(response_token.open_finite(&response)?, expected);
    }
    Ok(())
}

#[test]
fn empty_request_and_no_content_response_are_authenticated() -> Result<(), Error> {
    let (client, server) = engines()?;
    let protected = client.protect(&Request {
        method: Method::Get,
        authority: b"health.example.test".to_vec(),
        path: b"/health".to_vec(),
        headers: Vec::new(),
        body: Vec::new(),
    })?;
    let (envelope, response_token) = protected.into_parts();
    let preparsed = server.preparse(&envelope)?;
    let authenticated = server.authenticate(preparsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    assert!(opened.request.body.is_empty());

    let protected_response = opened.response.protect_finite(&Response {
        status: 204,
        headers: Vec::new(),
        body: Vec::new(),
    })?;
    let response = response_token.open_finite(&protected_response)?;
    assert_eq!(response.status, 204);
    assert!(response.body.is_empty());
    Ok(())
}

#[test]
fn replay_identity_is_stable_for_a_duplicate_envelope() -> Result<(), Error> {
    let (client, server) = engines()?;
    let protected = client.protect(&request(b"body"))?;
    let (envelope, _) = protected.into_parts();

    let first = server.preparse(&envelope)?;
    let first = server.authenticate(first.token, PSK)?;
    let duplicate = server.preparse(&envelope)?;
    let duplicate = server.authenticate(duplicate.token, PSK)?;
    assert_eq!(first.replay, duplicate.replay);
    assert_eq!(
        duplicate
            .token
            .admit(duplicate.replay.decision(false))
            .err(),
        Some(Error::ReplayRejected)
    );
    Ok(())
}

#[test]
fn authenticated_request_time_bounds_replay_retention() -> Result<(), Error> {
    let (client, server) = engines()?;
    let issued_at = 1_900_000_000;
    let envelope = client
        .protect_at(&request(b"body"), issued_at)?
        .envelope()
        .to_vec();

    let accepted = server.preparse(&envelope)?;
    let accepted = server.authenticate_at(accepted.token, PSK, issued_at + 329)?;
    assert_eq!(accepted.replay.retain_until_exclusive, issued_at + 330);

    let expired = server.preparse(&envelope)?;
    assert_eq!(
        server
            .authenticate_at(expired.token, PSK, issued_at + 330)
            .err(),
        Some(Error::InvalidRequestTime)
    );

    let future = server.preparse(&envelope)?;
    assert_eq!(
        server
            .authenticate_at(future.token, PSK, issued_at - 31)
            .err(),
        Some(Error::InvalidRequestTime)
    );
    Ok(())
}

#[test]
fn replay_admission_cannot_release_plaintext_at_or_after_expiry() -> Result<(), Error> {
    let (client, server) = engines()?;
    let issued_at = 1_900_000_000;
    let envelope = client
        .protect_at(&request(b"body"), issued_at)?
        .envelope()
        .to_vec();

    let before = server.preparse(&envelope)?;
    let before = server.authenticate_at(before.token, PSK, issued_at)?;
    let deadline = before.replay.retain_until_exclusive;
    let _ = before
        .token
        .admit_at(before.replay.decision(true), deadline - 1)?;

    let expired = server.preparse(&envelope)?;
    let expired = server.authenticate_at(expired.token, PSK, issued_at)?;
    assert_eq!(
        expired
            .token
            .admit_at(expired.replay.decision(true), deadline)
            .err(),
        Some(Error::InvalidRequestTime)
    );
    Ok(())
}

#[test]
fn replay_decision_is_bound_to_its_authenticated_request() -> Result<(), Error> {
    let (client, server) = engines()?;
    let first = client.protect(&request(b"first"))?;
    let first = server.preparse(first.envelope())?;
    let first = server.authenticate(first.token, PSK)?;

    let second = client.protect(&request(b"second"))?;
    let second = server.preparse(second.envelope())?;
    let second = server.authenticate(second.token, PSK)?;

    assert_eq!(
        second.token.admit(first.replay.decision(true)).err(),
        Some(Error::ReplayDecisionMismatch)
    );
    Ok(())
}

#[test]
fn wrong_psk_fails_before_plaintext_release() -> Result<(), Error> {
    let (client, server) = engines()?;
    let protected = client.protect(&request(b"secret body"))?;
    let preparsed = server.preparse(protected.envelope())?;
    assert_eq!(
        server
            .authenticate(preparsed.token, b"wrong credential with enough bytes!!")
            .err(),
        Some(Error::AuthenticationFailed)
    );
    Ok(())
}

#[test]
fn malformed_and_tampered_envelopes_are_rejected() -> Result<(), Error> {
    let (client, server) = engines()?;
    let protected = client.protect(&request(b"secret body"))?;
    let (envelope, _) = protected.into_parts();

    assert_eq!(
        server.preparse(&envelope[..20]).err(),
        Some(Error::MalformedEnvelope)
    );

    let mut trailing = envelope.clone();
    trailing.push(0);
    let preparsed = server.preparse(&trailing)?;
    assert_eq!(
        server.authenticate(preparsed.token, PSK).err(),
        Some(Error::MalformedEnvelope)
    );

    let mut tampered = envelope.clone();
    if let Some(last) = tampered.last_mut() {
        *last ^= 1;
    }
    let preparsed = server.preparse(&tampered)?;
    assert_eq!(
        server.authenticate(preparsed.token, PSK).err(),
        Some(Error::AuthenticationFailed)
    );
    Ok(())
}

#[test]
fn unsupported_suite_is_rejected_before_credential_lookup() -> Result<(), Error> {
    let (client, server) = engines()?;
    let protected = client.protect(&request(b"body"))?;
    let mut envelope = protected.envelope().to_vec();
    envelope[9] ^= 1;
    assert_eq!(
        server.preparse(&envelope).err(),
        Some(Error::UnsupportedSuite)
    );
    Ok(())
}

struct FailingEntropy;

impl EntropySource for FailingEntropy {
    fn fill(&mut self, _destination: &mut [u8]) -> Result<(), Error> {
        Err(Error::EntropyUnavailable)
    }
}

struct FixedEntropy(u8);

impl EntropySource for FixedEntropy {
    fn fill(&mut self, destination: &mut [u8]) -> Result<(), Error> {
        destination.fill(self.0);
        Ok(())
    }
}

#[test]
fn request_entropy_failure_returns_before_output() -> Result<(), Error> {
    let (client, _) = engines()?;
    assert_eq!(
        client
            .protect_with_entropy(&request(b"body"), &mut FailingEntropy)
            .err(),
        Some(Error::EntropyUnavailable)
    );
    Ok(())
}

#[test]
fn response_uses_fresh_injected_nonce_and_reports_entropy_failure() -> Result<(), Error> {
    let (client, server) = engines()?;

    let first = client.protect(&request(b"one"))?;
    let (first_envelope, _) = first.into_parts();
    let first = server.preparse(&first_envelope)?;
    let first = server.authenticate(first.token, PSK)?;
    let first = first.token.admit(first.replay.decision(true))?;
    let first_response = first
        .response
        .protect_finite_with_entropy(&response(), &mut FixedEntropy(1))?;
    assert_eq!(&first_response[5..37], &[1; 32]);

    let second = client.protect(&request(b"two"))?;
    let (second_envelope, _) = second.into_parts();
    let second = server.preparse(&second_envelope)?;
    let second = server.authenticate(second.token, PSK)?;
    let second = second.token.admit(second.replay.decision(true))?;
    let second_response = second
        .response
        .protect_finite_with_entropy(&response(), &mut FixedEntropy(2))?;
    assert_eq!(&second_response[5..37], &[2; 32]);
    assert_ne!(first_response, second_response);

    let third = client.protect(&request(b"three"))?;
    let (third_envelope, _) = third.into_parts();
    let third = server.preparse(&third_envelope)?;
    let third = server.authenticate(third.token, PSK)?;
    let third = third.token.admit(third.replay.decision(true))?;
    assert_eq!(
        third
            .response
            .protect_finite_with_entropy(&response(), &mut FailingEntropy)
            .err(),
        Some(Error::EntropyUnavailable)
    );
    Ok(())
}

#[test]
fn invalid_headers_and_content_lengths_are_rejected_before_protection() -> Result<(), Error> {
    let (client, server) = engines()?;
    let mut invalid = request(b"body");
    invalid.headers[0].name = b"Content-Type".to_vec();
    assert_eq!(
        client.protect(&invalid).err(),
        Some(Error::InvalidConfiguration)
    );

    let mut framing = request(b"body");
    framing.headers[0] = HeaderField {
        name: b"content-length".to_vec(),
        value: b"3".to_vec(),
    };
    assert_eq!(
        client.protect(&framing).err(),
        Some(Error::InvalidConfiguration)
    );

    framing.headers = vec![
        HeaderField {
            name: b"content-length".to_vec(),
            value: b"4".to_vec(),
        },
        HeaderField {
            name: b"content-length".to_vec(),
            value: b"4".to_vec(),
        },
    ];
    assert_eq!(
        client.protect(&framing).err(),
        Some(Error::InvalidConfiguration)
    );

    let protected = client.protect(&request(b"body"))?;
    let preparsed = server.preparse(protected.envelope())?;
    let authenticated = server.authenticate(preparsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    assert_eq!(
        opened
            .response
            .protect_finite(&Response {
                status: 204,
                headers: vec![HeaderField {
                    name: b"content-length".to_vec(),
                    value: b"0".to_vec(),
                }],
                body: Vec::new(),
            })
            .err(),
        Some(Error::InvalidConfiguration)
    );
    Ok(())
}

#[test]
fn secret_cannot_be_its_public_identifier() -> Result<(), Error> {
    let key_pair = generate_key_pair()?;
    assert_eq!(
        Client::new(
            key_pair.public_key(),
            KEY_ID.to_vec(),
            PSK.to_vec(),
            PSK.to_vec(),
            Limits::default(),
        )
        .err(),
        Some(Error::InvalidConfiguration)
    );
    Ok(())
}

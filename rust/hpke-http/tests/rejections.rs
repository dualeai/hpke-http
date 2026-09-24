//! Boundary, mutation, and cross-context rejection tests for protocol version 3.

#[test]
fn server_public_key_matches_its_private_key() -> Result<(), Error> {
    let pair = generate_key_pair()?;
    let expected = pair.public_key().to_vec();
    let (private_key, _) = pair.into_parts();
    let server = Server::new(&private_key, KEY_ID.to_vec(), Limits::default())?;
    assert_eq!(server.public_key(), expected);
    Ok(())
}

use hpke_http::{
    Client, EntropySource, Error, HARD_MAX_BODY_LEN, HeaderField, Limits, Method, Request,
    Response, ResponseToken, Server, generate_key_pair, generate_key_pair_with_entropy,
};

const KEY_ID: &[u8] = b"primary-2026-09";
const PSK: &[u8] = b"a 32-byte minimum test credential!";
const PSK_ID: &[u8] = b"tenant-42";

fn engines_with_limits(limits: Limits) -> Result<(Client, Server), Error> {
    let key_pair = generate_key_pair()?;
    let public_key = key_pair.public_key().to_vec();
    let (private_key, _) = key_pair.into_parts();
    Ok((
        Client::new(
            &public_key,
            KEY_ID.to_vec(),
            PSK.to_vec(),
            PSK_ID.to_vec(),
            limits,
        )?,
        Server::new(&private_key, KEY_ID.to_vec(), limits)?,
    ))
}

fn request(method: Method, body: &[u8]) -> Request {
    Request {
        method,
        authority: b"api.example.test".to_vec(),
        path: b"/items?limit=2".to_vec(),
        headers: vec![HeaderField {
            name: b"content-type".to_vec(),
            value: b"application/octet-stream".to_vec(),
        }],
        body: body.to_vec(),
    }
}

fn response(status: u16) -> Response {
    Response {
        status,
        headers: vec![HeaderField {
            name: b"content-type".to_vec(),
            value: b"application/octet-stream".to_vec(),
        }],
        body: b"response".to_vec(),
    }
}

fn protect_response(
    client: &Client,
    server: &Server,
    status: u16,
) -> Result<(ResponseToken, Vec<u8>), Error> {
    let protected = client.protect(&request(Method::Post, b"request"))?;
    let (envelope, response_token) = protected.into_parts();
    let preparsed = server.preparse(&envelope)?;
    let authenticated = server.authenticate(preparsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    let envelope = opened.response.protect_finite(&response(status))?;
    Ok((response_token, envelope))
}

fn request_is_rejected(server: &Server, envelope: &[u8]) -> bool {
    match server.preparse(envelope) {
        Ok(preparsed) => match server.authenticate(preparsed.token, PSK) {
            Ok(authenticated) => authenticated
                .token
                .admit(authenticated.replay.decision(true))
                .is_err(),
            Err(_) => true,
        },
        Err(_) => true,
    }
}

#[test]
fn every_supported_method_and_status_boundary_round_trips() -> Result<(), Error> {
    let (client, server) = engines_with_limits(Limits::default())?;
    let methods = [
        Method::Get,
        Method::Post,
        Method::Put,
        Method::Patch,
        Method::Delete,
        Method::Head,
        Method::Options,
    ];
    let statuses = [200, 204, 299, 300, 399, 400, 499, 500, 599];

    for (index, method) in methods.into_iter().enumerate() {
        let expected_request = request(method, &[u8::try_from(index).unwrap_or_default()]);
        let protected = client.protect(&expected_request)?;
        let (envelope, response_token) = protected.into_parts();
        let preparsed = server.preparse(&envelope)?;
        let authenticated = server.authenticate(preparsed.token, PSK)?;
        let opened = authenticated
            .token
            .admit(authenticated.replay.decision(true))?;
        assert_eq!(opened.request, expected_request);

        let mut expected_response = response(statuses[index]);
        if method == Method::Head || matches!(expected_response.status, 204 | 205 | 304) {
            expected_response.body.clear();
        }
        let envelope = opened.response.protect_finite(&expected_response)?;
        assert_eq!(response_token.open_finite(&envelope)?, expected_response);
    }
    Ok(())
}

#[test]
fn bodyless_http_responses_are_rejected_in_the_shared_engine() -> Result<(), Error> {
    let (client, server) = engines_with_limits(Limits::default())?;
    for (method, status) in [
        (Method::Head, 200),
        (Method::Get, 204),
        (Method::Get, 205),
        (Method::Get, 304),
    ] {
        let protected = client.protect(&request(method, b"request"))?;
        let preparsed = server.preparse(protected.envelope())?;
        let authenticated = server.authenticate(preparsed.token, PSK)?;
        let opened = authenticated
            .token
            .admit(authenticated.replay.decision(true))?;
        assert_eq!(
            opened.response.protect_finite(&response(status)).err(),
            Some(Error::InvalidConfiguration)
        );
    }
    Ok(())
}

#[test]
fn request_prefix_errors_are_classified_before_credentials() -> Result<(), Error> {
    let (client, server) = engines_with_limits(Limits::default())?;
    let envelope = client
        .protect(&request(Method::Post, b"body"))?
        .envelope()
        .to_vec();
    let cases = [
        (0, Error::MalformedEnvelope),
        (4, Error::UnsupportedVersion),
        (7, Error::UnsupportedSuite),
        (9, Error::UnsupportedSuite),
        (11, Error::UnsupportedSuite),
    ];
    for (offset, expected) in cases {
        let mut mutated = envelope.clone();
        mutated[offset] ^= 1;
        assert_eq!(server.preparse(&mutated).err(), Some(expected));
    }

    for offset in [5, 6] {
        let mut empty_id = envelope.clone();
        empty_id[offset] = 0;
        assert_eq!(
            server.preparse(&empty_id).err(),
            Some(Error::MalformedEnvelope)
        );
    }
    Ok(())
}

#[test]
fn every_request_truncation_and_single_byte_mutation_is_rejected() -> Result<(), Error> {
    let (client, server) = engines_with_limits(Limits::default())?;
    let envelope = client
        .protect_with_entropy(
            &request(Method::Post, b"sensitive body"),
            &mut FixedEntropy(7),
        )?
        .envelope()
        .to_vec();

    for length in 0..envelope.len() {
        assert!(request_is_rejected(&server, &envelope[..length]));
    }
    for offset in 0..envelope.len() {
        let mut mutated = envelope.clone();
        mutated[offset] ^= 1;
        assert!(request_is_rejected(&server, &mutated));
    }
    Ok(())
}

#[test]
fn one_shot_preparse_bounds_wire_before_it_clones() -> Result<(), Error> {
    let limits = Limits {
        max_body_len: 32,
        ..Limits::default()
    };
    let (_, server) = engines_with_limits(limits)?;
    let oversized = vec![0_u8; 50_000];
    assert_eq!(
        server.preparse(&oversized).err(),
        Some(Error::LimitExceeded)
    );
    assert_eq!(
        server.preparse_owned(oversized).err(),
        Some(Error::LimitExceeded)
    );
    Ok(())
}

#[test]
fn every_response_truncation_and_single_byte_mutation_is_rejected() -> Result<(), Error> {
    let (client, server) = engines_with_limits(Limits::default())?;
    let (_, reference) = protect_response(&client, &server, 200)?;

    for length in 0..reference.len() {
        let (token, envelope) = protect_response(&client, &server, 200)?;
        assert_eq!(envelope.len(), reference.len());
        assert!(token.open_finite(&envelope[..length]).is_err());
    }
    for offset in 0..reference.len() {
        let (token, mut envelope) = protect_response(&client, &server, 200)?;
        envelope[offset] ^= 1;
        assert!(token.open_finite(&envelope).is_err());
    }
    Ok(())
}

#[test]
fn response_tokens_reject_cross_request_envelopes() -> Result<(), Error> {
    let (client, server) = engines_with_limits(Limits::default())?;
    let (first_token, first_envelope) = protect_response(&client, &server, 200)?;
    let (second_token, second_envelope) = protect_response(&client, &server, 201)?;

    assert_eq!(
        first_token.open_finite(&second_envelope).err(),
        Some(Error::AuthenticationFailed)
    );
    assert_eq!(
        second_token.open_finite(&first_envelope).err(),
        Some(Error::AuthenticationFailed)
    );
    Ok(())
}

#[test]
fn configured_limits_apply_at_exact_boundaries() -> Result<(), Error> {
    let limits = Limits {
        max_body_len: 3,
        max_header_bytes: 7,
        max_header_count: 1,
        max_target_len: 5,
        max_request_bytes: 3,
    };
    let (client, _) = engines_with_limits(limits)?;
    let accepted = Request {
        method: Method::Options,
        authority: b"a.io".to_vec(),
        path: b"*".to_vec(),
        headers: vec![HeaderField {
            name: b"x".to_vec(),
            value: b"123456".to_vec(),
        }],
        body: b"123".to_vec(),
    };
    let _ = client.protect(&accepted)?;

    let mut too_large = accepted.clone();
    too_large.body.push(4);
    assert_eq!(client.protect(&too_large).err(), Some(Error::LimitExceeded));
    too_large = accepted.clone();
    too_large.headers[0].value.push(b'7');
    assert_eq!(client.protect(&too_large).err(), Some(Error::LimitExceeded));
    too_large = accepted.clone();
    too_large.headers.push(HeaderField {
        name: b"y".to_vec(),
        value: Vec::new(),
    });
    assert_eq!(client.protect(&too_large).err(), Some(Error::LimitExceeded));
    too_large = accepted;
    too_large.path = b"/x".to_vec();
    assert_eq!(client.protect(&too_large).err(), Some(Error::LimitExceeded));
    Ok(())
}

#[test]
fn invalid_configuration_is_rejected_before_protection() -> Result<(), Error> {
    let key_pair = generate_key_pair()?;
    let invalid_limits = Limits {
        max_body_len: HARD_MAX_BODY_LEN + 1,
        ..Limits::default()
    };
    assert_eq!(
        invalid_limits.validate().err(),
        Some(Error::InvalidConfiguration)
    );

    for invalid_psk in [b"short".as_slice(), PSK_ID] {
        assert!(
            Client::new(
                key_pair.public_key(),
                KEY_ID.to_vec(),
                invalid_psk.to_vec(),
                PSK_ID.to_vec(),
                Limits::default(),
            )
            .is_err()
        );
    }
    assert!(
        Client::new(
            key_pair.public_key(),
            Vec::new(),
            PSK.to_vec(),
            PSK_ID.to_vec(),
            Limits::default(),
        )
        .is_err()
    );
    assert!(Server::new(&[0; 31], KEY_ID.to_vec(), Limits::default()).is_err());
    Ok(())
}

#[test]
fn authority_syntax_is_validated_without_normalization() -> Result<(), Error> {
    let (client, _) = engines_with_limits(Limits::default())?;
    let mut invalid = request(Method::Post, b"body");
    for authority in [
        b"".as_slice(),
        b"user@example.test",
        b"example.test/path",
        b"::1",
        b"[broken",
        b"[]",
        b"[127.0.0.1]",
        b"host:abc",
        b"host:",
        b"host:65536",
        b"bad%2",
        b"999.999.999.999",
    ] {
        invalid.authority = authority.to_vec();
        assert_eq!(
            client.protect(&invalid).err(),
            Some(Error::InvalidConfiguration)
        );
    }

    for authority in [
        b"example.test".as_slice(),
        b"example.test:443",
        b"127.0.0.1",
        b"127.0.0.1:8443",
        b"[2001:db8::1]",
        b"[2001:db8::1]:443",
        b"[v1.fe80]",
        b"%65xample.test",
    ] {
        let mut valid = request(Method::Get, b"");
        valid.authority = authority.to_vec();
        let _ = client.protect(&valid)?;
    }

    Ok(())
}

#[test]
fn every_profile_forbidden_field_is_rejected() -> Result<(), Error> {
    let (client, _) = engines_with_limits(Limits::default())?;
    for name in [
        b"connection".as_slice(),
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
    ] {
        let mut invalid = request(Method::Post, b"body");
        invalid.headers = vec![HeaderField {
            name: name.to_vec(),
            value: b"field-value".to_vec(),
        }];
        assert_eq!(
            client.protect(&invalid).err(),
            Some(Error::InvalidConfiguration),
            "{name:?} must not cross the whole-message boundary"
        );
    }
    Ok(())
}

#[test]
fn invalid_http_values_are_rejected() -> Result<(), Error> {
    let (client, server) = engines_with_limits(Limits::default())?;
    let mut invalid = request(Method::Post, b"body");

    for path in [
        b"relative".as_slice(),
        b"/fragment#value",
        b"/line\nbreak",
        b"/back\\slash",
        b"/bad%2",
        b"/bad%xx",
        b"/a/../b",
        b"/a/%2E/b",
        b"/%FF",
        b"/%00",
        b"/%5cadmin",
        b"/safe/%2e%2e%2fadmin",
    ] {
        invalid.path = path.to_vec();
        assert_eq!(
            client.protect(&invalid).err(),
            Some(Error::InvalidConfiguration)
        );
    }

    invalid = request(Method::Post, b"body");
    invalid.path = b"/caf%C3%A9?next=%2Fitems".to_vec();
    let _ = client.protect(&invalid)?;

    invalid.path = b"/literal/%252e".to_vec();
    let _ = client.protect(&invalid)?;

    invalid = request(Method::Get, b"body");
    invalid.path = b"*".to_vec();
    assert_eq!(
        client.protect(&invalid).err(),
        Some(Error::InvalidConfiguration)
    );

    invalid = request(Method::Post, b"body");
    invalid.headers = vec![HeaderField {
        name: b"x-test".to_vec(),
        value: b"unsafe\r\nvalue".to_vec(),
    }];
    assert_eq!(
        client.protect(&invalid).err(),
        Some(Error::InvalidConfiguration)
    );

    for headers in [
        vec![HeaderField {
            name: b"x-test".to_vec(),
            value: b" padded".to_vec(),
        }],
        vec![HeaderField {
            name: b"x-test".to_vec(),
            value: "snowman: ☃".as_bytes().to_vec(),
        }],
    ] {
        invalid = request(Method::Post, b"body");
        invalid.headers = headers;
        assert_eq!(
            client.protect(&invalid).err(),
            Some(Error::InvalidConfiguration)
        );
    }

    let protected = client.protect(&request(Method::Post, b"body"))?;
    let (envelope, _) = protected.into_parts();
    let preparsed = server.preparse(&envelope)?;
    let authenticated = server.authenticate(preparsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    assert_eq!(
        opened.response.protect_finite(&response(199)).err(),
        Some(Error::InvalidConfiguration)
    );
    Ok(())
}

#[test]
fn key_generation_propagates_entropy_failure() {
    assert_eq!(
        generate_key_pair_with_entropy(&mut FailingEntropy).err(),
        Some(Error::EntropyUnavailable)
    );
}

#[test]
fn debug_output_redacts_private_keys_and_credentials() -> Result<(), Error> {
    let key_pair = generate_key_pair()?;
    let key_debug = format!("{key_pair:?}");
    assert!(key_debug.contains("[REDACTED]"));

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
    assert!(!format!("{client:?}").contains("minimum test credential"));
    assert!(format!("{server:?}").contains("[REDACTED]"));
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

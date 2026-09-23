//! Public transaction behavior of the opt-in authenticated body-coding extension.

use hpke_http::{
    Client, CompressionCoding, Error, HeaderField, Limits, Method, Request, Response, Server,
    generate_key_pair,
};

const KEY_ID: &[u8] = b"compression-key";
const PSK: &[u8] = b"a 32-byte minimum compression credential";
const PSK_ID: &[u8] = b"compression-tenant";

fn engines(client_limits: Limits, server_limits: Limits) -> Result<(Client, Server), Error> {
    let keys = generate_key_pair()?;
    let (private, public) = keys.into_parts();
    let client = Client::new(
        &public,
        KEY_ID.to_vec(),
        PSK.to_vec(),
        PSK_ID.to_vec(),
        client_limits,
    )?;
    let server = Server::new(&private, KEY_ID.to_vec(), server_limits)?;
    Ok((client, server))
}

fn request(body: Vec<u8>) -> Request {
    Request {
        method: Method::Post,
        authority: b"api.example.test".to_vec(),
        path: b"/compressed".to_vec(),
        headers: vec![HeaderField {
            name: b"content-length".to_vec(),
            value: body.len().to_string().into_bytes(),
        }],
        body,
    }
}

fn response(body: Vec<u8>) -> Response {
    Response {
        status: 200,
        headers: vec![HeaderField {
            name: b"content-length".to_vec(),
            value: body.len().to_string().into_bytes(),
        }],
        body,
    }
}

#[test]
fn gzip_and_zstd_round_trip_with_logical_content_length() -> Result<(), Error> {
    for coding in [CompressionCoding::Gzip, CompressionCoding::Zstd] {
        let (client, server) = engines(Limits::default(), Limits::default())?;
        let client = client.with_compression(coding);
        let server = server.with_compression();
        let expected_request = request(vec![b'a'; 16 * 1024]);
        let protected = client.protect(&expected_request)?;
        let envelope_len = protected.envelope().len();
        assert!(envelope_len < expected_request.body.len());
        let (envelope, response_token) = protected.into_parts();
        let preparsed = server.preparse(&envelope)?;
        let authenticated = server.authenticate(preparsed.token, PSK)?;
        let opened = authenticated
            .token
            .admit(authenticated.replay.decision(true))?;
        assert_eq!(opened.request, expected_request);

        let expected_response = response(vec![b'b'; 16 * 1024]);
        let protected_response = opened.response.protect(&expected_response)?;
        assert!(protected_response.len() < expected_response.body.len());
        assert_eq!(response_token.open(&protected_response)?, expected_response);
    }
    Ok(())
}

#[test]
fn opt_in_rejects_old_server_and_old_client_still_works_with_new_server() -> Result<(), Error> {
    let (client, server) = engines(Limits::default(), Limits::default())?;
    let old_request = client.protect(&request(vec![b'a'; 4096]))?;
    let preparsed = server.preparse(old_request.envelope())?;
    let authenticated = server.authenticate(preparsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    let old_response = opened.response.protect(&response(vec![b'b'; 4096]))?;
    assert_eq!(
        old_request.into_parts().1.open(&old_response)?,
        response(vec![b'b'; 4096])
    );

    let (client, server) = engines(Limits::default(), Limits::default())?;
    let new_request = client
        .with_compression(CompressionCoding::Gzip)
        .protect(&request(Vec::new()))?;
    let preparsed = server.preparse(new_request.envelope())?;
    assert_eq!(
        server.authenticate(preparsed.token, PSK).err(),
        Some(Error::MalformedEnvelope)
    );
    Ok(())
}

#[test]
fn decoded_request_limit_is_applied_only_after_replay_admission() -> Result<(), Error> {
    let small = Limits {
        max_body_len: 128,
        ..Limits::default()
    };
    let (client, server) = engines(Limits::default(), small)?;
    let envelope = client
        .with_compression(CompressionCoding::Zstd)
        .protect(&request(vec![b'a'; 4096]))?
        .envelope()
        .to_vec();
    let server = server.with_compression();
    let preparsed = server.preparse(&envelope)?;
    let authenticated = server.authenticate(preparsed.token, PSK)?;
    assert_eq!(
        authenticated
            .token
            .admit(authenticated.replay.decision(false))
            .err(),
        Some(Error::ReplayRejected)
    );
    let preparsed = server.preparse(&envelope)?;
    let authenticated = server.authenticate(preparsed.token, PSK)?;
    assert_eq!(
        authenticated
            .token
            .admit(authenticated.replay.decision(true))
            .err(),
        Some(Error::LimitExceeded)
    );
    Ok(())
}

#[test]
fn decoded_response_limit_is_enforced() -> Result<(), Error> {
    let small = Limits {
        max_body_len: 128,
        ..Limits::default()
    };
    let (client, server) = engines(small, Limits::default())?;
    let protected = client
        .with_compression(CompressionCoding::Gzip)
        .protect(&request(Vec::new()))?;
    let (envelope, response_token) = protected.into_parts();
    let server = server.with_compression();
    let preparsed = server.preparse(&envelope)?;
    let authenticated = server.authenticate(preparsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    let envelope = opened.response.protect(&response(vec![b'b'; 4096]))?;
    assert_eq!(response_token.open(&envelope), Err(Error::LimitExceeded));
    Ok(())
}

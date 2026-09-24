//! v3 native zstd coding applies to clear DATA before encryption.

use hpke_http::{Client, Error, Limits, Method, Request, Response, Server, generate_key_pair};

const PSK: &[u8] = b"a 32-byte minimum upload credential";

fn pair(limits: Limits) -> Result<(Client, Server), Error> {
    let keys = generate_key_pair()?;
    Ok((
        Client::new(
            keys.public_key(),
            b"key".to_vec(),
            PSK.to_vec(),
            b"psk".to_vec(),
            limits,
        )?,
        Server::new(&keys.into_parts().0, b"key".to_vec(), limits)?,
    ))
}

fn request(body: Vec<u8>) -> Request {
    Request {
        method: Method::Post,
        authority: b"api.example.test".to_vec(),
        path: b"/coded".to_vec(),
        headers: Vec::new(),
        body,
    }
}

#[test]
fn request_and_reply_auto_compress_before_encryption() -> Result<(), Error> {
    let (client, server) = pair(Limits::default())?;
    let clear = request(vec![b'a'; 16 * 1024]);
    let protected = client.protect(&clear)?;
    assert!(protected.envelope().len() < clear.body.len());
    let parsed = server.preparse(protected.envelope())?;
    let authenticated = server.authenticate(parsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    assert_eq!(opened.request, clear);
    let reply = Response {
        status: 200,
        headers: Vec::new(),
        body: vec![b'b'; 16 * 1024],
    };
    let wire = opened.response.protect_finite(&reply)?;
    assert!(wire.len() < reply.body.len());
    assert_eq!(protected.into_parts().1.open_finite(&wire)?, reply);
    Ok(())
}

#[test]
fn short_payloads_use_raw_records() -> Result<(), Error> {
    let (client, server) = pair(Limits::default())?;
    let clear = request(b"short".to_vec());
    let protected = client.protect(&clear)?;
    let parsed = server.preparse(protected.envelope())?;
    let authenticated = server.authenticate(parsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    assert_eq!(opened.request, clear);
    Ok(())
}

#[test]
fn complete_server_helper_bounds_decoded_body_before_append() -> Result<(), Error> {
    let keys = generate_key_pair()?;
    let client_limits = Limits {
        max_body_len: 128 * 1024,
        ..Limits::default()
    };
    let server_limits = Limits {
        max_body_len: 32,
        ..Limits::default()
    };
    let client = Client::new(
        keys.public_key(),
        b"key".to_vec(),
        PSK.to_vec(),
        b"psk".to_vec(),
        client_limits,
    )?;
    let server = Server::new(&keys.into_parts().0, b"key".to_vec(), server_limits)?;
    let protected = client.protect(&request(vec![b'a'; 128 * 1024]))?;
    assert!(protected.envelope().len() < 1024);
    let parsed = server.preparse(protected.envelope())?;
    assert_eq!(
        server.authenticate(parsed.token, PSK).err(),
        Some(Error::LimitExceeded)
    );
    Ok(())
}

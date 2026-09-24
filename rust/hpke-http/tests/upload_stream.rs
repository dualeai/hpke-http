//! The one v3 request wire: START, DATA, END, and true outer EOF.

use hpke_http::{
    Client, EntropySource, Error, HeaderField, Limits, Method, Request, RequestHead, Response,
    Server, StreamRequestRecord, generate_key_pair,
};

const KEY_ID: &[u8] = b"upload-key";
const PSK_ID: &[u8] = b"upload-tenant";
const PSK: &[u8] = b"a 32-byte minimum upload credential";

fn pair(client_limits: Limits, server_limits: Limits) -> Result<(Client, Server), Error> {
    let keys = generate_key_pair()?;
    Ok((
        Client::new(
            keys.public_key(),
            KEY_ID.to_vec(),
            PSK.to_vec(),
            PSK_ID.to_vec(),
            client_limits,
        )?,
        Server::new(&keys.into_parts().0, KEY_ID.to_vec(), server_limits)?,
    ))
}

fn head(length: Option<usize>) -> RequestHead {
    RequestHead {
        method: Method::Post,
        authority: b"api.example.test".to_vec(),
        path: b"/upload".to_vec(),
        headers: length.map_or_else(Vec::new, |value| {
            vec![HeaderField {
                name: b"content-length".to_vec(),
                value: value.to_string().into_bytes(),
            }]
        }),
    }
}

fn opened_reader(server: &Server, first: &[u8]) -> Result<hpke_http::StreamRequestOpener, Error> {
    let preparsed = server.preparse_stream(first)?;
    let authenticated = server.authenticate_stream(preparsed.token, PSK)?;
    Ok(authenticated
        .token
        .admit(authenticated.replay.decision(true))?
        .reader)
}

fn append_push(
    writer: &mut hpke_http::StreamRequestSealer,
    data: &[u8],
    wire: &mut Vec<u8>,
) -> Result<(), Error> {
    let mut remaining = data;
    while !remaining.is_empty() {
        let (used, record) = writer.push(remaining)?;
        assert!(used > 0);
        remaining = &remaining[used..];
        if let Some(record) = record {
            wire.extend_from_slice(&record);
        }
    }
    Ok(())
}

#[test]
fn one_byte_pushes_coalesce_and_any_byte_cuts_open() -> Result<(), Error> {
    let (client, server) = pair(Limits::default(), Limits::default())?;
    let data = vec![b'a'; 64 * 1024 + 9];
    let expected = head(Some(data.len()));
    let (mut writer, first) = client.begin_stream(&expected)?;
    assert_eq!(first[4], 3);
    assert_eq!(
        first[5],
        u8::try_from(KEY_ID.len()).map_err(|_| Error::LimitExceeded)?
    );
    assert_eq!(server.stream_start_length(&first)?, Some(first.len()));
    let mut wire = Vec::new();
    for byte in &data {
        append_push(&mut writer, &[*byte], &mut wire)?;
    }
    let (tail, response_token) = writer.finish()?;
    wire.extend_from_slice(&tail);
    let mut reader = opened_reader(&server, &first)?;
    let mut checked = Vec::new();
    for byte in wire {
        let (used, record) = reader.feed(&[byte])?;
        assert_eq!(used, 1);
        if let Some(StreamRequestRecord::Data(part)) = record {
            checked.extend_from_slice(&part);
        }
    }
    assert_eq!(checked, data);
    let response = reader.finish_eof()?.protect_finite(&Response {
        status: 200,
        headers: Vec::new(),
        body: b"ok".to_vec(),
    })?;
    assert_eq!(response_token.open_finite(&response)?.body, b"ok");
    Ok(())
}

#[test]
fn empty_request_and_one_shot_wrapper_use_v3_stream() -> Result<(), Error> {
    let (client, server) = pair(Limits::default(), Limits::default())?;
    let request = Request {
        method: Method::Post,
        authority: b"api.example.test".to_vec(),
        path: b"/upload".to_vec(),
        headers: Vec::new(),
        body: Vec::new(),
    };
    let protected = client.protect(&request)?;
    let envelope = protected.envelope();
    assert_eq!(envelope[4], 3);
    assert_eq!(
        envelope[5],
        u8::try_from(KEY_ID.len()).map_err(|_| Error::LimitExceeded)?
    );
    let parsed = server.preparse(envelope)?;
    let authenticated = server.authenticate(parsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    assert_eq!(opened.request, request);
    assert_eq!(
        server.stream_start_length(envelope)?,
        Some(envelope.len() - 21)
    );
    Ok(())
}

#[test]
fn checked_content_length_limits_bad_late_data_and_end_fail() -> Result<(), Error> {
    let small = Limits {
        max_request_bytes: 4,
        ..Limits::default()
    };
    let (client, server) = pair(small, small)?;
    let (mut writer, _) = client.begin_stream(&head(Some(4)))?;
    assert_eq!(writer.push(b"abcde").err(), Some(Error::LimitExceeded));
    assert!(writer.finish().is_err());
    let (mut writer, first) = client.begin_stream(&head(Some(4)))?;
    append_push(&mut writer, b"abcd", &mut Vec::new())?;
    let (wire, _) = writer.finish()?;
    let mut reader = opened_reader(&server, &first)?;
    let mut bad = wire.clone();
    *bad.last_mut().ok_or(Error::MalformedEnvelope)? ^= 1;
    let (used, _) = reader.feed(&bad)?;
    assert_eq!(
        reader.feed(&bad[used..]).err(),
        Some(Error::AuthenticationFailed)
    );
    assert!(reader.finish_eof().is_err());
    let mut reader = opened_reader(&server, &first)?;
    let (used, record) = reader.feed(&wire)?;
    assert_eq!(record, Some(StreamRequestRecord::Data(b"abcd".to_vec())));
    assert_eq!(
        reader.feed(&wire[used..])?.1,
        Some(StreamRequestRecord::End)
    );
    assert_eq!(reader.feed(b"x").err(), Some(Error::MalformedEnvelope));
    let mut reader = opened_reader(&server, &first)?;
    reader.feed(&wire[..wire.len() - 1])?;
    assert_eq!(reader.finish_eof().err(), Some(Error::MalformedEnvelope));
    let (mut writer, _) = client.begin_stream(&head(Some(4)))?;
    append_push(&mut writer, b"abc", &mut Vec::new())?;
    assert_eq!(writer.finish().err(), Some(Error::InvalidConfiguration));
    Ok(())
}

#[derive(Default)]
struct FixedEntropy;
impl EntropySource for FixedEntropy {
    fn fill(&mut self, output: &mut [u8]) -> Result<(), Error> {
        output.fill(7);
        Ok(())
    }
}

#[test]
fn complete_wrapper_matches_stream_writer_bytes() -> Result<(), Error> {
    let (client, _) = pair(Limits::default(), Limits::default())?;
    let request = Request {
        method: Method::Post,
        authority: b"api.example.test".to_vec(),
        path: b"/upload".to_vec(),
        headers: Vec::new(),
        body: vec![b'z'; 100_000],
    };
    let complete = client.protect_at_with_entropy(&request, 1_000_000, &mut FixedEntropy)?;
    let head = RequestHead {
        method: request.method,
        authority: request.authority.clone(),
        path: request.path.clone(),
        headers: request.headers.clone(),
    };
    let (mut writer, mut stream) =
        client.begin_stream_at_with_entropy(&head, 1_000_000, &mut FixedEntropy)?;
    append_push(&mut writer, &request.body, &mut stream)?;
    let (tail, _) = writer.finish()?;
    stream.extend_from_slice(&tail);
    assert_eq!(complete.envelope(), stream);
    Ok(())
}

#[test]
fn unsupported_request_version_fails_at_public_prefix() -> Result<(), Error> {
    let (client, server) = pair(Limits::default(), Limits::default())?;
    let (_, mut first) = client.begin_stream(&head(None))?;
    first[4] = 2;
    assert_eq!(
        server.stream_start_length(&first),
        Err(Error::UnsupportedVersion)
    );
    assert_eq!(
        server.preparse(&first).err(),
        Some(Error::UnsupportedVersion)
    );
    Ok(())
}

//! `CodSpeed` measures the public Rust transaction without transport or fixture setup.

use hpke_http::{
    Client, Error, HeaderField, Limits, Method, Request, Response, ResponseRecord, Server,
    generate_key_pair,
};

fn main() {
    divan::main();
}

fn transaction(
    client: &Client,
    server: &Server,
    request: &Request,
    response: &Response,
    psk: &[u8],
) -> Result<usize, Error> {
    let (envelope, response_token) = client.protect(request)?.into_parts();
    let preparsed = server.preparse_owned(envelope)?;
    let authenticated = server.authenticate(preparsed.token, psk)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    let protected_response = opened.response.protect_finite(response)?;
    Ok(response_token.open_finite(&protected_response)?.body.len())
}

#[divan::bench(args = [0, 1024, 1024 * 1024, 8 * 1024 * 1024])]
fn roundtrip(bencher: divan::Bencher, size: usize) {
    bench_roundtrip(bencher, size);
}

fn bench_roundtrip(bencher: divan::Bencher, size: usize) {
    let keys = match generate_key_pair() {
        Ok(keys) => keys,
        Err(error) => {
            eprintln!("benchmark setup failed: {error}");
            std::process::exit(1);
        }
    };
    let (private_key, public_key) = keys.into_parts();
    let key_id = b"benchmark-key".to_vec();
    let psk = b"a 32-byte minimum benchmark credential".to_vec();
    let client = match Client::new(
        &public_key,
        key_id.clone(),
        psk.clone(),
        b"benchmark-tenant".to_vec(),
        Limits::default(),
    ) {
        Ok(client) => client,
        Err(error) => {
            eprintln!("benchmark setup failed: {error}");
            std::process::exit(1);
        }
    };
    let server = match Server::new(&private_key, key_id, Limits::default()) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("benchmark setup failed: {error}");
            std::process::exit(1);
        }
    };
    let request = Request {
        method: Method::Post,
        authority: b"api.example.test".to_vec(),
        path: b"/benchmark".to_vec(),
        headers: Vec::new(),
        body: vec![0x42; size],
    };
    let response = Response {
        status: 200,
        headers: Vec::new(),
        body: vec![0x43; size],
    };

    bencher.bench(|| {
        let result = divan::black_box(transaction(&client, &server, &request, &response, &psk));
        assert_eq!(result, Ok(size));
    });
}

fn sse_transaction(
    client: &Client,
    server: &Server,
    request: &Request,
    block: &[u8],
    count: usize,
    psk: &[u8],
) -> Result<usize, Error> {
    let (envelope, response_token) = client.protect(request)?.into_parts();
    let preparsed = server.preparse_owned(envelope)?;
    let authenticated = server.authenticate(preparsed.token, psk)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    let headers = vec![HeaderField {
        name: b"content-type".to_vec(),
        value: b"text/event-stream".to_vec(),
    }];
    let (mut writer, start) = opened.response.into_sealer(200, headers)?;
    let mut reader = response_token.into_opener();
    if !matches!(reader.feed(&start)?.1, Some(ResponseRecord::Start(_))) {
        return Err(Error::MalformedEnvelope);
    }
    let mut total = 0;
    for _ in 0..count {
        let frame = writer.seal_sse_block(block)?;
        match reader.feed(&frame)?.1 {
            Some(ResponseRecord::SseData(data)) => total += data.len(),
            _ => return Err(Error::MalformedEnvelope),
        }
    }
    if !matches!(reader.feed(&writer.finish()?)?.1, Some(ResponseRecord::End)) {
        return Err(Error::MalformedEnvelope);
    }
    reader.finish_eof()?;
    Ok(total)
}

#[divan::bench]
fn sse_100_small_blocks(bencher: divan::Bencher) {
    bench_sse(bencher, 1, 100);
}

#[divan::bench]
fn sse_one_large_block(bencher: divan::Bencher) {
    bench_sse(bencher, 1024 * 1024, 1);
}

fn bench_sse(bencher: divan::Bencher, payload_len: usize, count: usize) {
    let keys = match generate_key_pair() {
        Ok(keys) => keys,
        Err(error) => {
            eprintln!("benchmark setup failed: {error}");
            std::process::exit(1);
        }
    };
    let (private_key, public_key) = keys.into_parts();
    let key_id = b"benchmark-key".to_vec();
    let psk = b"a 32-byte minimum benchmark credential".to_vec();
    let client = match Client::new(
        &public_key,
        key_id.clone(),
        psk.clone(),
        b"benchmark-tenant".to_vec(),
        Limits::default(),
    ) {
        Ok(client) => client,
        Err(error) => {
            eprintln!("benchmark setup failed: {error}");
            std::process::exit(1);
        }
    };
    let server = match Server::new(&private_key, key_id, Limits::default()) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("benchmark setup failed: {error}");
            std::process::exit(1);
        }
    };
    let request = Request {
        method: Method::Get,
        authority: b"api.example.test".to_vec(),
        path: b"/events".to_vec(),
        headers: Vec::new(),
        body: Vec::new(),
    };
    let mut block = Vec::with_capacity(payload_len + 7);
    block.extend_from_slice(b"data:");
    block.extend(std::iter::repeat_n(b'x', payload_len));
    block.extend_from_slice(b"\n\n");
    bencher.bench(|| {
        let result = divan::black_box(sse_transaction(
            &client, &server, &request, &block, count, &psk,
        ));
        assert_eq!(result, Ok(block.len() * count));
    });
}

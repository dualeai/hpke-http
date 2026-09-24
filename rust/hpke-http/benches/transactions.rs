//! `CodSpeed` measures the public Rust transaction without transport or fixture setup.

use hpke_http::{
    Client, Error, HeaderField, Limits, Method, Request, RequestHead, Response, ResponseRecord,
    Server, generate_key_pair,
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
    bench_roundtrip(bencher, size, false);
}

#[divan::bench(args = [1024 * 1024, 8 * 1024 * 1024])]
fn roundtrip_random(bencher: divan::Bencher, size: usize) {
    bench_roundtrip(bencher, size, true);
}

fn varied_bytes(size: usize) -> Vec<u8> {
    let mut state = 0x71f4_993d_5bea_2a51_u64;
    (0..size)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()[3]
        })
        .collect()
}

fn bench_roundtrip(bencher: divan::Bencher, size: usize, random: bool) {
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
        body: if random {
            varied_bytes(size)
        } else {
            vec![0x42; size]
        },
    };
    let response = Response {
        status: 200,
        headers: Vec::new(),
        body: if random {
            varied_bytes(size)
        } else {
            vec![0x43; size]
        },
    };

    bencher.bench(|| {
        let result = divan::black_box(transaction(&client, &server, &request, &response, &psk));
        assert_eq!(result, Ok(size));
    });
}

#[divan::bench(args = [false, true])]
fn upload_stream_1_mib(bencher: divan::Bencher, random: bool) {
    let keys = match generate_key_pair() {
        Ok(keys) => keys,
        Err(error) => {
            eprintln!("benchmark setup failed: {error}");
            std::process::exit(1);
        }
    };
    let client = match Client::new(
        keys.public_key(),
        b"benchmark-key".to_vec(),
        b"a 32-byte minimum benchmark credential".to_vec(),
        b"benchmark-tenant".to_vec(),
        Limits::default(),
    ) {
        Ok(client) => client,
        Err(error) => {
            eprintln!("benchmark setup failed: {error}");
            std::process::exit(1);
        }
    };
    let head = RequestHead {
        method: Method::Post,
        authority: b"api.example.test".to_vec(),
        path: b"/upload".to_vec(),
        headers: Vec::new(),
    };
    let body = if random {
        varied_bytes(1024 * 1024)
    } else {
        vec![0x42; 1024 * 1024]
    };
    bencher.bench(|| {
        let result = divan::black_box(seal_upload(&client, &head, &body));
        assert!(matches!(result, Ok(size) if size > 0));
    });
}

fn seal_upload(client: &Client, head: &RequestHead, body: &[u8]) -> Result<usize, Error> {
    let (mut writer, first) = client.begin_stream(head)?;
    let mut wire_bytes = first.len();
    let mut remaining = body;
    while !remaining.is_empty() {
        let (used, record) = writer.push(remaining)?;
        remaining = &remaining[used..];
        wire_bytes += record.as_ref().map_or(0, Vec::len);
    }
    let (end, response_token) = writer.finish()?;
    wire_bytes += end.len();
    drop(response_token);
    Ok(wire_bytes)
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

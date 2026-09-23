//! `CodSpeed` measures the public Rust transaction without transport or fixture setup.

use hpke_http::{
    Client, CompressionCoding, Error, Limits, Method, Request, Response, Server, generate_key_pair,
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
    let protected_response = opened.response.protect(response)?;
    Ok(response_token.open(&protected_response)?.body.len())
}

#[divan::bench(args = [0, 1024, 1024 * 1024, 8 * 1024 * 1024])]
fn roundtrip(bencher: divan::Bencher, size: usize) {
    bench_roundtrip(bencher, size, None);
}

#[divan::bench(args = [CompressionCoding::Gzip, CompressionCoding::Zstd])]
fn compressed_roundtrip(bencher: divan::Bencher, coding: CompressionCoding) {
    bench_roundtrip(bencher, 1024 * 1024, Some(coding));
}

fn bench_roundtrip(bencher: divan::Bencher, size: usize, compression: Option<CompressionCoding>) {
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
    let mut client = match Client::new(
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
    let mut server = match Server::new(&private_key, key_id, Limits::default()) {
        Ok(server) => server,
        Err(error) => {
            eprintln!("benchmark setup failed: {error}");
            std::process::exit(1);
        }
    };
    if let Some(coding) = compression {
        client = client.with_compression(coding);
        server = server.with_compression();
    }
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

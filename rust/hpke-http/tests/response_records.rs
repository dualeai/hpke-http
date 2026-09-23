//! Record order, byte-cut, and checked SSE block tests.

use hpke_http::{
    Client, Error, HeaderField, Limits, Method, Request, Response, ResponseMode, ResponseRecord,
    Server, generate_key_pair,
};

const KEY_ID: &[u8] = b"response-record-test";
const PSK_ID: &[u8] = b"tenant";
const PSK: &[u8] = &[0x73; 32];

fn pair() -> Result<(hpke_http::ResponseToken, hpke_http::ResponseCapability), Error> {
    let keys = generate_key_pair()?;
    let (private, public) = keys.into_parts();
    let client = Client::new(
        &public,
        KEY_ID.to_vec(),
        PSK.to_vec(),
        PSK_ID.to_vec(),
        Limits::default(),
    )?;
    let server = Server::new(&private, KEY_ID.to_vec(), Limits::default())?;
    let request = Request {
        method: Method::Get,
        authority: b"api.example.test".to_vec(),
        path: b"/events".to_vec(),
        headers: Vec::new(),
        body: Vec::new(),
    };
    let (envelope, token) = client.protect(&request)?.into_parts();
    let parsed = server.preparse(&envelope)?;
    let authenticated = server.authenticate(parsed.token, PSK)?;
    let opened = authenticated
        .token
        .admit(authenticated.replay.decision(true))?;
    Ok((token, opened.response))
}

fn stream() -> Result<(hpke_http::ResponseToken, Vec<u8>, usize, usize), Error> {
    let (token, capability) = pair()?;
    let headers = vec![HeaderField {
        name: b"content-type".to_vec(),
        value: b"text/event-stream; charset=utf-8".to_vec(),
    }];
    let (mut sealer, mut wire) = capability.into_sealer(200, headers, None)?;
    assert_eq!(sealer.head_mode(), ResponseMode::Sse);
    let start_end = wire.len();
    wire.extend_from_slice(&sealer.seal_sse_block(b": hi\n\n")?);
    let first_end = wire.len();
    wire.extend_from_slice(&sealer.seal_sse_block(b"data: two\n\n")?);
    wire.extend_from_slice(&sealer.finish()?);
    Ok((token, wire, start_end, first_end))
}

fn read_all(
    mut opener: hpke_http::ResponseOpener,
    wire: &[u8],
    cut: usize,
) -> Result<Vec<ResponseRecord>, Error> {
    let mut records = Vec::new();
    for chunk in [&wire[..cut], &wire[cut..]] {
        let mut offset = 0;
        while offset < chunk.len() {
            let (used, record) = opener.feed(&chunk[offset..])?;
            assert!(used > 0);
            offset += used;
            if let Some(record) = record {
                records.push(record);
            }
        }
    }
    assert_eq!(opener.finish_eof()?, None);
    Ok(records)
}

fn rejects(token: hpke_http::ResponseToken, wire: &[u8]) -> bool {
    let mut reader = token.into_opener();
    let mut offset = 0;
    while offset < wire.len() {
        match reader.feed(&wire[offset..]) {
            Ok((used, _)) if used > 0 => offset += used,
            _ => return true,
        }
    }
    reader.finish_eof().is_err()
}

#[test]
fn every_byte_cut_preserves_checked_blocks_and_start() -> Result<(), Error> {
    let (_, sample, _, _) = stream()?;
    for cut in 0..=sample.len() {
        let (token, wire, _, _) = stream()?;
        let records = read_all(token.into_opener(), &wire, cut)?;
        assert!(
            matches!(&records[0], ResponseRecord::Start(head) if head.mode == ResponseMode::Sse && head.status == 200)
        );
        assert_eq!(records[1], ResponseRecord::SseData(b": hi\n\n".to_vec()));
        assert_eq!(
            records[2],
            ResponseRecord::SseData(b"data: two\n\n".to_vec())
        );
        assert_eq!(records[3], ResponseRecord::End);
    }
    Ok(())
}

#[test]
fn record_mutation_loss_repeat_order_and_wrong_request_fail() -> Result<(), Error> {
    let (token, wire, _, first_end) = stream()?;
    let second_len = u32::from_be_bytes(
        wire[first_end..first_end + 4]
            .try_into()
            .map_err(|_| Error::MalformedEnvelope)?,
    ) as usize;
    let second_end = first_end + 4 + second_len;
    let mut changed_prefix = wire.clone();
    changed_prefix[5] ^= 1;
    assert!(rejects(token, &changed_prefix));

    let (token, wire, start_end, _) = stream()?;
    let mut changed_length = wire.clone();
    changed_length[start_end + 3] ^= 1;
    assert!(rejects(token, &changed_length));

    let (token, wire, _, first_end) = stream()?;
    let mut changed_tag = wire.clone();
    changed_tag[first_end - 1] ^= 1;
    assert!(rejects(token, &changed_tag));

    let (token, wire, start_end, first_end) = stream()?;
    let lost = [&wire[..start_end], &wire[first_end..]].concat();
    assert!(rejects(token, &lost));

    let (token, wire, start_end, first_end) = stream()?;
    let repeated = [
        &wire[..first_end],
        &wire[start_end..first_end],
        &wire[first_end..],
    ]
    .concat();
    assert!(rejects(token, &repeated));

    let (token, wire, start_end, first_end) = stream()?;
    let swapped = [
        &wire[..start_end],
        &wire[first_end..second_end],
        &wire[start_end..first_end],
        &wire[second_end..],
    ]
    .concat();
    assert!(rejects(token, &swapped));

    let (wrong_token, _) = pair()?;
    assert!(rejects(wrong_token, &wire));
    Ok(())
}

#[test]
fn finite_writer_allows_only_one_data_and_poison_after_error() -> Result<(), Error> {
    let (_, capability) = pair()?;
    let (mut writer, _) = capability.into_sealer(200, Vec::new(), None)?;
    assert!(writer.seal_finite_body(b"first")?.is_some());
    assert_eq!(
        writer.seal_finite_body(b"second"),
        Err(Error::InvalidConfiguration)
    );
    assert_eq!(writer.finish(), Err(Error::InvalidConfiguration));
    Ok(())
}

#[test]
fn finite_data_stays_private_until_end_and_outer_eof() -> Result<(), Error> {
    let (token, capability) = pair()?;
    let (mut writer, start) = capability.into_sealer(200, Vec::new(), None)?;
    let data = writer
        .seal_finite_body(b"complete")?
        .ok_or(Error::MalformedEnvelope)?;
    let end = writer.finish()?;
    let mut opener = token.into_opener();
    assert!(matches!(
        opener.feed(&start)?,
        (_, Some(ResponseRecord::Start(_)))
    ));
    assert_eq!(opener.feed(&data)?, (data.len(), None));
    assert_eq!(opener.feed(&end)?, (end.len(), Some(ResponseRecord::End)));
    assert_eq!(
        opener.finish_eof()?,
        Some(Response {
            status: 200,
            headers: Vec::new(),
            body: b"complete".to_vec(),
        })
    );
    Ok(())
}

#[test]
fn many_small_sse_records_can_exceed_the_finite_total_limit() -> Result<(), Error> {
    let (token, capability) = pair()?;
    let headers = vec![HeaderField {
        name: b"content-type".to_vec(),
        value: b"text/event-stream".to_vec(),
    }];
    let (mut writer, start) = capability.into_sealer(200, headers, None)?;
    let mut reader = token.into_opener();
    assert!(matches!(
        reader.feed(&start)?,
        (_, Some(ResponseRecord::Start(_)))
    ));
    let mut block = b"data:".to_vec();
    block.extend(std::iter::repeat_n(b'x', 1024 * 1024));
    block.extend_from_slice(b"\n\n");
    for _ in 0..9 {
        let frame = writer.seal_sse_block(&block)?;
        assert_eq!(
            reader.feed(&frame)?,
            (frame.len(), Some(ResponseRecord::SseData(block.clone())))
        );
    }
    let end = writer.finish()?;
    assert_eq!(reader.feed(&end)?, (end.len(), Some(ResponseRecord::End)));
    assert!(reader.finish_eof()?.is_none());
    Ok(())
}

#[test]
fn first_block_is_available_before_later_records() -> Result<(), Error> {
    let (token, wire, start_end, first_end) = stream()?;
    let mut opener = token.into_opener();
    let (used, start) = opener.feed(&wire[..start_end])?;
    assert_eq!(used, start_end);
    assert!(matches!(start, Some(ResponseRecord::Start(_))));
    let (used, first) = opener.feed(&wire[start_end..first_end])?;
    assert_eq!(used, first_end - start_end);
    assert_eq!(first, Some(ResponseRecord::SseData(b": hi\n\n".to_vec())));
    let (used, second) = opener.feed(&wire[first_end..])?;
    assert_eq!(
        second,
        Some(ResponseRecord::SseData(b"data: two\n\n".to_vec()))
    );
    let (_, end) = opener.feed(&wire[first_end + used..])?;
    assert_eq!(end, Some(ResponseRecord::End));
    assert_eq!(opener.finish_eof()?, None);
    Ok(())
}

#[test]
fn end_needs_transport_eof_and_no_extra_byte() -> Result<(), Error> {
    let (token, wire, _, _) = stream()?;
    let mut opener = token.into_opener();
    let mut offset = 0;
    while offset < wire.len() - 1 {
        let (used, _) = opener.feed(&wire[offset..wire.len() - 1])?;
        offset += used;
    }
    assert!(opener.finish_eof().is_err());

    let (token, mut wire, _, _) = stream()?;
    wire.push(0);
    let mut opener = token.into_opener();
    let mut offset = 0;
    let mut rejected = false;
    while offset < wire.len() {
        if let Ok((used, _)) = opener.feed(&wire[offset..]) {
            offset += used;
        } else {
            rejected = true;
            break;
        }
    }
    assert!(rejected);
    Ok(())
}

#[test]
fn checked_start_rejects_bad_sse_rules() -> Result<(), Error> {
    let (_, capability) = pair()?;
    let headers = vec![
        HeaderField {
            name: b"content-type".to_vec(),
            value: b"text/event-stream".to_vec(),
        },
        HeaderField {
            name: b"content-length".to_vec(),
            value: b"7".to_vec(),
        },
    ];
    assert!(capability.into_sealer(200, headers, None).is_err());
    let (_, capability) = pair()?;
    let headers = vec![HeaderField {
        name: b"content-type".to_vec(),
        value: b"text/event-stream".to_vec(),
    }];
    let (mut sealer, _) = capability.into_sealer(200, headers, None)?;
    assert!(sealer.seal_sse_block(b"data: incomplete\n").is_err());
    assert!(sealer.seal_sse_block(b"\n").is_err());
    Ok(())
}

#[test]
fn later_bad_record_cannot_revoke_earlier_block() -> Result<(), Error> {
    let (token, mut wire, _, first_end) = stream()?;
    let mut opener = token.into_opener();
    let mut offset = 0;
    let mut first = false;
    while offset < first_end {
        let (used, record) = opener.feed(&wire[offset..first_end])?;
        offset += used;
        first |= matches!(record, Some(ResponseRecord::SseData(_)));
    }
    assert!(first);
    *wire.last_mut().ok_or(Error::MalformedEnvelope)? ^= 1;
    let mut failed = false;
    while offset < wire.len() {
        if let Ok((used, _)) = opener.feed(&wire[offset..]) {
            offset += used;
        } else {
            failed = true;
            break;
        }
    }
    assert!(failed);
    assert!(opener.feed(b"").is_err());
    Ok(())
}

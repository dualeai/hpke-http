"""Check v3 request and response records with independent cryptography."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any, cast

import zstandard
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from hpke_http import Client, Limits, Method, Request, Response, Server

CORPUS = Path(__file__).resolve().parents[2] / "rust/hpke-http/tests/vectors/protocol-v3.json"
KEM_SUITE = b"KEM\x00\x20"
HPKE_SUITE = b"HPKE\x00\x20\x00\x01\x00\x03"


def _hex(value: str) -> bytes:
    return bytes.fromhex(value)


def _extract(salt: bytes, value: bytes) -> bytes:
    return hmac.digest(salt or bytes(32), value, "sha256")


def _expand(key: bytes, info: bytes, size: int) -> bytes:
    output = b""
    block = b""
    for counter in range(1, (size + 31) // 32 + 1):
        block = hmac.digest(key, block + info + bytes([counter]), "sha256")
        output += block
    return output[:size]


def _labeled_extract(salt: bytes, label: bytes, value: bytes, suite: bytes) -> bytes:
    return _extract(salt, b"HPKE-v1" + suite + label + value)


def _labeled_expand(key: bytes, label: bytes, info: bytes, size: int, suite: bytes) -> bytes:
    return _expand(key, size.to_bytes(2, "big") + b"HPKE-v1" + suite + label + info, size)


def _vector(value: bytes) -> bytes:
    size = len(value)
    if size < 64:
        return bytes([size]) + value
    if size < 16384:
        return (size | 0x4000).to_bytes(2, "big") + value
    raise ValueError("corpus vector is too long")


def _fields(fields: list[list[str]]) -> bytes:
    return b"".join(_vector(name.encode("ascii")) + _vector(value.encode("ascii")) for name, value in fields)


def _schedule(shared_secret: bytes, psk: bytes, psk_id: bytes, info: bytes) -> tuple[bytes, bytes, bytes]:
    context = (
        b"\x01"
        + _labeled_extract(b"", b"psk_id_hash", psk_id, HPKE_SUITE)
        + _labeled_extract(b"", b"info_hash", info, HPKE_SUITE)
    )
    secret = _labeled_extract(shared_secret, b"secret", psk, HPKE_SUITE)
    return (
        _labeled_expand(secret, b"key", context, 32, HPKE_SUITE),
        _labeled_expand(secret, b"base_nonce", context, 12, HPKE_SUITE),
        _labeled_expand(secret, b"exp", context, 32, HPKE_SUITE),
    )


def _records(
    key: bytes, base_nonce: bytes, prefix: bytes, label: bytes, plain: list[bytes]
) -> tuple[str, list[dict[str, object]]]:
    wire = bytearray(prefix)
    records: list[dict[str, object]] = []
    for sequence, plaintext in enumerate(plain):
        nonce = base_nonce[:4] + bytes(
            left ^ right for left, right in zip(base_nonce[4:], sequence.to_bytes(8, "big"), strict=True)
        )
        length = len(plaintext) + 16
        aad = label + prefix + sequence.to_bytes(8, "big") + length.to_bytes(4, "big")
        ciphertext = ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)
        frame = length.to_bytes(4, "big") + ciphertext
        wire.extend(frame)
        records.append(
            {
                "sequence": sequence,
                "nonce": nonce.hex(),
                "aad": aad.hex(),
                "plaintext": plaintext.hex(),
                "ciphertext": ciphertext.hex(),
                "frame": frame.hex(),
            }
        )
    return wire.hex(), records


def _open_records(
    wire: bytes, start_len: int, aad_prefix: bytes, label: bytes, key: bytes, base_nonce: bytes
) -> list[bytes]:
    plain: list[bytes] = []
    offset = start_len
    while offset < len(wire):
        assert offset + 4 <= len(wire)
        size = int.from_bytes(wire[offset : offset + 4], "big")
        offset += 4
        ciphertext = wire[offset : offset + size]
        assert len(ciphertext) == size
        sequence = len(plain)
        nonce = base_nonce[:4] + bytes(
            left ^ right for left, right in zip(base_nonce[4:], sequence.to_bytes(8, "big"), strict=True)
        )
        aad = label + aad_prefix + sequence.to_bytes(8, "big") + size.to_bytes(4, "big")
        plain.append(ChaCha20Poly1305(key).decrypt(nonce, ciphertext, aad))
        offset += size
    return plain


def _response_records(
    response_secret: bytes, enc: bytes, server_nonce: bytes, plain: list[bytes]
) -> tuple[str, dict[str, object]]:
    salt = enc + server_nonce
    prk = _extract(salt, response_secret)
    key = _expand(prk, b"hpke-http/3 response key", 32)
    base_nonce = _expand(prk, b"hpke-http/3 response nonce", 12)
    prefix = b"HHRP\x03" + server_nonce
    wire, records = _records(key, base_nonce, prefix, b"hpke-http/3 response record\x00", plain)
    return wire, {"key": key.hex(), "base_nonce": base_nonce.hex(), "prefix": prefix.hex(), "records": records}


def _derive(source: dict[str, Any]) -> dict[str, object]:
    recipient_private = X25519PrivateKey.from_private_bytes(_hex(str(source["recipient_private_key"])))
    recipient_public = recipient_private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    assert recipient_public.hex() == source["recipient_public_key"]
    ephemeral_private = X25519PrivateKey.from_private_bytes(_hex(str(source["ephemeral_private_key"])))
    enc = ephemeral_private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    dh = ephemeral_private.exchange(X25519PublicKey.from_public_bytes(recipient_public))
    shared_secret = _labeled_expand(
        _labeled_extract(b"", b"eae_prk", dh, KEM_SUITE), b"shared_secret", enc + recipient_public, 32, KEM_SUITE
    )
    key_id = _hex(str(source["recipient_key_id"]))
    psk_id = _hex(str(source["psk_id"]))
    header = (
        b"HHRQ\x03"
        + bytes((len(key_id), len(psk_id)))
        + b"\x00\x20\x00\x01\x00\x03"
        + int(source["issued_at_unix_s"]).to_bytes(8, "big")
        + key_id
        + psk_id
    )
    info = b"message/hpke-http request\x00v3\x00" + header
    psk = _hex(str(source["psk"]))
    key, base_nonce, exporter_secret = _schedule(shared_secret, psk, psk_id, info)
    response_secret = _labeled_expand(exporter_secret, b"sec", b"message/hpke-http response\x00v3", 32, HPKE_SUITE)
    request = source["request"]
    body = _hex(request["body"])
    request_plain = [
        b"\x01"
        + _vector(request["method"].encode("ascii"))
        + _vector(request["authority"].encode("ascii"))
        + _vector(request["path"].encode("ascii"))
        + _vector(_fields(request["headers"])),
        b"\x02\x00" + body[:8],
        b"\x02\x00" + body[8:],
        b"\x03",
    ]
    request_wire, request_records = _records(key, base_nonce, header, b"hpke-http/3 request record\x00", request_plain)
    request_wire = (header + enc).hex() + request_wire[len(header.hex()) :]
    finite = source["response"]
    finite_plain = [
        b"\x01" + int(finite["status"]).to_bytes(2, "big") + _fields(finite["headers"]),
        b"\x02\x00" + _hex(finite["body"]),
        b"\x03",
    ]
    sse = source["sse_response"]
    sse_plain = [
        b"\x01" + int(sse["status"]).to_bytes(2, "big") + _fields(sse["headers"]),
        *(b"\x02\x00" + _hex(block) for block in sse["blocks"]),
        b"\x03",
    ]
    finite_wire, finite_parts = _response_records(response_secret, enc, _hex(source["response_nonce"]), finite_plain)
    sse_wire, sse_parts = _response_records(response_secret, enc, _hex(source["sse_response_nonce"]), sse_plain)
    return {
        "request_envelope": request_wire,
        "response_envelope": finite_wire,
        "sse_response_envelope": sse_wire,
        "intermediates": {
            "header": header.hex(),
            "enc": enc.hex(),
            "dh": dh.hex(),
            "shared_secret": shared_secret.hex(),
            "request_key": key.hex(),
            "request_base_nonce": base_nonce.hex(),
            "request_records": request_records,
            "exporter_secret": exporter_secret.hex(),
            "response_secret": response_secret.hex(),
            "replay_id": hashlib.sha256(b"hpke-http/replay\x00v3\x00" + header + enc).hexdigest(),
            "finite_response": finite_parts,
            "sse_response": sse_parts,
        },
    }


def test_v3_corpus_matches_independent_cryptography() -> None:
    source = json.loads(CORPUS.read_text())
    assert source["schema"] == "hpke-http-protocol-corpus/3"
    assert source["protocol"] == "hpke-http/3"
    assert source["generator"] == "independent-python-cryptography/3"
    for name, expected in _derive(source).items():
        assert source[name] == expected


def test_many_small_data_records_fit_complete_request_limit() -> None:
    source = json.loads(CORPUS.read_text())
    body = b"x" * 4096
    source["issued_at_unix_s"] = int(time.time())
    source["request"]["body"] = body.hex()
    source["request"]["headers"] = [["content-length", str(len(body))]]
    derived = _derive(source)
    parts = cast(dict[str, Any], derived["intermediates"])
    header = _hex(parts["header"])
    enc = _hex(parts["enc"])
    start = _hex(parts["request_records"][0]["plaintext"])
    framed, _ = _records(
        _hex(parts["request_key"]),
        _hex(parts["request_base_nonce"]),
        header,
        b"hpke-http/3 request record\x00",
        [start, *([b"\x02\x00x"] * len(body)), b"\x03"],
    )
    envelope = header + enc + _hex(framed)[len(header) :]
    assert len(envelope) > 92_000

    server = Server(
        _hex(source["recipient_private_key"]),
        _hex(source["recipient_key_id"]),
        limits=Limits(max_body_len=10_000, max_request_bytes=10_000),
    )
    opened = server.preparse(envelope).authenticate(_hex(source["psk"])).admit(accepted=True)
    assert opened.request.body == body
    opened.close()
    server.close()


def test_v3_zstd_record_matches_independent_cryptography() -> None:
    source = json.loads(CORPUS.read_text())
    compressed = source["compressed_request"]
    recipient_private = X25519PrivateKey.from_private_bytes(_hex(source["recipient_private_key"]))
    recipient_public = recipient_private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ephemeral_private = X25519PrivateKey.from_private_bytes(_hex(compressed["ephemeral_private_key"]))
    enc = ephemeral_private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    assert enc.hex() == compressed["enc"]
    shared_secret = _labeled_expand(
        _labeled_extract(
            b"", b"eae_prk", ephemeral_private.exchange(X25519PublicKey.from_public_bytes(recipient_public)), KEM_SUITE
        ),
        b"shared_secret",
        enc + recipient_public,
        32,
        KEM_SUITE,
    )
    header = _hex(compressed["header"])
    assert header == (
        b"HHRQ\x03"
        + bytes((len(_hex(source["recipient_key_id"])), len(_hex(source["psk_id"]))))
        + b"\x00\x20\x00\x01\x00\x03"
        + int(compressed["issued_at_unix_s"]).to_bytes(8, "big")
        + _hex(source["recipient_key_id"])
        + _hex(source["psk_id"])
    )
    key, nonce, _exporter = _schedule(
        shared_secret,
        _hex(source["psk"]),
        _hex(source["psk_id"]),
        b"message/hpke-http request\x00v3\x00" + header,
    )
    assert key.hex() == compressed["request_key"]
    assert nonce.hex() == compressed["request_base_nonce"]
    head = compressed["head"]
    zstd_frame = _hex(compressed["zstd_frame"])
    assert zstd_frame.startswith(b"\x28\xb5\x2f\xfd")
    assert len(zstd_frame) < compressed["clear_body_size"]
    plain = [
        b"\x01"
        + _vector(head["method"].encode("ascii"))
        + _vector(head["authority"].encode("ascii"))
        + _vector(head["path"].encode("ascii"))
        + _vector(_fields(head["headers"])),
        b"\x02\x01" + zstd_frame,
        b"\x03",
    ]
    wire, records = _records(key, nonce, header, b"hpke-http/3 request record\x00", plain)
    envelope = (header + enc).hex() + wire[len(header.hex()) :]
    assert envelope == compressed["envelope"]
    for actual, expected in zip(records, compressed["records"], strict=True):
        for field in ("sequence", "nonce", "aad", "plaintext", "frame"):
            assert actual[field] == expected[field]


def test_native_senders_emit_zstd_that_an_independent_decoder_opens() -> None:
    source = json.loads(CORPUS.read_text())
    private_key = _hex(source["recipient_private_key"])
    public_key = _hex(source["recipient_public_key"])
    key_id = _hex(source["recipient_key_id"])
    psk = _hex(source["psk"])
    psk_id = _hex(source["psk_id"])
    client = Client(public_key, key_id, psk, psk_id)
    server = Server(private_key, key_id)
    request = Request(Method.POST, "api.example.test", "/compressed", body=b"a" * 4096)
    response = Response(200, body=b"b" * 4096)
    try:
        protected = client.protect(request)
        request_wire = protected.envelope
        header_len = len(_hex(source["intermediates"]["header"]))
        header = request_wire[:header_len]
        enc = request_wire[header_len : header_len + 32]
        recipient_private = X25519PrivateKey.from_private_bytes(private_key)
        shared_secret = _labeled_expand(
            _labeled_extract(
                b"", b"eae_prk", recipient_private.exchange(X25519PublicKey.from_public_bytes(enc)), KEM_SUITE
            ),
            b"shared_secret",
            enc + public_key,
            32,
            KEM_SUITE,
        )
        request_key, request_nonce, exporter_secret = _schedule(
            shared_secret, psk, psk_id, b"message/hpke-http request\x00v3\x00" + header
        )
        request_plain = _open_records(
            request_wire,
            header_len + 32,
            header,
            b"hpke-http/3 request record\x00",
            request_key,
            request_nonce,
        )
        assert len(request_plain) == 3
        assert request_plain[0][0] == 1
        assert request_plain[1][:2] == b"\x02\x01"
        assert (
            zstandard.ZstdDecompressor().decompress(request_plain[1][2:], max_output_size=len(request.body))
            == request.body
        )
        assert request_plain[2] == b"\x03"

        opened = server.preparse(request_wire).authenticate(psk).admit(accepted=True)
        response_wire = opened.protect_response(response)
        response_secret = _labeled_expand(exporter_secret, b"sec", b"message/hpke-http response\x00v3", 32, HPKE_SUITE)
        response_prefix = response_wire[: 5 + 32]
        response_prk = _extract(enc + response_prefix[5:], response_secret)
        response_plain = _open_records(
            response_wire,
            len(response_prefix),
            response_prefix,
            b"hpke-http/3 response record\x00",
            _expand(response_prk, b"hpke-http/3 response key", 32),
            _expand(response_prk, b"hpke-http/3 response nonce", 12),
        )
        assert len(response_plain) == 3
        assert response_plain[0][:3] == b"\x01\x00\xc8"
        assert response_plain[1][:2] == b"\x02\x01"
        assert (
            zstandard.ZstdDecompressor().decompress(response_plain[1][2:], max_output_size=len(response.body))
            == response.body
        )
        assert response_plain[2] == b"\x03"
    finally:
        client.close()
        server.close()

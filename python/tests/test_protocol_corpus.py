"""Check the frozen v2 bytes with an independent cryptography implementation."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

CORPUS = Path(__file__).resolve().parents[2] / "rust/hpke-http/tests/vectors/protocol-v2.json"
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


def _request_plaintext(request: dict[str, Any]) -> bytes:
    method = str(request["method"]).encode("ascii")
    authority = str(request["authority"]).encode("ascii")
    path = str(request["path"]).encode("ascii")
    headers = _fields(request["headers"])
    body = _hex(str(request["body"]))
    return b"\x00" + b"".join(
        (
            _vector(method),
            _vector(b"https"),
            _vector(authority),
            _vector(path),
            _vector(headers),
            _vector(body),
            _vector(b""),
        )
    )


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


def _response_records(
    response_secret: bytes, enc: bytes, server_nonce: bytes, plain: list[bytes]
) -> tuple[str, dict[str, object]]:
    salt = enc + server_nonce
    key = _expand(_extract(salt, response_secret), b"hpke-http/2 response key", 32)
    base_nonce = _expand(_extract(salt, response_secret), b"hpke-http/2 response nonce", 12)
    prefix = b"HHRP\x02" + server_nonce
    frames: list[dict[str, object]] = []
    wire = bytearray(prefix)
    for sequence, plaintext in enumerate(plain):
        nonce = base_nonce[:4] + bytes(
            left ^ right for left, right in zip(base_nonce[4:], sequence.to_bytes(8, "big"), strict=True)
        )
        length = len(plaintext) + 16
        aad = b"hpke-http/2 response record\x00" + prefix + sequence.to_bytes(8, "big") + length.to_bytes(4, "big")
        ciphertext = ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)
        frame = length.to_bytes(4, "big") + ciphertext
        wire.extend(frame)
        frames.append(
            {
                "sequence": sequence,
                "nonce": nonce.hex(),
                "aad": aad.hex(),
                "plaintext": plaintext.hex(),
                "ciphertext": ciphertext.hex(),
                "frame": frame.hex(),
            }
        )
    return wire.hex(), {"key": key.hex(), "base_nonce": base_nonce.hex(), "prefix": prefix.hex(), "records": frames}


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
        b"HHRQ\x02\x00"
        + bytes((len(key_id), len(psk_id)))
        + b"\x00\x20\x00\x01\x00\x03"
        + int(source["issued_at_unix_s"]).to_bytes(8, "big")
        + key_id
        + psk_id
    )
    info = b"message/hpke-http request\x00v2\x00" + header
    psk = _hex(str(source["psk"]))
    key, nonce, exporter_secret = _schedule(shared_secret, psk, psk_id, info)
    plaintext = _request_plaintext(source["request"])
    ciphertext = ChaCha20Poly1305(key).encrypt(nonce, plaintext, b"")
    response_secret = _labeled_expand(exporter_secret, b"sec", b"message/hpke-http response\x00v2", 32, HPKE_SUITE)
    finite = source["response"]
    finite_plain = [
        b"\x01" + int(finite["status"]).to_bytes(2, "big") + b"\x00" + _fields(finite["headers"]),
        b"\x02" + _hex(finite["body"]),
        b"\x03",
    ]
    sse = source["sse_response"]
    sse_plain = [
        b"\x01" + int(sse["status"]).to_bytes(2, "big") + b"\x00" + _fields(sse["headers"]),
        *(b"\x02" + _hex(block) for block in sse["blocks"]),
        b"\x03",
    ]
    finite_wire, finite_parts = _response_records(
        response_secret, enc, _hex(str(source["response_nonce"])), finite_plain
    )
    sse_wire, sse_parts = _response_records(response_secret, enc, _hex(str(source["sse_response_nonce"])), sse_plain)
    return {
        "request_envelope": (header + enc + ciphertext).hex(),
        "response_envelope": finite_wire,
        "sse_response_envelope": sse_wire,
        "intermediates": {
            "header": header.hex(),
            "enc": enc.hex(),
            "dh": dh.hex(),
            "shared_secret": shared_secret.hex(),
            "request_key": key.hex(),
            "request_base_nonce": nonce.hex(),
            "request_plaintext": plaintext.hex(),
            "request_ciphertext": ciphertext.hex(),
            "exporter_secret": exporter_secret.hex(),
            "response_secret": response_secret.hex(),
            "replay_id": hashlib.sha256(b"hpke-http/replay\x00v2\x00" + header + enc).hexdigest(),
            "finite_response": finite_parts,
            "sse_response": sse_parts,
        },
    }


def test_v2_corpus_matches_independent_cryptography() -> None:
    """Check every fixed request and response byte against Python cryptography."""
    source = json.loads(CORPUS.read_text())
    assert source["schema"] == "hpke-http-protocol-corpus/2"
    assert source["protocol"] == "hpke-http/2"
    for name, expected in _derive(source).items():
        assert source[name] == expected

    sender = _hex(source["sender_request_envelope"])
    header = _hex(source["intermediates"]["header"])
    assert sender.startswith(header)
    enc = sender[len(header) : len(header) + 32]
    recipient_private = X25519PrivateKey.from_private_bytes(_hex(source["recipient_private_key"]))
    recipient_public = _hex(source["recipient_public_key"])
    dh = recipient_private.exchange(X25519PublicKey.from_public_bytes(enc))
    shared_secret = _labeled_expand(
        _labeled_extract(b"", b"eae_prk", dh, KEM_SUITE), b"shared_secret", enc + recipient_public, 32, KEM_SUITE
    )
    info = b"message/hpke-http request\x00v2\x00" + header
    key, nonce, _ = _schedule(shared_secret, _hex(source["psk"]), _hex(source["psk_id"]), info)
    opened = ChaCha20Poly1305(key).decrypt(nonce, sender[len(header) + 32 :], b"")
    assert opened.hex() == source["intermediates"]["request_plaintext"]

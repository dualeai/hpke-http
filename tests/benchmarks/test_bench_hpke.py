"""Performance benchmarks for HPKE encryption operations.

Covers the hot paths: key exchange, seal/open, chunk streaming, compression,
high-level API (RequestEncryptor/ResponseDecryptor), primitives, and parsers.

Run with: uv run pytest tests/benchmarks/ --codspeed -v --no-cov -p no:xdist -o "addopts="

CodSpeed tracks both CPU (simulation mode) and memory (allocation count, peak RSS).
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

import pytest

from hpke_http.core import (
    RequestDecryptor,
    RequestEncryptor,
    ResponseDecryptor,
    ResponseEncryptor,
    SSEDecryptor,
    SSEEncryptor,
    _ChunkStreamParser,
)
from hpke_http.headers import b64url_decode, b64url_encode
from hpke_http.hpke import (
    open_psk,
    seal_psk,
    setup_recipient_psk,
    setup_sender_psk,
)
from hpke_http.primitives.aead import compute_nonce
from hpke_http.primitives.kdf import labeled_expand, labeled_extract
from hpke_http.primitives.kem import generate_keypair
from hpke_http.streaming import (
    ChunkDecryptor,
    ChunkEncryptor,
    RawFormat,
    SSEFormat,
    StreamingSession,
    create_session_from_context,
    gzip_compress,
    gzip_decompress,
)

pytestmark = pytest.mark.benchmark

# Type alias for the PSK params tuple: (info, psk, psk_id)
PSKParams = tuple[bytes, bytes, bytes]

# Payload sizes for parametrized pipeline benchmarks (request encrypt/decrypt).
# Covers: tiny JSON → single chunk → multi-chunk → large multi-chunk.
PAYLOAD_PARAMS = [
    pytest.param(36, id="36B"),
    pytest.param(1 * 1024, id="1KB"),
    pytest.param(64 * 1024, id="64KB"),
    pytest.param(1 * 1024 * 1024, id="1MB"),
    pytest.param(10 * 1024 * 1024, id="10MB"),
    pytest.param(50 * 1024 * 1024, id="50MB"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def psk_params() -> PSKParams:
    """PSK mode parameters: (info, psk, psk_id).

    Bundled because every HPKE function takes these three together.
    """
    psk = secrets.token_bytes(32)
    psk_id = hashlib.sha256(psk).digest()
    info = b"benchmark"
    return info, psk, psk_id


@pytest.fixture(scope="module")
def aad() -> bytes:
    return b"additional-data"


@pytest.fixture(scope="module")
def keypair() -> tuple[bytes, bytes]:
    return generate_keypair()


@pytest.fixture(scope="module")
def sk(keypair: tuple[bytes, bytes]) -> bytes:
    return keypair[0]


@pytest.fixture(scope="module")
def pk(keypair: tuple[bytes, bytes]) -> bytes:
    return keypair[1]


@pytest.fixture(scope="module")
def small_plaintext() -> bytes:
    return b'{"user": "alice", "action": "login"}'


@pytest.fixture(scope="module")
def medium_plaintext() -> bytes:
    return secrets.token_bytes(4 * 1024)  # 4 KB


@pytest.fixture(scope="module")
def large_plaintext() -> bytes:
    return secrets.token_bytes(64 * 1024)  # 64 KB (one full chunk)


@pytest.fixture(scope="module")
def sealed_message(
    pk: bytes,
    psk_params: PSKParams,
    aad: bytes,
    sk: bytes,
) -> tuple[bytes, bytes, bytes]:
    """Pre-sealed message for open benchmarks."""
    enc, ct = seal_psk(pk, *psk_params, aad, b'{"status": "ok"}')
    return enc, ct, sk


@pytest.fixture(scope="module")
def streaming_session(pk: bytes, psk_params: PSKParams) -> StreamingSession:
    """Streaming session derived from a sender HPKE context."""
    sender = setup_sender_psk(pk, *psk_params)
    return create_session_from_context(sender)


def _copy_session(session: StreamingSession) -> StreamingSession:
    """Create a fresh session with the same key material (resets counter state)."""
    return StreamingSession(session_key=session.session_key, session_salt=session.session_salt)


# ---------------------------------------------------------------------------
# Primitives: KEM, KDF, AEAD
# ---------------------------------------------------------------------------


def test_bench_generate_keypair(benchmark: Any) -> None:
    """Benchmark X25519 keypair generation."""
    benchmark(generate_keypair)


def test_bench_compute_nonce(benchmark: Any) -> None:
    """Benchmark nonce computation (XOR loop, called per seal/open)."""
    base_nonce = secrets.token_bytes(12)
    benchmark(compute_nonce, base_nonce, 42)


def test_bench_labeled_extract(benchmark: Any) -> None:
    """Benchmark HKDF-SHA256 LabeledExtract (HMAC, called in key schedule)."""
    salt = secrets.token_bytes(32)
    benchmark(labeled_extract, salt, b"test_label", b"input_keying_material")


def test_bench_labeled_expand(benchmark: Any) -> None:
    """Benchmark HKDF-SHA256 LabeledExpand (called in key schedule)."""
    prk = secrets.token_bytes(32)
    benchmark(labeled_expand, prk, b"test_label", b"context_info", 32)


# ---------------------------------------------------------------------------
# HPKE context setup (key exchange + key schedule)
# ---------------------------------------------------------------------------


def test_bench_setup_sender_context(benchmark: Any, pk: bytes, psk_params: PSKParams) -> None:
    """Benchmark sender context creation (encap + key schedule)."""
    benchmark(setup_sender_psk, pk, *psk_params)


def test_bench_setup_recipient_context(
    benchmark: Any,
    pk: bytes,
    sk: bytes,
    psk_params: PSKParams,
) -> None:
    """Benchmark recipient context creation (decap + key schedule)."""
    ctx = setup_sender_psk(pk, *psk_params)
    benchmark(setup_recipient_psk, ctx.enc, sk, *psk_params)


# ---------------------------------------------------------------------------
# Single-shot seal / open
# ---------------------------------------------------------------------------


def test_bench_seal_psk_small(
    benchmark: Any,
    pk: bytes,
    psk_params: PSKParams,
    aad: bytes,
    small_plaintext: bytes,
) -> None:
    """Benchmark single-shot encryption with small JSON payload."""
    benchmark(seal_psk, pk, *psk_params, aad, small_plaintext)


def test_bench_seal_psk_medium(
    benchmark: Any,
    pk: bytes,
    psk_params: PSKParams,
    aad: bytes,
    medium_plaintext: bytes,
) -> None:
    """Benchmark single-shot encryption with 4 KB payload."""
    benchmark(seal_psk, pk, *psk_params, aad, medium_plaintext)


def test_bench_open_psk(
    benchmark: Any,
    sealed_message: tuple[bytes, bytes, bytes],
    psk_params: PSKParams,
    aad: bytes,
) -> None:
    """Benchmark single-shot decryption (includes decap + key schedule)."""
    enc, ct, sk = sealed_message
    benchmark(open_psk, enc, sk, *psk_params, aad, ct)


# ---------------------------------------------------------------------------
# Context-based seal / open (amortised key exchange)
# ---------------------------------------------------------------------------


def test_bench_context_seal(
    benchmark: Any,
    pk: bytes,
    psk_params: PSKParams,
    aad: bytes,
    small_plaintext: bytes,
) -> None:
    """Benchmark seal on a reused context (no key exchange per call)."""
    ctx = setup_sender_psk(pk, *psk_params)
    benchmark(ctx.seal, aad, small_plaintext)


def test_bench_context_open(
    benchmark: Any,
    pk: bytes,
    sk: bytes,
    psk_params: PSKParams,
    aad: bytes,
    small_plaintext: bytes,
) -> None:
    """Benchmark open on a reused context (no key exchange per call).

    Uses matching sender/recipient seq counters so each iteration produces
    a fresh ciphertext and decrypts it. The seal cost is constant overhead
    per iteration but isolates the open from key exchange.
    """
    sender = setup_sender_psk(pk, *psk_params)
    recipient = setup_recipient_psk(sender.enc, sk, *psk_params)

    def _open() -> bytes:
        ct = sender.seal(aad, small_plaintext)
        return recipient.open(aad, ct)

    benchmark(_open)


# ---------------------------------------------------------------------------
# Chunk encryption / decryption (streaming)
# ---------------------------------------------------------------------------


def test_bench_chunk_encrypt_sse(
    benchmark: Any,
    streaming_session: StreamingSession,
    large_plaintext: bytes,
) -> None:
    """Benchmark SSE chunk encryption (64 KB)."""
    encryptor = ChunkEncryptor(session=streaming_session, format=SSEFormat())
    benchmark(encryptor.encrypt, large_plaintext)


def test_bench_chunk_encrypt_raw(
    benchmark: Any,
    streaming_session: StreamingSession,
    large_plaintext: bytes,
) -> None:
    """Benchmark raw binary chunk encryption (64 KB)."""
    encryptor = ChunkEncryptor(session=streaming_session, format=RawFormat())
    benchmark(encryptor.encrypt, large_plaintext)


def test_bench_chunk_decrypt_raw(
    benchmark: Any,
    streaming_session: StreamingSession,
    large_plaintext: bytes,
) -> None:
    """Benchmark raw binary chunk decryption (64 KB).

    Pre-encrypts a single chunk at counter=1. Each iteration creates a fresh
    ChunkDecryptor (expected_counter=1) to decrypt it. Includes decryptor
    construction overhead (~1µs) but isolates decrypt from encrypt.
    """
    fmt = RawFormat()

    # Pre-encrypt one chunk at counter=1
    enc = ChunkEncryptor(session=_copy_session(streaming_session), format=fmt)
    encrypted = enc.encrypt(large_plaintext)

    def _decrypt() -> bytes:
        dec = ChunkDecryptor(session=_copy_session(streaming_session), format=fmt)
        return dec.decrypt(encrypted)

    benchmark(_decrypt)


# ---------------------------------------------------------------------------
# High-level API: RequestEncryptor / ResponseDecryptor round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", PAYLOAD_PARAMS)
def test_bench_request_encrypt_all(
    benchmark: Any,
    pk: bytes,
    psk_params: PSKParams,
    size: int,
) -> None:
    """Benchmark RequestEncryptor.encrypt_all (setup + chunk + format)."""
    _info, psk, psk_id = psk_params
    plaintext = secrets.token_bytes(size)

    def _encrypt() -> bytes:
        enc = RequestEncryptor(pk, psk, psk_id)
        return enc.encrypt_all(plaintext)

    benchmark(_encrypt)


@pytest.mark.parametrize("size", PAYLOAD_PARAMS)
def test_bench_request_decrypt_all(
    benchmark: Any,
    pk: bytes,
    sk: bytes,
    psk_params: PSKParams,
    size: int,
) -> None:
    """Benchmark RequestDecryptor.decrypt_all (parse + decrypt + decompress)."""
    _info, psk, psk_id = psk_params
    plaintext = secrets.token_bytes(size)

    # Pre-encrypt once
    enc = RequestEncryptor(pk, psk, psk_id)
    ciphertext = enc.encrypt_all(plaintext)
    headers = enc.get_headers()

    def _decrypt() -> bytes:
        dec = RequestDecryptor(headers, sk, psk, psk_id)
        return dec.decrypt_all(ciphertext)

    benchmark(_decrypt)


def test_bench_response_encrypt_all(
    benchmark: Any,
    pk: bytes,
    sk: bytes,
    psk_params: PSKParams,
    large_plaintext: bytes,
) -> None:
    """Benchmark ResponseEncryptor.encrypt_all (64 KB response body)."""
    _info, psk, psk_id = psk_params
    # Create a recipient context (simulates server-side after decrypting request)
    sender = setup_sender_psk(pk, *psk_params)
    recipient = setup_recipient_psk(sender.enc, sk, *psk_params)

    def _encrypt() -> bytes:
        enc = ResponseEncryptor(recipient)
        return enc.encrypt_all(large_plaintext)

    benchmark(_encrypt)


def test_bench_response_decrypt_all(
    benchmark: Any,
    pk: bytes,
    sk: bytes,
    psk_params: PSKParams,
    large_plaintext: bytes,
) -> None:
    """Benchmark ResponseDecryptor.decrypt_all (64 KB response body)."""
    _info, psk, psk_id = psk_params
    sender = setup_sender_psk(pk, *psk_params)
    recipient = setup_recipient_psk(sender.enc, sk, *psk_params)

    # Pre-encrypt response
    resp_enc = ResponseEncryptor(recipient)
    ciphertext = resp_enc.encrypt_all(large_plaintext)
    headers = resp_enc.get_headers()

    def _decrypt() -> bytes:
        dec = ResponseDecryptor(headers, sender)
        return dec.decrypt_all(ciphertext)

    benchmark(_decrypt)


# ---------------------------------------------------------------------------
# High-level API: SSE encrypt / decrypt
# ---------------------------------------------------------------------------


def test_bench_sse_encrypt_event(
    benchmark: Any,
    pk: bytes,
    sk: bytes,
    psk_params: PSKParams,
) -> None:
    """Benchmark SSEEncryptor.encrypt (single event)."""
    sender = setup_sender_psk(pk, *psk_params)
    recipient = setup_recipient_psk(sender.enc, sk, *psk_params)
    sse_enc = SSEEncryptor(recipient)
    event = b'data: {"token": "hello"}\n\n'
    benchmark(sse_enc.encrypt, event)


def test_bench_sse_decrypt_event(
    benchmark: Any,
    pk: bytes,
    sk: bytes,
    psk_params: PSKParams,
) -> None:
    """Benchmark SSEDecryptor.decrypt (single event).

    Includes SSEEncryptor.encrypt overhead per iteration to produce matching
    ciphertexts with incrementing counters.
    """
    event = b'data: {"token": "hello"}\n\n'
    sender = setup_sender_psk(pk, *psk_params)
    recipient = setup_recipient_psk(sender.enc, sk, *psk_params)
    sse_enc = SSEEncryptor(recipient)
    headers = sse_enc.get_headers()
    sse_dec = SSEDecryptor(headers, sender)

    def _decrypt() -> bytes:
        encrypted_evt = sse_enc.encrypt(event)
        data = encrypted_evt.split(b"data: ", 1)[1].split(b"\n", 1)[0]
        return sse_dec.decrypt(data)

    benchmark(_decrypt)


# ---------------------------------------------------------------------------
# Parsers: chunk boundary detection and SSE parsing
# ---------------------------------------------------------------------------


def test_bench_chunk_stream_parser(benchmark: Any, streaming_session: StreamingSession) -> None:
    """Benchmark _ChunkStreamParser.feed with 10 chunks in one call."""
    fmt = RawFormat()
    encryptor = ChunkEncryptor(session=_copy_session(streaming_session), format=fmt)

    # Pre-encrypt 10 chunks, concatenate as a single wire blob
    chunks = [encryptor.encrypt(secrets.token_bytes(4096)) for _ in range(10)]
    wire_blob = b"".join(chunks)

    def _parse() -> int:
        parser = _ChunkStreamParser()
        return sum(1 for _ in parser.feed(wire_blob))

    benchmark(_parse)


# ---------------------------------------------------------------------------
# Wire format encoding / decoding
# ---------------------------------------------------------------------------


def test_bench_sse_format_encrypt_chunk(benchmark: Any, streaming_session: StreamingSession) -> None:
    """Benchmark SSEFormat.encrypt_chunk (combined-buffer encrypt + base64 SSE wrap)."""
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    fmt = SSEFormat()
    cipher = ChaCha20Poly1305(streaming_session.session_key)
    plaintext = secrets.token_bytes(64 * 1024)
    nonce = streaming_session.session_salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")
    benchmark(fmt.encrypt_chunk, 1, 0x00, plaintext, cipher, nonce)


def test_bench_raw_format_encrypt_chunk(benchmark: Any, streaming_session: StreamingSession) -> None:
    """Benchmark RawFormat.encrypt_chunk (combined-buffer encrypt + binary wire)."""
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    fmt = RawFormat()
    cipher = ChaCha20Poly1305(streaming_session.session_key)
    plaintext = secrets.token_bytes(64 * 1024)
    nonce = streaming_session.session_salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")
    benchmark(fmt.encrypt_chunk, 1, 0x00, plaintext, cipher, nonce)


def test_bench_sse_format_decode(benchmark: Any, streaming_session: StreamingSession) -> None:
    """Benchmark SSEFormat.decode (base64 → counter + encoding_id + ciphertext)."""
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    fmt = SSEFormat()
    cipher = ChaCha20Poly1305(streaming_session.session_key)
    plaintext = secrets.token_bytes(64 * 1024)
    nonce = streaming_session.session_salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")
    encoded = fmt.encrypt_chunk(1, 0x00, plaintext, cipher, nonce)
    # Extract the base64 data field
    data = encoded.split(b"data: ", 1)[1].split(b"\n", 1)[0]
    benchmark(fmt.decode, data)


def test_bench_raw_format_decode(benchmark: Any, streaming_session: StreamingSession) -> None:
    """Benchmark RawFormat.decode (binary wire → counter + encoding_id + ciphertext)."""
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    fmt = RawFormat()
    cipher = ChaCha20Poly1305(streaming_session.session_key)
    plaintext = secrets.token_bytes(64 * 1024)
    nonce = streaming_session.session_salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")
    encoded = fmt.encrypt_chunk(1, 0x00, plaintext, cipher, nonce)
    benchmark(fmt.decode, encoded)


# ---------------------------------------------------------------------------
# Base64url encoding / decoding
# ---------------------------------------------------------------------------


def test_bench_b64url_encode_32b(benchmark: Any) -> None:
    """Benchmark b64url_encode with 32-byte key (typical enc/salt header)."""
    data = secrets.token_bytes(32)
    benchmark(b64url_encode, data)


def test_bench_b64url_decode_32b(benchmark: Any) -> None:
    """Benchmark b64url_decode with 32-byte key."""
    data = secrets.token_bytes(32)
    encoded = b64url_encode(data)
    benchmark(b64url_decode, encoded)


# ---------------------------------------------------------------------------
# Compression (gzip — always available, no optional deps)
# ---------------------------------------------------------------------------


def test_bench_gzip_compress_4k(benchmark: Any, medium_plaintext: bytes) -> None:
    """Benchmark gzip compression of 4 KB payload."""
    benchmark(gzip_compress, medium_plaintext)


def test_bench_gzip_decompress_4k(benchmark: Any, medium_plaintext: bytes) -> None:
    """Benchmark gzip decompression of 4 KB payload."""
    compressed = gzip_compress(medium_plaintext)
    benchmark(gzip_decompress, compressed)


def test_bench_gzip_compress_64k(benchmark: Any, large_plaintext: bytes) -> None:
    """Benchmark gzip compression of 64 KB payload."""
    benchmark(gzip_compress, large_plaintext)


def test_bench_gzip_decompress_64k(benchmark: Any, large_plaintext: bytes) -> None:
    """Benchmark gzip decompression of 64 KB payload."""
    compressed = gzip_compress(large_plaintext)
    benchmark(gzip_decompress, compressed)

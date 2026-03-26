"""Performance benchmarks for HPKE encryption operations.

Covers the hot paths: key exchange, seal/open, chunk streaming, and compression.
"""

import hashlib
import secrets

import pytest

from hpke_http.hpke import (
    open_psk,
    seal_psk,
    setup_recipient_psk,
    setup_sender_psk,
)
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PSK = secrets.token_bytes(32)
PSK_ID = hashlib.sha256(PSK).digest()
INFO = b"benchmark"
AAD = b"additional-data"


@pytest.fixture(scope="module")
def keypair() -> tuple[bytes, bytes]:
    sk, pk = generate_keypair()
    return sk, pk


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
def sealed_message(keypair: tuple[bytes, bytes]) -> tuple[bytes, bytes, bytes]:
    """Pre-sealed message for open benchmarks."""
    sk, pk = keypair
    enc, ct = seal_psk(pk, INFO, PSK, PSK_ID, AAD, b'{"status": "ok"}')
    return enc, ct, sk


# ---------------------------------------------------------------------------
# HPKE context setup (key exchange + key schedule)
# ---------------------------------------------------------------------------


def test_bench_setup_sender_context(benchmark, keypair: tuple[bytes, bytes]) -> None:
    """Benchmark sender context creation (encap + key schedule)."""
    _sk, pk = keypair
    benchmark(setup_sender_psk, pk, INFO, PSK, PSK_ID)


def test_bench_setup_recipient_context(benchmark, keypair: tuple[bytes, bytes]) -> None:
    """Benchmark recipient context creation (decap + key schedule)."""
    sk, pk = keypair
    ctx = setup_sender_psk(pk, INFO, PSK, PSK_ID)
    benchmark(setup_recipient_psk, ctx.enc, sk, INFO, PSK, PSK_ID)


# ---------------------------------------------------------------------------
# Single-shot seal / open
# ---------------------------------------------------------------------------


def test_bench_seal_psk_small(benchmark, keypair: tuple[bytes, bytes], small_plaintext: bytes) -> None:
    """Benchmark single-shot encryption with small JSON payload."""
    _sk, pk = keypair
    benchmark(seal_psk, pk, INFO, PSK, PSK_ID, AAD, small_plaintext)


def test_bench_seal_psk_medium(benchmark, keypair: tuple[bytes, bytes], medium_plaintext: bytes) -> None:
    """Benchmark single-shot encryption with 4 KB payload."""
    _sk, pk = keypair
    benchmark(seal_psk, pk, INFO, PSK, PSK_ID, AAD, medium_plaintext)


def test_bench_open_psk(benchmark, sealed_message: tuple[bytes, bytes, bytes]) -> None:
    """Benchmark single-shot decryption."""
    enc, ct, sk = sealed_message
    benchmark(open_psk, enc, sk, INFO, PSK, PSK_ID, AAD, ct)


# ---------------------------------------------------------------------------
# Context-based seal / open (multi-message, amortises key exchange)
# ---------------------------------------------------------------------------


def test_bench_context_seal(benchmark, keypair: tuple[bytes, bytes], small_plaintext: bytes) -> None:
    """Benchmark seal on an existing context (no key exchange overhead)."""
    _sk, pk = keypair

    def _seal() -> None:
        ctx = setup_sender_psk(pk, INFO, PSK, PSK_ID)
        ctx.seal(AAD, small_plaintext)

    benchmark(_seal)


def test_bench_context_open(benchmark, keypair: tuple[bytes, bytes], small_plaintext: bytes) -> None:
    """Benchmark open on an existing context (no key exchange overhead)."""
    sk, pk = keypair

    def _open() -> None:
        sender = setup_sender_psk(pk, INFO, PSK, PSK_ID)
        ct = sender.seal(AAD, small_plaintext)
        recipient = setup_recipient_psk(sender.enc, sk, INFO, PSK, PSK_ID)
        recipient.open(AAD, ct)

    benchmark(_open)


# ---------------------------------------------------------------------------
# Chunk encryption / decryption (streaming)
# ---------------------------------------------------------------------------


def _make_session(keypair: tuple[bytes, bytes]) -> StreamingSession:
    _sk, pk = keypair
    sender = setup_sender_psk(pk, INFO, PSK, PSK_ID)
    return create_session_from_context(sender)


def test_bench_chunk_encrypt_sse(benchmark, keypair: tuple[bytes, bytes], large_plaintext: bytes) -> None:
    """Benchmark SSE chunk encryption (64 KB)."""
    session = _make_session(keypair)
    encryptor = ChunkEncryptor(session=session, format=SSEFormat())
    benchmark(encryptor.encrypt, large_plaintext)


def test_bench_chunk_encrypt_raw(benchmark, keypair: tuple[bytes, bytes], large_plaintext: bytes) -> None:
    """Benchmark raw binary chunk encryption (64 KB)."""
    session = _make_session(keypair)
    encryptor = ChunkEncryptor(session=session, format=RawFormat())
    benchmark(encryptor.encrypt, large_plaintext)


def test_bench_chunk_decrypt_raw(benchmark, keypair: tuple[bytes, bytes], large_plaintext: bytes) -> None:
    """Benchmark raw binary chunk decryption (64 KB)."""
    session = _make_session(keypair)
    fmt = RawFormat()

    def _encrypt_then_decrypt() -> bytes:
        enc = ChunkEncryptor(
            session=StreamingSession(session_key=session.session_key, session_salt=session.session_salt),
            format=fmt,
        )
        dec = ChunkDecryptor(
            session=StreamingSession(session_key=session.session_key, session_salt=session.session_salt),
            format=fmt,
        )
        encrypted = enc.encrypt(large_plaintext)
        return dec.decrypt(encrypted)

    benchmark(_encrypt_then_decrypt)


# ---------------------------------------------------------------------------
# Compression (gzip — always available, no optional deps)
# ---------------------------------------------------------------------------


def test_bench_gzip_compress_4k(benchmark, medium_plaintext: bytes) -> None:
    """Benchmark gzip compression of 4 KB payload."""
    benchmark(gzip_compress, medium_plaintext)


def test_bench_gzip_decompress_4k(benchmark, medium_plaintext: bytes) -> None:
    """Benchmark gzip decompression of 4 KB payload."""
    compressed = gzip_compress(medium_plaintext)
    benchmark(gzip_decompress, compressed)


def test_bench_gzip_compress_64k(benchmark, large_plaintext: bytes) -> None:
    """Benchmark gzip compression of 64 KB payload."""
    benchmark(gzip_compress, large_plaintext)

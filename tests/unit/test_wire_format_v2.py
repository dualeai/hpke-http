"""Wire format v2 tests: encoding_id authenticated via AAD.

Covers correctness, edge cases, downgrade-attack resistance, replay protection,
and buffer-reuse safety per the algorithmic verification plan.
"""

from __future__ import annotations

import os
import struct

import pytest
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from hpke_http.constants import SSEEncodingId
from hpke_http.exceptions import DecryptionError, ReplayAttackError
from hpke_http.streaming import (
    ChunkDecryptor,
    ChunkEncryptor,
    RawFormat,
    SSEFormat,
    StreamingSession,
)


def _make_pair(*, format_cls: type = RawFormat, compress: bool = False) -> tuple[ChunkEncryptor, ChunkDecryptor]:
    key = os.urandom(32)
    session = StreamingSession.create(key)
    enc = ChunkEncryptor(session, format=format_cls(), compress=compress)
    dec = ChunkDecryptor(
        StreamingSession(session_key=session.session_key, session_salt=session.session_salt),
        format=format_cls(),
    )
    return enc, dec


# =============================================================================
# A. Roundtrip matrix
# =============================================================================


@pytest.mark.parametrize("size", [0, 1, 16, 256, 4096, 65536, 65537])
@pytest.mark.parametrize("format_cls", [RawFormat, SSEFormat])
def test_roundtrip(size: int, format_cls: type) -> None:
    """Roundtrip across plaintext sizes including boundaries."""
    enc, dec = _make_pair(format_cls=format_cls)
    plaintext = os.urandom(size)
    wire = enc.encrypt(plaintext)
    if format_cls is SSEFormat:
        # SSE wire is wrapped; extract data field for decrypt
        data_line = bytes(wire).decode("ascii").split("\n")[1]
        data_field = data_line.replace("data: ", "")
        recovered = dec.decrypt(data_field)
    else:
        recovered = dec.decrypt(wire)
    assert bytes(recovered) == plaintext


def test_counter_sequence_1_to_100() -> None:
    """Counter increments correctly across many chunks."""
    enc, dec = _make_pair()
    for i in range(100):
        chunk = f"chunk-{i:04d}".encode()
        recovered = dec.decrypt(enc.encrypt(chunk))
        assert bytes(recovered) == chunk


# =============================================================================
# B. Edge cases
# =============================================================================


def test_empty_plaintext_wire_size_25() -> None:
    """Empty plaintext produces exactly 25-byte wire (length 4 + counter 4 + encoding_id 1 + tag 16)."""
    enc, _dec = _make_pair()
    wire = enc.encrypt(b"")
    assert len(wire) == 25


def test_empty_plaintext_roundtrip() -> None:
    enc, dec = _make_pair()
    assert bytes(dec.decrypt(enc.encrypt(b""))) == b""


def test_single_byte_plaintext() -> None:
    enc, dec = _make_pair()
    assert bytes(dec.decrypt(enc.encrypt(b"X"))) == b"X"


def test_chunk_size_64kb_boundary() -> None:
    enc, dec = _make_pair()
    plaintext = os.urandom(64 * 1024)
    assert bytes(dec.decrypt(enc.encrypt(plaintext))) == plaintext


def test_small_chunk_skips_compression() -> None:
    """Small chunks (< ZSTD_MIN_SIZE) use IDENTITY despite compress=True."""
    enc, dec = _make_pair(compress=True)
    plaintext = b"tiny"
    wire = enc.encrypt(plaintext)
    # Decode wire to inspect encoding_id
    counter, encoding_id, _ct = enc.format.decode(wire)
    assert counter == 1
    assert encoding_id == SSEEncodingId.IDENTITY  # 0x00
    # Roundtrip works
    assert bytes(dec.decrypt(wire)) == plaintext


# =============================================================================
# C. Out-of-bounds / DoS
# =============================================================================


def test_chunk_smaller_than_tag_rejected() -> None:
    """Wire with ciphertext < 16 bytes (no tag) raises DecryptionError."""
    _enc, dec = _make_pair()
    # Construct wire: length=5+15 (counter+eid+ct=20, ct<16), counter=1, eid=0, ct=15B garbage
    chunk_len = 4 + 1 + 15
    bogus_wire = struct.pack(">II", chunk_len, 1) + b"\x00" + b"\xaa" * 15
    with pytest.raises(DecryptionError, match="too short"):
        dec.decrypt(bogus_wire)


def test_oversized_length_prefix_rejected() -> None:
    """Length prefix > MAX_CHUNK_WIRE_SIZE raises immediately at parser."""
    from hpke_http.core import _ChunkStreamParser  # pyright: ignore[reportPrivateUsage]

    parser = _ChunkStreamParser()
    bogus = struct.pack(">I", 10_000_000) + b"\x00" * 10
    with pytest.raises(DecryptionError, match="too large"):
        list(parser.feed(bogus))


def test_truncated_wire_does_not_hang() -> None:
    """Partial wire bytes don't yield chunks; subsequent feed completes them."""
    from hpke_http.core import _ChunkStreamParser  # pyright: ignore[reportPrivateUsage]

    enc, _dec = _make_pair()
    full_wire = enc.encrypt(b"data")
    parser = _ChunkStreamParser()
    out1 = list(parser.feed(full_wire[:5]))  # only length + part of counter
    assert out1 == []
    out2 = list(parser.feed(full_wire[5:]))
    assert len(out2) == 1


def test_decompression_bomb_rejected() -> None:
    """Decompressed > MAX_DECOMPRESSED_CHUNK_SIZE raises (zip bomb prevention)."""
    enc, dec = _make_pair(compress=True)
    # 192KB of zeros compresses tiny; decompresses huge → exceeds MAX_DECOMPRESSED_CHUNK_SIZE
    huge = b"\x00" * (3 * 64 * 1024)
    wire = enc.encrypt(huge)
    with pytest.raises(DecryptionError, match="Decompressed chunk too large"):
        dec.decrypt(wire)


# =============================================================================
# D. Adversarial / tampering
# =============================================================================


def test_tampered_encoding_id_byte_raises() -> None:
    """Flipping encoding_id byte on wire fails AEAD authentication (downgrade resist)."""
    enc, dec = _make_pair()
    wire = bytearray(enc.encrypt(b"data" * 100))
    # encoding_id at offset 8 in RawFormat
    original = wire[8]
    wire[8] = original ^ 0x01  # flip a bit
    with pytest.raises(DecryptionError):
        dec.decrypt(bytes(wire))


def test_downgrade_zstd_to_identity_rejected() -> None:
    """Attacker can't change ZSTD encoding to IDENTITY to skip decompression."""
    enc, dec = _make_pair(compress=True)
    plaintext = b"x" * 1000
    wire = bytearray(enc.encrypt(plaintext))
    # encoding_id at offset 8 — change ZSTD (0x01) to IDENTITY (0x00)
    assert wire[8] == SSEEncodingId.ZSTD  # Verify we're testing the right path
    wire[8] = SSEEncodingId.IDENTITY
    with pytest.raises(DecryptionError):
        dec.decrypt(bytes(wire))


def test_tampered_ciphertext_byte_raises() -> None:
    """Flipping ciphertext byte fails AEAD."""
    enc, dec = _make_pair()
    wire = bytearray(enc.encrypt(b"data" * 100))
    wire[20] ^= 0x01
    with pytest.raises(DecryptionError):
        dec.decrypt(bytes(wire))


def test_tampered_tag_byte_raises() -> None:
    """Flipping tag byte fails AEAD."""
    enc, dec = _make_pair()
    wire = bytearray(enc.encrypt(b"data" * 100))
    wire[-1] ^= 0x01
    with pytest.raises(DecryptionError):
        dec.decrypt(bytes(wire))


def test_replay_same_chunk_raises() -> None:
    enc, dec = _make_pair()
    wire = enc.encrypt(b"data")
    dec.decrypt(wire)  # ok, counter=1
    with pytest.raises(ReplayAttackError):
        dec.decrypt(wire)  # replay: counter=1 again, but expected_counter=2


def test_out_of_order_chunks_raises() -> None:
    enc, dec = _make_pair()
    _w1 = enc.encrypt(b"first")
    w2 = enc.encrypt(b"second")
    with pytest.raises(ReplayAttackError):
        dec.decrypt(w2)  # expects counter=1, gets counter=2


def test_cross_session_replay_rejected() -> None:
    """Same key, different sessions (different salts) → wire from one session fails on another."""
    key = os.urandom(32)
    sess1 = StreamingSession.create(key)
    sess2 = StreamingSession.create(key)  # different salt
    enc1 = ChunkEncryptor(sess1, format=RawFormat())
    dec2 = ChunkDecryptor(sess2, format=RawFormat())
    wire = enc1.encrypt(b"data")
    with pytest.raises(DecryptionError):
        dec2.decrypt(wire)


# =============================================================================
# E. Buffer reuse safety (msgspec/picows pattern verification)
# =============================================================================


def test_reusable_scratch_does_not_corrupt_yielded_chunks() -> None:
    """100 chunks via reused Format scratch — all decrypt correctly even when held."""
    enc, dec = _make_pair()
    held: list[bytes] = [enc.encrypt(f"chunk-{i:04d}".encode()) for i in range(100)]
    for i, wire in enumerate(held):
        recovered = dec.decrypt(wire)
        assert bytes(recovered) == f"chunk-{i:04d}".encode()


def test_concurrent_encryptors_independent() -> None:
    """Two ChunkEncryptors must not share state (different sessions)."""
    enc1, dec1 = _make_pair()
    enc2, dec2 = _make_pair()
    w1 = enc1.encrypt(b"from-1")
    w2 = enc2.encrypt(b"from-2")
    assert bytes(dec1.decrypt(w1)) == b"from-1"
    assert bytes(dec2.decrypt(w2)) == b"from-2"


# =============================================================================
# F. Wire format size invariants
# =============================================================================


@pytest.mark.parametrize("pt_size", [0, 1, 100, 8000, 65536])
def test_wire_size_algebraic(pt_size: int) -> None:
    """Wire size = 25 + plaintext_size for RawFormat."""
    enc, _dec = _make_pair()
    wire = enc.encrypt(b"X" * pt_size)
    # length(4) + counter(4) + encoding_id(1) + ct(pt_size + 16) = 25 + pt_size
    assert len(wire) == 25 + pt_size


def test_length_prefix_value() -> None:
    """Length prefix = counter(4) + encoding_id(1) + ct+tag(N+16) for plaintext size N."""
    enc, _dec = _make_pair()
    pt_size = 100
    wire = enc.encrypt(b"X" * pt_size)
    length = struct.unpack_from(">I", wire, 0)[0]
    assert length == 4 + 1 + pt_size + 16


# =============================================================================
# G. v1 → v2 incompatibility
# =============================================================================


def test_v1_wire_rejected_by_v2_decoder() -> None:
    """Old v1 wire (encoding_id INSIDE plaintext, no AAD) fails on v2 decoder.

    v1 layout: ciphertext = AEAD(plaintext = encoding_id || data, aad=None)
    v2 decoder expects encoding_id at wire offset 8 (outside ct), passes as AAD.
    Decoding v1 wire under v2 misinterprets the wire structure and fails AEAD
    auth (wrong AAD) OR misreads encoding_id from a random byte.
    """
    key = os.urandom(32)
    salt = os.urandom(4)
    cipher = ChaCha20Poly1305(key)

    # Construct v1 wire: length || counter || ciphertext (encoding_id in plaintext)
    encoding_id_byte = bytes((0x00,))
    plaintext_v1 = encoding_id_byte + b"data"
    nonce = salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")
    ciphertext_v1 = cipher.encrypt(nonce, plaintext_v1, None)  # v1 had aad=None
    chunk_len_v1 = 4 + len(ciphertext_v1)
    v1_wire = struct.pack(">II", chunk_len_v1, 1) + ciphertext_v1

    # v2 decoder reads byte 8 as encoding_id (random ciphertext byte) and passes as AAD.
    # AEAD auth fails (AAD mismatch) OR encoding_id is out-of-range.
    session = StreamingSession(session_key=key, session_salt=salt)
    dec = ChunkDecryptor(session, format=RawFormat())
    with pytest.raises(DecryptionError):
        dec.decrypt(v1_wire)


# =============================================================================
# H. encrypt_into() public API (msgspec/picows pattern)
# =============================================================================


def test_encrypt_into_append_mode() -> None:
    """encrypt_into with offset=-1 (default) appends to dest, resizes."""
    enc, dec = _make_pair()
    dest = bytearray()
    n = enc.encrypt_into(b"hello", dest)
    assert n == len(dest) == 30  # 25 wire overhead + 5 plaintext
    assert bytes(dec.decrypt(bytes(dest))) == b"hello"


def test_encrypt_into_explicit_offset() -> None:
    """encrypt_into at explicit offset writes into pre-sized dest."""
    enc, dec = _make_pair()
    big_dest = bytearray(1000)
    n = enc.encrypt_into(b"data", big_dest, offset=100)
    # length(4) + counter(4) + encoding_id(1) + plaintext(4) + tag(16) = 29
    assert n == 29
    # Bytes at offset 100..100+29 are the wire chunk
    recovered = dec.decrypt(bytes(big_dest[100 : 100 + n]))
    assert bytes(recovered) == b"data"


def test_encrypt_into_dest_too_small_raises() -> None:
    """encrypt_into raises ValueError when dest can't fit at given offset."""
    enc, _dec = _make_pair()
    small_dest = bytearray(5)
    with pytest.raises(ValueError, match="dest buffer too small"):
        enc.encrypt_into(b"too big chunk", small_dest, offset=0)


def test_encrypt_into_reuse_dest_across_calls() -> None:
    """Same dest reused across many encrypt_into calls (zero alloc steady state)."""
    enc, dec = _make_pair()
    dest = bytearray()
    plaintexts = [b"chunk-1", b"chunk-2-medium", b"chunk-3-larger-content"]
    sizes: list[int] = []
    for pt in plaintexts:
        n = enc.encrypt_into(pt, dest)
        sizes.append(n)
    # Decrypt sequentially from accumulated dest
    cursor = 0
    for i, pt in enumerate(plaintexts):
        wire_chunk = bytes(dest[cursor : cursor + sizes[i]])
        assert bytes(dec.decrypt(wire_chunk)) == pt
        cursor += sizes[i]


def test_encrypt_into_rejects_sse_format() -> None:
    """encrypt_into requires RawFormat; SSEFormat raises TypeError."""
    key = os.urandom(32)
    session = StreamingSession.create(key)
    enc = ChunkEncryptor(session, format=SSEFormat())
    dest = bytearray()
    with pytest.raises(TypeError, match="encrypt_into requires RawFormat"):
        enc.encrypt_into(b"data", dest)


# =============================================================================
# I. AEAD AAD value verification (manual wire crafting)
# =============================================================================


def test_aad_is_encoding_id_byte() -> None:
    """Sender uses encoding_id byte as AAD; receiver passes same byte to verify."""
    key = os.urandom(32)
    salt = os.urandom(4)
    cipher = ChaCha20Poly1305(key)

    # Manually construct wire as encryptor would
    encoding_id = 0x00
    plaintext = b"hello"
    nonce = salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")
    ciphertext = cipher.encrypt(nonce, plaintext, bytes((encoding_id,)))

    # Build wire v2: length || counter || encoding_id || ct+tag
    chunk_len = 4 + 1 + len(ciphertext)
    wire = struct.pack(">II", chunk_len, 1) + bytes((encoding_id,)) + ciphertext

    # Decrypt via our library
    session = StreamingSession(session_key=key, session_salt=salt)
    dec = ChunkDecryptor(session, format=RawFormat())
    recovered = dec.decrypt(wire)
    assert bytes(recovered) == plaintext

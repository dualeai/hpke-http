"""Unit tests for standard response encryption (RawFormat)."""

import pytest
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from hpke_http.constants import (
    CHACHA20_POLY1305_KEY_SIZE,
    RESPONSE_KEY_LABEL,
    SSE_SESSION_KEY_LABEL,
    KemId,
)
from hpke_http.exceptions import DecryptionError, ReplayAttackError, SessionExpiredError
from hpke_http.hpke import setup_recipient_psk, setup_sender_psk
from hpke_http.streaming import (
    ChunkDecryptor,
    ChunkEncryptor,
    RawFormat,
    SSEFormat,
    StreamingSession,
    import_zstd,
)

_zstd_available = True
try:
    import_zstd()
except ImportError:
    _zstd_available = False

_requires_zstd = pytest.mark.skipif(not _zstd_available, reason="zstd not available")


def _make_cipher_and_nonce() -> tuple["ChaCha20Poly1305", bytes]:
    """Helper: build cipher + 12-byte nonce for direct Format tests."""
    key = b"0" * 32
    cipher = ChaCha20Poly1305(key)
    nonce = b"\x00" * 12
    return cipher, nonce


class TestRawFormat:
    """Test RawFormat binary encoding (wire format v2)."""

    def test_encrypt_chunk_decode_roundtrip(self) -> None:
        """Test RawFormat encrypt_chunk + decode is reversible."""

        fmt = RawFormat()
        cipher, nonce = _make_cipher_and_nonce()
        counter = 42
        encoding_id = 0x00
        plaintext = b"encrypted_data_here"

        encoded = fmt.encrypt_chunk(counter, encoding_id, plaintext, cipher, nonce)
        decoded_counter, decoded_eid, decoded_ct = fmt.decode(encoded)

        # Verify wire structure
        assert decoded_counter == counter
        assert decoded_eid == encoding_id
        # Decrypt to recover plaintext
        pt_buf = bytearray(len(decoded_ct) - 16)
        cipher.decrypt_into(nonce, decoded_ct, bytes((encoding_id,)), pt_buf)
        assert bytes(pt_buf) == plaintext

    def test_encrypt_chunk_format(self) -> None:
        """Test RawFormat wire structure: length(4) || counter(4) || encoding_id(1) || ct+tag."""
        fmt = RawFormat()
        cipher, nonce = _make_cipher_and_nonce()
        counter = 1
        encoding_id = 0x00
        plaintext = b"test"  # 4 bytes; ciphertext = 4 + 16 (tag) = 20

        encoded = fmt.encrypt_chunk(counter, encoding_id, plaintext, cipher, nonce)

        # Length prefix: counter(4) + encoding_id(1) + ct+tag(20) = 25
        assert encoded[:4] == b"\x00\x00\x00\x19"
        # Counter (big-endian)
        assert encoded[4:8] == b"\x00\x00\x00\x01"
        # Encoding ID
        assert encoded[8] == encoding_id
        # Total wire size: 4 + 4 + 1 + 4 + 16 = 29 bytes
        assert len(encoded) == 29

    def test_counter_boundary_max(self) -> None:
        """Test encoding max counter value (2**32 - 1)."""
        fmt = RawFormat()
        cipher, nonce = _make_cipher_and_nonce()
        counter = 2**32 - 1
        plaintext = b"x"

        encoded = fmt.encrypt_chunk(counter, 0x00, plaintext, cipher, nonce)
        decoded_counter, _eid, _ct = fmt.decode(encoded)

        assert decoded_counter == counter

    def test_length_prefix_value(self) -> None:
        """Test that length prefix correctly encodes chunk size for v2 layout."""
        fmt = RawFormat()
        cipher, nonce = _make_cipher_and_nonce()
        counter = 1
        plaintext = b"x" * 100  # 100-byte plaintext; ct+tag = 116

        encoded = fmt.encrypt_chunk(counter, 0x00, plaintext, cipher, nonce)

        # Length = counter(4) + encoding_id(1) + ct+tag(116) = 121
        length = int.from_bytes(encoded[:4], "big")
        assert length == 121
        assert len(encoded) == 4 + length  # length prefix + chunk


class TestSSEFormatComparison:
    """Test SSEFormat vs RawFormat differences."""

    def test_sse_format_is_text_based(self) -> None:
        """SSEFormat produces text (event: enc\\ndata: ...)."""
        fmt = SSEFormat()
        cipher, nonce = _make_cipher_and_nonce()
        encoded = fmt.encrypt_chunk(1, 0x00, b"test", cipher, nonce)

        assert encoded.startswith(b"event: enc\ndata: ")
        assert encoded.endswith(b"\n\n")

    def test_raw_format_is_binary(self) -> None:
        """RawFormat produces raw binary with length prefix + encoding_id."""
        fmt = RawFormat()
        cipher, nonce = _make_cipher_and_nonce()
        encoded = fmt.encrypt_chunk(1, 0x00, b"test", cipher, nonce)

        # length(4) + counter(4) + encoding_id(1) + ct+tag(20) = 29
        assert len(encoded) == 29
        # length=25 (counter 4 + encoding_id 1 + ct+tag 20), counter=1, encoding_id=0x00
        assert encoded[:9] == b"\x00\x00\x00\x19\x00\x00\x00\x01\x00"


class TestChunkEncryptorWithRawFormat:
    """Test ChunkEncryptor with RawFormat strategy."""

    def test_encrypt_decrypt_single_chunk(self) -> None:
        """Test single chunk roundtrip with RawFormat."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat())
        decryptor = ChunkDecryptor(session, format=RawFormat())

        original = b"Hello, encrypted response!"

        encrypted = encryptor.encrypt(original)
        decrypted = decryptor.decrypt(encrypted)

        assert decrypted == original

    def test_encrypt_decrypt_multiple_chunks(self) -> None:
        """Test multiple chunks with RawFormat (simulates chunked response)."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat())
        decryptor = ChunkDecryptor(session, format=RawFormat())

        chunks = [
            b'{"status": "ok"}',
            b'{"data": [1, 2, 3]}',
            b'{"end": true}',
        ]

        for original in chunks:
            encrypted = encryptor.encrypt(original)
            decrypted = decryptor.decrypt(encrypted)
            assert decrypted == original

    def test_counter_included_in_output(self) -> None:
        """Test that counter is included in RawFormat output."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session, format=RawFormat())

        encrypted1 = encryptor.encrypt(b"chunk1")
        encrypted2 = encryptor.encrypt(b"chunk2")

        # Counter is at bytes 4-8 (after 4-byte length prefix)
        counter1 = int.from_bytes(encrypted1[4:8], "big")
        counter2 = int.from_bytes(encrypted2[4:8], "big")

        assert counter1 == 1
        assert counter2 == 2

    def test_replay_attack_rejected(self) -> None:
        """Test that replayed chunks are rejected."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat())
        decryptor = ChunkDecryptor(session, format=RawFormat())

        enc1 = encryptor.encrypt(b"first")
        encryptor.encrypt(b"second")  # Advance counter

        # Decrypt first - works
        decryptor.decrypt(enc1)

        # Try to replay first chunk - should fail
        with pytest.raises(ReplayAttackError) as exc:
            decryptor.decrypt(enc1)

        assert exc.value.expected == 2
        assert exc.value.received == 1

    def test_out_of_order_rejected(self) -> None:
        """Test that out-of-order chunks are rejected."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat())
        decryptor = ChunkDecryptor(session, format=RawFormat())

        encryptor.encrypt(b"first")  # Advance counter
        enc2 = encryptor.encrypt(b"second")

        # Skip first, try to decrypt second - should fail
        with pytest.raises(ReplayAttackError) as exc:
            decryptor.decrypt(enc2)

        assert exc.value.expected == 1
        assert exc.value.received == 2


class TestResponseKeyDerivation:
    """Test response key derivation differs from SSE key."""

    def test_response_key_differs_from_sse_key(
        self,
        platform_keys: dict[KemId, tuple[bytes, bytes]],
        kem: KemId,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Response key and SSE key should be different (domain separation)."""
        _sk_r, pk_r = platform_keys[kem]

        # Sender context (client side)
        sender_ctx = setup_sender_psk(pk_r, b"", test_psk, test_psk_id, kem_id=kem)

        # Export both keys
        response_key = sender_ctx.export(RESPONSE_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
        sse_key = sender_ctx.export(SSE_SESSION_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)

        # Keys must be different
        assert response_key != sse_key
        assert len(response_key) == 32
        assert len(sse_key) == 32

    def test_sender_recipient_derive_same_key(
        self,
        platform_keys: dict[KemId, tuple[bytes, bytes]],
        kem: KemId,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Sender and recipient should derive the same response key."""
        sk_r, pk_r = platform_keys[kem]

        # Sender context (client)
        sender_ctx = setup_sender_psk(pk_r, b"", test_psk, test_psk_id, kem_id=kem)

        # Recipient context (server)
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"", test_psk, test_psk_id, kem_id=kem)

        # Derive response keys
        sender_response_key = sender_ctx.export(RESPONSE_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
        recipient_response_key = recipient_ctx.export(RESPONSE_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)

        assert sender_response_key == recipient_response_key


class TestResponseEncryptionPayloads:
    """Test various payload types with RawFormat."""

    def test_json_payload(self) -> None:
        """JSON response body should encrypt/decrypt correctly."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat())
        decryptor = ChunkDecryptor(session, format=RawFormat())

        original = b'{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'

        encrypted = encryptor.encrypt(original)
        decrypted = decryptor.decrypt(encrypted)

        assert decrypted == original

    def test_binary_payload(self) -> None:
        """Binary data should encrypt/decrypt correctly."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat())
        decryptor = ChunkDecryptor(session, format=RawFormat())

        # Binary data with null bytes and high bytes
        original = bytes(range(256)) * 4

        encrypted = encryptor.encrypt(original)
        decrypted = decryptor.decrypt(encrypted)

        assert decrypted == original

    def test_large_payload(self) -> None:
        """Large payload (~1MB) should work."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat())
        decryptor = ChunkDecryptor(session, format=RawFormat())

        original = b"x" * (1024 * 1024)  # 1MB

        encrypted = encryptor.encrypt(original)
        decrypted = decryptor.decrypt(encrypted)

        assert decrypted == original

    def test_empty_payload(self) -> None:
        """Empty payload should work."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat())
        decryptor = ChunkDecryptor(session, format=RawFormat())

        original = b""

        encrypted = encryptor.encrypt(original)
        decrypted = decryptor.decrypt(encrypted)

        assert decrypted == original


@_requires_zstd
class TestResponseWithCompression:
    """Test response encryption with Zstd compression."""

    def test_compressed_chunk_roundtrip(self) -> None:
        """Compressed chunk should roundtrip correctly."""

        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat(), compress=True)
        decryptor = ChunkDecryptor(session, format=RawFormat())

        # Large enough to trigger compression (>= ZSTD_MIN_SIZE)
        original = b"repeated_text_" * 100

        encrypted = encryptor.encrypt(original)
        decrypted = decryptor.decrypt(encrypted)

        assert decrypted == original

    def test_small_chunk_not_compressed(self) -> None:
        """Small chunks should not be compressed (identity encoding)."""

        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat(), compress=True)
        decryptor = ChunkDecryptor(session, format=RawFormat())

        # Small chunk (< ZSTD_MIN_SIZE)
        original = b"small"

        encrypted = encryptor.encrypt(original)
        decrypted = decryptor.decrypt(encrypted)

        assert decrypted == original


class TestDecryptionErrors:
    """Test error handling in decryption."""

    def test_truncated_ciphertext(self) -> None:
        """Truncated ciphertext should fail."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        decryptor = ChunkDecryptor(session, format=RawFormat())

        # Just counter, no ciphertext
        truncated = b"\x00\x00\x00\x01"

        with pytest.raises(DecryptionError):
            decryptor.decrypt(truncated)

    def test_corrupted_ciphertext(self) -> None:
        """Corrupted ciphertext should fail auth check."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat())
        decryptor = ChunkDecryptor(session, format=RawFormat())

        encrypted = encryptor.encrypt(b"test")

        # Corrupt the ciphertext (flip a byte)
        corrupted = bytearray(encrypted)
        corrupted[-1] ^= 0xFF
        corrupted = bytes(corrupted)

        with pytest.raises(DecryptionError):
            decryptor.decrypt(corrupted)

    def test_wrong_key(self) -> None:
        """Decryption with wrong key should fail."""
        session1 = StreamingSession.create(b"k" * 32)
        session2 = StreamingSession.create(b"x" * 32)

        encryptor = ChunkEncryptor(session1, format=RawFormat())
        decryptor = ChunkDecryptor(session2, format=RawFormat())

        encrypted = encryptor.encrypt(b"secret")

        with pytest.raises(DecryptionError):
            decryptor.decrypt(encrypted)


class TestMultiChunkResponse:
    """Test simulated multi-chunk response scenarios."""

    def test_concatenated_chunks(self) -> None:
        """Simulate reading concatenated chunks from response body."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat())

        # Server encrypts multiple chunks
        chunk1 = encryptor.encrypt(b"part1")
        chunk2 = encryptor.encrypt(b"part2")
        chunk3 = encryptor.encrypt(b"part3")

        # Wire: concatenated chunks
        wire_data = chunk1 + chunk2 + chunk3

        # Client parses and decrypts
        decryptor = ChunkDecryptor(session, format=RawFormat())
        result = bytearray()
        offset = 0

        # Process each chunk (we know the boundaries for this test)
        for expected_chunk in [chunk1, chunk2, chunk3]:
            chunk_len = len(expected_chunk)
            chunk_data = wire_data[offset : offset + chunk_len]
            plaintext = decryptor.decrypt(chunk_data)
            result.extend(plaintext)
            offset += chunk_len

        assert bytes(result) == b"part1part2part3"

    def test_1000_chunks_sequential(self) -> None:
        """1000 sequential chunks should work."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = ChunkEncryptor(session, format=RawFormat())
        decryptor = ChunkDecryptor(session, format=RawFormat())

        for i in range(1000):
            original = f"chunk_{i}".encode()
            encrypted = encryptor.encrypt(original)
            decrypted = decryptor.decrypt(encrypted)
            assert decrypted == original

        # Verify final counter state
        assert encryptor.counter == 1001
        assert decryptor.expected_counter == 1001


class TestRawFormatEdgeCases:
    """Edge case tests for RawFormat (wire format v2: counter || encoding_id || ct+tag)."""

    def test_decode_string_input(self) -> None:
        """RawFormat.decode accepts string input via latin-1 encoding."""
        fmt = RawFormat()
        # length=15 (counter=4 + encoding_id=1 + ciphertext=10), counter=1, eid=0x00, ct="ciphertext"
        binary_data = b"\x00\x00\x00\x0f" + b"\x00\x00\x00\x01" + b"\x00" + b"ciphertext"
        string_data = binary_data.decode("latin-1")

        counter, encoding_id, ciphertext = fmt.decode(string_data)

        assert counter == 1
        assert encoding_id == 0x00
        assert bytes(ciphertext) == b"ciphertext"

    def test_decode_counter_zero(self) -> None:
        """Counter=0 decodes correctly (edge case)."""
        fmt = RawFormat()
        # length=12 (counter=4 + encoding_id=1 + ciphertext=7), counter=0, eid=0x00, ct="payload"
        data = b"\x00\x00\x00\x0c" + b"\x00\x00\x00\x00" + b"\x00" + b"payload"

        counter, encoding_id, ciphertext = fmt.decode(data)

        assert counter == 0
        assert encoding_id == 0x00
        assert bytes(ciphertext) == b"payload"

    def test_decode_minimum_data_counter_only(self) -> None:
        """Decode with length + counter + encoding_id, no ciphertext."""
        fmt = RawFormat()
        # length=5 (counter + encoding_id), counter=5, eid=0x01
        data = b"\x00\x00\x00\x05" + b"\x00\x00\x00\x05" + b"\x01"

        counter, encoding_id, ciphertext = fmt.decode(data)

        assert counter == 5
        assert encoding_id == 0x01
        assert bytes(ciphertext) == b""

    def test_encrypt_chunk_counter_zero(self) -> None:
        """encrypt_chunk with counter=0 produces valid wire (encryptor starts at 1, but format itself accepts 0)."""
        fmt = RawFormat()
        cipher, nonce = _make_cipher_and_nonce()

        encoded = fmt.encrypt_chunk(0, 0x00, b"test", cipher, nonce)

        # length = counter(4) + encoding_id(1) + ct+tag(20) = 25 = 0x19
        assert encoded[:4] == b"\x00\x00\x00\x19"
        assert encoded[4:8] == b"\x00\x00\x00\x00"
        assert encoded[8] == 0x00


class TestSessionEdgeCases:
    """Edge case tests for StreamingSession."""

    def test_session_serialization_roundtrip(self) -> None:
        """Session serialize/deserialize preserves salt."""
        key = b"k" * 32
        original = StreamingSession.create(key)

        # Serialize (just the salt)
        serialized = original.serialize()
        assert len(serialized) == 4  # SSE_SESSION_SALT_SIZE

        # Deserialize with same key
        restored = StreamingSession.deserialize(serialized, key)

        assert restored.session_key == original.session_key
        assert restored.session_salt == original.session_salt

    def test_multiple_independent_sessions(self) -> None:
        """Multiple sessions with same key but different salts don't interfere."""
        key = b"k" * 32

        session1 = StreamingSession.create(key)
        session2 = StreamingSession.create(key)

        # Sessions should have different salts (random)
        assert session1.session_salt != session2.session_salt

        # Encrypt with session1
        encryptor1 = ChunkEncryptor(session1, format=RawFormat())
        encrypted1 = encryptor1.encrypt(b"from session 1")

        # Encrypt with session2
        encryptor2 = ChunkEncryptor(session2, format=RawFormat())
        encrypted2 = encryptor2.encrypt(b"from session 2")

        # Ciphertexts should be different (different salts → different nonces)
        assert encrypted1 != encrypted2

        # Each decryptor only works with its own session's data
        # Create fresh sessions with the same salts for decryption
        session1_clone = StreamingSession(session_key=key, session_salt=session1.session_salt)
        session2_clone = StreamingSession(session_key=key, session_salt=session2.session_salt)

        decryptor1 = ChunkDecryptor(session1_clone, format=RawFormat())
        decryptor2 = ChunkDecryptor(session2_clone, format=RawFormat())

        assert decryptor1.decrypt(encrypted1) == b"from session 1"
        assert decryptor2.decrypt(encrypted2) == b"from session 2"

        # Cross-session decryption fails (wrong salt → wrong nonce → auth fails)
        session_wrong = StreamingSession(session_key=key, session_salt=session1.session_salt)
        decryptor_wrong = ChunkDecryptor(session_wrong, format=RawFormat())
        with pytest.raises(DecryptionError):
            decryptor_wrong.decrypt(encrypted2)


class TestCounterEdgeCases:
    """Edge case tests for counter handling."""

    def test_counter_exactly_at_max(self) -> None:
        """Counter at exactly SSE_MAX_COUNTER works."""
        from hpke_http.constants import SSE_MAX_COUNTER

        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session, format=RawFormat())

        # Set counter to exactly max
        encryptor.counter = SSE_MAX_COUNTER

        # Should succeed (max is allowed)
        encrypted = encryptor.encrypt(b"at max")
        assert encrypted is not None

        # Next call should fail (counter now > max)
        with pytest.raises(SessionExpiredError):
            encryptor.encrypt(b"over max")

    def test_counter_starts_at_one(self) -> None:
        """Verify counter starts at 1, not 0."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session, format=RawFormat())

        assert encryptor.counter == 1

        encrypted = encryptor.encrypt(b"first")
        # Counter is at bytes 4-8 (after 4-byte length prefix)
        counter = int.from_bytes(encrypted[4:8], "big")
        assert counter == 1

    def test_decryptor_expects_counter_one(self) -> None:
        """Verify decryptor expects counter starting at 1."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        decryptor = ChunkDecryptor(session, format=RawFormat())

        assert decryptor.expected_counter == 1


class TestPayloadEdgeCases:
    """Edge case tests for various payload sizes."""

    def test_one_byte_payload(self) -> None:
        """Single byte payload works."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session, format=RawFormat())
        decryptor = ChunkDecryptor(session, format=RawFormat())

        encrypted = encryptor.encrypt(b"x")
        decrypted = decryptor.decrypt(encrypted)

        assert decrypted == b"x"

    @_requires_zstd
    def test_payload_at_compression_threshold(self) -> None:
        """Payload exactly at ZSTD_MIN_SIZE triggers compression."""
        from hpke_http.constants import ZSTD_MIN_SIZE

        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session, format=RawFormat(), compress=True)
        decryptor = ChunkDecryptor(session, format=RawFormat())

        # Exactly at threshold (64 bytes) - should compress
        payload = b"a" * ZSTD_MIN_SIZE
        encrypted = encryptor.encrypt(payload)
        decrypted = decryptor.decrypt(encrypted)

        assert decrypted == payload

    @_requires_zstd
    def test_payload_just_under_compression_threshold(self) -> None:
        """Payload just under ZSTD_MIN_SIZE doesn't compress."""
        from hpke_http.constants import ZSTD_MIN_SIZE

        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session, format=RawFormat(), compress=True)
        decryptor = ChunkDecryptor(session, format=RawFormat())

        # One byte under threshold - should NOT compress
        payload = b"a" * (ZSTD_MIN_SIZE - 1)
        encrypted = encryptor.encrypt(payload)
        decrypted = decryptor.decrypt(encrypted)

        assert decrypted == payload


class TestAdversarialInputs:
    """Tests for adversarial/weird inputs."""

    def test_unknown_encoding_id_raises(self) -> None:
        """Unknown encoding_id (not 0x00/0x01/0x02) raises DecryptionError (wire v2)."""
        import struct

        key = b"0" * 32
        session = StreamingSession.create(key)
        session2 = StreamingSession(session_key=key, session_salt=session.session_salt)

        # Wire v2: encoding_id is OUTSIDE ciphertext (in AAD).
        # Craft wire with unknown encoding_id (0x99). Sender uses 0x99 as AAD too,
        # so AEAD authenticates successfully, then decryptor's encoding_id range
        # check rejects it with "Unknown encoding".
        cipher = ChaCha20Poly1305(key)
        bad_eid = 0x99
        plaintext = b"data"
        nonce = session.session_salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")
        # Encrypt plaintext with AAD = bad encoding_id byte
        ciphertext = cipher.encrypt(nonce, plaintext, bytes((bad_eid,)))

        # Build wire v2: length(4) || counter(4) || encoding_id(1) || ct+tag
        chunk_len = 4 + 1 + len(ciphertext)
        bad_message = struct.pack(">II", chunk_len, 1) + bytes((bad_eid,)) + ciphertext

        decryptor = ChunkDecryptor(session2, format=RawFormat())
        with pytest.raises(DecryptionError, match="Unknown encoding"):
            decryptor.decrypt(bad_message)

    def test_raw_format_decode_very_short(self) -> None:
        """RawFormat.decode with minimal input (just counter + encoding_id) handles edge case."""
        fmt = RawFormat()

        # length=5 (counter + encoding_id), counter=5, eid=0x00
        data = b"\x00\x00\x00\x05\x00\x00\x00\x05\x00"
        counter, encoding_id, ciphertext = fmt.decode(data)

        assert counter == 5
        assert encoding_id == 0x00
        assert bytes(ciphertext) == b""

    def test_raw_format_decode_empty_raises(self) -> None:
        """RawFormat.decode with empty input raises IndexError (encoding_id byte missing)."""
        fmt = RawFormat()

        # Empty wire has no length prefix, no counter, no encoding_id.
        # Decoder attempts to read encoding_id at offset 8 → IndexError.
        # Real callers parse via _ChunkStreamParser which validates length first.
        with pytest.raises(IndexError):
            fmt.decode(b"")

    def test_all_zeros_ciphertext_fails_auth(self) -> None:
        """All-zeros ciphertext fails authentication."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        decryptor = ChunkDecryptor(session, format=RawFormat())

        # length(4) + Counter(4) + all-zeros ciphertext (17 = 1 encoding + 16 tag minimum)
        # length = 4 + 20 = 24 = 0x18
        fake_message = b"\x00\x00\x00\x18" + b"\x00\x00\x00\x01" + (b"\x00" * 20)

        with pytest.raises(DecryptionError):
            decryptor.decrypt(fake_message)

    def test_all_ff_ciphertext_fails_auth(self) -> None:
        """All-0xFF ciphertext fails authentication."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        decryptor = ChunkDecryptor(session, format=RawFormat())

        # length(4) + Counter(4) + all-FF ciphertext
        # length = 4 + 20 = 24 = 0x18
        fake_message = b"\x00\x00\x00\x18" + b"\x00\x00\x00\x01" + (b"\xff" * 20)

        with pytest.raises(DecryptionError):
            decryptor.decrypt(fake_message)

    def test_counter_manipulation_detected(self) -> None:
        """Modifying counter in encrypted message is detected."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session, format=RawFormat())
        decryptor = ChunkDecryptor(session, format=RawFormat())

        encrypted = encryptor.encrypt(b"legitimate data")

        # Tamper with counter at bytes 4-8 (change from 1 to 2)
        # Keep length prefix (bytes 0-4), change counter (bytes 4-8), keep rest
        tampered = encrypted[:4] + b"\x00\x00\x00\x02" + encrypted[8:]

        # Should fail - counter mismatch (expects 1, got 2) OR auth fail
        with pytest.raises((ReplayAttackError, DecryptionError)):
            decryptor.decrypt(tampered)


class TestThreadSafety:
    """Thread safety tests for ChunkEncryptor."""

    def test_concurrent_encryption(self) -> None:
        """Concurrent encryption from multiple threads doesn't corrupt state."""
        import threading

        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session, format=RawFormat())

        results: list[int] = []
        errors: list[BaseException] = []

        def encrypt_chunk(chunk_id: int) -> None:
            try:
                encrypted = encryptor.encrypt(f"chunk-{chunk_id}".encode())
                # Counter is at bytes 4-8 (after 4-byte length prefix)
                counter = int.from_bytes(encrypted[4:8], "big")
                results.append(counter)
            except (SessionExpiredError, DecryptionError, ValueError, RuntimeError) as e:
                errors.append(e)

        # Launch 100 concurrent encryptions
        threads = [threading.Thread(target=encrypt_chunk, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert not errors, f"Errors occurred: {errors}"

        # All counters should be unique (1 through 100)
        assert len(results) == 100
        assert len(set(results)) == 100  # All unique
        assert set(results) == set(range(1, 101))

        # Counter should be at 101
        assert encryptor.counter == 101


class TestSSEFormatEdgeCases:
    """Edge case tests for SSEFormat (wire format v2)."""

    def test_sse_format_decode_string_input(self) -> None:
        """SSEFormat.decode accepts string input (base64 data)."""
        fmt = SSEFormat()
        cipher, nonce = _make_cipher_and_nonce()
        # Encrypt some plaintext
        encoded = fmt.encrypt_chunk(1, 0x00, b"plaintext", cipher, nonce)

        # Extract just the base64 data field
        data_line = encoded.decode("ascii").split("\n")[1]
        data_str = data_line.replace("data: ", "")

        # Decode from string: returns (counter, encoding_id, ciphertext_mv)
        counter, encoding_id, ciphertext = fmt.decode(data_str)

        assert counter == 1
        assert encoding_id == 0x00
        # Decrypt to recover plaintext
        pt_buf = bytearray(len(ciphertext) - 16)
        cipher.decrypt_into(nonce, ciphertext, bytes((encoding_id,)), pt_buf)
        assert bytes(pt_buf) == b"plaintext"

    def test_sse_format_decode_bytes_input(self) -> None:
        """SSEFormat.decode accepts bytes input."""
        fmt = SSEFormat()
        cipher, nonce = _make_cipher_and_nonce()
        encoded = fmt.encrypt_chunk(1, 0x00, b"test", cipher, nonce)

        # Extract just the base64 data
        data_line = encoded.decode("ascii").split("\n")[1]
        data_bytes = data_line.replace("data: ", "").encode("ascii")

        counter, encoding_id, ciphertext = fmt.decode(data_bytes)

        assert counter == 1
        assert encoding_id == 0x00
        # Decrypt to recover plaintext
        pt_buf = bytearray(len(ciphertext) - 16)
        cipher.decrypt_into(nonce, ciphertext, bytes((encoding_id,)), pt_buf)
        assert bytes(pt_buf) == b"test"

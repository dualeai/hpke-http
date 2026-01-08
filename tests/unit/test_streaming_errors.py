"""SSE streaming error handling tests.

Tests all error paths in SSE encryption/decryption.
"""

import base64

import pytest

from hpke_http.exceptions import DecryptionError, ReplayAttackError, SessionExpiredError
from hpke_http.streaming import (
    ChunkDecryptor,
    ChunkEncryptor,
    StreamingSession,
)
from tests.conftest import extract_sse_data_field, make_sse_session


class TestDecryptionErrors:
    """Test DecryptionError paths in ChunkDecryptor."""

    def test_invalid_base64_raises(self) -> None:
        """Invalid base64 should raise DecryptionError."""
        session = make_sse_session()
        decryptor = ChunkDecryptor(session)

        # Single character causes base64 decode error (1 mod 4 = 1 is invalid)
        with pytest.raises(DecryptionError, match="Failed to decode chunk"):
            decryptor.decrypt("x")

    def test_truncated_payload_raises(self) -> None:
        """Payload shorter than counter + min ciphertext should raise."""
        session = make_sse_session()
        decryptor = ChunkDecryptor(session)

        # 4-byte counter + 16-byte tag minimum = 20 bytes
        # SSEFormat uses standard base64 (not base64url)
        short_payload = base64.b64encode(b"short").decode("ascii")  # Only 5 bytes

        with pytest.raises(DecryptionError, match="Ciphertext too short"):
            decryptor.decrypt(short_payload)

    def test_corrupted_ciphertext_raises(self) -> None:
        """Bit flip in ciphertext should raise DecryptionError."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session)
        decryptor = ChunkDecryptor(session)

        # Encrypt valid chunk
        encrypted = encryptor.encrypt(b"event: test\ndata: {}\n\n")
        encoded = extract_sse_data_field(encrypted)

        # Decode, corrupt, re-encode (SSEFormat uses standard base64)
        payload = bytearray(base64.b64decode(encoded))
        payload[10] ^= 0xFF  # Flip bits in ciphertext
        corrupted = base64.b64encode(bytes(payload)).decode("ascii")

        with pytest.raises(DecryptionError, match="Decryption failed"):
            decryptor.decrypt(corrupted)

    def test_wrong_session_key_raises(self) -> None:
        """Wrong session key should raise DecryptionError."""
        session1 = StreamingSession(session_key=b"a" * 32, session_salt=b"salt")
        session2 = StreamingSession(session_key=b"b" * 32, session_salt=b"salt")

        encryptor = ChunkEncryptor(session1)
        decryptor = ChunkDecryptor(session2)

        encrypted = encryptor.encrypt(b"event: test\ndata: {}\n\n")
        encoded = extract_sse_data_field(encrypted)

        with pytest.raises(DecryptionError, match="Decryption failed"):
            decryptor.decrypt(encoded)

    def test_wrong_session_salt_raises(self) -> None:
        """Wrong session salt should raise DecryptionError."""
        session1 = StreamingSession(session_key=b"k" * 32, session_salt=b"aaa1")
        session2 = StreamingSession(session_key=b"k" * 32, session_salt=b"bbb2")

        encryptor = ChunkEncryptor(session1)
        decryptor = ChunkDecryptor(session2)

        encrypted = encryptor.encrypt(b"event: test\ndata: {}\n\n")
        encoded = extract_sse_data_field(encrypted)

        with pytest.raises(DecryptionError, match="Decryption failed"):
            decryptor.decrypt(encoded)


class TestReplayOrderErrors:
    """Test replay attack and counter order error paths."""

    def test_replay_same_event_raises(self) -> None:
        """Decrypting same event twice should raise ReplayAttackError."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session)
        decryptor = ChunkDecryptor(session)

        encrypted = encryptor.encrypt(b"event: test\ndata: {}\n\n")
        data = extract_sse_data_field(encrypted)

        # First decryption succeeds
        decryptor.decrypt(data)

        # Second decryption of same event fails
        with pytest.raises(ReplayAttackError) as exc:
            decryptor.decrypt(data)

        assert exc.value.expected == 2
        assert exc.value.received == 1

    def test_out_of_order_raises(self) -> None:
        """Receiving event 2 when expecting 1 should raise ReplayAttackError."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session)
        decryptor = ChunkDecryptor(session)

        # Generate events 1 and 2
        _event1 = encryptor.encrypt(b"event: first\n\n")
        event2 = encryptor.encrypt(b"event: second\n\n")
        data2 = extract_sse_data_field(event2)

        # Try to decrypt event 2 first (expecting 1)
        with pytest.raises(ReplayAttackError) as exc:
            decryptor.decrypt(data2)

        assert exc.value.expected == 1
        assert exc.value.received == 2

    def test_skipped_counter_raises(self) -> None:
        """Skipping counter (1, then 3) should raise ReplayAttackError."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session)
        decryptor = ChunkDecryptor(session)

        # Generate events 1, 2, 3
        event1 = encryptor.encrypt(b"event: one\n\n")
        _event2 = encryptor.encrypt(b"event: two\n\n")
        event3 = encryptor.encrypt(b"event: three\n\n")

        # Decrypt event 1
        decryptor.decrypt(extract_sse_data_field(event1))

        # Skip event 2, try event 3 (expecting 2)
        with pytest.raises(ReplayAttackError) as exc:
            decryptor.decrypt(extract_sse_data_field(event3))

        assert exc.value.expected == 2
        assert exc.value.received == 3


class TestSessionErrors:
    """Test session serialization error paths."""

    def test_deserialize_wrong_length_short(self) -> None:
        """Deserialize with 3 bytes (too short) should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid session data length"):
            StreamingSession.deserialize(b"abc", session_key=b"0" * 32)

    def test_deserialize_wrong_length_long(self) -> None:
        """Deserialize with 5 bytes (too long) should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid session data length"):
            StreamingSession.deserialize(b"abcde", session_key=b"0" * 32)

    def test_deserialize_empty(self) -> None:
        """Deserialize with 0 bytes should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid session data length"):
            StreamingSession.deserialize(b"", session_key=b"0" * 32)


class TestCounterExhaustion:
    """Test counter exhaustion (SessionExpiredError)."""

    def test_counter_exhaustion_raises(self) -> None:
        """Counter exceeding max should raise SessionExpiredError."""
        from hpke_http.constants import SSE_MAX_COUNTER

        session = StreamingSession(session_key=b"0" * 32, session_salt=b"salt")
        encryptor = ChunkEncryptor(session)

        # Set counter to max value + 1
        encryptor.counter = SSE_MAX_COUNTER + 1

        with pytest.raises(SessionExpiredError, match="counter exhausted"):
            encryptor.encrypt(b"event: test\n\n")

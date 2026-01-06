"""SSE streaming error handling tests.

Tests all error paths in SSE encryption/decryption.
"""

import pytest

from hpke_http.exceptions import DecryptionError, ReplayAttackError, SessionExpiredError
from hpke_http.headers import b64url_encode
from hpke_http.streaming import (
    SSEDecryptor,
    SSEEncryptor,
    StreamingSession,
)


class TestDecryptionErrors:
    """Test DecryptionError paths in SSEDecryptor."""

    def _make_session(self) -> StreamingSession:
        """Create a test session."""
        return StreamingSession(
            session_key=b"0" * 32,
            session_salt=b"salt",
        )

    def test_invalid_base64_raises(self) -> None:
        """Invalid base64url should raise DecryptionError."""
        session = self._make_session()
        decryptor = SSEDecryptor(session)

        # Single character causes base64 decode error (1 mod 4 = 1 is invalid)
        with pytest.raises(DecryptionError, match="Invalid base64url encoding"):
            decryptor.decrypt_event("x")

    def test_truncated_payload_raises(self) -> None:
        """Payload shorter than counter + min ciphertext should raise."""
        session = self._make_session()
        decryptor = SSEDecryptor(session)

        # 4-byte counter + 16-byte tag minimum = 20 bytes
        short_payload = b64url_encode(b"short")  # Only 5 bytes

        with pytest.raises(DecryptionError, match="Payload too short"):
            decryptor.decrypt_event(short_payload)

    def test_corrupted_ciphertext_raises(self) -> None:
        """Bit flip in ciphertext should raise DecryptionError."""
        session = self._make_session()
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        # Encrypt valid event
        sse_event = encryptor.encrypt_event("test", {"key": "value"})
        data_line = next(line for line in sse_event.split("\n") if line.startswith("data:"))
        encoded = data_line.split(": ", 1)[1]

        # Decode, corrupt, re-encode
        from hpke_http.headers import b64url_decode

        payload = bytearray(b64url_decode(encoded))
        payload[10] ^= 0xFF  # Flip bits in ciphertext
        corrupted = b64url_encode(bytes(payload))

        with pytest.raises(DecryptionError, match="decryption failed"):
            decryptor.decrypt_event(corrupted)

    def test_wrong_session_key_raises(self) -> None:
        """Wrong session key should raise DecryptionError."""
        session1 = StreamingSession(session_key=b"a" * 32, session_salt=b"salt")
        session2 = StreamingSession(session_key=b"b" * 32, session_salt=b"salt")

        encryptor = SSEEncryptor(session1)
        decryptor = SSEDecryptor(session2)

        sse_event = encryptor.encrypt_event("test", {"key": "value"})
        data_line = next(line for line in sse_event.split("\n") if line.startswith("data:"))
        encoded = data_line.split(": ", 1)[1]

        with pytest.raises(DecryptionError, match="decryption failed"):
            decryptor.decrypt_event(encoded)

    def test_wrong_session_salt_raises(self) -> None:
        """Wrong session salt should raise DecryptionError."""
        session1 = StreamingSession(session_key=b"k" * 32, session_salt=b"aaa1")
        session2 = StreamingSession(session_key=b"k" * 32, session_salt=b"bbb2")

        encryptor = SSEEncryptor(session1)
        decryptor = SSEDecryptor(session2)

        sse_event = encryptor.encrypt_event("test", {"key": "value"})
        data_line = next(line for line in sse_event.split("\n") if line.startswith("data:"))
        encoded = data_line.split(": ", 1)[1]

        with pytest.raises(DecryptionError, match="decryption failed"):
            decryptor.decrypt_event(encoded)


class TestJSONStructureErrors:
    """Test JSON structure error paths."""

    def _make_encryptor_decryptor(self) -> tuple[SSEEncryptor, SSEDecryptor]:
        """Create matching encryptor/decryptor pair."""
        session = StreamingSession(session_key=b"0" * 32, session_salt=b"salt")
        return SSEEncryptor(session), SSEDecryptor(session)

    def _encrypt_raw_json(self, encryptor: SSEEncryptor, json_bytes: bytes) -> str:
        """Encrypt raw JSON bytes directly (bypassing normal structure)."""
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        cipher = ChaCha20Poly1305(encryptor.session.session_key)
        nonce = encryptor._compute_nonce(encryptor.counter)  # pyright: ignore[reportPrivateUsage]
        ciphertext = cipher.encrypt(nonce, json_bytes, associated_data=None)

        from hpke_http.constants import SSE_COUNTER_SIZE

        payload = encryptor.counter.to_bytes(SSE_COUNTER_SIZE, "big") + ciphertext
        encryptor.counter += 1
        return b64url_encode(payload)

    def test_missing_type_field_raises(self) -> None:
        """JSON without 't' field should raise DecryptionError."""
        encryptor, decryptor = self._make_encryptor_decryptor()

        # Encrypt {"d": {}} - missing "t" field
        encoded = self._encrypt_raw_json(encryptor, b'{"d": {}}')

        with pytest.raises(DecryptionError, match="Invalid SSE event structure"):
            decryptor.decrypt_event(encoded)

    def test_missing_data_field_raises(self) -> None:
        """JSON without 'd' field should raise DecryptionError."""
        encryptor, decryptor = self._make_encryptor_decryptor()

        # Encrypt {"t": "test"} - missing "d" field
        encoded = self._encrypt_raw_json(encryptor, b'{"t": "test"}')

        with pytest.raises(DecryptionError, match="Invalid SSE event structure"):
            decryptor.decrypt_event(encoded)

    def test_invalid_json_raises(self) -> None:
        """Non-JSON ciphertext should raise DecryptionError."""
        encryptor, decryptor = self._make_encryptor_decryptor()

        # Encrypt invalid JSON
        encoded = self._encrypt_raw_json(encryptor, b"not valid json")

        with pytest.raises(DecryptionError, match="Invalid SSE event structure"):
            decryptor.decrypt_event(encoded)

    def test_non_dict_data_accepted(self) -> None:
        """Non-dict 'd' field should be accepted (JSON allows any value)."""
        encryptor, decryptor = self._make_encryptor_decryptor()

        # Encrypt with string data - this is valid per the code
        encoded = self._encrypt_raw_json(encryptor, b'{"t": "test", "d": "string"}')

        event_type, data = decryptor.decrypt_event(encoded)
        assert event_type == "test"
        assert data == "string"


class TestReplayOrderErrors:
    """Test replay attack and counter order error paths."""

    def _make_session(self) -> StreamingSession:
        """Create a test session."""
        return StreamingSession(session_key=b"0" * 32, session_salt=b"salt")

    def _extract_data(self, sse_event: str) -> str:
        """Extract data field from SSE event string."""
        for line in sse_event.split("\n"):
            if line.startswith("data:"):
                return line.split(": ", 1)[1]
        raise ValueError("No data field found")

    def test_replay_same_event_raises(self) -> None:
        """Decrypting same event twice should raise ReplayAttackError."""
        session = self._make_session()
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        sse_event = encryptor.encrypt_event("test", {"i": 1})
        data = self._extract_data(sse_event)

        # First decryption succeeds
        decryptor.decrypt_event(data)

        # Second decryption of same event fails
        with pytest.raises(ReplayAttackError) as exc:
            decryptor.decrypt_event(data)

        assert exc.value.expected == 2
        assert exc.value.received == 1

    def test_out_of_order_raises(self) -> None:
        """Receiving event 2 when expecting 1 should raise ReplayAttackError."""
        session = self._make_session()
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        # Generate events 1 and 2
        _event1 = encryptor.encrypt_event("test", {"i": 1})
        event2 = encryptor.encrypt_event("test", {"i": 2})
        data2 = self._extract_data(event2)

        # Try to decrypt event 2 first (expecting 1)
        with pytest.raises(ReplayAttackError) as exc:
            decryptor.decrypt_event(data2)

        assert exc.value.expected == 1
        assert exc.value.received == 2

    def test_skipped_counter_raises(self) -> None:
        """Skipping counter (1, then 3) should raise ReplayAttackError."""
        session = self._make_session()
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        # Generate events 1, 2, 3
        event1 = encryptor.encrypt_event("test", {"i": 1})
        _event2 = encryptor.encrypt_event("test", {"i": 2})
        event3 = encryptor.encrypt_event("test", {"i": 3})

        # Decrypt event 1
        decryptor.decrypt_event(self._extract_data(event1))

        # Skip event 2, try event 3 (expecting 2)
        with pytest.raises(ReplayAttackError) as exc:
            decryptor.decrypt_event(self._extract_data(event3))

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
        encryptor = SSEEncryptor(session)

        # Set counter to max value
        encryptor.counter = SSE_MAX_COUNTER + 1

        with pytest.raises(SessionExpiredError, match="counter exhausted"):
            encryptor.encrypt_event("test", {"key": "value"})

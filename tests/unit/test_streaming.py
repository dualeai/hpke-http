"""Unit tests for SSE streaming encryption."""

import pytest

from hpke_http.exceptions import ReplayAttackError
from hpke_http.hpke import setup_recipient_psk, setup_sender_psk
from hpke_http.streaming import (
    SSEDecryptor,
    SSEEncryptor,
    StreamingSession,
    create_session_from_context,
)


class TestStreamingSession:
    """Test StreamingSession creation and serialization."""

    def test_create_session(self) -> None:
        """Test session creation with random salt."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        assert session.session_key == key
        assert len(session.session_salt) == 4

    def test_serialize_deserialize(self) -> None:
        """Test session serialization roundtrip."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        serialized = session.serialize()
        restored = StreamingSession.deserialize(serialized, key)

        assert restored.session_key == session.session_key
        assert restored.session_salt == session.session_salt


class TestSSEEncryption:
    """Test SSE chunk encryption/decryption."""

    @staticmethod
    def _extract_data_field(sse: str) -> str:
        """Extract data field from encrypted SSE output."""
        for line in sse.split("\n"):
            if line.startswith("data: "):
                return line[6:]
        raise ValueError("No data field found in SSE event")

    def test_encrypt_decrypt_chunk(self) -> None:
        """Test basic chunk encryption/decryption."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        # Raw SSE chunk
        original = 'event: progress\ndata: {"step": 1}\n\n'

        # Encrypt
        encrypted = encryptor.encrypt(original)

        # Extract data field and decrypt
        data_field = self._extract_data_field(encrypted)
        decrypted = decryptor.decrypt(data_field)

        assert decrypted == original

    def test_multiple_chunks(self) -> None:
        """Test encrypting/decrypting multiple chunks."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        chunks = [
            "event: start\ndata: {}\n\n",
            'event: progress\ndata: {"percent": 25}\n\n',
            ":keepalive\n\n",  # Comment
            'event: complete\ndata: {"result": "success"}\n\n',
        ]

        for original in chunks:
            encrypted = encryptor.encrypt(original)
            data_field = self._extract_data_field(encrypted)
            decrypted = decryptor.decrypt(data_field)
            assert decrypted == original

    def test_counter_monotonicity(self) -> None:
        """Test that out-of-order events are rejected."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        # Encrypt two chunks
        enc1 = encryptor.encrypt("event: first\n\n")
        enc2 = encryptor.encrypt("event: second\n\n")

        # Extract data fields
        data1 = self._extract_data_field(enc1)
        _ = self._extract_data_field(enc2)

        # Decrypt first - works
        decryptor.decrypt(data1)

        # Try to decrypt first again (replay) - should fail
        with pytest.raises(ReplayAttackError) as exc:
            decryptor.decrypt(data1)

        assert exc.value.expected == 2
        assert exc.value.received == 1

    def test_encrypted_sse_format(self) -> None:
        """Test that encrypted output is WHATWG compliant SSE."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        encrypted = encryptor.encrypt("event: test\ndata: {}\n\n")

        lines = encrypted.split("\n")
        assert lines[0] == "event: enc"
        assert lines[1].startswith("data: ")
        assert lines[2] == ""  # Empty line for event boundary


class TestCounterBoundaries:
    """Test counter boundary conditions."""

    @staticmethod
    def _extract_data_field(sse: str) -> str:
        """Extract data field from encrypted SSE output."""
        for line in sse.split("\n"):
            if line.startswith("data: "):
                return line[6:]
        raise ValueError("No data field found")

    def test_counter_starts_at_1(self) -> None:
        """First chunk should have counter=1."""
        from hpke_http.constants import SSE_COUNTER_SIZE
        from hpke_http.headers import b64url_decode

        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        encrypted = encryptor.encrypt("test")
        data = self._extract_data_field(encrypted)
        payload = b64url_decode(data)

        counter = int.from_bytes(payload[:SSE_COUNTER_SIZE], "big")
        assert counter == 1

    def test_counter_increments(self) -> None:
        """Counter should increment with each chunk."""
        from hpke_http.constants import SSE_COUNTER_SIZE
        from hpke_http.headers import b64url_decode

        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        for expected_counter in range(1, 6):
            encrypted = encryptor.encrypt(f"chunk {expected_counter}")
            data = self._extract_data_field(encrypted)
            payload = b64url_decode(data)
            counter = int.from_bytes(payload[:SSE_COUNTER_SIZE], "big")
            assert counter == expected_counter

    def test_counter_near_max(self) -> None:
        """Counter near max should still work."""
        from hpke_http.constants import SSE_MAX_COUNTER
        from hpke_http.exceptions import SessionExpiredError

        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        # Set counter to max
        encryptor.counter = SSE_MAX_COUNTER

        # This should work (counter == max)
        encrypted = encryptor.encrypt("at max")
        assert "data:" in encrypted

        # Next one should fail (counter > max)
        with pytest.raises(SessionExpiredError):
            encryptor.encrypt("over max")


class TestSessionSaltRandomness:
    """Test session salt randomness."""

    def test_different_sessions_have_different_salts(self) -> None:
        """Multiple sessions should have unique salts."""
        key = b"0" * 32
        salts: set[bytes] = set()

        for _ in range(100):
            session = StreamingSession.create(key)
            salts.add(session.session_salt)

        # With 4-byte salts and 100 samples, collisions are extremely unlikely
        assert len(salts) == 100


class TestPayloadVariations:
    """Test various payload types and sizes."""

    @staticmethod
    def _extract_data_field(sse: str) -> str:
        """Extract data field from encrypted SSE output."""
        for line in sse.split("\n"):
            if line.startswith("data: "):
                return line[6:]
        raise ValueError("No data field found")

    def test_large_chunk(self) -> None:
        """Large SSE chunk (~100KB) should encrypt/decrypt correctly."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        # Create ~100KB chunk
        large_data = "x" * 100000
        original = f"event: large\ndata: {large_data}\n\n"

        encrypted = encryptor.encrypt(original)
        data = self._extract_data_field(encrypted)
        decrypted = decryptor.decrypt(data)

        assert decrypted == original

    def test_unicode_chunk(self) -> None:
        """Unicode (Chinese, emoji, RTL) should work."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        original = "event: unicode\ndata: 你好世界 🔐 مرحبا\n\n"

        encrypted = encryptor.encrypt(original)
        data = self._extract_data_field(encrypted)
        decrypted = decryptor.decrypt(data)

        assert decrypted == original

    def test_empty_chunk(self) -> None:
        """Empty string should work."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        original = ""

        encrypted = encryptor.encrypt(original)
        data = self._extract_data_field(encrypted)
        decrypted = decryptor.decrypt(data)

        assert decrypted == original

    def test_comment_chunk(self) -> None:
        """SSE comment chunks should work."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        original = ":keepalive\n\n"

        encrypted = encryptor.encrypt(original)
        data = self._extract_data_field(encrypted)
        decrypted = decryptor.decrypt(data)

        assert decrypted == original

    def test_retry_chunk(self) -> None:
        """SSE retry directive should work."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        original = "retry: 5000\n\n"

        encrypted = encryptor.encrypt(original)
        data = self._extract_data_field(encrypted)
        decrypted = decryptor.decrypt(data)

        assert decrypted == original

    def test_special_characters(self) -> None:
        """Newlines, tabs, quotes should be handled."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        # Note: The chunk itself has controlled line endings,
        # but the data can contain escaped special chars
        original = 'event: special\ndata: {"msg": "line1\\nline2\\ttab"}\n\n'

        encrypted = encryptor.encrypt(original)
        data = self._extract_data_field(encrypted)
        decrypted = decryptor.decrypt(data)

        assert decrypted == original


class TestEventVolume:
    """Test high volume of events."""

    @staticmethod
    def _extract_data_field(sse: str) -> str:
        """Extract data field from encrypted SSE output."""
        for line in sse.split("\n"):
            if line.startswith("data: "):
                return line[6:]
        raise ValueError("No data field found")

    def test_many_chunks_100(self) -> None:
        """100 sequential chunks should work."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        for i in range(100):
            original = f"event: item\ndata: {i}\n\n"
            encrypted = encryptor.encrypt(original)
            data = self._extract_data_field(encrypted)
            decrypted = decryptor.decrypt(data)
            assert decrypted == original

    def test_many_chunks_1000(self) -> None:
        """1000 chunks should work with correct counters."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        for i in range(1000):
            original = f"event: bulk\ndata: {i}\n\n"
            encrypted = encryptor.encrypt(original)
            data = self._extract_data_field(encrypted)
            decrypted = decryptor.decrypt(data)
            assert decrypted == original

        # Verify final counter state
        assert encryptor.counter == 1001
        assert decryptor.expected_counter == 1001


class TestSessionFromContext:
    """Test session creation from HPKE context."""

    def test_session_from_hpke_context(
        self,
        platform_keypair: tuple[bytes, bytes],
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Test that sender and recipient derive same session key."""
        sk_r, pk_r = platform_keypair

        sender_ctx = setup_sender_psk(pk_r, b"", test_psk, test_psk_id)
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"", test_psk, test_psk_id)

        # Create sessions from both contexts
        sender_session = create_session_from_context(sender_ctx)
        recipient_session = create_session_from_context(recipient_ctx)

        # Keys should match
        assert sender_session.session_key == recipient_session.session_key

        # Salts are random, so they differ (each side generates their own)
        # In practice, server generates salt and sends to client via header

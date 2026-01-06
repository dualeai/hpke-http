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
    """Test SSE event encryption/decryption."""

    @staticmethod
    def _extract_data_field(sse: str) -> str:
        """Extract data field from SSE event."""
        for line in sse.split("\n"):
            if line.startswith("data: "):
                return line[6:]
        raise ValueError("No data field found in SSE event")

    def test_encrypt_decrypt_event(self) -> None:
        """Test basic event encryption/decryption."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        event_type = "task_progress"
        event_data = {"progress": 0.5, "message": "Processing..."}

        # Encrypt
        sse_event = encryptor.encrypt_event(event_type, event_data, event_id=1)

        # Parse SSE and extract data field
        data_field = self._extract_data_field(sse_event)

        # Decrypt
        decrypted_type, decrypted_data = decryptor.decrypt_event(data_field)

        assert decrypted_type == event_type
        assert decrypted_data == event_data

    def test_multiple_events(self) -> None:
        """Test encrypting/decrypting multiple events."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        events = [
            ("start", {"task_id": "123"}),
            ("progress", {"percent": 25}),
            ("progress", {"percent": 50}),
            ("complete", {"result": "success"}),
        ]

        for i, (event_type, event_data) in enumerate(events):
            sse_event = encryptor.encrypt_event(event_type, event_data, event_id=i + 1)

            # Extract data field and decrypt
            data_field = self._extract_data_field(sse_event)
            dec_type, dec_data = decryptor.decrypt_event(data_field)
            assert dec_type == event_type
            assert dec_data == event_data

    def test_counter_monotonicity(self) -> None:
        """Test that out-of-order events are rejected."""
        key = b"0" * 32
        session = StreamingSession.create(key)

        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        # Encrypt two events
        event1 = encryptor.encrypt_event("first", {}, event_id=1)
        event2 = encryptor.encrypt_event("second", {}, event_id=2)

        # Extract data fields
        data1 = self._extract_data_field(event1)
        _ = self._extract_data_field(event2)  # data2 prepared but not used in this test

        # Decrypt in order works
        decryptor.decrypt_event(data1)

        # Try to decrypt event1 again (replay) - should fail
        with pytest.raises(ReplayAttackError) as exc:
            decryptor.decrypt_event(data1)

        assert exc.value.expected == 2
        assert exc.value.received == 1

    def test_sse_format(self) -> None:
        """Test that SSE output is WHATWG compliant."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        sse_event = encryptor.encrypt_event("test", {"key": "value"}, event_id=42)

        lines = sse_event.split("\n")
        assert lines[0] == "id: 42"
        assert lines[1] == "event: enc"
        assert lines[2].startswith("data: ")
        assert lines[3] == ""  # Empty line for event boundary


class TestCounterBoundaries:
    """Test counter boundary conditions."""

    @staticmethod
    def _extract_data_field(sse: str) -> str:
        """Extract data field from SSE event."""
        for line in sse.split("\n"):
            if line.startswith("data: "):
                return line[6:]
        raise ValueError("No data field found in SSE event")

    def test_counter_starts_at_1(self) -> None:
        """First event should have counter=1."""
        from hpke_http.constants import SSE_COUNTER_SIZE
        from hpke_http.headers import b64url_decode

        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        sse_event = encryptor.encrypt_event("test", {})
        data = self._extract_data_field(sse_event)
        payload = b64url_decode(data)

        # Extract counter from first 4 bytes (big-endian)
        counter = int.from_bytes(payload[:SSE_COUNTER_SIZE], "big")
        assert counter == 1

    def test_counter_increments(self) -> None:
        """Counter should increment with each event."""
        from hpke_http.constants import SSE_COUNTER_SIZE
        from hpke_http.headers import b64url_decode

        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        for expected_counter in range(1, 6):
            sse_event = encryptor.encrypt_event("test", {"i": expected_counter})
            data = self._extract_data_field(sse_event)
            payload = b64url_decode(data)
            counter = int.from_bytes(payload[:SSE_COUNTER_SIZE], "big")
            assert counter == expected_counter

    def test_counter_near_max(self) -> None:
        """Counter near max should still work."""
        from hpke_http.constants import SSE_MAX_COUNTER

        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        # Set counter to max - 1
        encryptor.counter = SSE_MAX_COUNTER

        # This should work (counter == max)
        sse_event = encryptor.encrypt_event("test", {"at": "max"})
        assert "data:" in sse_event

        # Next one should fail (counter > max)
        from hpke_http.exceptions import SessionExpiredError

        with pytest.raises(SessionExpiredError):
            encryptor.encrypt_event("test", {"over": "max"})


class TestEventIDHandling:
    """Test event ID handling in SSE output."""

    def test_event_with_id(self) -> None:
        """Event with ID should include id: field."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        sse_event = encryptor.encrypt_event("test", {}, event_id=42)
        assert "id: 42\n" in sse_event

    def test_event_without_id(self) -> None:
        """Event without ID should omit id: field."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        sse_event = encryptor.encrypt_event("test", {}, event_id=None)
        assert "id:" not in sse_event

    def test_event_id_zero(self) -> None:
        """Event ID 0 should be valid."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        sse_event = encryptor.encrypt_event("test", {}, event_id=0)
        assert "id: 0\n" in sse_event


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
        """Extract data field from SSE event."""
        for line in sse.split("\n"):
            if line.startswith("data: "):
                return line[6:]
        raise ValueError("No data field found in SSE event")

    def test_large_event_payload(self) -> None:
        """100KB JSON data should encrypt/decrypt correctly."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        # Create ~100KB payload
        large_data = {"items": ["x" * 1000 for _ in range(100)]}

        sse_event = encryptor.encrypt_event("large", large_data)
        data = self._extract_data_field(sse_event)
        event_type, decrypted = decryptor.decrypt_event(data)

        assert event_type == "large"
        assert decrypted == large_data

    def test_unicode_event_data(self) -> None:
        """Unicode (Chinese, emoji, RTL) should work."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        unicode_data = {
            "chinese": "你好世界",
            "emoji": "🔐🔑🛡️",
            "rtl": "مرحبا بالعالم",
            "mixed": "Hello 世界 🌍",
        }

        sse_event = encryptor.encrypt_event("unicode", unicode_data)
        data = self._extract_data_field(sse_event)
        event_type, decrypted = decryptor.decrypt_event(data)

        assert event_type == "unicode"
        assert decrypted == unicode_data

    def test_empty_event_data(self) -> None:
        """Empty dict data should work."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        sse_event = encryptor.encrypt_event("ping", {})
        data = self._extract_data_field(sse_event)
        event_type, decrypted = decryptor.decrypt_event(data)

        assert event_type == "ping"
        assert decrypted == {}

    def test_nested_json_data(self) -> None:
        """Deeply nested JSON should work."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        nested_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": [1, 2, {"deep": True}],
                        }
                    }
                }
            }
        }

        sse_event = encryptor.encrypt_event("nested", nested_data)
        data = self._extract_data_field(sse_event)
        event_type, decrypted = decryptor.decrypt_event(data)

        assert event_type == "nested"
        assert decrypted == nested_data

    def test_special_characters_in_data(self) -> None:
        """Newlines, tabs, quotes should be handled."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        special_data = {
            "newlines": "line1\nline2\nline3",
            "tabs": "col1\tcol2\tcol3",
            "quotes": 'He said "hello"',
            "backslash": "path\\to\\file",
        }

        sse_event = encryptor.encrypt_event("special", special_data)
        data = self._extract_data_field(sse_event)
        event_type, decrypted = decryptor.decrypt_event(data)

        assert event_type == "special"
        assert decrypted == special_data


class TestEventVolume:
    """Test high volume of events."""

    @staticmethod
    def _extract_data_field(sse: str) -> str:
        """Extract data field from SSE event."""
        for line in sse.split("\n"):
            if line.startswith("data: "):
                return line[6:]
        raise ValueError("No data field found in SSE event")

    def test_many_events_100(self) -> None:
        """100 sequential events should work."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        for i in range(100):
            sse_event = encryptor.encrypt_event("event", {"index": i})
            data = self._extract_data_field(sse_event)
            event_type, decrypted = decryptor.decrypt_event(data)

            assert event_type == "event"
            assert decrypted["index"] == i

    def test_many_events_1000(self) -> None:
        """1000 events should work with correct counters."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        for i in range(1000):
            sse_event = encryptor.encrypt_event("bulk", {"n": i})
            data = self._extract_data_field(sse_event)
            event_type, decrypted = decryptor.decrypt_event(data)

            assert event_type == "bulk"
            assert decrypted["n"] == i

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

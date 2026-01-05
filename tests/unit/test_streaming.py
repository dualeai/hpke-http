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

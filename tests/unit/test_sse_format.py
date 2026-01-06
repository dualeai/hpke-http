"""SSE format conformance tests.

Tests encrypted SSE output format compliance.
"""

import re

from hpke_http.headers import b64url_decode
from hpke_http.streaming import SSEEncryptor, StreamingSession


class TestSSEOutputFormat:
    """Test SSEEncryptor output format compliance."""

    @staticmethod
    def _extract_data_field(sse: str) -> str:
        """Extract data field from encrypted SSE output."""
        for line in sse.split("\n"):
            if line.startswith("data: "):
                return line[6:]
        raise ValueError("No data field found")

    def test_sse_format_basic(self) -> None:
        """Encrypted SSE should have correct format."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        encrypted = encryptor.encrypt("event: test\ndata: {}\n\n")

        # Should be: event: enc\ndata: ...\n\n
        lines = encrypted.split("\n")
        assert lines[0] == "event: enc"
        assert lines[1].startswith("data: ")
        assert lines[2] == ""
        assert lines[3] == ""  # Final newline creates empty element

    def test_sse_line_endings_lf(self) -> None:
        """SSE output should use LF (not CRLF)."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        encrypted = encryptor.encrypt("event: test\n\n")

        # Should not contain CR
        assert "\r" not in encrypted
        # Should contain LF
        assert "\n" in encrypted

    def test_sse_event_boundary(self) -> None:
        """SSE event should end with \\n\\n."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        encrypted = encryptor.encrypt("event: test\n\n")

        assert encrypted.endswith("\n\n")

    def test_sse_data_is_base64url(self) -> None:
        """Data field should be valid base64url."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        encrypted = encryptor.encrypt("event: test\ndata: {}\n\n")

        # Extract data field
        data = self._extract_data_field(encrypted)

        # Should be valid base64url (no exceptions)
        decoded = b64url_decode(data)
        assert len(decoded) > 0

        # Should only contain base64url characters
        assert re.match(r"^[A-Za-z0-9_-]+$", data), f"Invalid base64url chars in: {data}"

    def test_event_type_always_enc(self) -> None:
        """Event type in SSE should always be 'enc'."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        # Different content types
        for content in [
            "event: start\n\n",
            "event: progress\ndata: {}\n\n",
            ":keepalive\n\n",
            "retry: 5000\n\n",
        ]:
            encrypted = encryptor.encrypt(content)
            assert "event: enc\n" in encrypted

    def test_raw_passthrough_preserves_content(self) -> None:
        """Raw content should roundtrip exactly."""
        from hpke_http.streaming import SSEDecryptor

        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)

        # Various SSE chunk types
        chunks = [
            "event: test\ndata: hello\n\n",
            ":keepalive comment\n\n",
            "retry: 5000\n\n",
            'id: 123\nevent: msg\ndata: {"foo": "bar"}\n\n',
            "",  # Empty chunk
        ]

        for original in chunks:
            encrypted = encryptor.encrypt(original)
            data = self._extract_data_field(encrypted)
            decrypted = decryptor.decrypt(data)
            assert decrypted == original, f"Failed for: {original!r}"

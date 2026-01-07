"""SSE format conformance tests.

Tests encrypted SSE output format compliance.
"""

import re

from hpke_http.headers import b64url_decode
from hpke_http.streaming import ChunkEncryptor, StreamingSession
from tests.conftest import extract_sse_data_field


class TestSSEOutputFormat:
    """Test ChunkEncryptor output format compliance."""

    def test_sse_format_basic(self) -> None:
        """Encrypted SSE should have correct format."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session)

        encrypted = encryptor.encrypt(b"event: test\ndata: {}\n\n")

        # Should be: event: enc\ndata: ...\n\n
        lines = encrypted.decode("ascii").split("\n")
        assert lines[0] == "event: enc"
        assert lines[1].startswith("data: ")
        assert lines[2] == ""
        assert lines[3] == ""  # Final newline creates empty element

    def test_sse_line_endings_lf(self) -> None:
        """SSE output should use LF (not CRLF)."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session)

        encrypted = encryptor.encrypt(b"event: test\n\n")

        # Should not contain CR
        assert b"\r" not in encrypted
        # Should contain LF
        assert b"\n" in encrypted

    def test_sse_event_boundary(self) -> None:
        """SSE event should end with \\n\\n."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session)

        encrypted = encryptor.encrypt(b"event: test\n\n")

        assert encrypted.endswith(b"\n\n")

    def test_sse_data_is_base64url(self) -> None:
        """Data field should be valid base64url."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session)

        encrypted = encryptor.encrypt(b"event: test\ndata: {}\n\n")

        # Extract data field
        data = extract_sse_data_field(encrypted)

        # Should be valid base64url (no exceptions)
        decoded = b64url_decode(data)
        assert len(decoded) > 0

        # Should only contain base64url characters
        assert re.match(r"^[A-Za-z0-9_-]+$", data), f"Invalid base64url chars in: {data}"

    def test_event_type_always_enc(self) -> None:
        """Event type in SSE should always be 'enc'."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session)

        # Different content types
        for content in [
            b"event: start\n\n",
            b"event: progress\ndata: {}\n\n",
            b":keepalive\n\n",
            b"retry: 5000\n\n",
        ]:
            encrypted = encryptor.encrypt(content)
            assert b"event: enc\n" in encrypted

    def test_raw_passthrough_preserves_content(self) -> None:
        """Raw content should roundtrip exactly."""
        from hpke_http.streaming import ChunkDecryptor

        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = ChunkEncryptor(session)
        decryptor = ChunkDecryptor(session)

        # Various SSE chunk types
        chunks = [
            b"event: test\ndata: hello\n\n",
            b":keepalive comment\n\n",
            b"retry: 5000\n\n",
            b'id: 123\nevent: msg\ndata: {"foo": "bar"}\n\n',
            b"",  # Empty chunk
        ]

        for original in chunks:
            encrypted = encryptor.encrypt(original)
            data = extract_sse_data_field(encrypted)
            decrypted = decryptor.decrypt(data)
            assert decrypted == original, f"Failed for: {original!r}"

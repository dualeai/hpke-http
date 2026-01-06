"""SSE format conformance tests.

Tests SSE output format compliance with WHATWG spec.
"""

import re

from hpke_http.headers import b64url_decode
from hpke_http.streaming import SSEEncryptor, StreamingSession, parse_sse_event


class TestSSEOutputFormat:
    """Test SSEEncryptor output format compliance."""

    def test_sse_format_with_id(self) -> None:
        """Event with ID should have correct format."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        sse_event = encryptor.encrypt_event("test", {"key": "value"}, event_id=1)

        # Should be: id: 1\nevent: enc\ndata: ...\n\n
        lines = sse_event.split("\n")
        assert lines[0] == "id: 1"
        assert lines[1] == "event: enc"
        assert lines[2].startswith("data: ")
        assert lines[3] == ""
        assert lines[4] == ""  # Final newline creates empty element

    def test_sse_format_without_id(self) -> None:
        """Event without ID should omit id field."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        sse_event = encryptor.encrypt_event("test", {"key": "value"}, event_id=None)

        # Should be: event: enc\ndata: ...\n\n
        lines = sse_event.split("\n")
        assert lines[0] == "event: enc"
        assert lines[1].startswith("data: ")
        assert lines[2] == ""
        assert "id:" not in sse_event

    def test_sse_line_endings_lf(self) -> None:
        """SSE output should use LF (not CRLF)."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        sse_event = encryptor.encrypt_event("test", {})

        # Should not contain CR
        assert "\r" not in sse_event
        # Should contain LF
        assert "\n" in sse_event

    def test_sse_event_boundary(self) -> None:
        """SSE event should end with \\n\\n."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        sse_event = encryptor.encrypt_event("test", {})

        assert sse_event.endswith("\n\n")

    def test_sse_data_is_base64url(self) -> None:
        """Data field should be valid base64url."""
        key = b"0" * 32
        session = StreamingSession.create(key)
        encryptor = SSEEncryptor(session)

        sse_event = encryptor.encrypt_event("test", {"key": "value"})

        # Extract data field
        data = next(line[6:] for line in sse_event.split("\n") if line.startswith("data: "))

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

        # Different event types in plaintext
        for event_type in ["start", "progress", "complete", "error"]:
            sse_event = encryptor.encrypt_event(event_type, {})
            assert "event: enc\n" in sse_event


class TestSSEParserFormat:
    """Test parse_sse_event format handling."""

    def test_parse_all_standard_fields(self) -> None:
        """Parser should extract id, event, data, retry."""
        raw = "id: 123\nevent: test\ndata: hello\nretry: 5000\n"
        result = parse_sse_event(raw)

        assert result["id"] == "123"
        assert result["event"] == "test"
        assert result["data"] == "hello"
        assert result["retry"] == "5000"

    def test_parse_unknown_field_preserved(self) -> None:
        """Unknown fields should be preserved in result."""
        raw = "custom: myvalue\ndata: test\n"
        result = parse_sse_event(raw)

        assert result["custom"] == "myvalue"
        assert result["data"] == "test"

    def test_parse_colon_without_space(self) -> None:
        """Field without space after colon should work."""
        raw = "data:test\n"
        result = parse_sse_event(raw)

        assert result["data"] == "test"

    def test_parse_colon_with_space(self) -> None:
        """Field with space after colon should work."""
        raw = "data: test\n"
        result = parse_sse_event(raw)

        assert result["data"] == "test"

    def test_parse_empty_value_colon_only(self) -> None:
        """Field with colon but no value should be empty string."""
        raw = "data:\n"
        result = parse_sse_event(raw)

        assert result["data"] == ""

    def test_parse_data_preserves_colons(self) -> None:
        """Colons in value should be preserved."""
        raw = "data: http://example.com:8080\n"
        result = parse_sse_event(raw)

        assert result["data"] == "http://example.com:8080"

    def test_parse_multiple_same_fields(self) -> None:
        """Last occurrence of same field wins."""
        raw = "data: first\ndata: second\n"
        result = parse_sse_event(raw)

        # Last value wins (per WHATWG, data should concatenate, but our impl overwrites)
        assert result["data"] == "second"

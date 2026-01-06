"""WHATWG EventSource conformance tests.

Tests SSE parsing against WHATWG HTML Living Standard requirements.
Reference: https://html.spec.whatwg.org/multipage/server-sent-events.html
"""

from hpke_http.streaming import parse_sse_event


class TestWHATWGLineEndings:
    """Test line ending handling per WHATWG spec."""

    def test_parse_crlf(self) -> None:
        """CRLF (\\r\\n) line endings should be handled."""
        raw = "id: 1\r\nevent: test\r\ndata: hello\r\n"
        result = parse_sse_event(raw)

        assert result["id"] == "1"
        assert result["event"] == "test"
        assert result["data"] == "hello"

    def test_parse_cr_only(self) -> None:
        """CR-only (\\r) line endings should be handled."""
        raw = "id: 2\revent: test\rdata: world\r"
        result = parse_sse_event(raw)

        assert result["id"] == "2"
        assert result["event"] == "test"
        assert result["data"] == "world"

    def test_parse_lf_only(self) -> None:
        """LF-only (\\n) line endings should be handled."""
        raw = "id: 3\nevent: test\ndata: foo\n"
        result = parse_sse_event(raw)

        assert result["id"] == "3"
        assert result["event"] == "test"
        assert result["data"] == "foo"

    def test_parse_mixed(self) -> None:
        """Mixed line endings should all be handled."""
        raw = "id: 4\r\nevent: test\rdata: bar\n"
        result = parse_sse_event(raw)

        assert result["id"] == "4"
        assert result["event"] == "test"
        assert result["data"] == "bar"


class TestWHATWGComments:
    """Test comment line handling per WHATWG spec."""

    def test_comment_ignored(self) -> None:
        """Lines starting with : should be ignored."""
        raw = ":this is a comment\ndata: test\n"
        result = parse_sse_event(raw)

        assert "this is a comment" not in result
        assert result["data"] == "test"

    def test_comment_as_keepalive(self) -> None:
        """Multiple comment lines (keepalive) should be ignored."""
        raw = ":\n:\n:keepalive\ndata: test\n"
        result = parse_sse_event(raw)

        assert result["data"] == "test"
        assert len(result) == 1  # Only data field

    def test_colon_in_value(self) -> None:
        """Colons in values should be preserved."""
        raw = "data: http://example.com:8080/path\n"
        result = parse_sse_event(raw)

        assert result["data"] == "http://example.com:8080/path"


class TestWHATWGFieldParsing:
    """Test field parsing rules per WHATWG spec."""

    def test_space_after_colon(self) -> None:
        """Single space after colon should be removed."""
        raw = "data: test\n"
        result = parse_sse_event(raw)

        assert result["data"] == "test"

    def test_no_space_after_colon(self) -> None:
        """No space after colon should work."""
        raw = "data:test\n"
        result = parse_sse_event(raw)

        assert result["data"] == "test"

    def test_multiple_spaces(self) -> None:
        """Only first space after colon should be removed."""
        raw = "data:  test with spaces\n"
        result = parse_sse_event(raw)

        # First space removed, second preserved
        assert result["data"] == " test with spaces"

    def test_field_without_colon(self) -> None:
        """Field without colon should have empty value."""
        raw = "data\nevent: test\n"
        result = parse_sse_event(raw)

        assert result["data"] == ""
        assert result["event"] == "test"

    def test_empty_value_with_colon(self) -> None:
        """Field with colon but no value should have empty string."""
        raw = "data:\nevent: test\n"
        result = parse_sse_event(raw)

        assert result["data"] == ""
        assert result["event"] == "test"

    def test_retry_field(self) -> None:
        """Retry field should be parsed."""
        raw = "retry: 5000\ndata: test\n"
        result = parse_sse_event(raw)

        assert result["retry"] == "5000"
        assert result["data"] == "test"

    def test_unknown_field_preserved(self) -> None:
        """Unknown fields should be preserved in result dict."""
        raw = "custom: value\ndata: test\n"
        result = parse_sse_event(raw)

        assert result["custom"] == "value"
        assert result["data"] == "test"

    def test_empty_lines_ignored(self) -> None:
        """Empty lines should be ignored."""
        raw = "data: test\n\n\nevent: foo\n"
        result = parse_sse_event(raw)

        assert result["data"] == "test"
        assert result["event"] == "foo"

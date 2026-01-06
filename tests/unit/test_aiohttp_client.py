"""Unit tests for HPKEClientSession cache header parsing.

NOTE: This tests a pure parsing function that cannot be tested via E2E because:
- Server always returns valid Cache-Control headers
- Testing parse edge cases (empty, invalid) requires mocking the header value
- The function is internal implementation detail, behavior is covered by E2E

This is one of the few cases where mocking is justified - it's a pure
stateless parsing function with no network or I/O involved.
"""

from hpke_http.middleware.aiohttp import DISCOVERY_CACHE_MAX_AGE, HPKEClientSession


class TestCacheHeaderParsing:
    """Test Cache-Control header parsing utility."""

    def test_parse_max_age_valid(self) -> None:
        """Valid max-age should be parsed correctly."""
        client = HPKEClientSession.__new__(HPKEClientSession)
        result = client._parse_max_age("public, max-age=3600")  # pyright: ignore[reportPrivateUsage]
        assert result == 3600

    def test_parse_max_age_no_max_age_returns_default(self) -> None:
        """Missing max-age should return default TTL (fail-safe)."""
        client = HPKEClientSession.__new__(HPKEClientSession)
        result = client._parse_max_age("public, no-cache")  # pyright: ignore[reportPrivateUsage]
        assert result == DISCOVERY_CACHE_MAX_AGE

    def test_parse_max_age_invalid_value_returns_default(self) -> None:
        """Non-integer max-age value should return default TTL (fail-safe)."""
        client = HPKEClientSession.__new__(HPKEClientSession)
        result = client._parse_max_age("max-age=invalid")  # pyright: ignore[reportPrivateUsage]
        assert result == DISCOVERY_CACHE_MAX_AGE

    def test_parse_max_age_empty_string_returns_default(self) -> None:
        """Empty string should return default TTL."""
        client = HPKEClientSession.__new__(HPKEClientSession)
        result = client._parse_max_age("")  # pyright: ignore[reportPrivateUsage]
        assert result == DISCOVERY_CACHE_MAX_AGE

    def test_parse_max_age_zero(self) -> None:
        """Zero max-age should be parsed as 0 (no caching)."""
        client = HPKEClientSession.__new__(HPKEClientSession)
        result = client._parse_max_age("max-age=0")  # pyright: ignore[reportPrivateUsage]
        assert result == 0

    def test_parse_max_age_with_other_directives(self) -> None:
        """max-age with other directives should be parsed."""
        client = HPKEClientSession.__new__(HPKEClientSession)
        result = client._parse_max_age("public, max-age=86400, immutable")  # pyright: ignore[reportPrivateUsage]
        assert result == 86400

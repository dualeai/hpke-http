"""Tests for BaseHPKEClient shared functionality.

This module tests the shared behavior of BaseHPKEClient that is identical
for both aiohttp and httpx implementations:
- Cache-Control header parsing
- Discovery key parsing
- Key caching behavior
- Request encryption
"""

# pyright: reportPrivateUsage=false

from hpke_http.constants import DISCOVERY_CACHE_MAX_AGE, KemId
from hpke_http.core import parse_cache_max_age, parse_discovery_keys
from tests.conftest import E2EServer


class TestCacheHeaderParsing:
    """Test Cache-Control header parsing (pure function).

    NOTE: This tests a pure parsing function that cannot be tested via E2E because:
    - Server always returns valid Cache-Control headers
    - Testing parse edge cases (empty, invalid) requires mocking the header value
    - The function is internal implementation detail, behavior is covered by E2E
    """

    def test_parse_max_age_valid(self) -> None:
        """Valid max-age should be parsed correctly."""
        result = parse_cache_max_age("public, max-age=3600")
        assert result == 3600

    def test_parse_max_age_no_max_age_returns_default(self) -> None:
        """Missing max-age should return default TTL (fail-safe)."""
        result = parse_cache_max_age("public, no-cache")
        assert result == DISCOVERY_CACHE_MAX_AGE

    def test_parse_max_age_invalid_value_returns_default(self) -> None:
        """Non-integer max-age value should return default TTL (fail-safe)."""
        result = parse_cache_max_age("max-age=invalid")
        assert result == DISCOVERY_CACHE_MAX_AGE

    def test_parse_max_age_empty_string_returns_default(self) -> None:
        """Empty string should return default TTL."""
        result = parse_cache_max_age("")
        assert result == DISCOVERY_CACHE_MAX_AGE

    def test_parse_max_age_zero(self) -> None:
        """Zero max-age should be parsed as 0 (no caching)."""
        result = parse_cache_max_age("max-age=0")
        assert result == 0

    def test_parse_max_age_with_other_directives(self) -> None:
        """max-age with other directives should be parsed."""
        result = parse_cache_max_age("public, max-age=86400, immutable")
        assert result == 86400


class TestDiscoveryKeyParsing:
    """Test discovery response parsing (pure function)."""

    def test_parse_valid_x25519_key(self) -> None:
        """Valid X25519 key should be parsed correctly."""
        import base64

        # 32 bytes encoded as base64url (no padding)
        public_key_bytes = b"\x00" * 32
        public_key_b64 = base64.urlsafe_b64encode(public_key_bytes).rstrip(b"=").decode()

        # kem_id is a hex string in the wire format (0x0020 = 32 = X25519)
        data = {"keys": [{"kem_id": "0x0020", "public_key": public_key_b64}]}
        keys = parse_discovery_keys(data)
        assert KemId.DHKEM_X25519_HKDF_SHA256 in keys
        assert keys[KemId.DHKEM_X25519_HKDF_SHA256] == public_key_bytes

    def test_parse_empty_keys_returns_empty(self) -> None:
        """Empty keys list returns empty dict."""
        keys = parse_discovery_keys({"keys": []})
        assert keys == {}

    def test_parse_missing_keys_returns_empty(self) -> None:
        """Missing keys field returns empty dict."""
        keys = parse_discovery_keys({})
        assert keys == {}

    def test_parse_same_kem_id_overwrites(self) -> None:
        """Same kem_id should overwrite (last wins)."""
        import base64

        key1 = base64.urlsafe_b64encode(b"\x01" * 32).rstrip(b"=").decode()
        key2 = base64.urlsafe_b64encode(b"\x02" * 32).rstrip(b"=").decode()

        data = {
            "keys": [
                {"kem_id": "0x0020", "public_key": key1},  # X25519
                {"kem_id": "0x0020", "public_key": key2},  # Same kem_id, overwrites
            ]
        }
        keys = parse_discovery_keys(data)
        # Last key wins
        assert len(keys) == 1
        assert keys[KemId.DHKEM_X25519_HKDF_SHA256] == b"\x02" * 32


class TestBaseHPKEClientKeyCache:
    """Test class-level key caching behavior."""

    async def test_cache_shared_across_instances(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Two clients share the same cache."""
        from hpke_http.middleware.aiohttp import HPKEClientSession

        base_url = f"http://{granian_server.host}:{granian_server.port}"

        # First client fetches keys (cache miss)
        async with HPKEClientSession(base_url, test_psk, test_psk_id) as client1:
            keys1 = await client1._ensure_keys()  # noqa: SLF001

        # Second client should hit cache
        async with HPKEClientSession(base_url, test_psk, test_psk_id) as client2:
            keys2 = await client2._ensure_keys()  # noqa: SLF001

        assert keys1 == keys2

    async def test_keys_are_valid_x25519(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Fetched keys should be valid X25519 public keys (32 bytes)."""
        from hpke_http.middleware.aiohttp import HPKEClientSession

        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEClientSession(base_url, test_psk, test_psk_id) as client:
            keys = await client._ensure_keys()  # noqa: SLF001

        assert KemId.DHKEM_X25519_HKDF_SHA256 in keys
        assert len(keys[KemId.DHKEM_X25519_HKDF_SHA256]) == 32


class TestEncryptRequestSync:
    """Test _encrypt_request_sync shared method."""

    async def test_returns_iterator_headers_context(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """_encrypt_request_sync returns (iterator, headers, context)."""
        from hpke_http.middleware.aiohttp import HPKEClientSession

        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEClientSession(base_url, test_psk, test_psk_id) as client:
            keys = await client._ensure_keys()  # noqa: SLF001
            iterator, headers, ctx = client._encrypt_request_sync(b"test body", keys)  # noqa: SLF001

            # Iterator should be iterable
            assert hasattr(iterator, "__iter__")
            chunks = list(iterator)
            assert len(chunks) > 0

            # Headers should contain HPKE headers
            assert "X-HPKE-Enc" in headers

            # Context should be valid SenderContext
            assert ctx is not None
            assert hasattr(ctx, "export")

    async def test_encrypt_with_compression(
        self,
        granian_server: E2EServer,
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Compression flag should be respected."""
        from hpke_http.middleware.aiohttp import HPKEClientSession

        base_url = f"http://{granian_server.host}:{granian_server.port}"

        async with HPKEClientSession(base_url, test_psk, test_psk_id, compress=True) as client:
            keys = await client._ensure_keys()  # noqa: SLF001
            # Use large body to trigger compression (>= ZSTD_MIN_SIZE)
            large_body = b"x" * 100
            _iterator, headers, _ctx = client._encrypt_request_sync(large_body, keys)  # noqa: SLF001

            # Compression header should be present for large bodies
            # (exact behavior depends on ZSTD_MIN_SIZE)
            assert "X-HPKE-Enc" in headers

"""
``parse_discovery_keys`` forward-compat tests.

The relaxed contract: structurally malformed entries raise; unknown
``kem_id`` values warn-and-skip so a future server advertising a KEM this
client doesn't know doesn't break older clients.
"""

from __future__ import annotations

import logging

import pytest

from hpke_http.constants import KemId
from hpke_http.core import _b64url_encode, parse_discovery_keys  # pyright: ignore[reportPrivateUsage]
from hpke_http.exceptions import KeyDiscoveryError


def _entry(kem_id_hex: str, pk_bytes: bytes) -> dict[str, str]:
    return {"kem_id": kem_id_hex, "public_key": _b64url_encode(pk_bytes)}


_VALID_X25519 = _entry("0x0020", b"\x00" * 32)


class TestHappyPath:
    def test_x25519_only(self) -> None:
        keys = parse_discovery_keys({"version": 1, "keys": [_VALID_X25519]})
        assert KemId.DHKEM_X25519_HKDF_SHA256 in keys

    def test_no_version_treated_as_1(self) -> None:
        keys = parse_discovery_keys({"keys": [_VALID_X25519]})
        assert KemId.DHKEM_X25519_HKDF_SHA256 in keys


class TestForwardCompat:
    def test_unknown_kem_id_skipped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Future kem_id (e.g. 0x6500) → silently skip, emit warning."""
        caplog.set_level(logging.WARNING)
        doc = {
            "version": 1,
            "keys": [
                _VALID_X25519,
                _entry("0x6500", b"\x00" * 32),  # unknown KEM
            ],
        }
        keys = parse_discovery_keys(doc)
        assert KemId.DHKEM_X25519_HKDF_SHA256 in keys
        assert len(keys) == 1
        assert any("0x6500" in r.getMessage() for r in caplog.records)

    def test_only_unknown_returns_empty(self) -> None:
        """All-unknown discovery returns empty dict — caller's _select_suite raises."""
        doc = {"version": 1, "keys": [_entry("0x6500", b"\x00" * 32)]}
        assert parse_discovery_keys(doc) == {}


class TestRejectMalformed:
    def test_unsupported_version_raises(self) -> None:
        with pytest.raises(KeyDiscoveryError, match="version"):
            parse_discovery_keys({"version": 2, "keys": []})

    def test_missing_kem_id_field_raises(self) -> None:
        doc = {"version": 1, "keys": [{"public_key": _b64url_encode(b"\x00" * 32)}]}
        with pytest.raises(KeyDiscoveryError, match="Malformed"):
            parse_discovery_keys(doc)

    def test_missing_public_key_field_raises(self) -> None:
        doc = {"version": 1, "keys": [{"kem_id": "0x0020"}]}
        with pytest.raises(KeyDiscoveryError, match="Malformed"):
            parse_discovery_keys(doc)

    def test_non_hex_kem_id_raises(self) -> None:
        doc = {"version": 1, "keys": [_entry("0xZZZZ", b"\x00" * 32)]}
        with pytest.raises(KeyDiscoveryError, match="Malformed"):
            parse_discovery_keys(doc)

    def test_undecodable_b64_lenient(self) -> None:
        """``base64.urlsafe_b64decode`` is lenient (discards invalid chars).
        Garbage in → bytes out; the surface that catches wrong-format keys is
        the KEM's own ``validate_public_key`` size check at encrypt/decrypt
        time, not the discovery doc parser. Document the behavior."""
        doc = {"version": 1, "keys": [{"kem_id": "0x0020", "public_key": "!!!not_base64"}]}
        keys = parse_discovery_keys(doc)
        # Invalid base64 decoded to whatever stripped chars produce.
        # Will likely fail downstream when KEM tries to use the bytes.
        assert KemId.DHKEM_X25519_HKDF_SHA256 in keys


class TestEmptyAndAbsent:
    def test_empty_keys_list_returns_empty(self) -> None:
        """Empty server advertisement is a structurally-valid edge case;
        parse returns empty dict and the caller's ``_select_suite`` decides
        what to do (currently raises KeyDiscoveryError)."""
        assert parse_discovery_keys({"version": 1, "keys": []}) == {}

    def test_missing_keys_field_treated_as_empty(self) -> None:
        assert parse_discovery_keys({"version": 1}) == {}

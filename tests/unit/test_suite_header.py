"""
``X-HPKE-Suite`` header parser tests.

Covers normal, edge, out-of-bound, and malformed cases. The parser is the
sole entry point on the server for KEM negotiation, so it must reject
adversarial input strictly without becoming a parsing oracle.
"""

from __future__ import annotations

import pytest

from hpke_http.constants import KemId
from hpke_http.core import _parse_suite_header  # pyright: ignore[reportPrivateUsage]
from hpke_http.exceptions import DecryptionError, UnsupportedKEMError


class TestAccept:
    def test_x25519(self) -> None:
        assert _parse_suite_header("kem=0x0020") == KemId.DHKEM_X25519_HKDF_SHA256

    def test_xwing(self) -> None:
        assert _parse_suite_header("kem=0x647a") == KemId.XWING


class TestRejectMalformed:
    @pytest.mark.parametrize(
        "bad",
        [
            "",  # empty
            "kem=0x0020;",  # trailing semicolon
            "kem=0x0020;kdf=0x0001",  # extra params reserved for the future
            "kem=0X0020",  # uppercase prefix
            "KEM=0x0020",  # uppercase key
            "kem=0x0020 ",  # trailing whitespace
            " kem=0x0020",  # leading whitespace
            "kem = 0x0020",  # spaces around =
            "kem=0x002",  # 3 hex digits
            "kem=0x00200",  # 5 hex digits
            "kem=0xZZZZ",  # non-hex
            "kem=0020",  # missing 0x prefix
            "0x0020",  # missing kem= key
            "kdf=0x0001",  # wrong key
        ],
    )
    def test_malformed_raises_decryption(self, bad: str) -> None:
        with pytest.raises(DecryptionError):
            _parse_suite_header(bad)

    def test_oversize_raises_decryption(self) -> None:
        # length-cap is enforced before regex
        oversize = "kem=0x" + "0" * 500
        with pytest.raises(DecryptionError, match="too long"):
            _parse_suite_header(oversize)


class TestRejectUnknown:
    def test_unknown_kem_id_raises_unsupported(self) -> None:
        # Well-formed but no KEM is registered for 0xfffe.
        with pytest.raises(UnsupportedKEMError):
            _parse_suite_header("kem=0xfffe")

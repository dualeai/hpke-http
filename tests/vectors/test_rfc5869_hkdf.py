"""RFC 5869 HKDF test vectors.

Official IETF test vectors for HMAC-based Extract-and-Expand Key Derivation Function.
Source: https://www.rfc-editor.org/rfc/rfc5869.html

Appendix A contains test cases for both SHA-256 and SHA-1.
We test SHA-256 cases (1-3) as that's what our HPKE implementation uses.

To update: These are hardcoded from the RFC specification.
"""

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import hmac as crypto_hmac
from cryptography.hazmat.primitives.kdf.hkdf import HKDF, HKDFExpand


def hkdf_extract(salt: bytes | None, ikm: bytes) -> bytes:
    """HKDF-Extract: PRK = HMAC-Hash(salt, IKM)."""
    if salt is None or len(salt) == 0:
        salt = b"\x00" * 32  # SHA-256 hash length
    h = crypto_hmac.HMAC(salt, hashes.SHA256())
    h.update(ikm)
    return h.finalize()


@pytest.mark.vectors
class TestRFC5869HKDF:
    """RFC 5869 Appendix A test vectors for HKDF-SHA256."""

    def test_case_1_basic(self) -> None:
        """Test Case 1: Basic test case with SHA-256."""
        ikm = bytes.fromhex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b")
        salt = bytes.fromhex("000102030405060708090a0b0c")
        info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
        length = 42

        expected_prk = bytes.fromhex("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5")
        expected_okm = bytes.fromhex(
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
        )

        # Verify Extract step
        prk = hkdf_extract(salt, ikm)
        assert prk == expected_prk, "PRK mismatch in test case 1"

        # Verify full HKDF
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            info=info,
        )
        okm = hkdf.derive(ikm)
        assert okm == expected_okm, "OKM mismatch in test case 1"

    def test_case_2_longer_inputs(self) -> None:
        """Test Case 2: Test with longer inputs/outputs."""
        ikm = bytes.fromhex(
            "000102030405060708090a0b0c0d0e0f"
            "101112131415161718191a1b1c1d1e1f"
            "202122232425262728292a2b2c2d2e2f"
            "303132333435363738393a3b3c3d3e3f"
            "404142434445464748494a4b4c4d4e4f"
        )
        salt = bytes.fromhex(
            "606162636465666768696a6b6c6d6e6f"
            "707172737475767778797a7b7c7d7e7f"
            "808182838485868788898a8b8c8d8e8f"
            "909192939495969798999a9b9c9d9e9f"
            "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf"
        )
        info = bytes.fromhex(
            "b0b1b2b3b4b5b6b7b8b9babbbcbdbebf"
            "c0c1c2c3c4c5c6c7c8c9cacbcccdcecf"
            "d0d1d2d3d4d5d6d7d8d9dadbdcdddedf"
            "e0e1e2e3e4e5e6e7e8e9eaebecedeeef"
            "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff"
        )
        length = 82

        expected_prk = bytes.fromhex("06a6b88c5853361a06104c9ceb35b45cef760014904671014a193f40c15fc244")
        expected_okm = bytes.fromhex(
            "b11e398dc80327a1c8e7f78c596a4934"
            "4f012eda2d4efad8a050cc4c19afa97c"
            "59045a99cac7827271cb41c65e590e09"
            "da3275600c2f09b8367793a9aca3db71"
            "cc30c58179ec3e87c14c01d5c1f3434f"
            "1d87"
        )

        # Verify Extract step
        prk = hkdf_extract(salt, ikm)
        assert prk == expected_prk, "PRK mismatch in test case 2"

        # Verify full HKDF
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=salt,
            info=info,
        )
        okm = hkdf.derive(ikm)
        assert okm == expected_okm, "OKM mismatch in test case 2"

    def test_case_3_zero_length_salt_info(self) -> None:
        """Test Case 3: Test with zero-length salt/info."""
        ikm = bytes.fromhex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b")
        salt = b""  # Zero-length salt
        info = b""  # Zero-length info
        length = 42

        expected_prk = bytes.fromhex("19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04")
        expected_okm = bytes.fromhex(
            "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8"
        )

        # Verify Extract step (empty salt treated as HashLen zeros)
        prk = hkdf_extract(salt, ikm)
        assert prk == expected_prk, "PRK mismatch in test case 3"

        # Verify full HKDF (pass None for empty salt per cryptography API)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=None,  # Empty salt
            info=info,
        )
        okm = hkdf.derive(ikm)
        assert okm == expected_okm, "OKM mismatch in test case 3"

    def test_expand_only(self) -> None:
        """Test HKDF-Expand separately using PRK from test case 1."""
        prk = bytes.fromhex("077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5")
        info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
        length = 42

        expected_okm = bytes.fromhex(
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
        )

        hkdf_expand = HKDFExpand(
            algorithm=hashes.SHA256(),
            length=length,
            info=info,
        )
        okm = hkdf_expand.derive(prk)
        assert okm == expected_okm, "HKDF-Expand mismatch"

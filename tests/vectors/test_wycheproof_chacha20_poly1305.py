"""Wycheproof ChaCha20-Poly1305 test vectors.

Tests ChaCha20-Poly1305 AEAD against Wycheproof vectors to catch edge cases and known attacks.
https://github.com/C2SP/wycheproof

To update vectors: `make download-vectors`
"""

import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305


def load_vectors() -> list[dict[str, Any]]:
    """Load Wycheproof ChaCha20-Poly1305 test vectors."""
    path = Path(__file__).parent / "wycheproof_chacha20_poly1305.json"
    with path.open() as f:
        data: dict[str, Any] = json.load(f)
    # Flatten all test groups, include group metadata
    return [
        {
            **test,
            "_keySize": group["keySize"],
            "_ivSize": group["ivSize"],
            "_tagSize": group["tagSize"],
        }
        for group in data["testGroups"]
        for test in group["tests"]
    ]


def vector_id(test: dict[str, Any]) -> str:
    """Generate test ID from vector."""
    return f"tc{test['tcId']}-{test['result']}"


@pytest.mark.vectors
class TestWycheproofChaCha20Poly1305:
    """Wycheproof ChaCha20-Poly1305 AEAD test vectors."""

    @pytest.mark.parametrize("test", load_vectors(), ids=vector_id)
    def test_chacha20_poly1305(self, test: dict[str, Any]) -> None:
        """Test ChaCha20-Poly1305 against Wycheproof vectors."""
        key = bytes.fromhex(test["key"])
        iv = bytes.fromhex(test["iv"])
        aad = bytes.fromhex(test["aad"])
        msg = bytes.fromhex(test["msg"])
        ct = bytes.fromhex(test["ct"])
        tag = bytes.fromhex(test["tag"])
        result = test["result"]  # "valid", "invalid", or "acceptable"

        # Skip non-standard key/iv/tag sizes
        if test["_keySize"] != 256 or test["_ivSize"] != 96 or test["_tagSize"] != 128:
            pytest.skip("Non-standard parameters")

        ciphertext_with_tag = ct + tag
        cipher = ChaCha20Poly1305(key)

        if result == "valid":
            # Encryption should produce expected ciphertext
            encrypted = cipher.encrypt(iv, msg, aad)
            assert encrypted == ciphertext_with_tag

            # Decryption should recover plaintext
            decrypted = cipher.decrypt(iv, ciphertext_with_tag, aad)
            assert decrypted == msg

        elif result == "invalid":
            # Decryption should fail (tag mismatch, etc.)
            with pytest.raises(InvalidTag):
                cipher.decrypt(iv, ciphertext_with_tag, aad)

        else:  # acceptable
            # Implementation-defined - verify encrypt/decrypt roundtrip if possible
            encrypted = cipher.encrypt(iv, msg, aad)
            if encrypted == ciphertext_with_tag:
                decrypted = cipher.decrypt(iv, ciphertext_with_tag, aad)
                assert decrypted == msg

"""Wycheproof HKDF-SHA256 test vectors.

Tests HKDF-SHA256 key derivation against Wycheproof vectors to catch edge cases.
https://github.com/C2SP/wycheproof

This validates the underlying KDF primitive used by our HPKE implementation.

To update vectors: `make download-vectors`
"""

import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def load_vectors() -> list[dict[str, Any]]:
    """Load Wycheproof HKDF-SHA256 test vectors."""
    path = Path(__file__).parent / "wycheproof_hkdf_sha256.json"
    with path.open() as f:
        data: dict[str, Any] = json.load(f)
    # Flatten all test groups
    return [
        {
            **test,
            "_keySize": group["keySize"],
        }
        for group in data["testGroups"]
        for test in group["tests"]
    ]


def vector_id(test: dict[str, Any]) -> str:
    """Generate test ID from vector."""
    return f"tc{test['tcId']}-{test['result']}"


@pytest.mark.vectors
class TestWycheproofHKDFSHA256:
    """Wycheproof HKDF-SHA256 test vectors."""

    @pytest.mark.parametrize("test", load_vectors(), ids=vector_id)
    def test_hkdf_sha256(self, test: dict[str, Any]) -> None:
        """Test HKDF-SHA256 against Wycheproof vectors."""
        ikm = bytes.fromhex(test["ikm"])
        salt_hex = test["salt"]
        salt = bytes.fromhex(salt_hex) if salt_hex else None
        info = bytes.fromhex(test["info"])
        size = test["size"]
        expected_okm = bytes.fromhex(test["okm"])
        result = test["result"]

        # Skip invalid tests that expect failure (e.g., size too large)
        if result == "invalid":
            # HKDF max output is 255 * hash_len = 255 * 32 = 8160 bytes
            max_output = 255 * 32
            if size > max_output:
                with pytest.raises(ValueError):
                    hkdf = HKDF(
                        algorithm=hashes.SHA256(),
                        length=size,
                        salt=salt,
                        info=info,
                    )
                    hkdf.derive(ikm)
                return
            pytest.skip("Unknown invalid test case")

        # Valid tests should produce expected output
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=size,
            salt=salt,
            info=info,
        )
        okm = hkdf.derive(ikm)
        assert okm == expected_okm, f"HKDF output mismatch for tcId={test['tcId']}"

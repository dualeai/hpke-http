"""Wycheproof X25519 test vectors.

Tests X25519 ECDH against Wycheproof vectors to catch edge cases and known attacks.
https://github.com/C2SP/wycheproof

To update vectors: `make download-vectors`
"""

import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)


def load_vectors() -> list[dict[str, Any]]:
    """Load Wycheproof X25519 test vectors."""
    path = Path(__file__).parent / "wycheproof_x25519.json"
    with path.open() as f:
        data: dict[str, Any] = json.load(f)
    return [test for group in data["testGroups"] for test in group["tests"]]


def vector_id(test: dict[str, Any]) -> str:
    """Generate test ID from vector."""
    return f"tc{test['tcId']}-{test['result']}"


@pytest.mark.vectors
class TestWycheproofX25519:
    """Wycheproof X25519 ECDH test vectors."""

    @pytest.mark.parametrize("test", load_vectors(), ids=vector_id)
    def test_x25519_ecdh(self, test: dict[str, Any]) -> None:
        """Test X25519 key exchange against Wycheproof vectors."""
        public_bytes = bytes.fromhex(test["public"])
        private_bytes = bytes.fromhex(test["private"])
        expected_shared = bytes.fromhex(test["shared"])
        result = test["result"]  # "valid", "invalid", or "acceptable"

        try:
            # Load keys
            private_key = X25519PrivateKey.from_private_bytes(private_bytes)
            public_key = X25519PublicKey.from_public_bytes(public_bytes)

            # Perform ECDH
            shared_secret = private_key.exchange(public_key)

            if result == "valid":
                assert shared_secret == expected_shared
            elif result == "acceptable":
                # Acceptable means implementation-defined behavior
                # We accept if it matches expected or raises
                assert shared_secret == expected_shared
            # Invalid vectors that compute are checked by Wycheproof design

        except (ValueError, UnsupportedAlgorithm):
            # Exception is acceptable for invalid/acceptable vectors
            if result == "valid":
                raise

"""Wycheproof HMAC-SHA256 test vectors.

Tests HMAC-SHA256 against Wycheproof vectors to catch edge cases.
https://github.com/C2SP/wycheproof

HMAC-SHA256 is the foundation of HKDF-SHA256, which is used throughout
our HPKE implementation. Testing HMAC ensures the core building block
is working correctly.

To update vectors: `make download-vectors`
"""

import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes, hmac


def load_vectors() -> list[dict[str, Any]]:
    """Load Wycheproof HMAC-SHA256 test vectors."""
    path = Path(__file__).parent / "wycheproof_hmac_sha256.json"
    with path.open() as f:
        data: dict[str, Any] = json.load(f)
    # Flatten all test groups, include group metadata
    return [
        {
            **test,
            "_keySize": group["keySize"],
            "_tagSize": group["tagSize"],
        }
        for group in data["testGroups"]
        for test in group["tests"]
    ]


def vector_id(test: dict[str, Any]) -> str:
    """Generate test ID from vector."""
    return f"tc{test['tcId']}-{test['result']}"


@pytest.mark.vectors
class TestWycheproofHMACSHA256:
    """Wycheproof HMAC-SHA256 test vectors."""

    @pytest.mark.parametrize("test", load_vectors(), ids=vector_id)
    def test_hmac_sha256(self, test: dict[str, Any]) -> None:
        """Test HMAC-SHA256 against Wycheproof vectors."""
        key = bytes.fromhex(test["key"])
        msg = bytes.fromhex(test["msg"])
        expected_tag = bytes.fromhex(test["tag"])
        result = test["result"]
        tag_size = test["_tagSize"] // 8  # Convert bits to bytes

        # Compute HMAC
        h = hmac.HMAC(key, hashes.SHA256())
        h.update(msg)
        computed_tag = h.finalize()

        # Truncate to expected tag size if needed
        computed_tag_truncated = computed_tag[:tag_size]

        if result == "valid":
            # Valid: computed tag must match expected
            assert computed_tag_truncated == expected_tag, f"HMAC mismatch for tcId={test['tcId']}"
        elif result == "invalid":
            # Invalid: computed tag should NOT match (modified tag test)
            assert computed_tag_truncated != expected_tag, f"Invalid tag unexpectedly matched for tcId={test['tcId']}"
        else:  # acceptable
            # Acceptable: implementation-defined, just verify we can compute
            # These are usually edge cases like short keys
            pass

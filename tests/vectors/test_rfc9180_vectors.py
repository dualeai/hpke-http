"""RFC 9180 HPKE test vector validation.

Tests against official CFRG test vectors:
https://github.com/cfrg/draft-irtf-cfrg-hpke/blob/master/test-vectors.json

Validates our implementation matches the reference for:
- mode=1 (PSK)
- kem_id=0x0020 (DHKEM(X25519, HKDF-SHA256))
- kdf_id=0x0001 (HKDF-SHA256)
- aead_id=0x0003 (ChaCha20-Poly1305)

To update vectors: `make download-vectors`
"""

import json
from pathlib import Path
from typing import Any

import pytest

from hpke_http.hpke import (
    _setup_sender_psk_deterministic,  # pyright: ignore[reportPrivateUsage] - test access
    setup_recipient_psk,
)


def load_test_vectors() -> list[dict[str, Any]]:
    """Load PSK test vectors from JSON file."""
    vectors_path = Path(__file__).parent / "rfc9180_psk_x25519_chacha.json"
    with vectors_path.open() as f:
        return json.load(f)


@pytest.mark.vectors
class TestRFC9180Vectors:
    """Validate against official RFC 9180 test vectors."""

    @pytest.fixture
    def vector(self) -> dict[str, Any]:
        """Load the PSK test vector."""
        vectors = load_test_vectors()
        assert len(vectors) == 1, "Expected exactly one PSK test vector"
        return vectors[0]

    def test_vector_metadata(self, vector: dict[str, Any]) -> None:
        """Verify test vector has correct suite configuration."""
        assert vector["mode"] == 1, "Must be PSK mode"
        assert vector["kem_id"] == 32, "Must be X25519"
        assert vector["kdf_id"] == 1, "Must be HKDF-SHA256"
        assert vector["aead_id"] == 3, "Must be ChaCha20-Poly1305"

    def test_sender_key_schedule(self, vector: dict[str, Any]) -> None:
        """Validate sender key schedule matches test vector.

        Uses deterministic encap with test vector's ephemeral key.
        """
        # Parse test vector inputs
        pk_r = bytes.fromhex(vector["pkRm"])
        sk_e = bytes.fromhex(vector["skEm"])
        info = bytes.fromhex(vector["info"])
        psk = bytes.fromhex(vector["psk"])
        psk_id = bytes.fromhex(vector["psk_id"])

        # Expected outputs
        expected_enc = bytes.fromhex(vector["enc"])
        expected_key = bytes.fromhex(vector["key"])
        expected_base_nonce = bytes.fromhex(vector["base_nonce"])
        expected_exporter_secret = bytes.fromhex(vector["exporter_secret"])

        # Run deterministic setup
        ctx = _setup_sender_psk_deterministic(pk_r, info, psk, psk_id, sk_e)

        # Validate all outputs match
        assert ctx.enc == expected_enc, "enc mismatch"
        assert ctx.key == expected_key, "key mismatch"
        assert ctx.base_nonce == expected_base_nonce, "base_nonce mismatch"
        assert ctx.exporter_secret == expected_exporter_secret, "exporter_secret mismatch"

    def test_recipient_key_schedule(self, vector: dict[str, Any]) -> None:
        """Validate recipient key schedule matches test vector."""
        # Parse test vector inputs
        enc = bytes.fromhex(vector["enc"])
        sk_r = bytes.fromhex(vector["skRm"])
        info = bytes.fromhex(vector["info"])
        psk = bytes.fromhex(vector["psk"])
        psk_id = bytes.fromhex(vector["psk_id"])

        # Expected outputs
        expected_key = bytes.fromhex(vector["key"])
        expected_base_nonce = bytes.fromhex(vector["base_nonce"])
        expected_exporter_secret = bytes.fromhex(vector["exporter_secret"])

        # Run setup
        ctx = setup_recipient_psk(enc, sk_r, info, psk, psk_id)

        # Validate all outputs match
        assert ctx.key == expected_key, "key mismatch"
        assert ctx.base_nonce == expected_base_nonce, "base_nonce mismatch"
        assert ctx.exporter_secret == expected_exporter_secret, "exporter_secret mismatch"

    def test_encryption_roundtrip(self, vector: dict[str, Any]) -> None:
        """Validate encryption produces expected ciphertexts."""
        # Parse test vector inputs
        pk_r = bytes.fromhex(vector["pkRm"])
        sk_r = bytes.fromhex(vector["skRm"])
        sk_e = bytes.fromhex(vector["skEm"])
        info = bytes.fromhex(vector["info"])
        psk = bytes.fromhex(vector["psk"])
        psk_id = bytes.fromhex(vector["psk_id"])

        # Setup sender with deterministic ephemeral key
        sender_ctx = _setup_sender_psk_deterministic(pk_r, info, psk, psk_id, sk_e)

        # Setup recipient
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, info, psk, psk_id)

        # Test each encryption in the vector
        encryptions = vector["encryptions"]
        for enc_data in encryptions:
            # Parse expected values
            aad = bytes.fromhex(enc_data["aad"])
            pt = bytes.fromhex(enc_data["pt"])
            expected_ct = bytes.fromhex(enc_data["ct"])
            expected_nonce = bytes.fromhex(enc_data["nonce"])

            # Verify nonce matches (before seal increments seq)
            actual_nonce = sender_ctx._compute_nonce()  # pyright: ignore[reportPrivateUsage]
            assert actual_nonce == expected_nonce, f"nonce mismatch at seq {sender_ctx.seq}"

            # Encrypt
            ct = sender_ctx.seal(aad, pt)
            assert ct == expected_ct, f"ciphertext mismatch at seq {sender_ctx.seq - 1}"

            # Decrypt
            decrypted = recipient_ctx.open(aad, ct)
            assert decrypted == pt, "decryption roundtrip failed"

    def test_all_encryptions_in_sequence(self, vector: dict[str, Any]) -> None:
        """Test all 256 encryptions maintain correct sequence."""
        # Parse test vector inputs
        pk_r = bytes.fromhex(vector["pkRm"])
        sk_r = bytes.fromhex(vector["skRm"])
        sk_e = bytes.fromhex(vector["skEm"])
        info = bytes.fromhex(vector["info"])
        psk = bytes.fromhex(vector["psk"])
        psk_id = bytes.fromhex(vector["psk_id"])

        # Setup contexts
        sender_ctx = _setup_sender_psk_deterministic(pk_r, info, psk, psk_id, sk_e)
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, info, psk, psk_id)

        # Test each encryption
        encryptions = vector["encryptions"]
        for i, enc_data in enumerate(encryptions):
            aad = bytes.fromhex(enc_data["aad"])
            pt = bytes.fromhex(enc_data["pt"])
            expected_ct = bytes.fromhex(enc_data["ct"])

            # Verify sequence counter
            assert sender_ctx.seq == i, f"sender seq mismatch: {sender_ctx.seq} != {i}"
            assert recipient_ctx.seq == i, f"recipient seq mismatch: {recipient_ctx.seq} != {i}"

            # Encrypt and verify
            ct = sender_ctx.seal(aad, pt)
            assert ct == expected_ct, f"ciphertext mismatch at index {i}"

            # Decrypt and verify
            decrypted = recipient_ctx.open(aad, ct)
            assert decrypted == pt, f"decryption failed at index {i}"

    def test_shared_secret_derivation(self, vector: dict[str, Any]) -> None:
        """Validate that sender and recipient derive same shared secret.

        This test validates the KEM Encap/Decap produces matching shared secrets.
        """
        # Use deterministic encap to get same enc as test vector
        pk_r = bytes.fromhex(vector["pkRm"])
        sk_r = bytes.fromhex(vector["skRm"])
        sk_e = bytes.fromhex(vector["skEm"])
        info = bytes.fromhex(vector["info"])
        psk = bytes.fromhex(vector["psk"])
        psk_id = bytes.fromhex(vector["psk_id"])

        # Both sides should derive same key schedule
        sender_ctx = _setup_sender_psk_deterministic(pk_r, info, psk, psk_id, sk_e)
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, info, psk, psk_id)

        assert sender_ctx.key == recipient_ctx.key
        assert sender_ctx.base_nonce == recipient_ctx.base_nonce
        assert sender_ctx.exporter_secret == recipient_ctx.exporter_secret

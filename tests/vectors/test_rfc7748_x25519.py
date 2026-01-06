"""RFC 7748 X25519 test vectors.

Official IETF test vectors for X25519 elliptic curve Diffie-Hellman.
Source: https://www.rfc-editor.org/rfc/rfc7748.html

Sections tested:
- Section 5.2: Direct scalar multiplication test vectors
- Section 6.1: Diffie-Hellman key exchange example

To update: These are hardcoded from the RFC specification.
"""

import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)


@pytest.mark.vectors
class TestRFC7748X25519:
    """RFC 7748 Section 5.2 and 6.1 test vectors."""

    def test_scalar_mult_vector_1(self) -> None:
        """RFC 7748 Section 5.2 - Test Vector 1."""
        scalar = bytes.fromhex("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4")
        u_coord = bytes.fromhex("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c")
        expected = bytes.fromhex("c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552")

        private_key = X25519PrivateKey.from_private_bytes(scalar)
        public_key = X25519PublicKey.from_public_bytes(u_coord)
        result = private_key.exchange(public_key)

        assert result == expected, "RFC 7748 vector 1 mismatch"

    def test_scalar_mult_vector_2(self) -> None:
        """RFC 7748 Section 5.2 - Test Vector 2."""
        scalar = bytes.fromhex("4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d")
        u_coord = bytes.fromhex("e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493")
        expected = bytes.fromhex("95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957")

        private_key = X25519PrivateKey.from_private_bytes(scalar)
        public_key = X25519PublicKey.from_public_bytes(u_coord)
        result = private_key.exchange(public_key)

        assert result == expected, "RFC 7748 vector 2 mismatch"

    def test_dh_key_exchange(self) -> None:
        """RFC 7748 Section 6.1 - Diffie-Hellman example."""
        # Alice's keys
        alice_private = bytes.fromhex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a")
        alice_public_expected = bytes.fromhex("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")

        # Bob's keys
        bob_private = bytes.fromhex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb")
        bob_public_expected = bytes.fromhex("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f")

        # Expected shared secret
        shared_secret_expected = bytes.fromhex("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742")

        # Load keys
        alice_sk = X25519PrivateKey.from_private_bytes(alice_private)
        bob_sk = X25519PrivateKey.from_private_bytes(bob_private)

        # Verify public key derivation
        alice_pk = alice_sk.public_key().public_bytes_raw()
        bob_pk = bob_sk.public_key().public_bytes_raw()

        assert alice_pk == alice_public_expected, "Alice public key mismatch"
        assert bob_pk == bob_public_expected, "Bob public key mismatch"

        # Verify shared secret (both directions)
        bob_public = X25519PublicKey.from_public_bytes(bob_public_expected)
        alice_public = X25519PublicKey.from_public_bytes(alice_public_expected)

        shared_alice = alice_sk.exchange(bob_public)
        shared_bob = bob_sk.exchange(alice_public)

        assert shared_alice == shared_secret_expected, "Alice shared secret mismatch"
        assert shared_bob == shared_secret_expected, "Bob shared secret mismatch"
        assert shared_alice == shared_bob, "Shared secrets don't match"

    def test_iterated_1(self) -> None:
        """RFC 7748 Section 5.2 - Iterated test (1 iteration)."""
        # Start with k = u = 9 (basepoint)
        k = bytes.fromhex("0900000000000000000000000000000000000000000000000000000000000000")

        # After 1 iteration
        expected_1 = bytes.fromhex("422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079")

        private_key = X25519PrivateKey.from_private_bytes(k)
        public_key = X25519PublicKey.from_public_bytes(k)
        result = private_key.exchange(public_key)

        assert result == expected_1, "Iterated test (1 iter) mismatch"

    def test_iterated_1000(self) -> None:
        """RFC 7748 Section 5.2 - Iterated test (1000 iterations)."""
        # Start with k = u = 9 (basepoint)
        k = bytes.fromhex("0900000000000000000000000000000000000000000000000000000000000000")
        u = k

        # After 1000 iterations
        expected_1000 = bytes.fromhex("684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51")

        for _ in range(1000):
            private_key = X25519PrivateKey.from_private_bytes(k)
            public_key = X25519PublicKey.from_public_bytes(u)
            result = private_key.exchange(public_key)
            u = k
            k = result

        assert k == expected_1000, "Iterated test (1000 iter) mismatch"

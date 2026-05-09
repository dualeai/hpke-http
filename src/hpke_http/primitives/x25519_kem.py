"""
DHKEM(X25519, HKDF-SHA256) implementation behind the KEM ABC.

RFC 9180 §4.1 KEM with kem_id 0x0020. Uses X25519 ECDH for the Diffie-Hellman
operation and HKDF-SHA256 for shared-secret derivation.

This module replaces the pre-abstraction logic that used to live in
primitives/kem.py at module level. The shim primitives/kem.py re-exports
classmethods on this class as module-level functions for backward
compatibility.

Reference: https://datatracker.ietf.org/doc/rfc9180/ §4.1
"""

from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric import x25519

from hpke_http.constants import (
    X25519_ENC_SIZE,
    X25519_PRIVATE_KEY_SIZE,
    X25519_PUBLIC_KEY_SIZE,
    X25519_SHARED_SECRET_SIZE,
    KemId,
)
from hpke_http.primitives.kdf import extract_and_expand
from hpke_http.primitives.kem_base import KEM, register_kem

__all__ = ["X25519KEM"]


@register_kem
class X25519KEM(KEM):
    """DHKEM(X25519, HKDF-SHA256) — RFC 9180 §4.1.

    Stateless. All operations are classmethods. Thread-safe.
    """

    KEM_ID = KemId.DHKEM_X25519_HKDF_SHA256
    PUBLIC_KEY_SIZE = X25519_PUBLIC_KEY_SIZE
    PRIVATE_KEY_SIZE = X25519_PRIVATE_KEY_SIZE
    ENC_SIZE = X25519_ENC_SIZE
    SHARED_SECRET_SIZE = X25519_SHARED_SECRET_SIZE
    DERIVE_KEYPAIR_IKM_MIN_SIZE = X25519_PRIVATE_KEY_SIZE  # RFC 9180 §4.1: ≥32 bytes

    @classmethod
    def generate_keypair(cls) -> tuple[bytes, bytes]:
        """Generate a fresh X25519 keypair using the system CSPRNG."""
        sk = x25519.X25519PrivateKey.generate()
        return (
            sk.private_bytes_raw(),
            sk.public_key().public_bytes_raw(),
        )

    @classmethod
    def derive_keypair(cls, ikm: bytes) -> tuple[bytes, bytes]:
        """RFC 9180 §4.1 DeriveKeyPair(ikm) for DHKEM(X25519, HKDF-SHA256).

        Uses LabeledExtract("dkp_prk") + LabeledExpand("sk") on the
        KEM-internal suite_id.
        """
        if len(ikm) < cls.DERIVE_KEYPAIR_IKM_MIN_SIZE:
            raise ValueError(f"X25519KEM: ikm length {len(ikm)} < {cls.DERIVE_KEYPAIR_IKM_MIN_SIZE}")

        sk_bytes = extract_and_expand(
            salt=b"",
            label_extract=b"dkp_prk",
            ikm=ikm,
            label_expand=b"sk",
            info=b"",
            length=cls.PRIVATE_KEY_SIZE,
            suite_id=cls.kem_suite_id(),
        )
        sk = x25519.X25519PrivateKey.from_private_bytes(sk_bytes)
        return (
            sk.private_bytes_raw(),
            sk.public_key().public_bytes_raw(),
        )

    @classmethod
    def encap(cls, pk_r: bytes) -> tuple[bytes, bytes]:
        """RFC 9180 §4.1 Encap(pkR).

        Generates an ephemeral keypair, performs X25519 with pk_r, derives the
        shared secret via ExtractAndExpand on (dh, enc || pk_r).
        """
        cls.validate_public_key(pk_r)

        sk_e, pk_e = cls.generate_keypair()
        ephemeral_private = x25519.X25519PrivateKey.from_private_bytes(sk_e)
        recipient_public = x25519.X25519PublicKey.from_public_bytes(pk_r)
        dh = ephemeral_private.exchange(recipient_public)

        enc = pk_e
        kem_context = enc + pk_r
        shared_secret = cls._extract_and_expand_dh(dh, kem_context)
        return (enc, shared_secret)

    @classmethod
    def decap(cls, enc: bytes, sk_r: bytes) -> bytes:
        """RFC 9180 §4.1 Decap(enc, skR).

        Performs X25519 between sk_r and enc (sender's ephemeral pk), derives
        the shared secret via ExtractAndExpand on (dh, enc || pk_r).

        Implementation note: pyca's X25519 ``exchange`` raises
        ``ValueError("Error computing shared key")`` on small-subgroup inputs
        per RFC 7748 §6.1 (e.g. all-zero enc). RFC 9180 §4.1 does not mandate
        either raise vs implicit rejection for DHKEM(X25519); we let the
        backend's behavior propagate. Callers requiring implicit-rejection
        semantics should use a hybrid KEM (e.g. ``XWingKEM``).
        """
        cls.validate_enc(enc)
        cls.validate_private_key(sk_r)

        recipient_private = x25519.X25519PrivateKey.from_private_bytes(sk_r)
        sender_public = x25519.X25519PublicKey.from_public_bytes(enc)
        dh = recipient_private.exchange(sender_public)

        pk_r = recipient_private.public_key().public_bytes_raw()
        kem_context = enc + pk_r
        return cls._extract_and_expand_dh(dh, kem_context)

    @classmethod
    def derive_public_key(cls, sk: bytes) -> bytes:
        """Derive X25519 public key from raw private key bytes."""
        cls.validate_private_key(sk)
        return x25519.X25519PrivateKey.from_private_bytes(sk).public_key().public_bytes_raw()

    # Internal helpers --------------------------------------------------------

    @classmethod
    def _extract_and_expand_dh(cls, dh: bytes, kem_context: bytes) -> bytes:
        """RFC 9180 §4.1 ExtractAndExpand(dh, kem_context).

        Derives the shared secret from the DH result and encapsulation context
        using the KEM-internal suite_id.
        """
        return extract_and_expand(
            salt=b"",
            label_extract=b"eae_prk",
            ikm=dh,
            label_expand=b"shared_secret",
            info=kem_context,
            length=cls.SHARED_SECRET_SIZE,
            suite_id=cls.kem_suite_id(),
        )

    @classmethod
    def _encap_deterministic(  # pyright: ignore[reportUnusedFunction]
        cls, pk_r: bytes, sk_e: bytes
    ) -> tuple[bytes, bytes]:
        """Deterministic Encap for RFC 9180 known-answer-test replay.

        WARNING: test-only. Production code MUST use encap() with system
        randomness. Tests reach in via `# noqa: SLF001`.
        """
        cls.validate_public_key(pk_r)
        cls.validate_private_key(sk_e)

        ephemeral_private = x25519.X25519PrivateKey.from_private_bytes(sk_e)
        recipient_public = x25519.X25519PublicKey.from_public_bytes(pk_r)
        pk_e = ephemeral_private.public_key().public_bytes_raw()
        dh = ephemeral_private.exchange(recipient_public)

        enc = pk_e
        kem_context = enc + pk_r
        shared_secret = cls._extract_and_expand_dh(dh, kem_context)
        return (enc, shared_secret)

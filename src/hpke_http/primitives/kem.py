"""
RFC 9180 §4.1 Key Encapsulation Mechanism for DHKEM(X25519, HKDF-SHA256).

Implements Encap/Decap for X25519 key exchange with HKDF-SHA256 key derivation.

Reference: https://datatracker.ietf.org/doc/rfc9180/ §4.1
"""

from cryptography.hazmat.primitives.asymmetric import x25519

from hpke_http.constants import (
    X25519_ENC_SIZE,
    X25519_PRIVATE_KEY_SIZE,
    X25519_PUBLIC_KEY_SIZE,
    X25519_SHARED_SECRET_SIZE,
)
from hpke_http.primitives.kdf import extract_and_expand

__all__ = [
    "decap",
    "derive_keypair",
    "encap",
    "generate_keypair",
]

# KEM suite_id for DHKEM(X25519, HKDF-SHA256)
# "KEM" || I2OSP(kem_id, 2)
KEM_SUITE_ID = b"KEM" + (0x0020).to_bytes(2, "big")


def generate_keypair() -> tuple[bytes, bytes]:
    """
    Generate a new X25519 keypair.

    Returns:
        Tuple of (private_key, public_key) as raw 32-byte values
    """
    private_key = x25519.X25519PrivateKey.generate()
    return (
        private_key.private_bytes_raw(),
        private_key.public_key().public_bytes_raw(),
    )


def derive_keypair(ikm: bytes) -> tuple[bytes, bytes]:
    """
    Derive a deterministic X25519 keypair from input keying material.

    RFC 9180 §4.1: DeriveKeyPair(ikm)

    Args:
        ikm: Input keying material (should be random, at least 32 bytes)

    Returns:
        Tuple of (private_key, public_key) as raw 32-byte values
    """
    # Extract-and-Expand with label "dkp_prk" for Extract and "sk" for Expand
    sk_bytes = extract_and_expand(
        salt=b"",
        label_extract=b"dkp_prk",
        ikm=ikm,
        label_expand=b"sk",
        info=b"",
        length=X25519_PRIVATE_KEY_SIZE,
        suite_id=KEM_SUITE_ID,
    )

    private_key = x25519.X25519PrivateKey.from_private_bytes(sk_bytes)
    return (
        private_key.private_bytes_raw(),
        private_key.public_key().public_bytes_raw(),
    )


def _extract_and_expand_dh(dh: bytes, kem_context: bytes) -> bytes:
    """
    RFC 9180 §4.1: ExtractAndExpand(dh, kem_context)

    Derives the shared secret from DH result and encapsulation context.

    Args:
        dh: Raw Diffie-Hellman shared value (X25519 output)
        kem_context: enc || pkR (encapsulated key || recipient public key)

    Returns:
        Derived shared secret (32 bytes)
    """
    return extract_and_expand(
        salt=b"",
        label_extract=b"eae_prk",
        ikm=dh,
        label_expand=b"shared_secret",
        info=kem_context,
        length=X25519_SHARED_SECRET_SIZE,
        suite_id=KEM_SUITE_ID,
    )


def encap(pk_r: bytes) -> tuple[bytes, bytes]:
    """
    RFC 9180 §4.1: Encap(pkR)

    Generates an ephemeral keypair, performs key exchange with recipient's
    public key, and returns encapsulated key and shared secret.

    Args:
        pk_r: Recipient's X25519 public key (32 bytes)

    Returns:
        Tuple of (enc, shared_secret):
        - enc: Encapsulated key (ephemeral public key, 32 bytes)
        - shared_secret: Derived shared secret (32 bytes)

    Raises:
        ValueError: If public key size is invalid
    """
    if len(pk_r) != X25519_PUBLIC_KEY_SIZE:
        raise ValueError(f"Invalid public key size: expected {X25519_PUBLIC_KEY_SIZE}, got {len(pk_r)}")

    # Generate ephemeral keypair
    sk_e, pk_e = generate_keypair()

    # Perform DH: dh = X25519(skE, pkR)
    ephemeral_private = x25519.X25519PrivateKey.from_private_bytes(sk_e)
    recipient_public = x25519.X25519PublicKey.from_public_bytes(pk_r)
    dh = ephemeral_private.exchange(recipient_public)

    # enc = pkE (encapsulated key is the ephemeral public key)
    enc = pk_e

    # kem_context = enc || pkR
    kem_context = enc + pk_r

    # shared_secret = ExtractAndExpand(dh, kem_context)
    shared_secret = _extract_and_expand_dh(dh, kem_context)

    return (enc, shared_secret)


def _encap_deterministic(pk_r: bytes, sk_e: bytes) -> tuple[bytes, bytes]:  # pyright: ignore[reportUnusedFunction]
    """
    Deterministic Encap for testing with known ephemeral key.

    WARNING: Only use for test vector validation. Production code must use encap().

    Args:
        pk_r: Recipient's X25519 public key (32 bytes)
        sk_e: Ephemeral private key (32 bytes) - MUST be random in production

    Returns:
        Tuple of (enc, shared_secret)

    Raises:
        ValueError: If key sizes are invalid
    """
    if len(pk_r) != X25519_PUBLIC_KEY_SIZE:
        raise ValueError(f"Invalid public key size: expected {X25519_PUBLIC_KEY_SIZE}, got {len(pk_r)}")
    if len(sk_e) != X25519_PRIVATE_KEY_SIZE:
        raise ValueError(f"Invalid private key size: expected {X25519_PRIVATE_KEY_SIZE}, got {len(sk_e)}")

    # Load keys
    ephemeral_private = x25519.X25519PrivateKey.from_private_bytes(sk_e)
    recipient_public = x25519.X25519PublicKey.from_public_bytes(pk_r)

    # Compute ephemeral public key and DH
    pk_e = ephemeral_private.public_key().public_bytes_raw()
    dh = ephemeral_private.exchange(recipient_public)

    enc = pk_e
    kem_context = enc + pk_r
    shared_secret = _extract_and_expand_dh(dh, kem_context)

    return (enc, shared_secret)


def decap(enc: bytes, sk_r: bytes) -> bytes:
    """
    RFC 9180 §4.1: Decap(enc, skR)

    Decapsulates the shared secret using recipient's private key.

    Args:
        enc: Encapsulated key (sender's ephemeral public key, 32 bytes)
        sk_r: Recipient's X25519 private key (32 bytes)

    Returns:
        Derived shared secret (32 bytes)

    Raises:
        ValueError: If enc or sk_r size is invalid
    """
    if len(enc) != X25519_ENC_SIZE:
        raise ValueError(f"Invalid enc size: expected {X25519_ENC_SIZE}, got {len(enc)}")
    if len(sk_r) != X25519_PRIVATE_KEY_SIZE:
        raise ValueError(f"Invalid private key size: expected {X25519_PRIVATE_KEY_SIZE}, got {len(sk_r)}")

    # Load keys
    recipient_private = x25519.X25519PrivateKey.from_private_bytes(sk_r)
    sender_public = x25519.X25519PublicKey.from_public_bytes(enc)

    # Perform DH: dh = X25519(skR, enc)
    dh = recipient_private.exchange(sender_public)

    # Get recipient's public key for kem_context
    pk_r = recipient_private.public_key().public_bytes_raw()

    # kem_context = enc || pkR
    kem_context = enc + pk_r

    # shared_secret = ExtractAndExpand(dh, kem_context)
    return _extract_and_expand_dh(dh, kem_context)

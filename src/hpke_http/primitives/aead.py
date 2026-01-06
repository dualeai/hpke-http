"""
RFC 9180 §5.2 AEAD operations for ChaCha20-Poly1305.

Implements Seal and Open with nonce computation from base_nonce and sequence number.

Reference:
- RFC 9180 §5.2 (AEAD operations)
- RFC 8439 (ChaCha20-Poly1305)
"""

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from hpke_http.constants import (
    CHACHA20_POLY1305_KEY_SIZE,
    CHACHA20_POLY1305_NONCE_SIZE,
    CHACHA20_POLY1305_TAG_SIZE,
)
from hpke_http.exceptions import DecryptionError

__all__ = [
    "aead_open",
    "aead_seal",
    "compute_nonce",
]


def compute_nonce(base_nonce: bytes, seq: int) -> bytes:
    """
    RFC 9180 §5.2: ComputeNonce(base_nonce, seq)

    XORs the base nonce with the sequence number to produce a unique nonce.

    nonce = base_nonce XOR I2OSP(seq, Nn)

    Args:
        base_nonce: Base nonce from key schedule (12 bytes)
        seq: Sequence number (0, 1, 2, ...)

    Returns:
        Computed nonce (12 bytes)

    Raises:
        ValueError: If base_nonce is not 12 bytes
    """
    if len(base_nonce) != CHACHA20_POLY1305_NONCE_SIZE:
        raise ValueError(f"Invalid base_nonce size: expected {CHACHA20_POLY1305_NONCE_SIZE}, got {len(base_nonce)}")

    # Convert seq to bytes (big-endian, same size as nonce)
    seq_bytes = seq.to_bytes(CHACHA20_POLY1305_NONCE_SIZE, "big")

    # XOR base_nonce with seq_bytes
    return bytes(a ^ b for a, b in zip(base_nonce, seq_bytes, strict=True))


def aead_seal(
    key: bytes,
    nonce: bytes,
    aad: bytes,
    plaintext: bytes,
) -> bytes:
    """
    RFC 9180 §5.2: Seal(key, nonce, aad, pt)

    Encrypts and authenticates plaintext with associated data.

    Args:
        key: Encryption key (32 bytes)
        nonce: Unique nonce (12 bytes)
        aad: Additional authenticated data (can be empty)
        plaintext: Data to encrypt

    Returns:
        Ciphertext with appended authentication tag (len = plaintext + 16)

    Raises:
        ValueError: If key or nonce size is invalid
    """
    if len(key) != CHACHA20_POLY1305_KEY_SIZE:
        raise ValueError(f"Invalid key size: expected {CHACHA20_POLY1305_KEY_SIZE}, got {len(key)}")
    if len(nonce) != CHACHA20_POLY1305_NONCE_SIZE:
        raise ValueError(f"Invalid nonce size: expected {CHACHA20_POLY1305_NONCE_SIZE}, got {len(nonce)}")

    cipher = ChaCha20Poly1305(key)
    return cipher.encrypt(nonce, plaintext, aad if aad else None)


def aead_open(
    key: bytes,
    nonce: bytes,
    aad: bytes,
    ciphertext: bytes,
) -> bytes:
    """
    RFC 9180 §5.2: Open(key, nonce, aad, ct)

    Decrypts and verifies ciphertext with associated data.

    Args:
        key: Encryption key (32 bytes)
        nonce: Nonce used for encryption (12 bytes)
        aad: Additional authenticated data (must match encryption)
        ciphertext: Encrypted data with authentication tag

    Returns:
        Decrypted plaintext

    Raises:
        ValueError: If key or nonce size is invalid
        DecryptionError: If authentication fails or ciphertext is invalid
    """
    if len(key) != CHACHA20_POLY1305_KEY_SIZE:
        raise ValueError(f"Invalid key size: expected {CHACHA20_POLY1305_KEY_SIZE}, got {len(key)}")
    if len(nonce) != CHACHA20_POLY1305_NONCE_SIZE:
        raise ValueError(f"Invalid nonce size: expected {CHACHA20_POLY1305_NONCE_SIZE}, got {len(nonce)}")

    if len(ciphertext) < CHACHA20_POLY1305_TAG_SIZE:
        raise DecryptionError("Ciphertext too short (missing authentication tag)")

    try:
        cipher = ChaCha20Poly1305(key)
        return cipher.decrypt(nonce, ciphertext, aad if aad else None)
    except Exception as e:
        # cryptography raises InvalidTag on auth failure
        raise DecryptionError("Authentication failed") from e


def aead_seal_with_seq(
    key: bytes,
    base_nonce: bytes,
    seq: int,
    aad: bytes,
    plaintext: bytes,
) -> bytes:
    """
    Convenience function: compute nonce from base_nonce + seq, then seal.

    Args:
        key: Encryption key (32 bytes)
        base_nonce: Base nonce from key schedule (12 bytes)
        seq: Sequence number
        aad: Additional authenticated data
        plaintext: Data to encrypt

    Returns:
        Ciphertext with authentication tag
    """
    nonce = compute_nonce(base_nonce, seq)
    return aead_seal(key, nonce, aad, plaintext)


def aead_open_with_seq(
    key: bytes,
    base_nonce: bytes,
    seq: int,
    aad: bytes,
    ciphertext: bytes,
) -> bytes:
    """
    Convenience function: compute nonce from base_nonce + seq, then open.

    Args:
        key: Encryption key (32 bytes)
        base_nonce: Base nonce from key schedule (12 bytes)
        seq: Sequence number
        aad: Additional authenticated data
        ciphertext: Encrypted data with tag

    Returns:
        Decrypted plaintext

    Raises:
        DecryptionError: If authentication fails
    """
    nonce = compute_nonce(base_nonce, seq)
    return aead_open(key, nonce, aad, ciphertext)

"""
RFC 9180 §5.2 AEAD nonce derivation for ChaCha20-Poly1305.

Reference:
- RFC 9180 §5.2 (AEAD operations)
- RFC 8439 (ChaCha20-Poly1305)
"""

from hpke_http.constants import CHACHA20_POLY1305_NONCE_SIZE

__all__ = [
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

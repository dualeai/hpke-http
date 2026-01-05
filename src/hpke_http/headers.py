"""
HTTP header utilities for HPKE encryption.

Uses base64url encoding (RFC 4648 §5) for HTTP header safety.
"""

import base64

from hpke_http.constants import HEADER_HPKE_ENC, HEADER_HPKE_STREAM

__all__ = [
    "HEADER_HPKE_ENC",
    "HEADER_HPKE_STREAM",
    "b64url_decode",
    "b64url_encode",
]


def b64url_encode(data: bytes) -> str:
    """
    Encode bytes to base64url string without padding.

    Args:
        data: Raw bytes to encode

    Returns:
        base64url encoded string (no padding)
    """
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(s: str) -> bytes:
    """
    Decode base64url string to bytes.

    Handles missing padding automatically.

    Args:
        s: base64url encoded string (with or without padding)

    Returns:
        Decoded bytes
    """
    # Add padding if needed (base64 uses 4-byte blocks)
    padding = len(s) % 4
    if padding:
        s += "=" * (4 - padding)
    return base64.urlsafe_b64decode(s)

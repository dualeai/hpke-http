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


def b64url_encode(data: bytes | memoryview) -> str:
    """
    Encode bytes to base64url string without padding.

    Accepts memoryview for zero-copy input (Python 3.4+).

    Args:
        data: Raw bytes or memoryview to encode

    Returns:
        base64url encoded string (no padding)
    """
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64url_decode(s: str) -> memoryview:
    """
    Decode base64url string to memoryview for zero-copy slicing.

    Handles missing padding automatically. Returns memoryview to enable
    zero-copy slicing of the decoded payload (e.g., extracting counter
    and ciphertext without allocating new buffers).

    Args:
        s: base64url encoded string (with or without padding)

    Returns:
        Decoded bytes as memoryview (supports zero-copy slicing, read-only)
    """
    # Add padding if needed (base64 uses 4-byte blocks)
    padding = len(s) % 4
    if padding:
        s += "=" * (4 - padding)
    # Return memoryview directly over decoded bytes (no extra copy)
    return memoryview(base64.urlsafe_b64decode(s))

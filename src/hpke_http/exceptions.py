"""
Exception hierarchy for hpke_http.

All crypto-related errors inherit from CryptoError for easy catching.
"""


class CryptoError(Exception):
    """Base exception for all cryptographic errors."""


class DecryptionError(CryptoError):
    """Failed to decrypt ciphertext.

    Possible causes:
    - Wrong key
    - Corrupted ciphertext
    - Invalid authentication tag
    - Nonce reuse detected
    """


class InvalidPSKError(CryptoError):
    """Pre-shared key validation failed.

    The PSK (API key) doesn't match what was used for encryption.
    """


class KeyDiscoveryError(CryptoError):
    """Failed to fetch or parse HPKE keys from discovery endpoint."""


class SequenceOverflowError(CryptoError):
    """Sequence counter has reached maximum value.

    The HPKE context can no longer be used for encryption/decryption.
    Nonce reuse would be catastrophic for ChaCha20-Poly1305 security.
    Create a new context to continue.
    """


class SessionExpiredError(CryptoError):
    """SSE streaming session has expired or exhausted its counter."""


class ReplayAttackError(CryptoError):
    """Detected out-of-order or duplicate SSE event.

    Counter validation failed, indicating potential replay attack.
    """

    def __init__(self, expected: int, received: int) -> None:
        self.expected = expected
        self.received = received
        # Don't expose counter values in message to prevent information leakage
        super().__init__("SSE event counter validation failed")

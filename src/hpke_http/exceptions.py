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
    """Failed to obtain a usable KEM key from the discovery endpoint.

    Raised when:
    - HTTP fetch of ``/.well-known/hpke-keys`` failed.
    - Discovery doc is structurally malformed (missing fields, bad version).
    - No entry in the client's ``kem_priority`` matches a server-advertised
      KEM (strict suite negotiation; see ``_select_suite``).
    """


class UnsupportedKEMError(CryptoError):
    """KEM is not registered.

    Raised by ``primitives.get_kem(kem_id)`` when no implementation is
    registered for the given identifier, and by the ``X-HPKE-Suite`` header
    parser when a client advertises a KEM the server doesn't recognize. The
    server-side middleware maps this to HTTP 415.
    """


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


class EncryptionRequiredError(CryptoError):
    """Encryption was required but request/response was plaintext.

    Raised when:
    - Server has require_encryption=True and receives a plaintext request
    - Client has require_encryption=True and receives a plaintext response
    """

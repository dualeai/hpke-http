"""
SSE (Server-Sent Events) streaming encryption.

Transparent encryption layer for SSE streams. Server sends normal SSE,
client receives normal SSE - encryption is invisible to application code.

Wire format:
    event: enc
    data: <base64url(counter_be32 || ciphertext)>

The ciphertext contains the raw SSE chunk exactly as the server sent it.
Perfect fidelity: comments, retry, id, events - everything preserved.

Reference: RFC-065 §6
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from hpke_http.constants import (
    CHACHA20_POLY1305_KEY_SIZE,
    SSE_COUNTER_SIZE,
    SSE_MAX_COUNTER,
    SSE_SESSION_KEY_LABEL,
    SSE_SESSION_SALT_SIZE,
)
from hpke_http.exceptions import DecryptionError, ReplayAttackError, SessionExpiredError
from hpke_http.headers import b64url_decode, b64url_encode
from hpke_http.hpke import HPKEContext

__all__ = [
    "SSEDecryptor",
    "SSEEncryptor",
    "StreamingSession",
    "create_session_from_context",
]


@dataclass
class StreamingSession:
    """
    SSE streaming session parameters.

    Created by server after decrypting initial request.
    Sent to client in X-HPKE-Stream header.
    """

    session_key: bytes
    """32-byte key derived from HPKE context."""

    session_salt: bytes
    """4-byte random salt for nonce construction."""

    @classmethod
    def create(cls, session_key: bytes) -> StreamingSession:
        """
        Create a new streaming session with random salt.

        Args:
            session_key: 32-byte key (from HPKE context.export())

        Returns:
            New StreamingSession
        """
        return cls(
            session_key=session_key,
            session_salt=secrets.token_bytes(SSE_SESSION_SALT_SIZE),
        )

    def serialize(self) -> bytes:
        """
        Serialize session parameters for transmission.

        Returns:
            session_salt (4 bytes) - key is derived, not transmitted
        """
        # Only transmit salt; key is derived from HPKE context
        return self.session_salt

    @classmethod
    def deserialize(cls, data: bytes, session_key: bytes) -> StreamingSession:
        """
        Deserialize session parameters.

        Args:
            data: Serialized session (4 bytes salt)
            session_key: Key derived from HPKE context

        Returns:
            StreamingSession
        """
        if len(data) != SSE_SESSION_SALT_SIZE:
            raise ValueError(f"Invalid session data length: {len(data)}")
        return cls(session_key=session_key, session_salt=data)


def create_session_from_context(ctx: HPKEContext) -> StreamingSession:
    """
    Create SSE streaming session from HPKE context.

    Uses HPKE export secret to derive session key.

    Args:
        ctx: HPKE context (sender or recipient)

    Returns:
        StreamingSession ready for encryption/decryption
    """
    session_key = ctx.export(SSE_SESSION_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
    return StreamingSession.create(session_key)


@dataclass
class SSEEncryptor:
    """
    Server-side SSE encryptor.

    Encrypts raw SSE chunks with monotonic counter for replay protection.
    Thread-safe: uses a lock to protect counter operations.

    The encryption is transparent - any valid SSE chunk (events, comments,
    retry directives) is encrypted as-is and will be decrypted identically.
    """

    session: StreamingSession
    counter: int = field(default=1)  # Start at 1 (0 reserved)
    _cipher: ChaCha20Poly1305 = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._cipher = ChaCha20Poly1305(self.session.session_key)

    def _compute_nonce(self, counter: int) -> bytes:
        """
        Compute 12-byte nonce from salt and counter.

        nonce = session_salt (4B) || zero_pad (4B) || counter_le32 (4B)
        """
        return self.session.session_salt + b"\x00\x00\x00\x00" + counter.to_bytes(4, "little")

    def encrypt(self, chunk: str) -> str:
        """
        Encrypt a raw SSE chunk.

        Args:
            chunk: Raw SSE chunk exactly as server would send it.
                   Can be event, comment, retry directive, or any valid SSE.

        Returns:
            Encrypted SSE event string (event: enc, data: <encrypted>)

        Raises:
            SessionExpiredError: If counter exhausted
        """
        with self._lock:
            if self.counter > SSE_MAX_COUNTER:
                raise SessionExpiredError("SSE session counter exhausted")

            # Encrypt raw chunk as-is (UTF-8 bytes)
            plaintext = chunk.encode("utf-8")

            # Encrypt with counter nonce
            nonce = self._compute_nonce(self.counter)
            ciphertext = self._cipher.encrypt(nonce, plaintext, associated_data=None)

            # Wire format: counter_be32 || ciphertext
            payload = self.counter.to_bytes(SSE_COUNTER_SIZE, "big") + ciphertext
            encoded = b64url_encode(payload)

            # Increment counter for next chunk
            self.counter += 1

        # Format as encrypted SSE event
        return f"event: enc\ndata: {encoded}\n\n"


@dataclass
class SSEDecryptor:
    """
    Client-side SSE decryptor.

    Decrypts SSE chunks and validates counter monotonicity for replay protection.
    Returns the exact raw SSE chunk the server originally sent.
    """

    session: StreamingSession
    expected_counter: int = field(default=1)  # Expect counter starting at 1
    _cipher: ChaCha20Poly1305 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cipher = ChaCha20Poly1305(self.session.session_key)

    def _compute_nonce(self, counter: int) -> bytes:
        """Compute 12-byte nonce from salt and counter."""
        return self.session.session_salt + b"\x00\x00\x00\x00" + counter.to_bytes(4, "little")

    def decrypt(self, sse_data: str) -> str:
        """
        Decrypt an SSE data field to recover the original chunk.

        Args:
            sse_data: base64url-encoded payload from SSE data field

        Returns:
            Original raw SSE chunk exactly as server sent it

        Raises:
            ReplayAttackError: If counter is out of order
            DecryptionError: If decryption fails
        """
        # Decode payload
        try:
            payload = b64url_decode(sse_data)
        except Exception as e:
            raise DecryptionError("Invalid base64url encoding") from e

        if len(payload) < SSE_COUNTER_SIZE + 16:  # Counter + minimum ciphertext (tag only)
            raise DecryptionError("Payload too short")

        # Extract counter and ciphertext
        counter = int.from_bytes(payload[:SSE_COUNTER_SIZE], "big")
        ciphertext = payload[SSE_COUNTER_SIZE:]

        # Validate counter monotonicity
        if counter != self.expected_counter:
            raise ReplayAttackError(self.expected_counter, counter)

        # Decrypt
        nonce = self._compute_nonce(counter)
        try:
            plaintext = self._cipher.decrypt(nonce, ciphertext, associated_data=None)
        except Exception as e:
            raise DecryptionError("SSE decryption failed") from e

        # Increment expected counter
        self.expected_counter += 1

        # Return raw chunk as string
        return plaintext.decode("utf-8")

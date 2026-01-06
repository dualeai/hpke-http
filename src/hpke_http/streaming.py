"""
SSE (Server-Sent Events) streaming encryption.

Provides encrypted SSE format that remains WHATWG-compliant while
hiding event types and data from MITM attackers.

Wire format per event:
    id: {seq}
    event: enc
    data: <base64url(counter_be32 || ciphertext)>

Plaintext structure (encrypted):
    {"t": "event_type", "d": {data}}

Reference: RFC-065 §6
"""

from __future__ import annotations

import json
import re
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any

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
    Server-side SSE event encryptor.

    Encrypts events with monotonic counter for replay protection.
    Thread-safe: uses a lock to protect counter operations.
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

    def encrypt_event(
        self,
        event_type: str,
        data: dict[str, Any],
        event_id: int | None = None,
    ) -> str:
        """
        Encrypt an SSE event.

        Args:
            event_type: Original event type (e.g., "task_progress")
            data: Event data payload
            event_id: Optional SSE event ID for reconnection

        Returns:
            WHATWG-compliant SSE event string

        Raises:
            SessionExpiredError: If counter exhausted
        """
        with self._lock:
            if self.counter > SSE_MAX_COUNTER:
                raise SessionExpiredError("SSE session counter exhausted")

            # Build plaintext: {"t": type, "d": data}
            plaintext = json.dumps({"t": event_type, "d": data}, separators=(",", ":")).encode()

            # Encrypt with counter nonce
            nonce = self._compute_nonce(self.counter)
            ciphertext = self._cipher.encrypt(nonce, plaintext, associated_data=None)

            # Wire format: counter_be32 || ciphertext
            payload = self.counter.to_bytes(SSE_COUNTER_SIZE, "big") + ciphertext
            encoded = b64url_encode(payload)

            # Increment counter for next event
            self.counter += 1

        # Format as SSE (outside lock - no shared state modified)
        lines: list[str] = []
        if event_id is not None:
            lines.append(f"id: {event_id}")
        lines.append("event: enc")
        lines.append(f"data: {encoded}")
        lines.append("")  # Empty line = event boundary

        return "\n".join(lines) + "\n"


@dataclass
class SSEDecryptor:
    """
    Client-side SSE event decryptor.

    Validates counter monotonicity for replay protection.
    """

    session: StreamingSession
    expected_counter: int = field(default=1)  # Expect counter starting at 1
    _cipher: ChaCha20Poly1305 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cipher = ChaCha20Poly1305(self.session.session_key)

    def _compute_nonce(self, counter: int) -> bytes:
        """Compute 12-byte nonce from salt and counter."""
        return self.session.session_salt + b"\x00\x00\x00\x00" + counter.to_bytes(4, "little")

    def decrypt_event(self, sse_data: str) -> tuple[str, dict[str, Any]]:
        """
        Decrypt an SSE event data field.

        Args:
            sse_data: base64url-encoded payload from SSE data field

        Returns:
            Tuple of (event_type, data)

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
            raise DecryptionError("SSE event decryption failed") from e

        # Increment expected counter
        self.expected_counter += 1

        # Parse event structure
        try:
            event = json.loads(plaintext)
            return (event["t"], event["d"])
        except (json.JSONDecodeError, KeyError) as e:
            raise DecryptionError("Invalid SSE event structure") from e


def parse_sse_event(raw_event: str) -> dict[str, str]:
    """
    Parse raw SSE event into components per WHATWG EventSource spec.

    Handles:
    - All line endings: CR (\\r), LF (\\n), CRLF (\\r\\n)
    - Comment lines (starting with :)
    - Field values with single leading space removed
    - Lines without colons (empty value)

    Reference: https://html.spec.whatwg.org/multipage/server-sent-events.html

    Args:
        raw_event: Raw SSE event text

    Returns:
        Dict with 'id', 'event', 'data', 'retry' keys (if present)
    """
    result: dict[str, str] = {}
    # Split on any line ending: CRLF, LF, or CR
    lines = re.split(r"\r\n|\r|\n", raw_event.strip())

    for line in lines:
        # Skip empty lines
        if not line:
            continue
        # Skip comment lines (start with :)
        if line.startswith(":"):
            continue
        # Parse field
        if ":" in line:
            key, _, value = line.partition(":")
            # Remove single leading space if present (per WHATWG spec)
            if value.startswith(" "):
                value = value[1:]
            result[key] = value
        else:
            # Line without colon: field name with empty value
            result[line] = ""

    return result

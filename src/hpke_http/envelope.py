"""
Wire format for HPKE encrypted envelopes.

Envelope format (sent in HTTP body):
┌─────────┬─────────┬─────────┬─────────┬──────┬────────────┐
│Version  │ KEM_ID  │ KDF_ID  │ AEAD_ID │ Mode │ Ciphertext │
│ (1B)    │  (2B)   │  (2B)   │  (2B)   │ (1B) │  (N+16B)   │
└─────────┴─────────┴─────────┴─────────┴──────┴────────────┘

The encapsulated key (enc) is sent in the X-HPKE-Enc HTTP header.

Reference: RFC-065 §4.1
"""

from dataclasses import dataclass

from hpke_http.constants import (
    AEAD_ID,
    CHACHA20_POLY1305_TAG_SIZE,
    ENVELOPE_HEADER_SIZE,
    ENVELOPE_VERSION,
    KDF_ID,
    KEM_ID,
    MODE_PSK,
)
from hpke_http.exceptions import EnvelopeError, UnsupportedSuiteError

__all__ = [
    "EnvelopeHeader",
    "decode_envelope",
    "encode_envelope",
    "parse_header",
]


@dataclass
class EnvelopeHeader:
    """Parsed envelope header."""

    version: int
    kem_id: int
    kdf_id: int
    aead_id: int
    mode: int

    def validate(self) -> None:
        """
        Validate that header uses supported cipher suite.

        Raises:
            EnvelopeError: If version is unsupported
            UnsupportedSuiteError: If cipher suite is not supported
        """
        if self.version != ENVELOPE_VERSION:
            raise EnvelopeError(f"Unsupported envelope version: {self.version}")

        if self.kem_id != KEM_ID or self.kdf_id != KDF_ID or self.aead_id != AEAD_ID:
            raise UnsupportedSuiteError(self.kem_id, self.kdf_id, self.aead_id)

        if self.mode != MODE_PSK:
            raise EnvelopeError(f"Unsupported HPKE mode: {self.mode} (only PSK mode 0x01 supported)")


def encode_header(
    version: int = ENVELOPE_VERSION,
    kem_id: int = KEM_ID,
    kdf_id: int = KDF_ID,
    aead_id: int = AEAD_ID,
    mode: int = MODE_PSK,
) -> bytes:
    """
    Encode envelope header.

    Args:
        version: Envelope format version
        kem_id: KEM algorithm identifier
        kdf_id: KDF algorithm identifier
        aead_id: AEAD algorithm identifier
        mode: HPKE mode

    Returns:
        8-byte header
    """
    return (
        version.to_bytes(1, "big")
        + kem_id.to_bytes(2, "big")
        + kdf_id.to_bytes(2, "big")
        + aead_id.to_bytes(2, "big")
        + mode.to_bytes(1, "big")
    )


def parse_header(data: bytes) -> EnvelopeHeader:
    """
    Parse envelope header from bytes.

    Args:
        data: At least 8 bytes starting with header

    Returns:
        Parsed EnvelopeHeader

    Raises:
        EnvelopeError: If data is too short
    """
    if len(data) < ENVELOPE_HEADER_SIZE:
        raise EnvelopeError(f"Envelope too short: {len(data)} bytes (minimum {ENVELOPE_HEADER_SIZE})")

    return EnvelopeHeader(
        version=data[0],
        kem_id=int.from_bytes(data[1:3], "big"),
        kdf_id=int.from_bytes(data[3:5], "big"),
        aead_id=int.from_bytes(data[5:7], "big"),
        mode=data[7],
    )


def encode_envelope(ciphertext: bytes) -> bytes:
    """
    Encode ciphertext into envelope format.

    The encapsulated key (enc) should be sent separately in X-HPKE-Enc header.

    Args:
        ciphertext: AEAD-encrypted data with authentication tag

    Returns:
        Complete envelope: header || ciphertext
    """
    header = encode_header()
    return header + ciphertext


def decode_envelope(envelope: bytes) -> tuple[EnvelopeHeader, bytes]:
    """
    Decode envelope into header and ciphertext.

    Args:
        envelope: Complete envelope bytes

    Returns:
        Tuple of (header, ciphertext)

    Raises:
        EnvelopeError: If envelope is malformed
        UnsupportedSuiteError: If cipher suite is not supported
    """
    if len(envelope) < ENVELOPE_HEADER_SIZE + CHACHA20_POLY1305_TAG_SIZE:
        raise EnvelopeError(
            f"Envelope too short: {len(envelope)} bytes (minimum {ENVELOPE_HEADER_SIZE + CHACHA20_POLY1305_TAG_SIZE})"
        )

    header = parse_header(envelope)
    header.validate()

    ciphertext = envelope[ENVELOPE_HEADER_SIZE:]
    return (header, ciphertext)


def envelope_overhead() -> int:
    """
    Calculate total overhead added by envelope encoding.

    Returns:
        Overhead in bytes (header + AEAD tag)
    """
    return ENVELOPE_HEADER_SIZE + CHACHA20_POLY1305_TAG_SIZE

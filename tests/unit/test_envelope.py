"""Unit tests for envelope encoding/decoding."""

import pytest

from hpke_http.constants import (
    AEAD_ID,
    CHACHA20_POLY1305_TAG_SIZE,
    ENVELOPE_HEADER_SIZE,
    ENVELOPE_VERSION,
    KDF_ID,
    KEM_ID,
    MODE_PSK,
)
from hpke_http.envelope import (
    EnvelopeHeader,
    decode_envelope,
    encode_envelope,
    envelope_overhead,
    parse_header,
)
from hpke_http.exceptions import EnvelopeError, UnsupportedSuiteError


class TestEnvelopeHeader:
    """Test envelope header parsing and validation."""

    def test_parse_valid_header(self) -> None:
        """Test parsing a valid header."""
        header_bytes = (
            ENVELOPE_VERSION.to_bytes(1, "big")
            + KEM_ID.to_bytes(2, "big")
            + KDF_ID.to_bytes(2, "big")
            + AEAD_ID.to_bytes(2, "big")
            + MODE_PSK.to_bytes(1, "big")
        )

        header = parse_header(header_bytes)

        assert header.version == ENVELOPE_VERSION
        assert header.kem_id == KEM_ID
        assert header.kdf_id == KDF_ID
        assert header.aead_id == AEAD_ID
        assert header.mode == MODE_PSK

    def test_parse_short_data_fails(self) -> None:
        """Test that parsing short data raises EnvelopeError."""
        with pytest.raises(EnvelopeError, match="too short"):
            parse_header(b"short")

    def test_validate_unsupported_version(self) -> None:
        """Test that unsupported version raises EnvelopeError."""
        header = EnvelopeHeader(
            version=0xFF,  # Invalid version
            kem_id=KEM_ID,
            kdf_id=KDF_ID,
            aead_id=AEAD_ID,
            mode=MODE_PSK,
        )

        with pytest.raises(EnvelopeError, match="Unsupported envelope version"):
            header.validate()

    def test_validate_unsupported_suite(self) -> None:
        """Test that unsupported cipher suite raises UnsupportedSuiteError."""
        header = EnvelopeHeader(
            version=ENVELOPE_VERSION,
            kem_id=0x9999,  # Invalid KEM
            kdf_id=KDF_ID,
            aead_id=AEAD_ID,
            mode=MODE_PSK,
        )

        with pytest.raises(UnsupportedSuiteError) as exc:
            header.validate()

        assert exc.value.kem_id == 0x9999


class TestEnvelope:
    """Test full envelope encoding/decoding."""

    def test_encode_decode_roundtrip(self) -> None:
        """Test envelope encode/decode roundtrip."""
        # Simulate ciphertext (tag size ensures minimum length)
        ciphertext = b"encrypted-data" + b"\x00" * CHACHA20_POLY1305_TAG_SIZE

        envelope = encode_envelope(ciphertext)
        header, decoded_ct = decode_envelope(envelope)

        assert header.version == ENVELOPE_VERSION
        assert header.kem_id == KEM_ID
        assert decoded_ct == ciphertext

    def test_decode_too_short_fails(self) -> None:
        """Test that decoding too-short envelope fails."""
        with pytest.raises(EnvelopeError, match="too short"):
            decode_envelope(b"short")

    def test_overhead_calculation(self) -> None:
        """Test envelope overhead calculation."""
        overhead = envelope_overhead()
        assert overhead == ENVELOPE_HEADER_SIZE + CHACHA20_POLY1305_TAG_SIZE

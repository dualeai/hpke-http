"""Unit tests for HPKE core functionality."""

import pytest

from hpke_http.exceptions import DecryptionError, InvalidPSKError
from hpke_http.hpke import (
    open_psk,
    seal_psk,
    setup_recipient_psk,
    setup_sender_psk,
)


class TestHPKESealOpen:
    """Test HPKE seal/open operations."""

    def test_seal_open_roundtrip(
        self,
        platform_keypair: tuple[bytes, bytes],
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Test basic encryption/decryption roundtrip."""
        sk_r, pk_r = platform_keypair
        plaintext = b"Hello, HPKE!"
        info = b"test-context"

        # Encrypt
        enc, ciphertext = seal_psk(
            pk_r=pk_r,
            info=info,
            psk=test_psk,
            psk_id=test_psk_id,
            aad=b"",
            plaintext=plaintext,
        )

        # Decrypt
        decrypted = open_psk(
            enc=enc,
            sk_r=sk_r,
            info=info,
            psk=test_psk,
            psk_id=test_psk_id,
            aad=b"",
            ciphertext=ciphertext,
        )

        assert decrypted == plaintext

    def test_seal_open_with_aad(
        self,
        platform_keypair: tuple[bytes, bytes],
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Test encryption/decryption with additional authenticated data."""
        sk_r, pk_r = platform_keypair
        plaintext = b"Secret message"
        aad = b"authenticated-but-not-encrypted"

        enc, ciphertext = seal_psk(
            pk_r=pk_r,
            info=b"",
            psk=test_psk,
            psk_id=test_psk_id,
            aad=aad,
            plaintext=plaintext,
        )

        decrypted = open_psk(
            enc=enc,
            sk_r=sk_r,
            info=b"",
            psk=test_psk,
            psk_id=test_psk_id,
            aad=aad,
            ciphertext=ciphertext,
        )

        assert decrypted == plaintext

    def test_wrong_aad_fails(
        self,
        platform_keypair: tuple[bytes, bytes],
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Test that wrong AAD causes decryption failure."""
        sk_r, pk_r = platform_keypair

        enc, ciphertext = seal_psk(
            pk_r=pk_r,
            info=b"",
            psk=test_psk,
            psk_id=test_psk_id,
            aad=b"correct-aad",
            plaintext=b"test",
        )

        with pytest.raises(DecryptionError):
            open_psk(
                enc=enc,
                sk_r=sk_r,
                info=b"",
                psk=test_psk,
                psk_id=test_psk_id,
                aad=b"wrong-aad",
                ciphertext=ciphertext,
            )

    def test_wrong_psk_fails(
        self,
        platform_keypair: tuple[bytes, bytes],
        test_psk: bytes,
        test_psk_id: bytes,
        wrong_psk: bytes,
    ) -> None:
        """Test that wrong PSK causes decryption failure."""
        sk_r, pk_r = platform_keypair

        enc, ciphertext = seal_psk(
            pk_r=pk_r,
            info=b"",
            psk=test_psk,
            psk_id=test_psk_id,
            aad=b"",
            plaintext=b"test",
        )

        with pytest.raises(DecryptionError):
            open_psk(
                enc=enc,
                sk_r=sk_r,
                info=b"",
                psk=wrong_psk,
                psk_id=test_psk_id,
                aad=b"",
                ciphertext=ciphertext,
            )

    def test_empty_psk_fails(
        self,
        platform_keypair: tuple[bytes, bytes],
    ) -> None:
        """Test that empty PSK raises InvalidPSKError."""
        _, pk_r = platform_keypair

        with pytest.raises(InvalidPSKError):
            seal_psk(
                pk_r=pk_r,
                info=b"",
                psk=b"",  # Empty PSK
                psk_id=b"id",
                aad=b"",
                plaintext=b"test",
            )

    def test_empty_psk_id_fails(
        self,
        platform_keypair: tuple[bytes, bytes],
    ) -> None:
        """Test that empty PSK ID raises InvalidPSKError."""
        _, pk_r = platform_keypair

        with pytest.raises(InvalidPSKError):
            seal_psk(
                pk_r=pk_r,
                info=b"",
                psk=b"some-psk",
                psk_id=b"",  # Empty PSK ID
                aad=b"",
                plaintext=b"test",
            )


class TestHPKEContext:
    """Test HPKE context operations."""

    def test_multiple_messages(
        self,
        platform_keypair: tuple[bytes, bytes],
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Test encrypting/decrypting multiple messages with context."""
        sk_r, pk_r = platform_keypair

        sender_ctx = setup_sender_psk(pk_r, b"", test_psk, test_psk_id)
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"", test_psk, test_psk_id)

        messages = [b"First message", b"Second message", b"Third message"]

        for msg in messages:
            ct = sender_ctx.seal(b"", msg)
            pt = recipient_ctx.open(b"", ct)
            assert pt == msg

    def test_export_secret(
        self,
        platform_keypair: tuple[bytes, bytes],
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Test that export secret derivation is consistent."""
        sk_r, pk_r = platform_keypair

        sender_ctx = setup_sender_psk(pk_r, b"", test_psk, test_psk_id)
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"", test_psk, test_psk_id)

        # Both contexts should derive the same export secret
        label = b"test-export"
        sender_export = sender_ctx.export(label, 32)
        recipient_export = recipient_ctx.export(label, 32)

        assert sender_export == recipient_export
        assert len(sender_export) == 32

    def test_different_export_labels_differ(
        self,
        platform_keypair: tuple[bytes, bytes],
        test_psk: bytes,
        test_psk_id: bytes,
    ) -> None:
        """Test that different export labels produce different secrets."""
        _sk_r, pk_r = platform_keypair

        ctx = setup_sender_psk(pk_r, b"", test_psk, test_psk_id)

        export1 = ctx.export(b"label-1", 32)
        export2 = ctx.export(b"label-2", 32)

        assert export1 != export2

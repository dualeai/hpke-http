"""
RFC 9180 HPKE (Hybrid Public Key Encryption) with PSK mode.

This module implements the high-level HPKE API for encrypting/decrypting
messages using the PSK (Pre-Shared Key) mode.

Cipher Suite:
- KEM: dispatched via the KEM ABC + registry
  (``primitives/kem_base.py``). Currently registered: DHKEM(X25519,
  HKDF-SHA256) (0x0020) and X-Wing (0x647A, post-quantum hybrid).
- KDF: HKDF-SHA256
- AEAD: ChaCha20-Poly1305
- Mode: PSK

Reference: https://datatracker.ietf.org/doc/rfc9180/
"""

from dataclasses import dataclass, field
from typing import Final

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from hpke_http.constants import (
    AEAD_ID,
    CHACHA20_POLY1305_KEY_SIZE,
    CHACHA20_POLY1305_NONCE_SIZE,
    HKDF_SHA256_N_H,
    KDF_ID,
    MODE_PSK,
    PSK_MIN_SIZE,
    KemId,
    build_suite_id,
)
from hpke_http.exceptions import DecryptionError, InvalidPSKError, SequenceOverflowError
from hpke_http.primitives.aead import compute_nonce
from hpke_http.primitives.kdf import labeled_expand, labeled_extract
from hpke_http.primitives.kem_base import get_kem
from hpke_http.primitives.x25519_kem import X25519KEM

__all__ = [
    "HPKEContext",
    "RecipientContext",
    "SenderContext",
    "open_psk",
    "seal_psk",
    "setup_recipient_psk",
    "setup_sender_psk",
]

# Mode-specific constants
MODE_PSK_VALUE: Final[int] = MODE_PSK


@dataclass
class HPKEContext:
    """
    HPKE encryption context with key and nonce state.

    Maintains the sequence number for nonce computation and provides
    encryption/decryption operations.

    The cipher instance is cached for efficient batch operations, avoiding
    the overhead of creating a new ChaCha20Poly1305 instance per seal/open call.

    TODO: cryptography 47.0+ adds ``ChaCha20Poly1305.encrypt_into`` /
    ``decrypt_into``; switching eliminates per-message allocation in the AEAD
    layer. Project pins cryptography~=48.0 so the dependency is already
    satisfied; tracked separately as a perf opt.
    See: https://cryptography.io/en/latest/hazmat/primitives/aead/
    """

    key: bytes
    """AEAD encryption key (32 bytes)."""

    base_nonce: bytes
    """Base nonce for computing per-message nonces (12 bytes)."""

    exporter_secret: bytes
    """Secret for deriving additional keys via export() (32 bytes)."""

    seq: int = 0
    """Sequence number, incremented after each encryption/decryption."""

    kem_id: KemId = KemId.DHKEM_X25519_HKDF_SHA256
    """KEM identifier this context was set up under. Drives the suite_id used
    by export() and is exposed to consumers for observability (OTel attribute,
    metrics labelling)."""

    _cipher: ChaCha20Poly1305 = field(init=False, repr=False)
    """Cached AEAD cipher instance for efficient batch operations."""

    # Max sequence: (1 << (8 * Nn)) - 1 = 2^96 - 1 for 12-byte nonce
    # Nonce reuse is catastrophic for ChaCha20-Poly1305
    _MAX_SEQ: Final[int] = (1 << (8 * CHACHA20_POLY1305_NONCE_SIZE)) - 1

    def __post_init__(self) -> None:
        """Initialize cached cipher instance."""
        self._cipher = ChaCha20Poly1305(self.key)

    def _increment_seq(self) -> None:
        """Increment sequence number, checking for overflow."""
        if self.seq >= self._MAX_SEQ:
            raise SequenceOverflowError("Sequence counter exhausted, cannot encrypt more messages")
        self.seq += 1

    def _compute_nonce(self) -> bytes:
        """Compute nonce for current sequence number."""
        return compute_nonce(self.base_nonce, self.seq)

    def export(self, exporter_context: bytes, length: int) -> bytes:
        """
        RFC 9180 §5.3: Export(exporter_context, L)

        Derive additional secrets from the HPKE context.
        Useful for deriving session keys for SSE streaming.

        Args:
            exporter_context: Context string for domain separation
            length: Desired output length

        Returns:
            Derived key material
        """
        return labeled_expand(
            prk=self.exporter_secret,
            label=b"sec",
            info=exporter_context,
            length=length,
            suite_id=build_suite_id(self.kem_id, KDF_ID, AEAD_ID),
        )


@dataclass
class SenderContext(HPKEContext):
    """HPKE context for the sender (encryptor)."""

    enc: bytes = b""
    """Encapsulated key to send to recipient."""

    def seal(self, aad: bytes, plaintext: bytes) -> bytes:
        """
        Encrypt and authenticate a message.

        Args:
            aad: Additional authenticated data
            plaintext: Message to encrypt

        Returns:
            Ciphertext with authentication tag
        """
        nonce = self._compute_nonce()
        ct = self._cipher.encrypt(nonce, plaintext, aad if aad else None)
        self._increment_seq()
        return ct


@dataclass
class RecipientContext(HPKEContext):
    """HPKE context for the recipient (decryptor)."""

    def open(self, aad: bytes, ciphertext: bytes) -> bytes:
        """
        Decrypt and verify a message.

        Args:
            aad: Additional authenticated data (must match sender)
            ciphertext: Encrypted message with authentication tag

        Returns:
            Decrypted plaintext

        Raises:
            DecryptionError: If authentication fails
        """
        nonce = self._compute_nonce()
        try:
            pt = self._cipher.decrypt(nonce, ciphertext, aad if aad else None)
        except Exception as e:
            raise DecryptionError("Authentication failed") from e
        self._increment_seq()
        return pt


def _verify_psk_inputs(psk: bytes, psk_id: bytes) -> None:
    """
    RFC 9180 §5.1: VerifyPSKInputs(mode, psk, psk_id)

    For PSK mode, both psk and psk_id must be non-empty.
    PSK must be at least 32 bytes (256 bits) for cryptographic security.
    """
    if not psk or not psk_id:
        raise InvalidPSKError("PSK mode requires both psk and psk_id to be non-empty")
    if len(psk) < PSK_MIN_SIZE:
        raise InvalidPSKError("PSK does not meet minimum security requirements")


def _key_schedule_psk(
    shared_secret: bytes,
    info: bytes,
    psk: bytes,
    psk_id: bytes,
    kem_id: KemId = KemId.DHKEM_X25519_HKDF_SHA256,
) -> tuple[bytes, bytes, bytes]:
    """
    RFC 9180 §5.1: KeySchedule for PSK mode.

    Derives key, base_nonce, and exporter_secret from shared secret and PSK.

    Args:
        shared_secret: Output from KEM Encap/Decap
        info: Application-supplied context
        psk: Pre-shared key
        psk_id: Pre-shared key identifier
        kem_id: KEM identifier; drives the HPKE suite_id used in labels.
            Defaults to X25519 for back-compat.

    Returns:
        Tuple of (key, base_nonce, exporter_secret)
    """
    _verify_psk_inputs(psk, psk_id)

    mode = MODE_PSK_VALUE
    suite_id = build_suite_id(kem_id, KDF_ID, AEAD_ID)

    # psk_id_hash = LabeledExtract("", "psk_id_hash", psk_id)
    psk_id_hash = labeled_extract(b"", b"psk_id_hash", psk_id, suite_id=suite_id)

    # info_hash = LabeledExtract("", "info_hash", info)
    info_hash = labeled_extract(b"", b"info_hash", info, suite_id=suite_id)

    # ks_context = mode || psk_id_hash || info_hash
    ks_context = mode.to_bytes(1, "big") + psk_id_hash + info_hash

    # secret = LabeledExtract(shared_secret, "secret", psk)
    secret = labeled_extract(shared_secret, b"secret", psk, suite_id=suite_id)

    # key = LabeledExpand(secret, "key", ks_context, Nk)
    key = labeled_expand(secret, b"key", ks_context, CHACHA20_POLY1305_KEY_SIZE, suite_id=suite_id)

    # base_nonce = LabeledExpand(secret, "base_nonce", ks_context, Nn)
    base_nonce = labeled_expand(secret, b"base_nonce", ks_context, CHACHA20_POLY1305_NONCE_SIZE, suite_id=suite_id)

    # exporter_secret = LabeledExpand(secret, "exp", ks_context, Nh)
    exporter_secret = labeled_expand(secret, b"exp", ks_context, HKDF_SHA256_N_H, suite_id=suite_id)

    return (key, base_nonce, exporter_secret)


def setup_sender_psk(
    pk_r: bytes,
    info: bytes,
    psk: bytes,
    psk_id: bytes,
    kem_id: KemId = KemId.DHKEM_X25519_HKDF_SHA256,
) -> SenderContext:
    """
    RFC 9180 §5.1.2: SetupPSKS(pkR, info, psk, psk_id)

    Set up a sender context for PSK mode encryption.

    Args:
        pk_r: Recipient's public key
        info: Application context (e.g., tenant_id)
        psk: Pre-shared key (e.g., API key)
        psk_id: Pre-shared key identifier (e.g., tenant_id)
        kem_id: KEM to use. Defaults to X25519 for back-compat.

    Returns:
        SenderContext ready for encryption
    """
    kem = get_kem(kem_id)
    enc, shared_secret = kem.encap(pk_r)
    key, base_nonce, exporter_secret = _key_schedule_psk(shared_secret, info, psk, psk_id, kem_id)

    return SenderContext(
        key=key,
        base_nonce=base_nonce,
        exporter_secret=exporter_secret,
        kem_id=kem_id,
        enc=enc,
    )


def _setup_sender_psk_deterministic(  # pyright: ignore[reportUnusedFunction]
    pk_r: bytes,
    info: bytes,
    psk: bytes,
    psk_id: bytes,
    sk_e: bytes,
    kem_id: KemId = KemId.DHKEM_X25519_HKDF_SHA256,
) -> SenderContext:
    """
    Deterministic SetupPSKS for test vector validation.

    WARNING: Only use for testing. Production code must use setup_sender_psk().
    Only KEMs that expose a deterministic encap path support this. X25519 does;
    X-Wing does not (pyca's ML-KEM API has no deterministic encaps).

    Args:
        pk_r: Recipient's public key
        info: Application context
        psk: Pre-shared key
        psk_id: Pre-shared key identifier
        sk_e: Ephemeral private key bytes (MUST be random in production)
        kem_id: KEM to use. Defaults to X25519.

    Returns:
        SenderContext with deterministic enc
    """
    if kem_id != KemId.DHKEM_X25519_HKDF_SHA256:
        raise NotImplementedError(f"Deterministic encap not supported for kem_id=0x{int(kem_id):04x}")

    enc, shared_secret = X25519KEM._encap_deterministic(pk_r, sk_e)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    key, base_nonce, exporter_secret = _key_schedule_psk(shared_secret, info, psk, psk_id, kem_id)

    return SenderContext(
        key=key,
        base_nonce=base_nonce,
        exporter_secret=exporter_secret,
        kem_id=kem_id,
        enc=enc,
    )


def setup_recipient_psk(
    enc: bytes,
    sk_r: bytes,
    info: bytes,
    psk: bytes,
    psk_id: bytes,
    kem_id: KemId = KemId.DHKEM_X25519_HKDF_SHA256,
) -> RecipientContext:
    """
    RFC 9180 §5.1.2: SetupPSKR(enc, skR, info, psk, psk_id)

    Set up a recipient context for PSK mode decryption.

    Args:
        enc: Encapsulated key from sender
        sk_r: Recipient's private key
        info: Application context (must match sender)
        psk: Pre-shared key (must match sender)
        psk_id: Pre-shared key identifier (must match sender)
        kem_id: KEM to use. Must match the sender. Defaults to X25519.

    Returns:
        RecipientContext ready for decryption
    """
    kem = get_kem(kem_id)
    shared_secret = kem.decap(enc, sk_r)
    key, base_nonce, exporter_secret = _key_schedule_psk(shared_secret, info, psk, psk_id, kem_id)

    return RecipientContext(
        key=key,
        base_nonce=base_nonce,
        exporter_secret=exporter_secret,
        kem_id=kem_id,
    )


def seal_psk(
    pk_r: bytes,
    info: bytes,
    psk: bytes,
    psk_id: bytes,
    aad: bytes,
    plaintext: bytes,
    kem_id: KemId = KemId.DHKEM_X25519_HKDF_SHA256,
) -> tuple[bytes, bytes]:
    """
    Single-shot PSK mode encryption.

    Convenience function that sets up context and encrypts in one call.
    Use setup_sender_psk() if encrypting multiple messages.

    Args:
        pk_r: Recipient's public key
        info: Application context
        psk: Pre-shared key
        psk_id: Pre-shared key identifier
        aad: Additional authenticated data
        plaintext: Message to encrypt
        kem_id: KEM to use. Defaults to X25519.

    Returns:
        Tuple of (enc, ciphertext)
    """
    ctx = setup_sender_psk(pk_r, info, psk, psk_id, kem_id)
    ct = ctx.seal(aad, plaintext)
    return (ctx.enc, ct)


def open_psk(
    enc: bytes,
    sk_r: bytes,
    info: bytes,
    psk: bytes,
    psk_id: bytes,
    aad: bytes,
    ciphertext: bytes,
    kem_id: KemId = KemId.DHKEM_X25519_HKDF_SHA256,
) -> bytes:
    """
    Single-shot PSK mode decryption.

    Convenience function that sets up context and decrypts in one call.
    Use setup_recipient_psk() if decrypting multiple messages.

    Args:
        enc: Encapsulated key from sender
        sk_r: Recipient's private key
        info: Application context (must match sender)
        psk: Pre-shared key (must match sender)
        psk_id: Pre-shared key identifier (must match sender)
        aad: Additional authenticated data (must match sender)
        ciphertext: Encrypted message
        kem_id: KEM to use. Must match the sender.

    Returns:
        Decrypted plaintext

    Raises:
        DecryptionError: If authentication fails
        InvalidPSKError: If PSK doesn't match
    """
    ctx = setup_recipient_psk(enc, sk_r, info, psk, psk_id, kem_id)
    return ctx.open(aad, ciphertext)

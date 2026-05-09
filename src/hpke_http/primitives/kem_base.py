"""
KEM (Key Encapsulation Mechanism) abstraction + registry.

Concrete KEM algorithms (X25519, X-Wing, future) implement the KEM ABC and
register at module-import time via the @register_kem decorator. Higher layers
(hpke.py, core.py, middleware) dispatch by KemId via get_kem(kem_id).

Adding a new KEM:
    1. Add the IANA-assigned identifier to KemId in constants.py.
    2. Create primitives/<name>_kem.py with a @register_kem-decorated class
       subclassing KEM.
    3. Import the module from primitives/__init__.py so registration runs.
    No changes to hpke.py, core.py, or middleware are required.

External packages MAY register their own KEM subclasses; the registry is a
public extension point but callers assume responsibility for IANA-ID
coordination and wire-format stability.

Reference: RFC 9180 §4 (https://datatracker.ietf.org/doc/rfc9180/).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, TypeVar

from hpke_http.constants import KemId
from hpke_http.exceptions import UnsupportedKEMError

_KEM_T = TypeVar("_KEM_T", bound=type["KEM"])

__all__ = [
    "KEM",
    "get_kem",
    "register_kem",
    "registered_kem_ids",
]


class KEM(ABC):
    """RFC 9180 §4 Key Encapsulation Mechanism abstraction.

    Subclasses implement a single KEM algorithm using only classmethods
    (KEM impls are stateless; instances are not used).
    """

    KEM_ID: ClassVar[KemId]
    """IANA HPKE KEM identifier (RFC 9180 §7.1 registry)."""

    PUBLIC_KEY_SIZE: ClassVar[int]
    """Npk: encoded public key size in bytes."""

    PRIVATE_KEY_SIZE: ClassVar[int]
    """Nsk: encoded private key size in bytes."""

    ENC_SIZE: ClassVar[int]
    """Nenc: encapsulated key (ciphertext) size in bytes."""

    SHARED_SECRET_SIZE: ClassVar[int]
    """Nsecret: derived shared-secret size in bytes."""

    DERIVE_KEYPAIR_IKM_MIN_SIZE: ClassVar[int]
    """Minimum input keying material length accepted by derive_keypair."""

    @classmethod
    @abstractmethod
    def generate_keypair(cls) -> tuple[bytes, bytes]:
        """Generate a fresh keypair using the system CSPRNG.

        Returns:
            (sk, pk): private key bytes, public key bytes.
        """

    @classmethod
    @abstractmethod
    def derive_keypair(cls, ikm: bytes) -> tuple[bytes, bytes]:
        """RFC 9180 §4.1 DeriveKeyPair: deterministic keypair from ikm.

        Args:
            ikm: input keying material; must satisfy
                len(ikm) >= DERIVE_KEYPAIR_IKM_MIN_SIZE plus any
                KEM-specific exact-length constraints.

        Raises:
            ValueError: if ikm length violates constraints.
        """

    @classmethod
    @abstractmethod
    def encap(cls, pk_r: bytes) -> tuple[bytes, bytes]:
        """RFC 9180 §4 Encap: encapsulate a fresh shared secret to pk_r.

        Uses the system CSPRNG for ephemeral randomness.

        Returns:
            (enc, shared_secret): encapsulated key, shared secret.

        Raises:
            ValueError: if pk_r size is invalid.
        """

    @classmethod
    @abstractmethod
    def decap(cls, enc: bytes, sk_r: bytes) -> bytes:
        """RFC 9180 §4 Decap: recover the shared secret from enc and sk_r.

        Implementations MAY use implicit rejection on adversarial enc, or MAY
        let underlying primitives raise. RFC 9180 §4.1 DHKEM does not mandate
        either behavior for X25519; FIPS 203 ML-KEM uses implicit rejection
        deterministically. The shipped KEMs:

        - ``X25519KEM``: lets pyca's X25519 raise on small-subgroup inputs.
        - ``XWingKEM``: catches the X25519 leg's exception and substitutes an
          all-zero ``ss_X``, so the combiner runs to completion. Matches the
          implicit-rejection behavior of Go crypto/x25519 and BoringSSL.

        Callers MUST treat decap as best-effort and rely on AEAD authentication
        downstream to detect tampering. Do not depend on a specific exception
        for adversarial enc.

        Raises:
            ValueError: if sizes are invalid; possibly more (KEM-specific) for
                small-subgroup or non-canonical inputs depending on backend.
        """

    @classmethod
    @abstractmethod
    def derive_public_key(cls, sk: bytes) -> bytes:
        """Derive the public key from a private key.

        Used by middleware to populate the discovery doc at construction time.
        May be expensive for some KEMs (X-Wing re-runs ML-KEM KeyGen); cache
        externally if hot.

        Raises:
            ValueError: if sk size is invalid.
        """

    @classmethod
    def validate_public_key(cls, pk: bytes) -> None:
        """Reject malformed public keys before allocating crypto state.

        Default: size check only. Subclasses MAY override for stronger
        invariants. The shipped KEMs (X25519, X-Wing) keep the default and
        delegate full validation to the underlying primitives' constructors
        (``X25519PublicKey.from_public_bytes``,
        ``MLKEM768PublicKey.from_public_bytes``), which run during encap.

        Raises:
            ValueError: if pk size != PUBLIC_KEY_SIZE.
        """
        if len(pk) != cls.PUBLIC_KEY_SIZE:
            raise ValueError(f"{cls.__name__}: pk size {len(pk)} != {cls.PUBLIC_KEY_SIZE}")

    @classmethod
    def validate_private_key(cls, sk: bytes) -> None:
        """Reject malformed private keys before allocating crypto state.

        Default: size check only. Subclasses override if a KEM has further
        invariants (e.g., curve-point validation for keys not stored as a
        wire-canonical seed).

        Raises:
            ValueError: if sk size != PRIVATE_KEY_SIZE.
        """
        if len(sk) != cls.PRIVATE_KEY_SIZE:
            raise ValueError(f"{cls.__name__}: sk size {len(sk)} != {cls.PRIVATE_KEY_SIZE}")

    @classmethod
    def validate_enc(cls, enc: bytes) -> None:
        """Reject malformed enc before allocating crypto state.

        Default: size check only. Critical defense against malicious oversize
        inputs driving allocations in subsequent crypto code.

        Raises:
            ValueError: if enc size != ENC_SIZE.
        """
        if len(enc) != cls.ENC_SIZE:
            raise ValueError(f"{cls.__name__}: enc size {len(enc)} != {cls.ENC_SIZE}")

    @classmethod
    def kem_suite_id(cls) -> bytes:
        """RFC 9180 KEM suite_id for KEM-internal HKDF labels.

        suite_id = b"KEM" || I2OSP(kem_id, 2)

        Different from the HPKE-level suite_id (which also includes kdf_id
        and aead_id). Use this inside KEM implementations only.
        """
        return b"KEM" + int(cls.KEM_ID).to_bytes(2, "big")


# Registry --------------------------------------------------------------------

_KEM_REGISTRY: dict[KemId, type[KEM]] = {}


def register_kem(cls: _KEM_T) -> _KEM_T:
    """Decorator: register a KEM subclass at module import time.

    Raises:
        RuntimeError: if another KEM is already registered for cls.KEM_ID.
            Different impls of the same algorithm must not coexist; consumers
            would have no way to tell which one will be dispatched.
    """
    if cls.KEM_ID in _KEM_REGISTRY:
        existing = _KEM_REGISTRY[cls.KEM_ID]
        raise RuntimeError(
            f"Duplicate KEM registration for kem_id=0x{int(cls.KEM_ID):04x}: "
            f"{existing.__name__} already registered, cannot register {cls.__name__}"
        )
    _KEM_REGISTRY[cls.KEM_ID] = cls
    return cls


def get_kem(kem_id: KemId) -> type[KEM]:
    """Look up a registered KEM implementation by IANA identifier.

    Raises:
        UnsupportedKEMError: if no KEM is registered for the given id.
    """
    if kem_id not in _KEM_REGISTRY:
        raise UnsupportedKEMError(f"KEM not registered: kem_id=0x{int(kem_id):04x}")
    return _KEM_REGISTRY[kem_id]


def registered_kem_ids() -> tuple[KemId, ...]:
    """Return the tuple of currently-registered KemId values.

    Order is registration order (insertion order of the underlying dict).
    Suite-selection logic must not rely on iteration order; use explicit
    preference flags instead.
    """
    return tuple(_KEM_REGISTRY)

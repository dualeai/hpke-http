"""
KEM registry tests.

Verifies the abstraction's enforcement points: ABC instantiation blocked,
incomplete subclasses rejected by ABC, duplicate registrations refused,
unknown KemId raises UnsupportedKEMError.
"""

from __future__ import annotations

from enum import IntEnum
from typing import ClassVar

import pytest

from hpke_http.constants import KemId
from hpke_http.exceptions import UnsupportedKEMError
from hpke_http.primitives import KEM, X25519KEM, get_kem, register_kem, registered_kem_ids


class TestRegistryLookup:
    def test_get_kem_x25519(self) -> None:
        assert get_kem(KemId.DHKEM_X25519_HKDF_SHA256) is X25519KEM

    def test_registered_ids_includes_x25519(self) -> None:
        assert KemId.DHKEM_X25519_HKDF_SHA256 in registered_kem_ids()


class TestUnknownKem:
    def test_get_kem_unknown_raises(self) -> None:
        # Use a KemId-shaped value that is NOT in the registry. We can't
        # instantiate KemId(0xFFFE) directly (IntEnum rejects unknown values),
        # so build a sibling enum that aliases the int and pass that.
        class _BogusKemId(IntEnum):
            UNKNOWN = 0xFFFE

        with pytest.raises(UnsupportedKEMError):
            get_kem(_BogusKemId.UNKNOWN)  # type: ignore[arg-type]


class TestABCEnforcement:
    def test_cannot_instantiate_kem_abc(self) -> None:
        """KEM is abstract — direct instantiation must fail."""
        with pytest.raises(TypeError):
            KEM()  # type: ignore[abstract]

    def test_subclass_missing_abstract_method_uninstantiable(self) -> None:
        """A subclass that omits a required classmethod is still abstract.

        We only check instantiability (Python's ABC contract). Registration
        of the class itself is independent.
        """

        class IncompleteKEM(KEM):
            # Missing: derive_keypair, encap, decap, derive_public_key,
            # generate_keypair. The class is still abstract.
            KEM_ID: ClassVar[KemId] = KemId.DHKEM_X25519_HKDF_SHA256
            PUBLIC_KEY_SIZE: ClassVar[int] = 32
            PRIVATE_KEY_SIZE: ClassVar[int] = 32
            ENC_SIZE: ClassVar[int] = 32
            SHARED_SECRET_SIZE: ClassVar[int] = 32
            DERIVE_KEYPAIR_IKM_MIN_SIZE: ClassVar[int] = 32

        with pytest.raises(TypeError):
            IncompleteKEM()  # type: ignore[abstract]


class TestRegisterKemDuplicate:
    def test_duplicate_kem_id_rejected(self) -> None:
        """Two classes claiming the same KEM_ID cannot coexist in the registry."""

        class Doppelganger(KEM):
            # Same KEM_ID as X25519KEM — registry must reject.
            KEM_ID: ClassVar[KemId] = KemId.DHKEM_X25519_HKDF_SHA256
            PUBLIC_KEY_SIZE: ClassVar[int] = 32
            PRIVATE_KEY_SIZE: ClassVar[int] = 32
            ENC_SIZE: ClassVar[int] = 32
            SHARED_SECRET_SIZE: ClassVar[int] = 32
            DERIVE_KEYPAIR_IKM_MIN_SIZE: ClassVar[int] = 32

            @classmethod
            def generate_keypair(cls) -> tuple[bytes, bytes]:
                return (b"\x00" * 32, b"\x00" * 32)

            @classmethod
            def derive_keypair(cls, ikm: bytes) -> tuple[bytes, bytes]:
                return (ikm[:32], ikm[:32])

            @classmethod
            def encap(cls, pk_r: bytes) -> tuple[bytes, bytes]:  # noqa: ARG003
                return (b"\x00" * 32, b"\x00" * 32)

            @classmethod
            def decap(cls, enc: bytes, sk_r: bytes) -> bytes:  # noqa: ARG003
                return b"\x00" * 32

            @classmethod
            def derive_public_key(cls, sk: bytes) -> bytes:
                return sk[:32]

        with pytest.raises(RuntimeError, match="Duplicate"):
            register_kem(Doppelganger)

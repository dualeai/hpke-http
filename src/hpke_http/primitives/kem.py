"""
Backward-compatibility shim.

The KEM logic for DHKEM(X25519, HKDF-SHA256) was moved to
``primitives/x25519_kem.py`` behind the ``KEM`` ABC defined in
``primitives/kem_base.py``. New code should call
``get_kem(kem_id).encap(...)`` from ``hpke_http.primitives``.

This module re-exports the X25519KEM classmethods at module scope so existing
``from hpke_http.primitives.kem import encap, decap, ...`` imports keep
working.
"""

from __future__ import annotations

from hpke_http.primitives.x25519_kem import X25519KEM

__all__ = [
    "decap",
    "derive_keypair",
    "encap",
    "generate_keypair",
]


def generate_keypair() -> tuple[bytes, bytes]:
    """Deprecated: use ``X25519KEM.generate_keypair()``."""
    return X25519KEM.generate_keypair()


def derive_keypair(ikm: bytes) -> tuple[bytes, bytes]:
    """Deprecated: use ``X25519KEM.derive_keypair(ikm)``."""
    return X25519KEM.derive_keypair(ikm)


def encap(pk_r: bytes) -> tuple[bytes, bytes]:
    """Deprecated: use ``X25519KEM.encap(pk_r)``."""
    return X25519KEM.encap(pk_r)


def decap(enc: bytes, sk_r: bytes) -> bytes:
    """Deprecated: use ``X25519KEM.decap(enc, sk_r)``."""
    return X25519KEM.decap(enc, sk_r)

"""
Low-level cryptographic primitives for RFC 9180 HPKE.

These are internal implementation details. Use the high-level API in hpke.py instead.

The KEM layer is built on a small abstraction: subclasses of ``KEM`` implement a
single algorithm and self-register at module-import time via ``@register_kem``.
Higher layers (``hpke.py``, ``core.py``, middleware) dispatch by ``KemId`` via
``get_kem(kem_id)``. Importing this package triggers registration of every
shipped KEM module.
"""

from hpke_http.primitives.kdf import labeled_expand, labeled_extract
from hpke_http.primitives.kem import decap, encap, generate_keypair
from hpke_http.primitives.kem_base import KEM, get_kem, register_kem, registered_kem_ids

# Importing each KEM module triggers self-registration at import time. The
# re-exports below also keep classes available on the package surface for
# tests and consumers that need them directly.
from hpke_http.primitives.x25519_kem import X25519KEM
from hpke_http.primitives.xwing_kem import XWingKEM

__all__ = [
    "KEM",
    "X25519KEM",
    "XWingKEM",
    "decap",
    "encap",
    "generate_keypair",
    "get_kem",
    "labeled_expand",
    "labeled_extract",
    "register_kem",
    "registered_kem_ids",
]

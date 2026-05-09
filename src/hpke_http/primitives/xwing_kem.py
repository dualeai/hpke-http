"""
X-Wing hybrid KEM (X25519 + ML-KEM-768) behind the KEM ABC.

Implements draft-connolly-cfrg-xwing-kem-10. KEM identifier 0x647A is an
IANA HPKE registry early allocation referencing draft-06; the wire format
matches the draft revision pinned in
``hpke_http.constants.XWING_DRAFT_REVISION``.

Wire layout (raw bytes):
    sk     = 32-byte seed
    pk     = pk_M (1184) || pk_X (32)              == 1216 bytes
    enc    = ct_M (1088) || ct_X (32)              == 1120 bytes
    ss     = 32 bytes (SHA3-256 of the combiner)

Encap (§5.4, randomized by the system CSPRNG):
    1. Split pk_r into pk_M (ML-KEM-768) || pk_X (X25519).
    2. (ss_M, ct_M) = MLKEM768.encapsulate(pk_M).
    3. ek_X = random(32); ct_X = X25519(ek_X, BASE); ss_X = X25519(ek_X, pk_X).
    4. ss = SHA3-256(ss_M || ss_X || ct_X || pk_X || XWingLabel).
    5. enc = ct_M || ct_X.

Decap (§5.5, deterministic given sk_seed and ct):
    1. (sk_M, sk_X, pk_X) = expandDecapsulationKey(sk_seed)    (cached;
       pk_X is precomputed inside the cache to avoid a per-decap X25519
       base-point multiplication).
    2. ct_M = ct[0:1088]; ct_X = ct[1088:1120].
    3. ss_M = MLKEM768.decapsulate(ct_M, sk_M).
    4. ss_X = X25519(sk_X, ct_X), or all-zero fallback if pyca raises on a
       small-subgroup ct_X (RFC 7748 §6.1) — implicit rejection that matches
       Go crypto/x25519 + BoringSSL behavior.
    5. ss = SHA3-256(ss_M || ss_X || ct_X || pk_X || XWingLabel).

Performance:
    ``expandDecapsulationKey`` runs SHAKE256 + FIPS 203 KeyGen + X25519 keypair
    construction — ~1 ms total. ``functools.lru_cache(maxsize=8)`` on
    ``_expand_decapsulation_key`` reduces this to ~0 on warm path; server
    typically uses one seed across all requests. ``_load_mlkem_public``
    (maxsize=16) caches ``MLKEM768PublicKey.from_public_bytes(pk_M)`` on the
    client encap path. Both caches are bounded; adversarial inputs cannot
    grow them past ``maxsize``. Bytes are hashable; cache works without
    wrapping.

Backend availability:
    pyca/cryptography ML-KEM ships on OpenSSL 3.5+, which is bundled in every
    ``cryptography>=48.0`` wheel. The pin in ``pyproject.toml`` makes the
    primitive a hard dependency; ML-KEM is always available in supported
    deployments. Source builds against a system OpenSSL <3.5 raise
    ``cryptography.exceptions.UnsupportedAlgorithm`` on first use.

Reference:
    https://datatracker.ietf.org/doc/draft-connolly-cfrg-xwing-kem/
"""

from __future__ import annotations

import functools
import hashlib
import secrets
from typing import Final

from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.asymmetric.mlkem import (
    MLKEM768PrivateKey,
    MLKEM768PublicKey,
)

from hpke_http.constants import (
    MLKEM768_CIPHERTEXT_SIZE,
    MLKEM768_PUBLIC_KEY_SIZE,
    X25519_SHARED_SECRET_SIZE,
    XWING_ENC_SIZE,
    XWING_LABEL,
    XWING_PRIVATE_KEY_SEED_SIZE,
    XWING_PUBLIC_KEY_SIZE,
    XWING_SHARED_SECRET_SIZE,
    KemId,
)
from hpke_http.primitives.kem_base import KEM, register_kem

__all__ = ["XWingKEM"]


# All-zero rejection for X25519 small-subgroup inputs (RFC 7748 §6.1).
# pyca raises on these; we substitute a fixed all-zero ss_X so the combiner
# still produces a deterministic ss that won't match what Encap produces,
# matching implementations that do not check (Go crypto/x25519, BoringSSL).
# AEAD authentication downstream rejects either way; the only observable
# difference is exception vs derived ss.
_X25519_REJECT_SS_X: Final = b"\x00" * X25519_SHARED_SECRET_SIZE


# ---------------------------------------------------------------------------
# Caches + spec helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _expand_decapsulation_key(
    sk_seed: bytes,
) -> tuple[MLKEM768PrivateKey, x25519.X25519PrivateKey, bytes]:
    """draft-10 §5.2 ``expandDecapsulationKey``, cached per 32-byte seed.

    Returns the full triple needed by ``derive_keypair`` and ``decap``:

    - ``mlkem_priv``: ML-KEM private key (also exposes pk_M via
      ``.public_key().public_bytes_raw()``).
    - ``x25519_priv``: X25519 private key.
    - ``pk_X`` (32 bytes): X25519 public key, precomputed. Decap blends it
      into the combiner; deriving it on every call would re-run X25519 base-
      point scalar multiplication (~30 µs) for no algorithmic reason since
      it is fully determined by the seed.

    One SHAKE256 + one FIPS 203 KeyGen + one X25519 keypair load + one
    base-point mul per cache miss. Subsequent calls hit the cache. Server
    side uses one seed → ~100% hit rate.

    Internal helper. Callers must validate ``len(sk_seed) == 32`` before
    calling; this function trusts the input to keep the cache key well-formed.
    """
    expanded = hashlib.shake_256(sk_seed).digest(96)
    d, z, sk_x_bytes = expanded[0:32], expanded[32:64], expanded[64:96]
    mlkem_priv = MLKEM768PrivateKey.from_seed_bytes(d + z)
    x25519_priv = x25519.X25519PrivateKey.from_private_bytes(sk_x_bytes)
    pk_x = x25519_priv.public_key().public_bytes_raw()
    return (mlkem_priv, x25519_priv, pk_x)


@functools.lru_cache(maxsize=16)
def _load_mlkem_public(pk_M: bytes) -> MLKEM768PublicKey:
    """Cache loaded ML-KEM public key per pk_M bytes.

    Client side hits the same server pk repeatedly → ~100% hit rate. Without
    the cache, every encap rebuilds the public-key handle from raw bytes
    (~100 µs). With the cache, that overhead is amortized away.
    """
    return MLKEM768PublicKey.from_public_bytes(pk_M)


def _combine(ss_M: bytes, ss_X: bytes, ct_X: bytes, pk_X: bytes) -> bytes:
    """draft-10 §5.3 combiner: SHA3-256(ss_M || ss_X || ct_X || pk_X || XWingLabel)."""
    h = hashlib.sha3_256()
    h.update(ss_M)
    h.update(ss_X)
    h.update(ct_X)
    h.update(pk_X)
    h.update(XWING_LABEL)
    return h.digest()


# ---------------------------------------------------------------------------
# KEM class
# ---------------------------------------------------------------------------


@register_kem
class XWingKEM(KEM):
    """X-Wing hybrid KEM (X25519 + ML-KEM-768) — draft-10.

    Stateless. All operations are classmethods. Thread-safe.

    The decapsulation key on the wire and in storage is the 32-byte seed;
    expanded ML-KEM and X25519 secret key material is recomputed on demand
    (cached behind the module-level ``_expand_decapsulation_key``).
    """

    KEM_ID = KemId.XWING
    PUBLIC_KEY_SIZE = XWING_PUBLIC_KEY_SIZE
    PRIVATE_KEY_SIZE = XWING_PRIVATE_KEY_SEED_SIZE  # 32-byte seed only
    ENC_SIZE = XWING_ENC_SIZE
    SHARED_SECRET_SIZE = XWING_SHARED_SECRET_SIZE
    DERIVE_KEYPAIR_IKM_MIN_SIZE = XWING_PRIVATE_KEY_SEED_SIZE  # X-Wing requires exactly 32

    @classmethod
    def generate_keypair(cls) -> tuple[bytes, bytes]:
        """Generate a fresh X-Wing keypair using the system CSPRNG."""
        seed = secrets.token_bytes(XWING_PRIVATE_KEY_SEED_SIZE)
        return cls.derive_keypair(seed)

    @classmethod
    def derive_keypair(cls, ikm: bytes) -> tuple[bytes, bytes]:
        """draft-10 §5.2.1 GenerateKeyPairDerand: deterministic keypair from
        a 32-byte seed.

        X-Wing requires ``len(ikm) == 32`` exactly (differs from RFC 9180 §4.1
        DHKEM, which says ``≥32``). The seed is the canonical wire form of
        the private key.
        """
        if len(ikm) != XWING_PRIVATE_KEY_SEED_SIZE:
            raise ValueError(
                f"XWingKEM: ikm length {len(ikm)} != {XWING_PRIVATE_KEY_SEED_SIZE} (X-Wing requires exact-length seed)"
            )

        mlkem_priv, _x25519_priv, pk_X = _expand_decapsulation_key(ikm)
        pk_M = mlkem_priv.public_key().public_bytes_raw()
        return (ikm, pk_M + pk_X)

    @classmethod
    def encap(cls, pk_r: bytes) -> tuple[bytes, bytes]:
        """draft-10 §5.4 Encapsulate. Randomized by system CSPRNG."""
        cls.validate_public_key(pk_r)

        pk_M = pk_r[:MLKEM768_PUBLIC_KEY_SIZE]
        pk_X = pk_r[MLKEM768_PUBLIC_KEY_SIZE:]

        ss_M, ct_M = _load_mlkem_public(pk_M).encapsulate()

        eph_priv = x25519.X25519PrivateKey.generate()
        ct_X = eph_priv.public_key().public_bytes_raw()
        ss_X = eph_priv.exchange(x25519.X25519PublicKey.from_public_bytes(pk_X))

        return (ct_M + ct_X, _combine(ss_M, ss_X, ct_X, pk_X))

    @classmethod
    def decap(cls, enc: bytes, sk_r: bytes) -> bytes:
        """draft-10 §5.5 Decapsulate.

        Implicit-rejection semantics: a malformed ``enc`` does not raise. The
        ML-KEM-768 leg returns a FIPS 203-derived rejection secret on bad
        ``ct_M``. The X25519 leg returns an all-zero ``ss_X`` if pyca raises
        on a small-subgroup ``ct_X`` (RFC 7748 §6.1) — matching impls that
        don't check (Go crypto/x25519, BoringSSL). The combiner blends both;
        downstream AEAD authentication catches any mismatch.
        """
        cls.validate_enc(enc)
        cls.validate_private_key(sk_r)

        ct_M = enc[:MLKEM768_CIPHERTEXT_SIZE]
        ct_X = enc[MLKEM768_CIPHERTEXT_SIZE:]

        # Single cache lookup recovers both legs' private handles + pk_X.
        mlkem_priv, x25519_priv, pk_X = _expand_decapsulation_key(sk_r)
        ss_M = mlkem_priv.decapsulate(ct_M)

        try:
            ss_X = x25519_priv.exchange(x25519.X25519PublicKey.from_public_bytes(ct_X))
        except ValueError:
            # Small-subgroup ct_X: implicit-reject with all-zero ss_X.
            ss_X = _X25519_REJECT_SS_X

        return _combine(ss_M, ss_X, ct_X, pk_X)

    @classmethod
    def derive_public_key(cls, sk: bytes) -> bytes:
        """Derive 1216-byte X-Wing public key from a 32-byte seed."""
        _sk, pk = cls.derive_keypair(sk)
        return pk

    # ``validate_public_key`` inherits the ABC default (size check). ML-KEM-
    # side lattice validation runs implicitly inside
    # ``MLKEM768PublicKey.from_public_bytes`` during the cached load on
    # ``encap``; rejecting malformed coefficients is delegated there.

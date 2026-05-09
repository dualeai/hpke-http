"""
XWingKEM internal cache tests.

Verifies the perf-critical caches in ``primitives/xwing_kem.py``:

- ``_expand_decapsulation_key`` (maxsize=8): drops decap from ~1 ms to ~150 µs
  by caching the (mlkem_priv, x25519_priv) handle pair per 32-byte sk_seed.
  Implements draft-10 §5.2 ``expandDecapsulationKey`` cached in one place.
- ``_load_mlkem_public`` (maxsize=16): drops encap by caching the loaded
  ML-KEM public-key handle per pk_m bytes.

Cache misses are expensive (FIPS 203 KeyGen / public-key construction); the
production assumption is ~100% hit rate (server uses one seed, clients hit
one server pk repeatedly). Tests pin that property.
"""

from __future__ import annotations

import secrets

import pytest

from hpke_http.primitives.xwing_kem import (
    _expand_decapsulation_key,  # pyright: ignore[reportPrivateUsage]
    _load_mlkem_public,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture(autouse=True)
def reset_caches() -> None:
    """Each test starts with empty caches so hit/miss counts are deterministic."""
    _expand_decapsulation_key.cache_clear()
    _load_mlkem_public.cache_clear()


class TestPrivateKeyCache:
    def test_repeat_calls_hit_cache(self) -> None:
        seed = secrets.token_bytes(32)
        for _ in range(50):
            _expand_decapsulation_key(seed)
        info = _expand_decapsulation_key.cache_info()
        assert info.misses == 1, "first call must miss"
        assert info.hits == 49, "remaining calls must hit"

    def test_distinct_seeds_evict_lru(self) -> None:
        """Cache maxsize is 8; 20 distinct seeds force evictions."""
        for _ in range(20):
            _expand_decapsulation_key(secrets.token_bytes(32))
        info = _expand_decapsulation_key.cache_info()
        assert info.currsize == 8, f"cache currsize must equal maxsize=8, got {info.currsize}"
        assert info.misses == 20, "all 20 distinct seeds miss on first call"

    def test_cache_returns_same_object_for_same_input(self) -> None:
        """Identity, not just equality — cached handle pair is reused."""
        seed = secrets.token_bytes(32)
        a = _expand_decapsulation_key(seed)
        b = _expand_decapsulation_key(seed)
        assert a is b

    def test_cache_returns_handles_and_pk_x(self) -> None:
        """One cache entry covers ML-KEM private, X25519 private, and pk_X."""
        seed = secrets.token_bytes(32)
        mlkem_priv, x25519_priv, pk_x = _expand_decapsulation_key(seed)
        # All three must be usable + correctly sized.
        assert len(mlkem_priv.public_key().public_bytes_raw()) == 1184
        assert len(x25519_priv.public_key().public_bytes_raw()) == 32
        # pk_X must equal x25519 base-point mul of the derived secret.
        assert pk_x == x25519_priv.public_key().public_bytes_raw()
        assert len(pk_x) == 32

    def test_cache_isolation_across_seeds(self) -> None:
        """Different seeds → different handle pairs."""
        seed_a = secrets.token_bytes(32)
        seed_b = secrets.token_bytes(32)
        pair_a = _expand_decapsulation_key(seed_a)
        pair_b = _expand_decapsulation_key(seed_b)
        assert pair_a is not pair_b


class TestPublicKeyCache:
    def test_repeat_calls_hit_cache(self) -> None:
        # Generate a real ML-KEM-768 pk via XWingKEM keypair to feed the cache.
        from hpke_http.primitives import XWingKEM

        _sk, pk = XWingKEM.generate_keypair()
        pk_m = pk[:1184]

        # Reset to discount the warm-up call XWingKEM.generate_keypair did.
        _load_mlkem_public.cache_clear()

        for _ in range(50):
            _load_mlkem_public(pk_m)
        info = _load_mlkem_public.cache_info()
        assert info.misses == 1
        assert info.hits == 49

    def test_lru_eviction_at_maxsize_16(self) -> None:
        from hpke_http.primitives import XWingKEM

        # Generate 20 distinct pk_m values
        pk_ms: list[bytes] = []
        for _ in range(20):
            _, pk = XWingKEM.generate_keypair()
            pk_ms.append(pk[:1184])

        _load_mlkem_public.cache_clear()
        for pk_m in pk_ms:
            _load_mlkem_public(pk_m)
        assert _load_mlkem_public.cache_info().currsize == 16


class TestProductionHitRateAssumption:
    """Acceptance criterion (plan §performance): ≥99/100 sequential decaps hit."""

    def test_sequential_decaps_hit_at_least_99_pct(self) -> None:
        from hpke_http.primitives import XWingKEM

        sk, pk = XWingKEM.generate_keypair()

        _expand_decapsulation_key.cache_clear()
        for _ in range(100):
            enc, _ss = XWingKEM.encap(pk)
            XWingKEM.decap(enc, sk)

        info = _expand_decapsulation_key.cache_info()
        # 100 decaps → 99 hits, 1 miss (warm-up).
        assert info.hits >= 99, f"cache hit rate too low: {info}"

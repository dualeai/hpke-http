"""
Generic ABC contract suite — runs against every registered KEM.

Adding a new KEM (drop-in `primitives/<name>_kem.py` + `@register_kem`) is
covered by these tests automatically. They catch abstraction violations
(wrong sizes, broken roundtrip, missing classmethod, identifier drift) at
the layer that matters most.
"""

from __future__ import annotations

from threading import Thread

import pytest

from hpke_http.constants import KemId
from hpke_http.primitives import KEM, get_kem, registered_kem_ids


@pytest.fixture(params=registered_kem_ids(), ids=lambda k: k.name)
def kem(request: pytest.FixtureRequest) -> type[KEM]:
    """Yields each currently-registered KEM class."""
    kem_id: KemId = request.param
    return get_kem(kem_id)


class TestKEMContract:
    def test_kem_id_classvar(self, kem: type[KEM]) -> None:
        assert isinstance(kem.KEM_ID, KemId)
        assert get_kem(kem.KEM_ID) is kem

    def test_size_classvars_positive(self, kem: type[KEM]) -> None:
        for attr in ("PUBLIC_KEY_SIZE", "PRIVATE_KEY_SIZE", "ENC_SIZE", "SHARED_SECRET_SIZE"):
            assert getattr(kem, attr) > 0, f"{kem.__name__}.{attr} must be positive"

    def test_kem_suite_id_format(self, kem: type[KEM]) -> None:
        sid = kem.kem_suite_id()
        assert len(sid) == 5
        assert sid[:3] == b"KEM"
        assert int.from_bytes(sid[3:], "big") == int(kem.KEM_ID)

    def test_keypair_sizes_match_classvars(self, kem: type[KEM]) -> None:
        sk, pk = kem.generate_keypair()
        assert len(sk) == kem.PRIVATE_KEY_SIZE
        assert len(pk) == kem.PUBLIC_KEY_SIZE

    def test_encap_sizes_match_classvars(self, kem: type[KEM]) -> None:
        _sk, pk = kem.generate_keypair()
        enc, ss = kem.encap(pk)
        assert len(enc) == kem.ENC_SIZE
        assert len(ss) == kem.SHARED_SECRET_SIZE

    def test_encap_decap_roundtrip(self, kem: type[KEM]) -> None:
        sk, pk = kem.generate_keypair()
        enc, ss_a = kem.encap(pk)
        assert kem.decap(enc, sk) == ss_a

    def test_many_roundtrips_no_state_pollution(self, kem: type[KEM]) -> None:
        """Stateless classmethods: 50 fresh roundtrips all consistent."""
        for _ in range(50):
            sk, pk = kem.generate_keypair()
            enc, ss = kem.encap(pk)
            assert kem.decap(enc, sk) == ss

    def test_validate_public_key_wrong_size(self, kem: type[KEM]) -> None:
        with pytest.raises(ValueError):
            kem.validate_public_key(b"\x00" * (kem.PUBLIC_KEY_SIZE - 1))
        with pytest.raises(ValueError):
            kem.validate_public_key(b"\x00" * (kem.PUBLIC_KEY_SIZE + 1))
        with pytest.raises(ValueError):
            kem.validate_public_key(b"")

    def test_validate_enc_wrong_size(self, kem: type[KEM]) -> None:
        with pytest.raises(ValueError):
            kem.validate_enc(b"\x00" * (kem.ENC_SIZE + 1))

    def test_oversize_pk_rejected_cheaply(self, kem: type[KEM]) -> None:
        """10 MB input must hit size-check fast path (no crypto allocation)."""
        with pytest.raises(ValueError):
            kem.validate_public_key(b"\x00" * 10_000_000)

    def test_validate_pk_correct_size_no_raise(self, kem: type[KEM]) -> None:
        _sk, pk = kem.generate_keypair()
        kem.validate_public_key(pk)

    def test_derive_keypair_deterministic(self, kem: type[KEM]) -> None:
        ikm = b"\x42" * max(kem.DERIVE_KEYPAIR_IKM_MIN_SIZE, 32)
        sk1, pk1 = kem.derive_keypair(ikm)
        sk2, pk2 = kem.derive_keypair(ikm)
        assert (sk1, pk1) == (sk2, pk2)

    def test_derive_keypair_too_short(self, kem: type[KEM]) -> None:
        if kem.DERIVE_KEYPAIR_IKM_MIN_SIZE > 0:
            with pytest.raises(ValueError):
                kem.derive_keypair(b"\x00" * (kem.DERIVE_KEYPAIR_IKM_MIN_SIZE - 1))

    def test_derive_public_key_matches_generation(self, kem: type[KEM]) -> None:
        sk, pk = kem.generate_keypair()
        assert kem.derive_public_key(sk) == pk

    def test_decap_with_wrong_sk_yields_different_ss(self, kem: type[KEM]) -> None:
        _sk_a, pk_a = kem.generate_keypair()
        sk_b, _ = kem.generate_keypair()
        enc, ss_correct = kem.encap(pk_a)
        # Both decaps must run (ML-KEM implicit rejection); only matching sk recovers ss.
        assert kem.decap(enc, sk_b) != ss_correct

    def test_concurrent_decap_safe(self, kem: type[KEM]) -> None:
        """Stateless classmethods must be safe under concurrent access."""
        sk, pk = kem.generate_keypair()
        enc, ss = kem.encap(pk)
        results: list[bytes] = []

        def worker() -> None:
            results.append(kem.decap(enc, sk))

        threads = [Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r == ss for r in results)

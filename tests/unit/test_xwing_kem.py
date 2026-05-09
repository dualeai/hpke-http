"""
XWingKEM unit tests — normal, edge, out-of-bound, and weird cases.

Plays a complementary role to the spec KAT in tests/vectors/test_xwing_vectors.py:
the KAT verifies wire-format compatibility against draft-10; this module
verifies behavior of the Python API itself (validation, error paths, cache
correctness, statelessness, implicit-rejection semantics).
"""

from __future__ import annotations

import secrets
from threading import Thread

import pytest

from hpke_http.constants import (
    XWING_ENC_SIZE,
    XWING_PRIVATE_KEY_SEED_SIZE,
    XWING_PUBLIC_KEY_SIZE,
    XWING_SHARED_SECRET_SIZE,
    KemId,
)
from hpke_http.primitives import XWingKEM, get_kem

# ---------------------------------------------------------------------------
# Class-level invariants
# ---------------------------------------------------------------------------


class TestClassInvariants:
    def test_kem_id(self) -> None:
        assert XWingKEM.KEM_ID == KemId.XWING
        assert int(XWingKEM.KEM_ID) == 0x647A

    def test_sizes(self) -> None:
        assert XWingKEM.PUBLIC_KEY_SIZE == 1216
        assert XWingKEM.PRIVATE_KEY_SIZE == 32
        assert XWingKEM.ENC_SIZE == 1120
        assert XWingKEM.SHARED_SECRET_SIZE == 32
        assert XWingKEM.DERIVE_KEYPAIR_IKM_MIN_SIZE == 32

    def test_kem_suite_id(self) -> None:
        assert XWingKEM.kem_suite_id() == b"KEM" + (0x647A).to_bytes(2, "big")

    def test_registry_lookup(self) -> None:
        assert get_kem(KemId.XWING) is XWingKEM


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_generate_keypair_sizes(self) -> None:
        sk, pk = XWingKEM.generate_keypair()
        assert len(sk) == XWING_PRIVATE_KEY_SEED_SIZE
        assert len(pk) == XWING_PUBLIC_KEY_SIZE

    def test_encap_sizes(self) -> None:
        _sk, pk = XWingKEM.generate_keypair()
        enc, ss = XWingKEM.encap(pk)
        assert len(enc) == XWING_ENC_SIZE
        assert len(ss) == XWING_SHARED_SECRET_SIZE

    def test_encap_decap_roundtrip(self) -> None:
        sk, pk = XWingKEM.generate_keypair()
        enc, ss_send = XWingKEM.encap(pk)
        ss_recv = XWingKEM.decap(enc, sk)
        assert ss_send == ss_recv

    def test_many_roundtrips_no_state_pollution(self) -> None:
        """Each fresh keypair + encap+decap must agree, in any order."""
        for _ in range(20):
            sk, pk = XWingKEM.generate_keypair()
            enc, ss_send = XWingKEM.encap(pk)
            assert XWingKEM.decap(enc, sk) == ss_send

    def test_same_pk_different_encaps_differ(self) -> None:
        """Encap is randomized: same pk → different enc and ss."""
        _sk, pk = XWingKEM.generate_keypair()
        enc1, ss1 = XWingKEM.encap(pk)
        enc2, ss2 = XWingKEM.encap(pk)
        assert enc1 != enc2
        assert ss1 != ss2

    def test_derive_keypair_deterministic(self) -> None:
        seed = secrets.token_bytes(32)
        sk1, pk1 = XWingKEM.derive_keypair(seed)
        sk2, pk2 = XWingKEM.derive_keypair(seed)
        assert sk1 == sk2 == seed
        assert pk1 == pk2

    def test_derive_public_key_matches_generation(self) -> None:
        sk, pk = XWingKEM.generate_keypair()
        assert XWingKEM.derive_public_key(sk) == pk

    def test_all_zero_seed_succeeds_and_is_deterministic(self) -> None:
        seed = b"\x00" * 32
        sk1, pk1 = XWingKEM.derive_keypair(seed)
        sk2, pk2 = XWingKEM.derive_keypair(seed)
        assert sk1 == sk2
        assert pk1 == pk2

    def test_all_ff_seed_succeeds(self) -> None:
        seed = b"\xff" * 32
        XWingKEM.derive_keypair(seed)  # no raise


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSeedLength:
    @pytest.mark.parametrize("size", [0, 1, 16, 31, 33, 64, 1000])
    def test_derive_keypair_wrong_size_raises(self, size: int) -> None:
        with pytest.raises(ValueError, match=r"seed|ikm"):
            XWingKEM.derive_keypair(b"\x00" * size)

    def test_derive_keypair_exact_size_ok(self) -> None:
        XWingKEM.derive_keypair(b"\x00" * 32)


class TestPublicKeyValidation:
    @pytest.mark.parametrize("size", [0, 1, 32, 1184, 1215, 1217, 10_000])
    def test_validate_public_key_wrong_size(self, size: int) -> None:
        with pytest.raises(ValueError, match="pk size"):
            XWingKEM.validate_public_key(b"\x00" * size)

    def test_validate_public_key_correct_size(self) -> None:
        _sk, pk = XWingKEM.generate_keypair()
        XWingKEM.validate_public_key(pk)  # no raise

    def test_oversize_pk_rejected_cheaply(self) -> None:
        """10 MB input must be rejected at the size check, not allocated through."""
        big = b"\x00" * 10_000_000
        with pytest.raises(ValueError):
            XWingKEM.validate_public_key(big)

    def test_encap_rejects_wrong_size_pk(self) -> None:
        with pytest.raises(ValueError, match="pk size"):
            XWingKEM.encap(b"\x00" * 1215)


class TestEncValidation:
    @pytest.mark.parametrize("size", [0, 1, 32, 1088, 1119, 1121, 10_000])
    def test_validate_enc_wrong_size(self, size: int) -> None:
        with pytest.raises(ValueError, match="enc size"):
            XWingKEM.validate_enc(b"\x00" * size)

    def test_decap_rejects_wrong_size_enc(self) -> None:
        sk, _pk = XWingKEM.generate_keypair()
        with pytest.raises(ValueError, match="enc size"):
            XWingKEM.decap(b"\x00" * 1119, sk)


class TestSkValidation:
    @pytest.mark.parametrize("size", [0, 1, 31, 33, 64])
    def test_decap_rejects_wrong_size_sk(self, size: int) -> None:
        _sk, pk = XWingKEM.generate_keypair()
        enc, _ss = XWingKEM.encap(pk)
        with pytest.raises(ValueError, match="sk size"):
            XWingKEM.decap(enc, b"\x00" * size)


# ---------------------------------------------------------------------------
# Implicit rejection / weird inputs
# ---------------------------------------------------------------------------


class TestImplicitRejection:
    """ML-KEM-768 (FIPS 203 §6.3) uses implicit rejection: decap with bad ct
    returns a deterministically-derived secret rather than raising.

    The X25519 leg in pyca raises ``ValueError("Error computing shared key")``
    on small-subgroup inputs (RFC 7748 §6.1). XWingKEM.decap catches this
    and substitutes an all-zero ``ss_X`` so behavior matches implementations
    that do not check (Go crypto/x25519, BoringSSL). The combiner blends the
    rejection bytes with the ML-KEM rejection secret to produce a
    deterministic but useless ``ss``; AEAD authentication downstream
    rejects either way.

    Net contract: ``decap`` MUST NOT raise on adversarial ``ct``.
    """

    def test_random_enc_does_not_raise(self) -> None:
        """Random adversarial enc → decap returns a 32-byte ss."""
        sk, _pk = XWingKEM.generate_keypair()
        for _ in range(10):
            random_enc = secrets.token_bytes(XWING_ENC_SIZE)
            ss = XWingKEM.decap(random_enc, sk)
            assert len(ss) == XWING_SHARED_SECRET_SIZE

    def test_all_zero_enc_does_not_raise(self) -> None:
        """All-zero enc — ct_X is a small-subgroup point; decap must implicit-reject."""
        sk, _pk = XWingKEM.generate_keypair()
        ss = XWingKEM.decap(b"\x00" * XWING_ENC_SIZE, sk)
        assert len(ss) == XWING_SHARED_SECRET_SIZE

    def test_all_ff_enc_does_not_raise(self) -> None:
        """All-0xff ct_X is reduced mod p by X25519 — not small-subgroup."""
        sk, _pk = XWingKEM.generate_keypair()
        ss = XWingKEM.decap(b"\xff" * XWING_ENC_SIZE, sk)
        assert len(ss) == XWING_SHARED_SECRET_SIZE

    def test_implicit_rejection_is_deterministic(self) -> None:
        """Two calls with the same (sk, bad_ct) return the same ss.

        Required for HPKE: AEAD key derivation must be deterministic, even
        on rejection paths, so attackers cannot detect rejection via
        non-determinism.
        """
        sk, _pk = XWingKEM.generate_keypair()
        bad_ct = b"\x00" * XWING_ENC_SIZE
        ss1 = XWingKEM.decap(bad_ct, sk)
        ss2 = XWingKEM.decap(bad_ct, sk)
        assert ss1 == ss2


class TestTampering:
    def test_tampered_enc_yields_different_ss(self) -> None:
        sk, pk = XWingKEM.generate_keypair()
        enc, ss = XWingKEM.encap(pk)
        # Flip the last byte (X25519 leg)
        tampered = bytearray(enc)
        tampered[-1] ^= 0x01
        assert XWingKEM.decap(bytes(tampered), sk) != ss

    def test_tampered_ml_kem_half_yields_different_ss(self) -> None:
        sk, pk = XWingKEM.generate_keypair()
        enc, ss = XWingKEM.encap(pk)
        # Flip a byte inside ct_M (ML-KEM portion)
        tampered = bytearray(enc)
        tampered[100] ^= 0x01
        assert XWingKEM.decap(bytes(tampered), sk) != ss

    def test_decap_with_wrong_sk_yields_different_ss(self) -> None:
        sk_a, pk_a = XWingKEM.generate_keypair()
        sk_b, _pk_b = XWingKEM.generate_keypair()
        enc, ss = XWingKEM.encap(pk_a)
        # Both decaps must run (implicit rejection); only the matching sk recovers ss.
        assert XWingKEM.decap(enc, sk_a) == ss
        assert XWingKEM.decap(enc, sk_b) != ss


# ---------------------------------------------------------------------------
# Statelessness / concurrency
# ---------------------------------------------------------------------------


class TestStatelessness:
    def test_concurrent_decap_produces_same_ss(self) -> None:
        """Stateless classmethods must be safe under concurrent access."""
        sk, pk = XWingKEM.generate_keypair()
        enc, ss_expected = XWingKEM.encap(pk)
        results: list[bytes] = []

        def worker() -> None:
            results.append(XWingKEM.decap(enc, sk))

        threads = [Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r == ss_expected for r in results)
        assert len(results) == 20


# Note: registry-level unknown-KEM behavior is covered in
# tests/unit/test_kem_registry.py::TestUnknownKem to avoid duplication.

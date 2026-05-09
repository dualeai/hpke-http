"""
Decap-side KAT replay against draft-connolly-cfrg-xwing-kem-10 Appendix C.

The encap-side KAT is not exercised: pyca/cryptography's ML-KEM-768 has no
deterministic-randomness API (encapsulate() takes no `m` parameter), so we
cannot reproduce the spec's `enc` byte-for-byte from a fixed `eseed`. The
decap path is deterministic given `(sk_seed, ct)`, which is what we replay
here. This still verifies:

- SHAKE256 seed expansion (sk_seed → d || z || sk_X)
- ML-KEM-768 KeyGen via from_seed_bytes(d || z) (matches FIPS 203)
- ML-KEM-768 Decap on the fixture ct
- X25519 base-point scalar mul (sk_X → pk_X)
- X25519 ECDH against ct_X
- SHA3-256 combiner field ordering with XWING_LABEL (draft §5.2)

If the spec changes any of these, the fixture's `ss` will fail to match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hpke_http.primitives import XWingKEM

_FIXTURE_PATH = Path(__file__).parent / "xwing_kem_draft10.json"


def _load_vectors() -> list[dict[str, Any]]:
    with _FIXTURE_PATH.open() as f:
        data: dict[str, Any] = json.load(f)
    return data["vectors"]


@pytest.fixture(params=_load_vectors(), ids=lambda v: v["sk_seed"][:16])
def vector(request: pytest.FixtureRequest) -> dict[str, bytes]:
    v: dict[str, str] = request.param
    return {
        "sk_seed": bytes.fromhex(v["sk_seed"]),
        "pk": bytes.fromhex(v["pk"]),
        "ct": bytes.fromhex(v["ct"]),
        "ss": bytes.fromhex(v["ss"]),
    }


@pytest.mark.vectors
class TestXWingDraft10Vectors:
    """Replay against draft-10 Appendix C."""

    def test_keypair_determinism(self, vector: dict[str, bytes]) -> None:
        """derive_keypair(sk_seed) must produce the spec's pk."""
        sk, pk = XWingKEM.derive_keypair(vector["sk_seed"])
        assert sk == vector["sk_seed"], "sk must echo the seed"
        assert pk == vector["pk"], "pk must match draft-10 fixture"

    def test_decap_kat(self, vector: dict[str, bytes]) -> None:
        """decap(ct, sk_seed) must produce the spec's ss."""
        ss = XWingKEM.decap(vector["ct"], vector["sk_seed"])
        assert ss == vector["ss"], "ss must match draft-10 fixture"

    def test_derive_public_key_matches(self, vector: dict[str, bytes]) -> None:
        """derive_public_key matches derive_keypair output."""
        assert XWingKEM.derive_public_key(vector["sk_seed"]) == vector["pk"]

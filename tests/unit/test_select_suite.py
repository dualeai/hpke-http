"""
Client-side suite selection helper tests.

``_select_suite`` walks a priority list (default ``DEFAULT_KEM_PRIORITY``)
left-to-right and picks the first match in the discovery-doc result. PQ
hybrids come before classical in the default; explicit overrides via
``priority=`` pin specific orderings.
"""

from __future__ import annotations

import pytest

from hpke_http.constants import DEFAULT_KEM_PRIORITY, KemId
from hpke_http.core import _select_suite  # pyright: ignore[reportPrivateUsage]
from hpke_http.exceptions import KeyDiscoveryError

_FAKE_X25519_PK = b"\x00" * 32
_FAKE_XWING_PK = b"\x00" * 1216


class TestDefaultPriority:
    def test_default_is_pq_first(self) -> None:
        """``DEFAULT_KEM_PRIORITY`` is a fixed tuple with X-Wing before X25519."""
        assert DEFAULT_KEM_PRIORITY == (KemId.XWING, KemId.DHKEM_X25519_HKDF_SHA256)

    def test_x25519_only_picks_x25519(self) -> None:
        kem_id, pk = _select_suite({KemId.DHKEM_X25519_HKDF_SHA256: _FAKE_X25519_PK})
        assert kem_id == KemId.DHKEM_X25519_HKDF_SHA256
        assert pk is _FAKE_X25519_PK

    def test_xwing_only_picks_xwing(self) -> None:
        kem_id, pk = _select_suite({KemId.XWING: _FAKE_XWING_PK})
        assert kem_id == KemId.XWING
        assert pk is _FAKE_XWING_PK

    def test_dual_picks_xwing(self) -> None:
        """PQ-first default — X-Wing wins when server advertises both."""
        keys = {
            KemId.DHKEM_X25519_HKDF_SHA256: _FAKE_X25519_PK,
            KemId.XWING: _FAKE_XWING_PK,
        }
        kem_id, pk = _select_suite(keys)
        assert kem_id == KemId.XWING
        assert pk is _FAKE_XWING_PK


class TestExplicitPriority:
    def test_force_classical(self) -> None:
        """Caller pins X25519 — overrides PQ-first default even if X-Wing present."""
        keys = {
            KemId.DHKEM_X25519_HKDF_SHA256: _FAKE_X25519_PK,
            KemId.XWING: _FAKE_XWING_PK,
        }
        kem_id, _pk = _select_suite(keys, priority=[KemId.DHKEM_X25519_HKDF_SHA256])
        assert kem_id == KemId.DHKEM_X25519_HKDF_SHA256

    def test_force_xwing_only_succeeds(self) -> None:
        keys = {KemId.XWING: _FAKE_XWING_PK}
        kem_id, _pk = _select_suite(keys, priority=[KemId.XWING])
        assert kem_id == KemId.XWING

    def test_priority_order_respected(self) -> None:
        """Multi-entry priority — first match wins."""
        keys = {
            KemId.DHKEM_X25519_HKDF_SHA256: _FAKE_X25519_PK,
            KemId.XWING: _FAKE_XWING_PK,
        }
        kem_id, _pk = _select_suite(
            keys,
            priority=[KemId.DHKEM_X25519_HKDF_SHA256, KemId.XWING],
        )
        assert kem_id == KemId.DHKEM_X25519_HKDF_SHA256


class TestStrictNoFallthrough:
    """Priority list is strict — no KEM outside the list is ever picked.

    This protects user pinning intent (e.g. classical-only) and avoids
    crashing later at encap if the server advertises a KEM the client
    doesn't have a primitive for.
    """

    def test_pin_classical_against_xwing_only_server_raises(self) -> None:
        """User pinned X25519. Server only advertises X-Wing. Raise rather
        than silently using X-Wing despite the pin."""
        with pytest.raises(KeyDiscoveryError, match="No KEM in priority"):
            _select_suite(
                {KemId.XWING: _FAKE_XWING_PK},
                priority=[KemId.DHKEM_X25519_HKDF_SHA256],
            )

    def test_pin_xwing_against_x25519_only_server_raises(self) -> None:
        """Symmetric: user pinned X-Wing. Server only advertises X25519. Raise."""
        with pytest.raises(KeyDiscoveryError, match="No KEM in priority"):
            _select_suite(
                {KemId.DHKEM_X25519_HKDF_SHA256: _FAKE_X25519_PK},
                priority=[KemId.XWING],
            )


class TestEmpty:
    def test_empty_dict_raises(self) -> None:
        with pytest.raises(KeyDiscoveryError):
            _select_suite({})

    def test_empty_priority_raises(self) -> None:
        """Empty priority = client accepts no KEM = always raise."""
        with pytest.raises(KeyDiscoveryError, match="No KEM in priority"):
            _select_suite({KemId.DHKEM_X25519_HKDF_SHA256: _FAKE_X25519_PK}, priority=[])

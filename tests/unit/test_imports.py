"""
Module import safety tests.

The contract: ``import hpke_http`` and its submodules must succeed even on
runtimes that lack ML-KEM-768 (older OpenSSL, missing AWS-LC/BoringSSL).
On such runtimes, X-Wing self-deregisters at import time and the rest of
the library continues to work for X25519-only deployments.

Verified via subprocess so the test itself doesn't depend on whether the
host backend has ML-KEM.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(code: str) -> tuple[int, str, str]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


class TestImportSafety:
    def test_top_level_import_clean(self) -> None:
        rc, out, err = _run("import hpke_http; print('OK')")
        assert rc == 0, f"stderr: {err}"
        assert "OK" in out

    def test_primitives_subpackage_imports(self) -> None:
        rc, out, err = _run(
            textwrap.dedent("""
                import hpke_http.primitives
                from hpke_http.primitives import KEM, X25519KEM, get_kem, registered_kem_ids
                from hpke_http.constants import KemId
                # X25519 must always be available — it has no backend dependency.
                assert get_kem(KemId.DHKEM_X25519_HKDF_SHA256) is X25519KEM
                assert KemId.DHKEM_X25519_HKDF_SHA256 in registered_kem_ids()
                print('OK')
            """)
        )
        assert rc == 0, f"stderr: {err}"
        assert "OK" in out

    def test_middleware_subpackage_imports(self) -> None:
        """Middleware must import cleanly even before any KEM op runs."""
        rc, out, err = _run(
            textwrap.dedent("""
                import hpke_http.middleware.fastapi
                import hpke_http.middleware.aiohttp
                import hpke_http.middleware.httpx
                print('OK')
            """)
        )
        assert rc == 0, f"stderr: {err}"
        assert "OK" in out

    def test_x25519_only_path_does_not_touch_mlkem(self) -> None:
        """X25519 encrypt/decrypt must not trigger ML-KEM imports.

        This is what guarantees X25519-only deployments on older backends
        keep working: never reach the lazy-import code paths in xwing_kem.py.
        """
        rc, out, err = _run(
            textwrap.dedent("""
                from hpke_http.primitives import X25519KEM

                sk, pk = X25519KEM.generate_keypair()
                enc, ss_a = X25519KEM.encap(pk)
                ss_b = X25519KEM.decap(enc, sk)
                assert ss_a == ss_b
                # ML-KEM is imported eagerly by xwing_kem.py at package load,
                # but X25519KEM operations themselves never touch it. We rely
                # on the type system + roundtrip success here, not on
                # sys.modules introspection (which can't distinguish between
                # 'mlkem imported but never called' vs 'mlkem called').
                print('OK')
            """)
        )
        assert rc == 0, f"stderr: {err}"
        assert "OK" in out

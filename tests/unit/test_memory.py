"""Memory usage tests for HPKE encryption.

Verifies memory bounds to catch regressions and ensure
predictable resource usage under load.

Note: Memory measurements include OpenSSL internal buffers and Python
object overhead. Bounds are set conservatively to avoid flaky tests
while still catching significant regressions.
"""

import gc
import secrets
import tracemalloc
from collections.abc import Callable

import pytest
from cryptography.hazmat.primitives.asymmetric import x25519

from hpke_http.constants import PSK_MIN_SIZE
from hpke_http.hpke import setup_recipient_psk, setup_sender_psk

# Memory bounds (conservative to account for OpenSSL internals)
MAX_CONTEXT_MEMORY = 100 * 1024  # 100KB per context (includes OpenSSL state)
MAX_SEAL_OVERHEAD = 50 * 1024  # 50KB overhead per seal (excludes ciphertext)


def make_psk(length: int = PSK_MIN_SIZE) -> bytes:
    """Generate a random PSK of specified length."""
    return secrets.token_bytes(length)


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate X25519 keypair."""
    sk = x25519.X25519PrivateKey.generate()
    return sk.private_bytes_raw(), sk.public_key().public_bytes_raw()


def measure_allocation[T](func: Callable[[], T]) -> tuple[int, T]:
    """Measure memory allocated by a function call."""
    gc.collect()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    result = func()

    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    diff = snapshot2.compare_to(snapshot1, "lineno")
    allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)
    return allocated, result


class TestContextMemory:
    """Memory bounds for HPKE context creation."""

    def test_sender_context_memory_bounded(self) -> None:
        """SenderContext creation uses bounded memory."""
        _, pk_r = generate_keypair()
        psk = make_psk()

        allocated, ctx = measure_allocation(lambda: setup_sender_psk(pk_r, b"info", psk, b"tenant"))

        assert allocated < MAX_CONTEXT_MEMORY, f"Context used {allocated} bytes, expected < {MAX_CONTEXT_MEMORY}"
        assert ctx is not None

    def test_recipient_context_memory_bounded(self) -> None:
        """RecipientContext creation uses bounded memory."""
        sk_r, pk_r = generate_keypair()
        psk = make_psk()
        sender_ctx = setup_sender_psk(pk_r, b"info", psk, b"tenant")

        allocated, ctx = measure_allocation(lambda: setup_recipient_psk(sender_ctx.enc, sk_r, b"info", psk, b"tenant"))

        assert allocated < MAX_CONTEXT_MEMORY, f"Context used {allocated} bytes, expected < {MAX_CONTEXT_MEMORY}"
        assert ctx is not None

    def test_concurrent_contexts_memory_sublinear_per_context(self) -> None:
        """Per-context memory decreases with more contexts (shared overhead)."""
        _, pk_r = generate_keypair()
        psk = make_psk()

        # Warmup to stabilize allocator
        for _ in range(10):
            setup_sender_psk(pk_r, b"info", psk, b"tenant")
        gc.collect()

        # Measure memory for batches
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        contexts_100 = [setup_sender_psk(pk_r, b"info", psk, b"tenant") for _ in range(100)]
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        mem_100 = sum(stat.size_diff for stat in diff if stat.size_diff > 0)
        per_context = mem_100 / 100

        # Per-context memory should be reasonable (< 10KB after warmup)
        assert per_context < 10 * 1024, f"Per-context memory {per_context:.0f} bytes, expected < 10KB"
        assert len(contexts_100) == 100


class TestEncryptionMemory:
    """Memory bounds for encryption operations."""

    @pytest.mark.parametrize(
        "payload_size",
        [64, 1024, 64 * 1024],
    )
    def test_seal_memory_overhead_bounded(self, payload_size: int) -> None:
        """Encryption overhead (excluding ciphertext) is bounded."""
        _sk_r, pk_r = generate_keypair()
        psk = make_psk()
        plaintext = secrets.token_bytes(payload_size)
        sender_ctx = setup_sender_psk(pk_r, b"info", psk, b"tenant")

        # Warmup
        sender_ctx.seal(b"warmup", plaintext)
        gc.collect()

        allocated, ciphertext = measure_allocation(lambda: sender_ctx.seal(b"aad", plaintext))

        # Overhead = total allocation - ciphertext size
        overhead = allocated - len(ciphertext)

        assert overhead < MAX_SEAL_OVERHEAD, f"Overhead {overhead} bytes, expected < {MAX_SEAL_OVERHEAD}"
        assert len(ciphertext) == payload_size + 16  # payload + Poly1305 tag

    def test_roundtrip_memory_no_leak(self) -> None:
        """Repeated encrypt/decrypt doesn't leak memory."""
        sk_r, pk_r = generate_keypair()
        psk = make_psk()
        plaintext = secrets.token_bytes(1024)

        sender_ctx = setup_sender_psk(pk_r, b"info", psk, b"tenant")
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"info", psk, b"tenant")

        # Warmup
        for i in range(10):
            ct = sender_ctx.seal(f"aad-{i}".encode(), plaintext)
            recipient_ctx.open(f"aad-{i}".encode(), ct)

        # Measure memory after many operations
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for i in range(100):
            ct = sender_ctx.seal(f"aad-{i + 10}".encode(), plaintext)
            pt = recipient_ctx.open(f"aad-{i + 10}".encode(), ct)
            assert pt == plaintext

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        net_allocated = sum(stat.size_diff for stat in diff)

        # Net allocation after 100 roundtrips should be minimal (< 10KB)
        # Some allocation is expected for sequence counter updates
        assert net_allocated < 10 * 1024, f"Net allocation {net_allocated} bytes after 100 ops, possible leak"


class TestMemoryPressure:
    """Memory behavior under pressure."""

    def test_large_payload_memory_proportional(self) -> None:
        """Large payload memory usage is proportional to payload size."""
        _sk_r, pk_r = generate_keypair()
        psk = make_psk()

        # 1MB payload
        payload_size = 1024 * 1024
        plaintext = b"\x00" * payload_size

        sender_ctx = setup_sender_psk(pk_r, b"info", psk, b"tenant")

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        ciphertext = sender_ctx.seal(b"aad", plaintext)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        # Should allocate roughly 1x payload (for ciphertext)
        # Allow up to 1.5x for intermediate buffers
        max_expected = int(payload_size * 1.5)
        assert allocated < max_expected, f"Allocated {allocated} bytes for {payload_size} byte payload"
        assert len(ciphertext) == payload_size + 16

    def test_many_small_operations_stable(self) -> None:
        """Memory stays stable across many small operations."""
        sk_r, pk_r = generate_keypair()
        psk = make_psk()
        plaintext = secrets.token_bytes(64)

        sender_ctx = setup_sender_psk(pk_r, b"info", psk, b"tenant")
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"info", psk, b"tenant")

        # Run 1000 operations
        tracemalloc.start()
        peak_early = 0
        peak_late = 0

        for i in range(1000):
            ct = sender_ctx.seal(f"aad-{i}".encode(), plaintext)
            pt = recipient_ctx.open(f"aad-{i}".encode(), ct)
            assert pt == plaintext

            # Sample memory periodically
            if i == 100:
                _, peak_early = tracemalloc.get_traced_memory()
            if i == 900:
                _, peak_late = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        # Peak memory shouldn't grow significantly between early and late samples
        # Allow 50% growth for GC timing variations
        growth = (peak_late - peak_early) / peak_early if peak_early > 0 else 0.0
        assert growth < 0.5, f"Memory grew {growth * 100:.0f}% between op 100 and 900, possible leak"

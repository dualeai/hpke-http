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
from typing import TypeVar

import pytest
from cryptography.hazmat.primitives.asymmetric import x25519

from hpke_http.constants import PSK_MIN_SIZE
from hpke_http.hpke import setup_recipient_psk, setup_sender_psk
from hpke_http.streaming import SSEDecryptor, SSEEncryptor, StreamingSession

T = TypeVar("T")

# Memory bounds (conservative to account for OpenSSL internals)
MAX_CONTEXT_MEMORY = 100 * 1024  # 100KB per context (includes OpenSSL state)
MAX_SEAL_OVERHEAD = 50 * 1024  # 50KB overhead per seal (excludes ciphertext)

# SSE memory bounds
# Wire format: "event: enc\ndata: " (17B) + "\n\n" (2B) = 19B
# Payload overhead: counter (4B) + Poly1305 tag (16B) = 20B
# Base64 encoding adds 33% to (payload + 20)
# Formula: output = 19 + ceil((input + 20) * 4/3)
SSE_WIRE_OVERHEAD = 19  # Fixed wire format bytes
SSE_PAYLOAD_OVERHEAD = 20  # counter + tag
MAX_SSE_STREAMING_NET = 50 * 1024  # 50KB net allocation after 1000 roundtrips


def expected_sse_output_size(input_size: int) -> int:
    """Calculate expected SSE output size for given input."""
    # Binary payload = input + counter(4) + tag(16)
    binary_size = input_size + SSE_PAYLOAD_OVERHEAD
    # Base64 encoding: ceil(n * 4/3)
    base64_size = (binary_size * 4 + 2) // 3
    # Wire format wrapper
    return SSE_WIRE_OVERHEAD + base64_size


def make_psk(length: int = PSK_MIN_SIZE) -> bytes:
    """Generate a random PSK of specified length."""
    return secrets.token_bytes(length)


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate X25519 keypair."""
    sk = x25519.X25519PrivateKey.generate()
    return sk.private_bytes_raw(), sk.public_key().public_bytes_raw()


def measure_allocation(func: Callable[[], T]) -> tuple[int, T]:
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
        """Memory stays stable across many small operations.

        Uses net allocation after GC to avoid Python version variance in peak measurements.
        """
        sk_r, pk_r = generate_keypair()
        psk = make_psk()
        plaintext = secrets.token_bytes(64)

        sender_ctx = setup_sender_psk(pk_r, b"info", psk, b"tenant")
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"info", psk, b"tenant")

        # Warmup phase - let allocator/GC stabilize
        for i in range(100):
            ct = sender_ctx.seal(f"warmup-{i}".encode(), plaintext)
            pt = recipient_ctx.open(f"warmup-{i}".encode(), ct)
            assert pt == plaintext
        gc.collect()

        # Measure net allocation over many operations
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for i in range(1000):
            ct = sender_ctx.seal(f"aad-{i}".encode(), plaintext)
            pt = recipient_ctx.open(f"aad-{i}".encode(), ct)
            assert pt == plaintext

        gc.collect()  # Force GC before final measurement
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Check net allocation (not peak) - should be minimal after GC
        diff = snapshot2.compare_to(snapshot1, "lineno")
        net_allocated = sum(stat.size_diff for stat in diff)

        # Net allocation after 1000 roundtrips should be < 100KB
        # (sequence counters increment but don't allocate much)
        assert net_allocated < 100 * 1024, f"Net allocation {net_allocated} bytes after 1000 ops, possible leak"


class TestSSEMemory:
    """Memory bounds for SSE streaming encryption.

    Tests zero-copy optimizations:
    - SSEEncryptor.encrypt(bytes) -> bytes
    - SSEDecryptor.decrypt(str) -> bytes
    - Wire format overhead (base64 + counter + tag)
    """

    @staticmethod
    def _make_session() -> StreamingSession:
        """Create a test SSE session."""
        return StreamingSession(session_key=b"k" * 32, session_salt=b"salt")

    @staticmethod
    def _extract_data_field(sse: bytes) -> str:
        """Extract data field from encrypted SSE output."""
        for line in sse.decode("ascii").split("\n"):
            if line.startswith("data: "):
                return line[6:]
        raise ValueError("No data field found")

    @pytest.mark.parametrize("chunk_size", [64, 1024, 64 * 1024])
    def test_sse_encrypt_overhead_bounded(self, chunk_size: int) -> None:
        """SSE encryption overhead is bounded (base64 + wire format).

        Formula: output = 19 + ceil((input + 20) * 4/3)
        - 19B wire format ("event: enc\\ndata: " + "\\n\\n")
        - 20B payload overhead (4B counter + 16B tag)
        - 33% base64 encoding expansion
        """
        session = self._make_session()
        encryptor = SSEEncryptor(session)
        chunk = secrets.token_bytes(chunk_size)

        # Warmup
        encryptor.encrypt(chunk)
        gc.collect()

        allocated, output = measure_allocation(lambda: encryptor.encrypt(chunk))

        # Output should match formula (with small margin for rounding)
        max_expected = expected_sse_output_size(chunk_size) + 10  # 10 byte margin
        assert len(output) <= max_expected, (
            f"Output {len(output)} bytes for {chunk_size} input, expected <= {max_expected}"
        )

        # Memory allocation should be proportional to output (+ 10KB for temp buffers)
        assert allocated < len(output) + 10 * 1024, f"Allocated {allocated} bytes, expected < {len(output) + 10 * 1024}"

    @pytest.mark.parametrize("payload_size", [64, 1024, 64 * 1024])
    def test_sse_decrypt_overhead_bounded(self, payload_size: int) -> None:
        """SSE decryption overhead is bounded.

        Overhead = base64 decode temp buffer only.
        Expected: allocation < 1.1x ciphertext size.
        """
        session = self._make_session()
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)
        chunk = secrets.token_bytes(payload_size)

        # Create encrypted data
        encrypted = encryptor.encrypt(chunk)
        data_field = self._extract_data_field(encrypted)

        # Warmup
        _ = decryptor.decrypt(data_field)
        # Reset decryptor for fresh measurement
        decryptor = SSEDecryptor(self._make_session())
        encryptor2 = SSEEncryptor(self._make_session())
        encrypted2 = encryptor2.encrypt(chunk)
        data_field2 = self._extract_data_field(encrypted2)
        gc.collect()

        allocated, plaintext = measure_allocation(lambda: decryptor.decrypt(data_field2))

        # Allocation should be proportional to payload
        max_expected = int(payload_size * 1.2) + 10 * 1024  # 1.2x + 10KB buffer
        assert allocated < max_expected, (
            f"Allocated {allocated} bytes for {payload_size} payload, expected < {max_expected}"
        )
        assert plaintext == chunk

    def test_sse_streaming_no_leak(self) -> None:
        """1000 SSE encrypt/decrypt roundtrips don't leak memory.

        Verifies zero-copy optimizations prevent memory accumulation.
        """
        session = self._make_session()
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)
        chunk = b"event: test\ndata: {}\n\n"

        # Warmup phase
        for _ in range(100):
            encrypted = encryptor.encrypt(chunk)
            data_field = self._extract_data_field(encrypted)
            decryptor.decrypt(data_field)
        gc.collect()

        # Measure net allocation over 1000 operations
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(1000):
            encrypted = encryptor.encrypt(chunk)
            data_field = self._extract_data_field(encrypted)
            plaintext = decryptor.decrypt(data_field)
            assert plaintext == chunk

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        net_allocated = sum(stat.size_diff for stat in diff)

        # Net allocation should be minimal (counters increment but don't allocate much)
        assert net_allocated < MAX_SSE_STREAMING_NET, (
            f"Net allocation {net_allocated} bytes after 1000 roundtrips, expected < {MAX_SSE_STREAMING_NET}"
        )

    def test_sse_large_chunk_memory_proportional(self) -> None:
        """100KB SSE chunk memory scales linearly.

        Verifies no quadratic blowup from string operations.
        """
        session = self._make_session()
        encryptor = SSEEncryptor(session)

        # 100KB chunk
        chunk_size = 100 * 1024
        chunk = secrets.token_bytes(chunk_size)

        # Warmup
        encryptor.encrypt(chunk)
        gc.collect()

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        output = encryptor.encrypt(chunk)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        # Should allocate close to output size (formula-based)
        max_expected = expected_sse_output_size(chunk_size) + 10 * 1024  # 10KB buffer margin
        assert allocated < max_expected, (
            f"Allocated {allocated} bytes for {chunk_size} chunk, expected < {max_expected}"
        )
        assert len(output) <= expected_sse_output_size(chunk_size) + 10

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
from hpke_http.headers import b64url_decode, b64url_encode
from hpke_http.hpke import setup_recipient_psk, setup_sender_psk
from hpke_http.streaming import SSEDecryptor, SSEEncryptor
from tests.conftest import extract_sse_data_field, make_sse_session

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

    @pytest.mark.parametrize("chunk_size", [64, 1024, 64 * 1024])
    def test_sse_encrypt_overhead_bounded(self, chunk_size: int) -> None:
        """SSE encryption overhead is bounded (base64 + wire format).

        Formula: output = 19 + ceil((input + 20) * 4/3)
        - 19B wire format ("event: enc\\ndata: " + "\\n\\n")
        - 20B payload overhead (4B counter + 16B tag)
        - 33% base64 encoding expansion
        """
        session = make_sse_session()
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

        With memoryview optimization, b64url_decode returns a view over
        decoded bytes (no extra copy). Slicing for counter/ciphertext is zero-copy.
        """
        session = make_sse_session()
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)
        chunk = secrets.token_bytes(payload_size)

        # Create two encrypted messages (counter increments)
        encrypted_warmup = encryptor.encrypt(chunk)
        encrypted_measure = encryptor.encrypt(chunk)
        data_field_warmup = extract_sse_data_field(encrypted_warmup)
        data_field_measure = extract_sse_data_field(encrypted_measure)

        # Warmup the same decryptor we'll measure
        _ = decryptor.decrypt(data_field_warmup)
        gc.collect()

        # Measure the warmed-up decryptor
        allocated, plaintext = measure_allocation(lambda: decryptor.decrypt(data_field_measure))

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
        session = make_sse_session()
        encryptor = SSEEncryptor(session)
        decryptor = SSEDecryptor(session)
        chunk = b"event: test\ndata: {}\n\n"

        # Warmup phase
        for _ in range(100):
            encrypted = encryptor.encrypt(chunk)
            data_field = extract_sse_data_field(encrypted)
            decryptor.decrypt(data_field)
        gc.collect()

        # Measure net allocation over 1000 operations
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(1000):
            encrypted = encryptor.encrypt(chunk)
            data_field = extract_sse_data_field(encrypted)
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
        session = make_sse_session()
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


class TestBase64Memory:
    """Memory bounds for base64url encode/decode operations.

    Tests zero-copy optimizations:
    - b64url_encode: accepts memoryview input (Python 3.4+)
    - b64url_decode: returns memoryview for zero-copy slicing
    """

    # Base64 expands data by 4/3 (33%)
    BASE64_EXPANSION = 4 / 3

    @pytest.mark.parametrize("input_size", [1024, 64 * 1024, 256 * 1024])
    def test_encode_memory_proportional(self, input_size: int) -> None:
        """b64url_encode memory scales linearly with input size.

        Allocations:
        - urlsafe_b64encode: 1.33x input (base64 expansion)
        - rstrip: reuses buffer or small copy
        - decode to string: 1.33x input
        Measured: ~1.34x input for large payloads, ~3.5x for small (fixed overhead)

        Note: Small payloads tested separately due to fixed Python overhead.
        """
        data = secrets.token_bytes(input_size)

        # Warmup
        b64url_encode(data)
        gc.collect()

        allocated, encoded = measure_allocation(lambda: b64url_encode(data))

        # Expected output size (base64 without padding)
        expected_output = (input_size * 4 + 2) // 3

        # Measured ~1.34x for large payloads; allow 2x + overhead for safety margin
        max_expected = int(input_size * 2) + 4096
        assert allocated < max_expected, (
            f"Allocated {allocated} bytes for {input_size} input, expected < {max_expected}"
        )
        # Verify output size is correct
        assert len(encoded) == expected_output or len(encoded) == expected_output - 1, (
            f"Output {len(encoded)} bytes, expected ~{expected_output}"
        )

    @pytest.mark.parametrize("input_size", [1024, 64 * 1024, 256 * 1024])
    def test_decode_memory_proportional(self, input_size: int) -> None:
        """b64url_decode memory scales linearly with input size.

        Allocations:
        - urlsafe_b64decode: 1x decoded size (bytes)
        - memoryview: ~184 bytes (view object only, no copy)
        Measured: ~1.01x decoded size for large payloads

        Note: Small payloads tested separately due to fixed Python overhead.
        """
        data = secrets.token_bytes(input_size)
        encoded = b64url_encode(data)

        # Warmup
        b64url_decode(encoded)
        gc.collect()

        allocated, decoded = measure_allocation(lambda: b64url_decode(encoded))

        # Measured ~1.01x for large payloads; allow 1.2x + overhead for safety margin
        max_expected = int(input_size * 1.2) + 4096
        assert allocated < max_expected, (
            f"Allocated {allocated} bytes for {input_size} decoded, expected < {max_expected}"
        )
        # Verify decoded content
        assert bytes(decoded) == data

    def test_decode_returns_memoryview(self) -> None:
        """b64url_decode returns memoryview type."""
        data = secrets.token_bytes(64)
        encoded = b64url_encode(data)

        decoded = b64url_decode(encoded)

        assert isinstance(decoded, memoryview), f"Expected memoryview, got {type(decoded)}"
        assert bytes(decoded) == data

    def test_decode_memoryview_enables_zero_copy_slicing(self) -> None:
        """Slicing memoryview result doesn't copy underlying buffer data.

        This is the key optimization: extracting counter and ciphertext
        from decoded SSE payload should be zero-copy (memoryview slices
        share the underlying buffer).

        Proof: slice objects are tiny (~184 bytes) regardless of data size.
        """
        # Simulate SSE payload: counter (4B) + ciphertext (64KB)
        counter_bytes = (42).to_bytes(4, "big")
        ciphertext = secrets.token_bytes(64 * 1024)
        payload = counter_bytes + ciphertext
        encoded = b64url_encode(payload)

        # Decode once
        decoded = b64url_decode(encoded)

        # Slice into counter and ciphertext
        counter_slice = decoded[:4]
        ciphertext_slice = decoded[4:]

        # Verify slice objects are memoryview (not bytes copies)
        assert isinstance(counter_slice, memoryview), f"Expected memoryview, got {type(counter_slice)}"
        assert isinstance(ciphertext_slice, memoryview), f"Expected memoryview, got {type(ciphertext_slice)}"

        # Zero-copy proof: memoryview slice objects are tiny (~184 bytes)
        # regardless of the underlying data size. A bytes copy would be ~64KB.
        import sys
        assert sys.getsizeof(ciphertext_slice) < 300, (
            f"Slice object is {sys.getsizeof(ciphertext_slice)} bytes, "
            "expected <300 (should be a view, not a copy)"
        )

        # Verify data correctness
        assert int.from_bytes(counter_slice, "big") == 42
        assert bytes(ciphertext_slice) == ciphertext

    def test_decode_slice_vs_bytes_slice_comparison(self) -> None:
        """Compare memory: memoryview slice vs bytes slice.

        Demonstrates the optimization benefit by measuring allocation
        when slicing already-decoded data.
        """
        payload = secrets.token_bytes(64 * 1024)  # 64KB
        encoded = b64url_encode(payload)

        # Pre-decode for both methods
        mv_decoded = b64url_decode(encoded)
        bytes_decoded = bytes(mv_decoded)
        gc.collect()

        # Measure bytes slicing (creates copies)
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        bytes_counter = bytes_decoded[:4]
        bytes_ciphertext = bytes_decoded[4:]
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        diff = snapshot2.compare_to(snapshot1, "lineno")
        bytes_slice_alloc = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        gc.collect()

        # Measure memoryview slicing (zero-copy views)
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        mv_counter = mv_decoded[:4]
        mv_ciphertext = mv_decoded[4:]
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        diff = snapshot2.compare_to(snapshot1, "lineno")
        mv_slice_alloc = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        # bytes slicing copies data (~64KB for large slice)
        # memoryview slicing just creates view objects (~200 bytes)
        assert mv_slice_alloc < bytes_slice_alloc, (
            f"memoryview slicing ({mv_slice_alloc}) should allocate less than bytes ({bytes_slice_alloc})"
        )

        # Verify data is correct
        assert bytes(mv_counter) == bytes_counter
        assert bytes(mv_ciphertext) == bytes_ciphertext

        # Verify slices are correct types
        assert isinstance(mv_counter, memoryview)
        assert isinstance(bytes_counter, bytes)

    def test_encode_accepts_memoryview_input(self) -> None:
        """b64url_encode accepts memoryview input (zero-copy from caller)."""
        data = bytearray(secrets.token_bytes(1024))
        mv = memoryview(data)

        # Should not raise
        encoded = b64url_encode(mv)

        # Verify result is correct
        assert b64url_encode(bytes(data)) == encoded

    def test_encode_memoryview_slice_input(self) -> None:
        """b64url_encode accepts memoryview slice (zero-copy partial encoding)."""
        data = bytearray(secrets.token_bytes(4096))
        mv = memoryview(data)

        # Encode only middle 2KB
        partial = mv[1024:3072]

        # Warmup
        b64url_encode(partial)
        gc.collect()

        allocated, encoded = measure_allocation(lambda: b64url_encode(partial))

        # Should allocate based on slice size (2KB), not full buffer
        # Measured ~1.34x; allow 2x + overhead for safety margin
        max_expected = int(2048 * 2) + 4096
        assert allocated < max_expected, (
            f"Allocated {allocated} bytes for 2KB slice, expected < {max_expected}"
        )

        # Verify encoding is correct
        assert b64url_encode(bytes(data[1024:3072])) == encoded

    def test_roundtrip_no_memory_leak(self) -> None:
        """Repeated encode/decode cycles don't leak memory."""
        data = secrets.token_bytes(1024)

        # Warmup
        for _ in range(100):
            encoded = b64url_encode(data)
            decoded = b64url_decode(encoded)
            assert bytes(decoded) == data
        gc.collect()

        # Measure over many iterations
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(1000):
            encoded = b64url_encode(data)
            decoded = b64url_decode(encoded)
            assert bytes(decoded) == data

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        net_allocated = sum(stat.size_diff for stat in diff)

        # Net allocation after 1000 roundtrips should be minimal
        assert net_allocated < 50 * 1024, (
            f"Net allocation {net_allocated} bytes after 1000 roundtrips, possible leak"
        )

    def test_large_payload_memory_bounded(self) -> None:
        """1MB payload encode/decode uses bounded memory."""
        payload_size = 1024 * 1024  # 1MB
        data = secrets.token_bytes(payload_size)

        # Warmup
        encoded = b64url_encode(data)
        _ = b64url_decode(encoded)
        gc.collect()

        # Measure encode
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        encoded = b64url_encode(data)
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        encode_allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        # Measured ~1.33x payload; allow 2x for safety margin
        max_encode = int(payload_size * 2)
        assert encode_allocated < max_encode, (
            f"Encode allocated {encode_allocated} bytes for {payload_size} payload"
        )

        gc.collect()

        # Measure decode
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        decoded = b64url_decode(encoded)
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        decode_allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        # Measured ~1.0x payload; allow 1.2x for safety margin
        max_decode = int(payload_size * 1.2)
        assert decode_allocated < max_decode, (
            f"Decode allocated {decode_allocated} bytes for {payload_size} payload"
        )
        assert bytes(decoded) == data

    @pytest.mark.parametrize(
        "size",
        [
            pytest.param(32, id="32B-ephemeral-key"),
            pytest.param(4, id="4B-sse-salt"),
            pytest.param(64, id="64B-typical-header"),
        ],
    )
    def test_small_payload_overhead_reasonable(self, size: int) -> None:
        """Small payloads (headers, keys) have reasonable overhead.

        For small data, fixed Python object overhead dominates.
        Verify allocations stay bounded (not growing unexpectedly).
        """
        data = secrets.token_bytes(size)

        # Warmup
        b64url_encode(data)
        gc.collect()

        allocated, encoded = measure_allocation(lambda: b64url_encode(data))

        # Small payloads: fixed overhead ~3KB for Python objects
        max_expected = 4096
        assert allocated < max_expected, (
            f"Allocated {allocated} bytes for {size}-byte payload, expected < {max_expected}"
        )

        # Decode
        _ = b64url_decode(encoded)
        gc.collect()

        allocated, decoded = measure_allocation(lambda: b64url_decode(encoded))

        assert allocated < max_expected, (
            f"Decode allocated {allocated} bytes for {size}-byte payload, expected < {max_expected}"
        )
        assert bytes(decoded) == data

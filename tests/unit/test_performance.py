"""Performance and memory tests for HPKE-HTTP encryption.

Tests:
1. Memory bounds (context, seal, SSE, base64)
2. Memory ratios (peak memory / payload size)
3. Scaling behavior (linear vs quadratic)
4. Relative overhead (response vs request)
5. Memory leaks
"""

import gc
import os
import secrets
import sys
import time
import tracemalloc
from collections.abc import Callable
from typing import TypeVar

import pytest
from cryptography.hazmat.primitives.asymmetric import x25519

from hpke_http.constants import CHUNK_SIZE, PSK_MIN_SIZE
from hpke_http.core import (
    RequestDecryptor,
    RequestEncryptor,
    ResponseDecryptor,
    ResponseEncryptor,
)
from hpke_http.headers import b64url_decode, b64url_encode
from hpke_http.hpke import setup_recipient_psk, setup_sender_psk
from hpke_http.streaming import ChunkDecryptor, ChunkEncryptor, RawFormat, StreamingSession
from tests.conftest import extract_sse_data_field, make_sse_session

T = TypeVar("T")


# =============================================================================
# BOUNDS - All thresholds centralized here
# =============================================================================

# --- Context & Encryption Memory ---
# Single HPKE context creation (X25519 + HKDF). Measured: ~3-11KB
MAX_CONTEXT_MEMORY = 12 * 1024
# Per-context in batch after warmup. Measured: ~500 bytes
MAX_CONTEXT_MEMORY_BATCH = 1 * 1024
# Seal overhead excluding ciphertext. Measured: ~1.4-2.6KB
MAX_SEAL_OVERHEAD = 3 * 1024

# --- SSE Wire Format ---
# Fixed: "event: enc\ndata: " (17B) + "\n\n" (2B)
SSE_WIRE_OVERHEAD = 19
# Per-chunk: counter (4B) + Poly1305 tag (16B)
SSE_PAYLOAD_OVERHEAD = 20
# Margin for SSE output size calculation
SSE_OUTPUT_MARGIN = 10
# Extra allocation margin for SSE operations. Measured: ~1-2KB
SSE_ALLOC_MARGIN = 3 * 1024

# --- Base64 Memory ---
# Small payload overhead (Python object overhead). Measured: ~650-2000 bytes
MAX_BASE64_SMALL_OVERHEAD = 2 * 1024
# Encode ratio + fixed overhead. Measured: ~1.34x
MAX_BASE64_ENCODE_RATIO = 2.0
MAX_BASE64_ENCODE_FIXED = 4 * 1024
# Decode ratio + fixed overhead. Measured: ~1.01x
MAX_BASE64_DECODE_RATIO = 1.2
MAX_BASE64_DECODE_FIXED = 4 * 1024

# --- Memory Ratios (peak / payload) ---
# Buffered encrypt/decrypt. Measured: ~1.03x for 100KB+
MAX_BUFFERED_MEMORY_RATIO = 1.1
# Streaming encrypt. Measured: ~1.03x for 100KB+
MAX_STREAMING_MEMORY_RATIO = 1.1
# Large payload seal. Measured: ~1.001x
MAX_LARGE_PAYLOAD_RATIO = 1.1

# --- Scaling Factors ---
# Time scaling: 10x payload → Nx time. Measured: 2.4x (sublinear due to HPKE setup)
# CI environments have ~2x variance, so allow 6x to avoid flaky failures
MAX_TIME_SCALING_FACTOR = 0.6
# Memory scaling: 10x payload → Nx memory. Measured: 10x (linear)
MAX_MEMORY_SCALING_FACTOR = 1.5

# --- Overhead Ratios ---
# Setup dominates small payloads (1KB). Measured: ~80%
MIN_SETUP_RATIO_SMALL = 0.2
# Crypto dominates large payloads (5MB). Measured: ~5%
MAX_SETUP_RATIO_LARGE = 0.3

# --- Memory Leak Thresholds ---
# Net allocation after N operations (should be minimal)
# These thresholds accommodate variance across Python versions (3.10-3.14+)
# HPKE seal/open 100 ops. Measured: ~1.2KB (py3.10)
MAX_LEAK_HPKE_100 = 10 * 1024
# RequestEncryptor/Decryptor 100 roundtrips. Measured: ~34KB (py3.10), ~41KB (py3.14)
MAX_LEAK_CORE_100 = 50 * 1024
# SSE encrypt/decrypt 1000 roundtrips. Measured: ~4KB (py3.10)
MAX_LEAK_SSE_1000 = 15 * 1024
# Streaming chunk 500 ops. Measured: ~129KB (py3.10)
MAX_LEAK_STREAMING_500 = 150 * 1024
# Base64 encode/decode 1000 roundtrips. Measured: ~730B (py3.10), ~11KB (py3.14)
MAX_LEAK_BASE64_1000 = 15 * 1024
# Many small HPKE ops 1000. Measured: ~1.3KB (py3.10), ~9KB (py3.14)
MAX_LEAK_SMALL_OPS_1000 = 15 * 1024

# --- Misc ---
# Memoryview slice object size (proves zero-copy). Measured: 184 bytes
MAX_MEMORYVIEW_SLICE_SIZE = 250

# --- Test Payload Sizes ---
# Full range: 10KB, 500KB, 10MB, 50MB, 250MB
PAYLOAD_SIZES_KB = [10, 500, 10 * 1024, 50 * 1024, 250 * 1024]
PAYLOAD_SIZES_KB_IDS = ["10KB", "500KB", "10MB", "50MB", "250MB"]
# Ratio tests: start at 100KB (smaller sizes have too much fixed overhead variance)
PAYLOAD_SIZES_KB_RATIO = [100, 500, 10 * 1024, 50 * 1024]
PAYLOAD_SIZES_KB_RATIO_IDS = ["100KB", "500KB", "10MB", "50MB"]
# Small payloads for overhead tests
PAYLOAD_SIZES_SMALL = [64, 1024, 64 * 1024]
PAYLOAD_SIZES_SMALL_IDS = ["64B", "1KB", "64KB"]


# =============================================================================
# UTILITIES
# =============================================================================


def expected_sse_output_size(input_size: int) -> int:
    """Calculate expected SSE output size for given input."""
    binary_size = input_size + SSE_PAYLOAD_OVERHEAD
    base64_size = (binary_size * 4 + 2) // 3
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


def measure_cpu_time(fn: Callable[[], T], iterations: int = 5) -> tuple[float, T]:
    """Measure median CPU time. Returns (seconds, last_result)."""
    for _ in range(2):
        fn()
    gc.collect()
    times: list[float] = []
    result: T = fn()
    for _ in range(iterations):
        t0 = time.process_time()
        result = fn()
        times.append(time.process_time() - t0)
    times.sort()
    return times[len(times) // 2], result


def measure_peak_memory(fn: Callable[[], T]) -> tuple[int, T]:
    """Measure peak memory allocation. Returns (bytes, result)."""
    gc.collect()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    result = fn()
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    diff = snapshot2.compare_to(snapshot1, "lineno")
    peak = sum(stat.size_diff for stat in diff if stat.size_diff > 0)
    return peak, result


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def perf_keypair() -> tuple[bytes, bytes]:
    """Generate X25519 keypair."""
    sk = x25519.X25519PrivateKey.generate()
    return sk.private_bytes_raw(), sk.public_key().public_bytes_raw()


@pytest.fixture
def perf_psk() -> bytes:
    """32-byte PSK."""
    return b"perf-test-psk-32-bytes-exactly!!"


@pytest.fixture
def perf_psk_id() -> bytes:
    """PSK identifier."""
    return b"perf-tenant"


@pytest.fixture
def streaming_session() -> StreamingSession:
    """Pre-initialized streaming session."""
    return StreamingSession.create(os.urandom(32))


# =============================================================================
# CONTEXT MEMORY TESTS
# =============================================================================


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

    def test_concurrent_contexts_memory_sublinear(self) -> None:
        """Per-context memory decreases with more contexts (shared overhead)."""
        _, pk_r = generate_keypair()
        psk = make_psk()

        for _ in range(10):
            setup_sender_psk(pk_r, b"info", psk, b"tenant")
        gc.collect()

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        contexts = [setup_sender_psk(pk_r, b"info", psk, b"tenant") for _ in range(100)]
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        mem = sum(stat.size_diff for stat in diff if stat.size_diff > 0)
        per_context = mem / 100

        assert per_context < MAX_CONTEXT_MEMORY_BATCH, (
            f"Per-context memory {per_context:.0f} bytes, expected < {MAX_CONTEXT_MEMORY_BATCH}"
        )
        assert len(contexts) == 100


# =============================================================================
# ENCRYPTION MEMORY TESTS
# =============================================================================


class TestEncryptionMemory:
    """Memory bounds for encryption operations."""

    @pytest.mark.parametrize("payload_size", PAYLOAD_SIZES_SMALL, ids=PAYLOAD_SIZES_SMALL_IDS)
    def test_seal_memory_overhead_bounded(self, payload_size: int) -> None:
        """Encryption overhead (excluding ciphertext) is bounded."""
        _, pk_r = generate_keypair()
        psk = make_psk()
        plaintext = secrets.token_bytes(payload_size)
        sender_ctx = setup_sender_psk(pk_r, b"info", psk, b"tenant")

        sender_ctx.seal(b"warmup", plaintext)
        gc.collect()

        allocated, ciphertext = measure_allocation(lambda: sender_ctx.seal(b"aad", plaintext))
        overhead = allocated - len(ciphertext)

        assert overhead < MAX_SEAL_OVERHEAD, f"Overhead {overhead} bytes, expected < {MAX_SEAL_OVERHEAD}"
        assert len(ciphertext) == payload_size + 16

    def test_large_payload_memory_proportional(self) -> None:
        """Large payload memory usage is proportional to payload size."""
        _, pk_r = generate_keypair()
        psk = make_psk()
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

        max_expected = int(payload_size * MAX_LARGE_PAYLOAD_RATIO)
        assert allocated < max_expected, (
            f"Allocated {allocated} bytes for {payload_size} byte payload, expected < {max_expected}"
        )
        assert len(ciphertext) == payload_size + 16


# =============================================================================
# MEMORY RATIOS TESTS
# =============================================================================


class TestMemoryRatios:
    """Tests for memory usage relative to payload size."""

    @pytest.mark.parametrize("size_kb", PAYLOAD_SIZES_KB_RATIO, ids=PAYLOAD_SIZES_KB_RATIO_IDS)
    def test_buffered_encrypt_memory_ratio(
        self, perf_keypair: tuple[bytes, bytes], perf_psk: bytes, perf_psk_id: bytes, size_kb: int
    ) -> None:
        """Buffered encryption memory stays within ratio bound."""
        _, pk = perf_keypair
        size = size_kb * 1024
        plaintext = os.urandom(size)

        RequestEncryptor(pk, perf_psk, perf_psk_id).encrypt_all(plaintext)
        gc.collect()

        def encrypt() -> bytes:
            return RequestEncryptor(pk, perf_psk, perf_psk_id).encrypt_all(plaintext)

        peak_bytes, _ = measure_peak_memory(encrypt)
        ratio = peak_bytes / size

        assert ratio < MAX_BUFFERED_MEMORY_RATIO, (
            f"Buffered encrypt used {ratio:.2f}x, expected < {MAX_BUFFERED_MEMORY_RATIO}x"
        )

    @pytest.mark.parametrize("size_kb", PAYLOAD_SIZES_KB_RATIO, ids=PAYLOAD_SIZES_KB_RATIO_IDS)
    def test_buffered_decrypt_memory_ratio(
        self, perf_keypair: tuple[bytes, bytes], perf_psk: bytes, perf_psk_id: bytes, size_kb: int
    ) -> None:
        """Buffered decryption memory stays within ratio bound."""
        sk, pk = perf_keypair
        size = size_kb * 1024
        plaintext = os.urandom(size)

        enc = RequestEncryptor(pk, perf_psk, perf_psk_id)
        ciphertext = enc.encrypt_all(plaintext)
        headers = enc.get_headers()

        RequestDecryptor(headers, sk, perf_psk, perf_psk_id).decrypt_all(ciphertext)
        gc.collect()

        def decrypt() -> bytes:
            return RequestDecryptor(headers, sk, perf_psk, perf_psk_id).decrypt_all(ciphertext)

        peak_bytes, _ = measure_peak_memory(decrypt)
        ratio = peak_bytes / size

        assert ratio < MAX_BUFFERED_MEMORY_RATIO, (
            f"Buffered decrypt used {ratio:.2f}x, expected < {MAX_BUFFERED_MEMORY_RATIO}x"
        )

    @pytest.mark.parametrize("size_kb", PAYLOAD_SIZES_KB_RATIO, ids=PAYLOAD_SIZES_KB_RATIO_IDS)
    def test_streaming_encrypt_memory_ratio(self, streaming_session: StreamingSession, size_kb: int) -> None:
        """Streaming encryption memory stays within ratio bound."""
        size = size_kb * 1024
        plaintext = os.urandom(size)
        num_chunks = (size + CHUNK_SIZE - 1) // CHUNK_SIZE

        enc = ChunkEncryptor(streaming_session, format=RawFormat(), compress=False)
        for i in range(num_chunks):
            enc.encrypt(plaintext[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE])
        gc.collect()

        def encrypt_streaming() -> bytes:
            e = ChunkEncryptor(streaming_session, format=RawFormat(), compress=False)
            chunks = [e.encrypt(plaintext[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]) for i in range(num_chunks)]
            return b"".join(chunks)

        peak_bytes, _ = measure_peak_memory(encrypt_streaming)
        ratio = peak_bytes / size

        assert ratio < MAX_STREAMING_MEMORY_RATIO, (
            f"Streaming encrypt used {ratio:.2f}x, expected < {MAX_STREAMING_MEMORY_RATIO}x"
        )


# =============================================================================
# SSE MEMORY TESTS
# =============================================================================


class TestSSEMemory:
    """Memory bounds for SSE streaming encryption."""

    @pytest.mark.parametrize("chunk_size", PAYLOAD_SIZES_SMALL, ids=PAYLOAD_SIZES_SMALL_IDS)
    def test_sse_encrypt_overhead_bounded(self, chunk_size: int) -> None:
        """SSE encryption overhead is bounded (base64 + wire format)."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session)
        chunk = secrets.token_bytes(chunk_size)

        encryptor.encrypt(chunk)
        gc.collect()

        allocated, output = measure_allocation(lambda: encryptor.encrypt(chunk))

        max_expected = expected_sse_output_size(chunk_size) + SSE_OUTPUT_MARGIN
        assert len(output) <= max_expected, f"Output {len(output)} bytes, expected <= {max_expected}"
        assert allocated < len(output) + SSE_ALLOC_MARGIN, (
            f"Allocated {allocated} bytes, expected < {len(output) + SSE_ALLOC_MARGIN}"
        )

    @pytest.mark.parametrize("payload_size", PAYLOAD_SIZES_SMALL, ids=PAYLOAD_SIZES_SMALL_IDS)
    def test_sse_decrypt_overhead_bounded(self, payload_size: int) -> None:
        """SSE decryption overhead is bounded."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session)
        decryptor = ChunkDecryptor(session)
        chunk = secrets.token_bytes(payload_size)

        encrypted_warmup = encryptor.encrypt(chunk)
        encrypted_measure = encryptor.encrypt(chunk)
        data_field_warmup = extract_sse_data_field(encrypted_warmup)
        data_field_measure = extract_sse_data_field(encrypted_measure)

        decryptor.decrypt(data_field_warmup)
        gc.collect()

        allocated, plaintext = measure_allocation(lambda: decryptor.decrypt(data_field_measure))

        max_expected = int(payload_size * MAX_LARGE_PAYLOAD_RATIO) + SSE_ALLOC_MARGIN
        assert allocated < max_expected, f"Allocated {allocated} bytes, expected < {max_expected}"
        assert plaintext == chunk

    def test_sse_large_chunk_memory_proportional(self) -> None:
        """100KB SSE chunk memory scales linearly."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session)
        chunk_size = 100 * 1024
        chunk = secrets.token_bytes(chunk_size)

        encryptor.encrypt(chunk)
        gc.collect()

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        output = encryptor.encrypt(chunk)
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        max_expected = expected_sse_output_size(chunk_size) + SSE_ALLOC_MARGIN
        assert allocated < max_expected, f"Allocated {allocated} bytes, expected < {max_expected}"
        assert len(output) <= expected_sse_output_size(chunk_size) + SSE_OUTPUT_MARGIN


# =============================================================================
# BASE64 MEMORY TESTS
# =============================================================================


class TestBase64Memory:
    """Memory bounds for base64url encode/decode operations."""

    @pytest.mark.parametrize("input_size", [1024, 64 * 1024, 256 * 1024], ids=["1KB", "64KB", "256KB"])
    def test_encode_memory_proportional(self, input_size: int) -> None:
        """b64url_encode memory scales linearly with input size."""
        data = secrets.token_bytes(input_size)

        b64url_encode(data)
        gc.collect()

        allocated, encoded = measure_allocation(lambda: b64url_encode(data))

        expected_output = (input_size * 4 + 2) // 3
        max_expected = int(input_size * MAX_BASE64_ENCODE_RATIO) + MAX_BASE64_ENCODE_FIXED
        assert allocated < max_expected, f"Allocated {allocated} bytes, expected < {max_expected}"
        assert len(encoded) == expected_output or len(encoded) == expected_output - 1

    @pytest.mark.parametrize("input_size", [1024, 64 * 1024, 256 * 1024], ids=["1KB", "64KB", "256KB"])
    def test_decode_memory_proportional(self, input_size: int) -> None:
        """b64url_decode memory scales linearly with input size."""
        data = secrets.token_bytes(input_size)
        encoded = b64url_encode(data)

        b64url_decode(encoded)
        gc.collect()

        allocated, decoded = measure_allocation(lambda: b64url_decode(encoded))

        max_expected = int(input_size * MAX_BASE64_DECODE_RATIO) + MAX_BASE64_DECODE_FIXED
        assert allocated < max_expected, f"Allocated {allocated} bytes, expected < {max_expected}"
        assert bytes(decoded) == data

    def test_decode_returns_memoryview(self) -> None:
        """b64url_decode returns memoryview type."""
        data = secrets.token_bytes(64)
        encoded = b64url_encode(data)
        decoded = b64url_decode(encoded)

        assert isinstance(decoded, memoryview), f"Expected memoryview, got {type(decoded)}"
        assert bytes(decoded) == data

    def test_decode_memoryview_enables_zero_copy_slicing(self) -> None:
        """Slicing memoryview result doesn't copy underlying buffer data."""
        counter_bytes = (42).to_bytes(4, "big")
        ciphertext = secrets.token_bytes(64 * 1024)
        payload = counter_bytes + ciphertext
        encoded = b64url_encode(payload)

        decoded = b64url_decode(encoded)
        counter_slice = decoded[:4]
        ciphertext_slice = decoded[4:]

        assert isinstance(counter_slice, memoryview)
        assert isinstance(ciphertext_slice, memoryview)
        assert sys.getsizeof(ciphertext_slice) < MAX_MEMORYVIEW_SLICE_SIZE, "Slice should be a view, not a copy"
        assert int.from_bytes(counter_slice, "big") == 42
        assert bytes(ciphertext_slice) == ciphertext

    def test_decode_slice_vs_bytes_slice_comparison(self) -> None:
        """memoryview slice allocates less than bytes slice."""
        payload = secrets.token_bytes(64 * 1024)
        encoded = b64url_encode(payload)

        mv_decoded = b64url_decode(encoded)
        bytes_decoded = bytes(mv_decoded)
        gc.collect()

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        _ = bytes_decoded[:4]
        _ = bytes_decoded[4:]
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        diff = snapshot2.compare_to(snapshot1, "lineno")
        bytes_alloc = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        gc.collect()

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        _ = mv_decoded[:4]
        _ = mv_decoded[4:]
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()
        diff = snapshot2.compare_to(snapshot1, "lineno")
        mv_alloc = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        assert mv_alloc < bytes_alloc, f"memoryview ({mv_alloc}) should allocate less than bytes ({bytes_alloc})"

    def test_encode_accepts_memoryview_input(self) -> None:
        """b64url_encode accepts memoryview input."""
        data = bytearray(secrets.token_bytes(1024))
        mv = memoryview(data)
        encoded = b64url_encode(mv)
        assert b64url_encode(bytes(data)) == encoded

    @pytest.mark.parametrize("size", [32, 4, 64], ids=["32B-key", "4B-salt", "64B-header"])
    def test_small_payload_overhead_reasonable(self, size: int) -> None:
        """Small payloads have reasonable overhead."""
        data = secrets.token_bytes(size)

        b64url_encode(data)
        gc.collect()

        allocated, encoded = measure_allocation(lambda: b64url_encode(data))
        assert allocated < MAX_BASE64_SMALL_OVERHEAD, f"Allocated {allocated} bytes for {size}-byte payload"

        b64url_decode(encoded)
        gc.collect()

        allocated, decoded = measure_allocation(lambda: b64url_decode(encoded))
        assert allocated < MAX_BASE64_SMALL_OVERHEAD, f"Decode allocated {allocated} bytes for {size}-byte payload"
        assert bytes(decoded) == data


# =============================================================================
# SCALING BEHAVIOR TESTS
# =============================================================================


class TestScalingBehavior:
    """Tests that operations scale linearly, not quadratically."""

    def test_encrypt_scales_linearly(
        self, perf_keypair: tuple[bytes, bytes], perf_psk: bytes, perf_psk_id: bytes
    ) -> None:
        """Encryption time scales linearly with payload size."""
        _, pk = perf_keypair

        small_size = 100 * 1024
        small_payload = os.urandom(small_size)

        def encrypt_small() -> bytes:
            return RequestEncryptor(pk, perf_psk, perf_psk_id).encrypt_all(small_payload)

        small_time, _ = measure_cpu_time(encrypt_small, iterations=5)

        large_size = 1024 * 1024
        large_payload = os.urandom(large_size)

        def encrypt_large() -> bytes:
            return RequestEncryptor(pk, perf_psk, perf_psk_id).encrypt_all(large_payload)

        large_time, _ = measure_cpu_time(encrypt_large, iterations=5)

        size_ratio = large_size / small_size
        time_ratio = large_time / small_time if small_time > 0 else float("inf")
        max_allowed = size_ratio * MAX_TIME_SCALING_FACTOR

        assert time_ratio < max_allowed, (
            f"Encryption scaled {time_ratio:.1f}x for {size_ratio:.0f}x payload, expected < {max_allowed:.0f}x"
        )

    def test_decrypt_scales_linearly(
        self, perf_keypair: tuple[bytes, bytes], perf_psk: bytes, perf_psk_id: bytes
    ) -> None:
        """Decryption time scales linearly with payload size."""
        sk, pk = perf_keypair

        small_size = 100 * 1024
        small_payload = os.urandom(small_size)
        enc_small = RequestEncryptor(pk, perf_psk, perf_psk_id)
        ct_small = enc_small.encrypt_all(small_payload)
        headers_small = enc_small.get_headers()

        def decrypt_small() -> bytes:
            return RequestDecryptor(headers_small, sk, perf_psk, perf_psk_id).decrypt_all(ct_small)

        small_time, _ = measure_cpu_time(decrypt_small, iterations=5)

        large_size = 1024 * 1024
        large_payload = os.urandom(large_size)
        enc_large = RequestEncryptor(pk, perf_psk, perf_psk_id)
        ct_large = enc_large.encrypt_all(large_payload)
        headers_large = enc_large.get_headers()

        def decrypt_large() -> bytes:
            return RequestDecryptor(headers_large, sk, perf_psk, perf_psk_id).decrypt_all(ct_large)

        large_time, _ = measure_cpu_time(decrypt_large, iterations=5)

        size_ratio = large_size / small_size
        time_ratio = large_time / small_time if small_time > 0 else float("inf")
        max_allowed = size_ratio * MAX_TIME_SCALING_FACTOR

        assert time_ratio < max_allowed, (
            f"Decryption scaled {time_ratio:.1f}x for {size_ratio:.0f}x payload, expected < {max_allowed:.0f}x"
        )

    def test_memory_scales_linearly(
        self, perf_keypair: tuple[bytes, bytes], perf_psk: bytes, perf_psk_id: bytes
    ) -> None:
        """Memory usage scales linearly with payload size."""
        _, pk = perf_keypair

        small_size = 100 * 1024
        small_payload = os.urandom(small_size)

        def encrypt_small() -> bytes:
            return RequestEncryptor(pk, perf_psk, perf_psk_id).encrypt_all(small_payload)

        encrypt_small()
        gc.collect()
        small_mem, _ = measure_peak_memory(encrypt_small)

        large_size = 1024 * 1024
        large_payload = os.urandom(large_size)

        def encrypt_large() -> bytes:
            return RequestEncryptor(pk, perf_psk, perf_psk_id).encrypt_all(large_payload)

        encrypt_large()
        gc.collect()
        large_mem, _ = measure_peak_memory(encrypt_large)

        size_ratio = large_size / small_size
        mem_ratio = large_mem / small_mem if small_mem > 0 else float("inf")
        max_allowed = size_ratio * MAX_MEMORY_SCALING_FACTOR

        assert mem_ratio < max_allowed, (
            f"Memory scaled {mem_ratio:.1f}x for {size_ratio:.0f}x payload, expected < {max_allowed:.0f}x"
        )


# =============================================================================
# RELATIVE OVERHEAD TESTS
# =============================================================================


class TestRelativeOverhead:
    """Tests for relative performance characteristics."""

    def test_response_encrypt_faster_than_request(
        self, perf_keypair: tuple[bytes, bytes], perf_psk: bytes, perf_psk_id: bytes
    ) -> None:
        """Response encryption is faster than request (no key exchange)."""
        sk, pk = perf_keypair
        payload = os.urandom(512 * 1024)

        def request_encrypt() -> bytes:
            return RequestEncryptor(pk, perf_psk, perf_psk_id).encrypt_all(payload)

        request_time, _ = measure_cpu_time(request_encrypt, iterations=5)

        req_enc = RequestEncryptor(pk, perf_psk, perf_psk_id)
        req_dec = RequestDecryptor(req_enc.get_headers(), sk, perf_psk, perf_psk_id)
        req_dec.decrypt_all(req_enc.encrypt_all(b"setup"))

        def response_encrypt() -> bytes:
            return ResponseEncryptor(req_dec.context).encrypt_all(payload)

        response_time, _ = measure_cpu_time(response_encrypt, iterations=5)

        # Allow 10% tolerance for measurement noise in CI
        assert response_time < request_time * 1.1, (
            f"Response ({response_time * 1000:.2f}ms) should be faster than request ({request_time * 1000:.2f}ms)"
        )

    def test_response_decrypt_faster_than_request(
        self, perf_keypair: tuple[bytes, bytes], perf_psk: bytes, perf_psk_id: bytes
    ) -> None:
        """Response decryption is faster than request (no key exchange)."""
        sk, pk = perf_keypair
        payload = os.urandom(512 * 1024)

        req_enc = RequestEncryptor(pk, perf_psk, perf_psk_id)
        req_ct = req_enc.encrypt_all(payload)
        req_headers = req_enc.get_headers()

        def request_decrypt() -> bytes:
            return RequestDecryptor(req_headers, sk, perf_psk, perf_psk_id).decrypt_all(req_ct)

        request_time, _ = measure_cpu_time(request_decrypt, iterations=5)

        req_dec = RequestDecryptor(req_headers, sk, perf_psk, perf_psk_id)
        req_dec.decrypt_all(req_ct)
        resp_enc = ResponseEncryptor(req_dec.context)
        resp_ct = resp_enc.encrypt_all(payload)
        resp_headers = resp_enc.get_headers()

        def response_decrypt() -> bytes:
            return ResponseDecryptor(resp_headers, req_enc.context).decrypt_all(resp_ct)

        response_time, _ = measure_cpu_time(response_decrypt, iterations=5)

        # Allow 10% tolerance for measurement noise in CI
        assert response_time < request_time * 1.1, (
            f"Response ({response_time * 1000:.2f}ms) should be faster than request ({request_time * 1000:.2f}ms)"
        )

    def test_setup_dominates_small_payloads(
        self, perf_keypair: tuple[bytes, bytes], perf_psk: bytes, perf_psk_id: bytes
    ) -> None:
        """HPKE setup is significant portion of time for small payloads."""
        _, pk = perf_keypair
        small_payload = os.urandom(1024)

        def full_encrypt() -> bytes:
            return RequestEncryptor(pk, perf_psk, perf_psk_id).encrypt_all(small_payload)

        full_time, _ = measure_cpu_time(full_encrypt, iterations=10)

        def setup_only() -> RequestEncryptor:
            return RequestEncryptor(pk, perf_psk, perf_psk_id)

        setup_time, _ = measure_cpu_time(setup_only, iterations=10)

        setup_ratio = setup_time / full_time if full_time > 0 else 0
        assert setup_ratio > MIN_SETUP_RATIO_SMALL, (
            f"HPKE setup is only {setup_ratio * 100:.1f}% for 1KB, expected >{MIN_SETUP_RATIO_SMALL * 100:.0f}%"
        )

    def test_crypto_dominates_large_payloads(
        self, perf_keypair: tuple[bytes, bytes], perf_psk: bytes, perf_psk_id: bytes
    ) -> None:
        """Crypto dominates for large payloads (setup is small fraction)."""
        _, pk = perf_keypair
        large_payload = os.urandom(5 * 1024 * 1024)

        def full_encrypt() -> bytes:
            return RequestEncryptor(pk, perf_psk, perf_psk_id).encrypt_all(large_payload)

        full_time, _ = measure_cpu_time(full_encrypt, iterations=3)

        def setup_only() -> RequestEncryptor:
            return RequestEncryptor(pk, perf_psk, perf_psk_id)

        setup_time, _ = measure_cpu_time(setup_only, iterations=10)

        setup_ratio = setup_time / full_time if full_time > 0 else 1
        assert setup_ratio < MAX_SETUP_RATIO_LARGE, (
            f"HPKE setup is {setup_ratio * 100:.1f}% for 5MB, expected <{MAX_SETUP_RATIO_LARGE * 100:.0f}%"
        )


# =============================================================================
# MEMORY LEAK TESTS
# =============================================================================


class TestMemoryLeaks:
    """Tests for memory leaks over repeated operations."""

    def test_hpke_roundtrip_no_leak(self) -> None:
        """Repeated HPKE seal/open doesn't leak memory."""
        sk_r, pk_r = generate_keypair()
        psk = make_psk()
        plaintext = secrets.token_bytes(1024)

        sender_ctx = setup_sender_psk(pk_r, b"info", psk, b"tenant")
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"info", psk, b"tenant")

        for i in range(10):
            ct = sender_ctx.seal(f"aad-{i}".encode(), plaintext)
            recipient_ctx.open(f"aad-{i}".encode(), ct)

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

        assert net_allocated < MAX_LEAK_HPKE_100, (
            f"Net allocation {net_allocated} bytes after 100 ops, expected < {MAX_LEAK_HPKE_100}"
        )

    def test_encrypt_decrypt_no_leak(
        self, perf_keypair: tuple[bytes, bytes], perf_psk: bytes, perf_psk_id: bytes
    ) -> None:
        """Repeated RequestEncryptor/Decryptor doesn't leak memory."""
        sk, pk = perf_keypair
        plaintext = secrets.token_bytes(10 * 1024)

        for _ in range(20):
            enc = RequestEncryptor(pk, perf_psk, perf_psk_id)
            ct = enc.encrypt_all(plaintext)
            dec = RequestDecryptor(enc.get_headers(), sk, perf_psk, perf_psk_id)
            dec.decrypt_all(ct)
        gc.collect()

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(100):
            enc = RequestEncryptor(pk, perf_psk, perf_psk_id)
            ct = enc.encrypt_all(plaintext)
            dec = RequestDecryptor(enc.get_headers(), sk, perf_psk, perf_psk_id)
            pt = dec.decrypt_all(ct)
            assert pt == plaintext

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        net_allocated = sum(stat.size_diff for stat in diff)

        max_kb = MAX_LEAK_CORE_100 // 1024
        assert net_allocated < MAX_LEAK_CORE_100, (
            f"Memory grew by {net_allocated / 1024:.1f}KB after 100 roundtrips, expected < {max_kb}KB"
        )

    def test_sse_streaming_no_leak(self) -> None:
        """1000 SSE encrypt/decrypt roundtrips don't leak memory."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session)
        decryptor = ChunkDecryptor(session)
        chunk = b"event: test\ndata: {}\n\n"

        for _ in range(100):
            encrypted = encryptor.encrypt(chunk)
            data_field = extract_sse_data_field(encrypted)
            decryptor.decrypt(data_field)
        gc.collect()

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

        assert net_allocated < MAX_LEAK_SSE_1000, (
            f"Net allocation {net_allocated} bytes after 1000 roundtrips, expected < {MAX_LEAK_SSE_1000}"
        )

    def test_streaming_chunk_no_leak(self, streaming_session: StreamingSession) -> None:
        """Repeated streaming encrypt/decrypt doesn't leak memory."""
        chunk = secrets.token_bytes(CHUNK_SIZE)

        enc = ChunkEncryptor(streaming_session, format=RawFormat(), compress=False)
        dec = ChunkDecryptor(streaming_session, format=RawFormat())
        for _ in range(20):
            ct = enc.encrypt(chunk)
            dec.decrypt(ct)
        gc.collect()

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(500):
            ct = enc.encrypt(chunk)
            pt = dec.decrypt(ct)
            assert pt == chunk

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        net_allocated = sum(stat.size_diff for stat in diff)

        max_kb = MAX_LEAK_STREAMING_500 // 1024
        assert net_allocated < MAX_LEAK_STREAMING_500, (
            f"Memory grew by {net_allocated / 1024:.1f}KB after 500 chunks, expected < {max_kb}KB"
        )

    def test_base64_roundtrip_no_leak(self) -> None:
        """Repeated base64 encode/decode doesn't leak memory."""
        data = secrets.token_bytes(1024)

        for _ in range(100):
            encoded = b64url_encode(data)
            decoded = b64url_decode(encoded)
            assert bytes(decoded) == data
        gc.collect()

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

        assert net_allocated < MAX_LEAK_BASE64_1000, (
            f"Net allocation {net_allocated} bytes after 1000 roundtrips, expected < {MAX_LEAK_BASE64_1000}"
        )

    def test_many_small_operations_stable(self) -> None:
        """Memory stays stable across many small operations."""
        sk_r, pk_r = generate_keypair()
        psk = make_psk()
        plaintext = secrets.token_bytes(64)

        sender_ctx = setup_sender_psk(pk_r, b"info", psk, b"tenant")
        recipient_ctx = setup_recipient_psk(sender_ctx.enc, sk_r, b"info", psk, b"tenant")

        for i in range(100):
            ct = sender_ctx.seal(f"warmup-{i}".encode(), plaintext)
            pt = recipient_ctx.open(f"warmup-{i}".encode(), ct)
            assert pt == plaintext
        gc.collect()

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for i in range(1000):
            ct = sender_ctx.seal(f"aad-{i}".encode(), plaintext)
            pt = recipient_ctx.open(f"aad-{i}".encode(), ct)
            assert pt == plaintext

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        net_allocated = sum(stat.size_diff for stat in diff)

        assert net_allocated < MAX_LEAK_SMALL_OPS_1000, (
            f"Net allocation {net_allocated} bytes after 1000 ops, expected < {MAX_LEAK_SMALL_OPS_1000}"
        )

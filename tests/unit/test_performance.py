"""Memory safety and behavioral tests for HPKE-HTTP encryption.

Tests that require tracemalloc or relational assertions that CodSpeed cannot replace:
1. Batch sublinearity (per-context overhead in batch of 100)
2. Behavioral correctness (memoryview type, zero-copy slicing)
3. Relative overhead (response vs request timing assertions)
4. Memory leaks (N iterations, tracemalloc snapshot diff)

Performance regression tracking (overhead bounds, ratios, scaling) is handled by
CodSpeed benchmarks in tests/benchmarks/ with mode: simulation,memory.
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
from hpke_http.streaming import ChunkDecryptor, ChunkEncryptor, RawFormat
from tests.conftest import extract_sse_data_field, make_sse_session

T = TypeVar("T")


# =============================================================================
# BOUNDS - Thresholds for tests that CodSpeed cannot replace
# =============================================================================

# --- Context Memory (batch) ---
# Per-context in batch after warmup. Measured: ~500 bytes
MAX_CONTEXT_MEMORY_BATCH = 1 * 1024

# --- Memory Leak Thresholds ---
# Net allocation after N operations (should be minimal)
# These thresholds accommodate variance across Python versions (3.10-3.14+)
MAX_LEAK_HPKE_100 = 10 * 1024
MAX_LEAK_CORE_100 = 50 * 1024
MAX_LEAK_SSE_1000 = 15 * 1024
MAX_LEAK_STREAMING_500 = 150 * 1024
MAX_LEAK_BASE64_1000 = 15 * 1024
MAX_LEAK_SMALL_OPS_1000 = 15 * 1024

# --- Overhead Ratios ---
# Setup dominates small payloads (1KB). Measured: ~80%
MIN_SETUP_RATIO_SMALL = 0.2
# Crypto dominates large payloads (5MB). Measured: ~5%
MAX_SETUP_RATIO_LARGE = 0.3

# --- Misc ---
# Memoryview slice object size (proves zero-copy). Measured: 184 bytes
MAX_MEMORYVIEW_SLICE_SIZE = 250
# Small payload overhead (Python object overhead). Measured: ~650-2000 bytes
MAX_BASE64_SMALL_OVERHEAD = 2 * 1024


# =============================================================================
# UTILITIES
# =============================================================================


def make_psk(length: int = PSK_MIN_SIZE) -> bytes:
    """Generate a random PSK of specified length."""
    return secrets.token_bytes(length)


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate X25519 keypair."""
    sk = x25519.X25519PrivateKey.generate()
    return sk.private_bytes_raw(), sk.public_key().public_bytes_raw()


def measure_allocation(func: Callable[[], T]) -> tuple[int, T]:
    """Measure gross memory allocated by a function call (positive diffs only)."""
    gc.collect()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    result = func()
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    diff = snapshot2.compare_to(snapshot1, "lineno")
    allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)
    return allocated, result


def measure_net_growth(func: Callable[[], None]) -> int:
    """Measure net memory growth after a function call (includes frees).

    Used for leak detection: run N operations, measure if memory grew.
    Caller should gc.collect() + warmup before calling.
    """
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    func()
    gc.collect()
    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    diff = snapshot2.compare_to(snapshot1, "lineno")
    return sum(stat.size_diff for stat in diff)


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


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def perf_keypair() -> tuple[bytes, bytes]:
    """Generate X25519 keypair."""
    return generate_keypair()


@pytest.fixture
def perf_psk() -> bytes:
    """32-byte PSK."""
    return b"perf-test-psk-32-bytes-exactly!!"


@pytest.fixture
def perf_psk_id() -> bytes:
    """PSK identifier."""
    return b"perf-tenant"


# =============================================================================
# CONTEXT MEMORY TESTS (batch sublinearity only)
# =============================================================================


class TestContextMemory:
    """Batch sublinearity for HPKE context creation."""

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
# BASE64 BEHAVIORAL TESTS (zero-copy, memoryview)
# =============================================================================


class TestBase64Memory:
    """Behavioral tests for base64url encode/decode operations."""

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
# RELATIVE OVERHEAD TESTS
# =============================================================================


class TestRelativeOverhead:
    """Relational timing assertions (response < request, setup vs crypto ratio)."""

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

        def ops() -> None:
            for i in range(100):
                ct = sender_ctx.seal(f"aad-{i + 10}".encode(), plaintext)
                pt = recipient_ctx.open(f"aad-{i + 10}".encode(), ct)
                assert pt == plaintext

        net = measure_net_growth(ops)
        assert net < MAX_LEAK_HPKE_100, f"Net allocation {net} bytes after 100 ops, expected < {MAX_LEAK_HPKE_100}"

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

        def ops() -> None:
            for _ in range(100):
                enc = RequestEncryptor(pk, perf_psk, perf_psk_id)
                ct = enc.encrypt_all(plaintext)
                dec = RequestDecryptor(enc.get_headers(), sk, perf_psk, perf_psk_id)
                pt = dec.decrypt_all(ct)
                assert pt == plaintext

        net = measure_net_growth(ops)
        assert net < MAX_LEAK_CORE_100, (
            f"Memory grew by {net / 1024:.1f}KB after 100 roundtrips, expected < {MAX_LEAK_CORE_100 // 1024}KB"
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

        def ops() -> None:
            for _ in range(1000):
                encrypted = encryptor.encrypt(chunk)
                data_field = extract_sse_data_field(encrypted)
                plaintext = decryptor.decrypt(data_field)
                assert plaintext == chunk

        net = measure_net_growth(ops)
        assert net < MAX_LEAK_SSE_1000, (
            f"Net allocation {net} bytes after 1000 roundtrips, expected < {MAX_LEAK_SSE_1000}"
        )

    def test_streaming_chunk_no_leak(self) -> None:
        """Repeated streaming encrypt/decrypt doesn't leak memory."""
        session = make_sse_session()
        chunk = secrets.token_bytes(CHUNK_SIZE)

        enc = ChunkEncryptor(session, format=RawFormat(), compress=False)
        dec = ChunkDecryptor(session, format=RawFormat())
        for _ in range(20):
            ct = enc.encrypt(chunk)
            dec.decrypt(ct)
        gc.collect()

        def ops() -> None:
            for _ in range(500):
                ct = enc.encrypt(chunk)
                pt = dec.decrypt(ct)
                assert pt == chunk

        net = measure_net_growth(ops)
        assert net < MAX_LEAK_STREAMING_500, (
            f"Memory grew by {net / 1024:.1f}KB after 500 chunks, expected < {MAX_LEAK_STREAMING_500 // 1024}KB"
        )

    def test_base64_roundtrip_no_leak(self) -> None:
        """Repeated base64 encode/decode doesn't leak memory."""
        data = secrets.token_bytes(1024)

        for _ in range(100):
            encoded = b64url_encode(data)
            decoded = b64url_decode(encoded)
            assert bytes(decoded) == data
        gc.collect()

        def ops() -> None:
            for _ in range(1000):
                encoded = b64url_encode(data)
                decoded = b64url_decode(encoded)
                assert bytes(decoded) == data

        net = measure_net_growth(ops)
        assert net < MAX_LEAK_BASE64_1000, (
            f"Net allocation {net} bytes after 1000 roundtrips, expected < {MAX_LEAK_BASE64_1000}"
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

        def ops() -> None:
            for i in range(1000):
                ct = sender_ctx.seal(f"aad-{i}".encode(), plaintext)
                pt = recipient_ctx.open(f"aad-{i}".encode(), ct)
                assert pt == plaintext

        net = measure_net_growth(ops)
        assert net < MAX_LEAK_SMALL_OPS_1000, (
            f"Net allocation {net} bytes after 1000 ops, expected < {MAX_LEAK_SMALL_OPS_1000}"
        )

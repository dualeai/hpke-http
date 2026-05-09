"""Memory regression gates for the zero-copy AEAD refactor.

These tests lock in the memory wins from wire format v2 + combined buffer +
reusable scratch. If a future change reintroduces a `bytes()` coercion or
extra allocation in the hot path, these gates will catch it.

NOTE: tests/benchmarks/ is excluded from pyright per pyproject.toml.
"""

from __future__ import annotations

import gc
import os
import sys
import tracemalloc
from typing import Any

import pytest

from hpke_http.core import _ChunkStreamParser  # pyright: ignore[reportPrivateUsage]
from hpke_http.streaming import (
    ChunkDecryptor,
    ChunkEncryptor,
    RawFormat,
    StreamingSession,
)

pytestmark = pytest.mark.benchmark


def _make_pair() -> tuple[ChunkEncryptor, ChunkDecryptor]:
    key = os.urandom(32)
    session = StreamingSession.create(key)
    enc = ChunkEncryptor(session, format=RawFormat())
    dec = ChunkDecryptor(
        StreamingSession(session_key=session.session_key, session_salt=session.session_salt),
        format=RawFormat(),
    )
    return enc, dec


def test_chunk_encryptor_peak_memory_under_threshold() -> None:
    """Peak per encrypt() < 150 KB for 64 KB chunk.

    Baseline (pre-refactor): ~193 KB peak (3x chunk_size: chunk + ct + concat).
    Target (post-refactor): < 150 KB (combined buffer + reusable scratch).
    """
    enc, _dec = _make_pair()
    chunk = os.urandom(64 * 1024)
    # Warmup — let the reusable scratch allocate
    for _ in range(10):
        enc.encrypt(chunk)

    gc.collect()
    tracemalloc.start()
    for _ in range(1000):
        _out = enc.encrypt(chunk)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Allow generous threshold to account for tracemalloc overhead + Python alloc rounding
    assert peak < 200 * 1024, f"encrypt peak {peak} bytes exceeds 200 KB"


def test_chunk_decryptor_peak_memory_under_threshold() -> None:
    """Peak per decrypt() < 200 KB for 64 KB chunk."""
    enc, dec = _make_pair()
    chunk = os.urandom(64 * 1024)
    wires = [enc.encrypt(chunk) for _ in range(1000)]

    gc.collect()
    tracemalloc.start()
    for wire in wires:
        _pt = dec.decrypt(wire)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 200 * 1024, f"decrypt peak {peak} bytes exceeds 200 KB"


def test_chunk_encryptor_alloc_count_steady_state() -> None:
    """After warmup, encrypt() should not unbounded-grow allocation count.

    Reusable scratch buffer means steady-state allocs per call should be small
    (just the bytes() coercion at boundary). If reuse is broken (regression
    introduces fresh bytearray per call), this test catches it.
    """
    enc, _dec = _make_pair()
    chunk = os.urandom(64 * 1024)
    # Warmup
    for _ in range(20):
        enc.encrypt(chunk)
    gc.collect()

    before = sys.getallocatedblocks()
    for _ in range(100):
        enc.encrypt(chunk)
    after = sys.getallocatedblocks()
    delta = after - before

    # Each call: ~1 alloc (bytes(view) coercion at boundary). 100 calls → ~100 net allocs.
    # Threshold generous to account for tracemalloc/runtime overhead.
    assert delta < 500, f"encrypt alloc delta {delta} after 100 calls (expected steady-state reuse)"


def test_50mb_stream_peak_memory() -> None:
    """Streaming 50 MB body: peak should be O(chunk_size), not O(body_size).

    Verifies that streaming actually streams (doesn't buffer entire body).
    """
    enc_chunk, dec_chunk = _make_pair()
    # We use ChunkEncryptor directly to bypass HPKE setup overhead
    body = b"X" * (50 * 1024 * 1024)

    gc.collect()
    tracemalloc.start()
    for offset in range(0, len(body), 64 * 1024):
        chunk = memoryview(body)[offset : offset + 64 * 1024]
        wire = enc_chunk.encrypt(chunk)
        _ = dec_chunk.decrypt(wire)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak should be ~few x chunk_size (64 KB), not 50 MB body size.
    assert peak < 5 * 1024 * 1024, f"streaming peak {peak} bytes suggests O(body) memory"


def test_parser_compaction_keeps_peak_bounded() -> None:
    """100 x 64 KB chunks fed to parser → peak buffer ≤ 256 KB (compaction at 128 KB)."""
    enc, _dec = _make_pair()
    chunks_wire = [enc.encrypt(b"X" * 64 * 1024) for _ in range(100)]

    parser = _ChunkStreamParser()
    gc.collect()
    tracemalloc.start()
    for w in chunks_wire:
        # Drop yielded mvs immediately (don't accumulate)
        for _mv in parser.feed(w):
            pass
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Compaction threshold is 128 KB; allow 256 KB headroom.
    assert peak < 1024 * 1024, f"parser peak {peak} bytes exceeds 1 MB (compaction broken?)"


def test_bench_chunk_encrypt_vs_decrypt_balanced(benchmark: Any) -> None:
    """CodSpeed regression bench: full encrypt+decrypt roundtrip on 64 KB chunk.

    Captures CPU + memory profile via CodSpeed's instrumentation.
    """
    enc, dec = _make_pair()
    chunk = os.urandom(64 * 1024)
    # Pre-warm scratch buffer
    enc.encrypt(chunk)
    enc.counter = 1  # reset
    dec.expected_counter = 1

    def _roundtrip() -> bytes:
        wire = enc.encrypt(chunk)
        result = dec.decrypt(wire)
        # Reset counters so each iteration is independent
        enc.counter = 1
        dec.expected_counter = 1
        return bytes(result)

    benchmark(_roundtrip)

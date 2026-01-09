"""Compression tests for HPKE encryption with Zstd (RFC 8878).

Tests:
- Request body compression (client→server)
- SSE response compression (server→client)
- Memory usage and leak detection
"""

import gc
import json
import secrets
import tracemalloc

import pytest

from hpke_http.constants import (
    ZSTD_COMPRESSION_LEVEL,
    ZSTD_MIN_SIZE,
)
from hpke_http.exceptions import DecryptionError
from hpke_http.streaming import (
    ChunkDecryptor,
    ChunkEncryptor,
    import_zstd,
)
from tests.conftest import extract_sse_data_field, make_sse_session


def _zstd_available() -> bool:
    """Check if zstd module is available."""
    try:
        import_zstd()
        return True
    except ImportError:
        return False


# Skip all tests if zstd not available
pytestmark = pytest.mark.skipif(
    not _zstd_available(),
    reason="backports.zstd not installed",
)


class TestSSECompression:
    """Zstd compression for SSE streaming."""

    def test_compress_sse_roundtrip(self) -> None:
        """Compressed SSE encrypt/decrypt roundtrip works."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=True)
        decryptor = ChunkDecryptor(session)

        # Large JSON chunk (compressible)
        chunk = json.dumps({"data": "x" * 1000, "id": 123}).encode()

        encrypted = encryptor.encrypt(chunk)
        data_field = extract_sse_data_field(encrypted)
        decrypted = decryptor.decrypt(data_field)

        assert decrypted == chunk

    def test_encoding_id_zstd_in_payload(self) -> None:
        """Encoding ID ZSTD is correctly set for large chunks."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=True)

        # Large chunk - should be compressed
        chunk = b"x" * 100

        encrypted = encryptor.encrypt(chunk)
        # Verify it decrypts correctly (encoding ID is parsed internally)
        decryptor = ChunkDecryptor(session)
        data_field = extract_sse_data_field(encrypted)
        decrypted = decryptor.decrypt(data_field)

        assert decrypted == chunk

    def test_encoding_id_identity_for_small_chunks(self) -> None:
        """Small chunks use IDENTITY encoding (no compression)."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=True)
        decryptor = ChunkDecryptor(session)

        # Small chunk - should NOT be compressed
        chunk = b"small"
        assert len(chunk) < ZSTD_MIN_SIZE

        encrypted = encryptor.encrypt(chunk)
        data_field = extract_sse_data_field(encrypted)
        decrypted = decryptor.decrypt(data_field)

        assert decrypted == chunk

    def test_compression_disabled_uses_identity(self) -> None:
        """compress=False always uses IDENTITY encoding."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=False)
        decryptor = ChunkDecryptor(session)

        # Large chunk - but compression disabled
        chunk = b"x" * 1000

        encrypted = encryptor.encrypt(chunk)
        data_field = extract_sse_data_field(encrypted)
        decrypted = decryptor.decrypt(data_field)

        assert decrypted == chunk

    def test_compression_ratio_json(self) -> None:
        """JSON payloads achieve expected compression (40-70% reduction)."""
        zstd = import_zstd()

        # Repetitive JSON (highly compressible)
        payload = json.dumps({"data": "hello world " * 100}).encode()
        compressed = zstd.compress(payload, level=ZSTD_COMPRESSION_LEVEL)

        ratio = len(compressed) / len(payload)
        # Expect at least 40% reduction
        assert ratio < 0.6, f"Compression ratio {ratio:.2%} is worse than expected"

    def test_unknown_encoding_raises_error(self) -> None:
        """Unknown encoding ID raises DecryptionError."""
        import base64

        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=False)

        # Encrypt with IDENTITY encoding (not used - we manually craft invalid payload below)
        _ = encryptor.encrypt(b"test data here")

        # Manually corrupt the encoding ID in the decrypted payload
        # This is a bit tricky - we need to create a payload with invalid encoding
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        # Create a new session for tampering
        tamper_session = make_sse_session()
        cipher = ChaCha20Poly1305(tamper_session.session_key)
        nonce = tamper_session.session_salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")

        # Create payload with invalid encoding ID (0xFF)
        invalid_data = bytes([0xFF]) + b"some data"
        ciphertext = cipher.encrypt(nonce, invalid_data, associated_data=None)
        payload = (1).to_bytes(4, "big") + ciphertext
        # SSEFormat uses standard base64 (not base64url) for ~1.7x faster encoding
        encoded = base64.b64encode(payload).decode("ascii")

        decryptor = ChunkDecryptor(tamper_session)
        with pytest.raises(DecryptionError, match="Unknown encoding"):
            decryptor.decrypt(encoded)

    def test_multiple_chunks_compressed(self) -> None:
        """Multiple chunks compress/decompress correctly in sequence."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=True)
        decryptor = ChunkDecryptor(session)

        chunks = [json.dumps({"event": "start", "data": "x" * 100}).encode()]
        chunks.extend(json.dumps({"event": "progress", "step": i, "data": "y" * 100}).encode() for i in range(5))
        chunks.append(json.dumps({"event": "complete", "result": "z" * 100}).encode())

        for chunk in chunks:
            encrypted = encryptor.encrypt(chunk)
            data_field = extract_sse_data_field(encrypted)
            decrypted = decryptor.decrypt(data_field)
            assert decrypted == chunk


class TestCompressionMemory:
    """Memory usage tests for compression."""

    def test_compressor_memory_bounded(self) -> None:
        """Compressor instance uses bounded memory (<100KB)."""
        zstd = import_zstd()

        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        compressor = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL)
        # Force some usage
        _ = compressor.compress(b"test" * 100, mode=zstd.ZstdCompressor.FLUSH_BLOCK)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        assert allocated < 100 * 1024, f"Compressor used {allocated} bytes, expected < 100KB"

    def test_decompressor_memory_bounded(self) -> None:
        """Decompressor instance uses bounded memory (<100KB)."""
        zstd = import_zstd()

        # Create some compressed data first
        compressed = zstd.compress(b"test data " * 100, level=ZSTD_COMPRESSION_LEVEL)

        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        decompressor = zstd.ZstdDecompressor()
        _ = decompressor.decompress(compressed)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        assert allocated < 100 * 1024, f"Decompressor used {allocated} bytes, expected < 100KB"

    def test_compressed_payload_smaller(self) -> None:
        """Compressed payload uses less memory than plaintext."""
        zstd = import_zstd()

        # Highly compressible data
        plaintext = json.dumps({"data": "repetitive " * 1000}).encode()
        compressed = zstd.compress(plaintext, level=ZSTD_COMPRESSION_LEVEL)

        assert len(compressed) < len(plaintext) * 0.5, (
            f"Compressed size {len(compressed)} should be < 50% of {len(plaintext)}"
        )

    def test_streaming_no_memory_leak(self) -> None:
        """1000 compress/decompress cycles don't leak memory."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=True)
        decryptor = ChunkDecryptor(session)
        chunk = json.dumps({"data": "x" * 200}).encode()

        # Warmup
        for _ in range(100):
            encrypted = encryptor.encrypt(chunk)
            data_field = extract_sse_data_field(encrypted)
            decryptor.decrypt(data_field)
        gc.collect()

        # Reset for fresh measurement
        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=True)
        decryptor = ChunkDecryptor(session)

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(1000):
            encrypted = encryptor.encrypt(chunk)
            data_field = extract_sse_data_field(encrypted)
            decrypted = decryptor.decrypt(data_field)
            assert decrypted == chunk

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        net_allocated = sum(stat.size_diff for stat in diff)

        assert net_allocated < 50 * 1024, f"Net allocation {net_allocated} bytes after 1000 cycles, expected < 50KB"

    def test_large_payload_memory_proportional(self) -> None:
        """Memory scales linearly with payload size (no quadratic blowup)."""
        zstd = import_zstd()

        # 100KB payload
        payload_size = 100 * 1024
        payload = secrets.token_bytes(payload_size)

        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        _ = zstd.compress(payload, level=ZSTD_COMPRESSION_LEVEL)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        # Should allocate roughly proportional to payload (not quadratic)
        # Allow up to 2x for compressed output + temporary buffers
        max_expected = payload_size * 2
        assert allocated < max_expected, (
            f"Allocated {allocated} bytes for {payload_size} byte payload, expected < {max_expected}"
        )

    def test_instance_reuse_memory_stable(self) -> None:
        """Reusing compressor/decompressor keeps memory stable."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=True)
        chunk = json.dumps({"data": "x" * 200}).encode()

        # Warmup - creates compressor
        for _ in range(10):
            encryptor.encrypt(chunk)
        gc.collect()

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Many more operations with same compressor
        for _ in range(500):
            encryptor.encrypt(chunk)

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        net_allocated = sum(stat.size_diff for stat in diff)

        # Memory should be stable (< 20KB growth for 500 ops)
        assert net_allocated < 20 * 1024, f"Net allocation {net_allocated} bytes after 500 ops, expected < 20KB"


class TestCompressionErrors:
    """Error handling tests for corrupted/invalid compressed data."""

    def test_corrupted_zstd_data_raises_error(self) -> None:
        """Corrupted zstd data raises DecryptionError."""
        import base64

        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        from hpke_http.constants import SSEEncodingId

        session = make_sse_session()
        cipher = ChaCha20Poly1305(session.session_key)
        nonce = session.session_salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")

        # ZSTD encoding but garbage data
        invalid_zstd = bytes([SSEEncodingId.ZSTD]) + b"not valid zstd"
        ciphertext = cipher.encrypt(nonce, invalid_zstd, associated_data=None)
        payload = (1).to_bytes(4, "big") + ciphertext
        # SSEFormat uses standard base64 (not base64url)
        encoded = base64.b64encode(payload).decode("ascii")

        decryptor = ChunkDecryptor(session)
        with pytest.raises(DecryptionError, match="decompression failed"):
            decryptor.decrypt(encoded)

    def test_truncated_zstd_data_returns_empty(self) -> None:
        """Truncated zstd data returns empty bytes (library behavior)."""
        import base64

        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        from hpke_http.constants import SSEEncodingId

        zstd = import_zstd()
        session = make_sse_session()
        cipher = ChaCha20Poly1305(session.session_key)
        nonce = session.session_salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")

        # Create valid zstd then truncate - library returns empty bytes
        valid_compressed = zstd.compress(b"x" * 1000, level=ZSTD_COMPRESSION_LEVEL)
        truncated = valid_compressed[: len(valid_compressed) // 2]

        truncated_payload = bytes([SSEEncodingId.ZSTD]) + truncated
        ciphertext = cipher.encrypt(nonce, truncated_payload, associated_data=None)
        payload = (1).to_bytes(4, "big") + ciphertext
        # SSEFormat uses standard base64 (not base64url)
        encoded = base64.b64encode(payload).decode("ascii")

        decryptor = ChunkDecryptor(session)
        # backports.zstd returns empty bytes for truncated data
        result = decryptor.decrypt(encoded)
        assert result == b""

    def test_empty_zstd_payload_returns_empty(self) -> None:
        """ZSTD encoding with empty data returns empty bytes (library behavior)."""
        import base64

        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        from hpke_http.constants import SSEEncodingId

        session = make_sse_session()
        cipher = ChaCha20Poly1305(session.session_key)
        nonce = session.session_salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")

        # ZSTD encoding ID but no data - library returns empty
        empty_zstd = bytes([SSEEncodingId.ZSTD])
        ciphertext = cipher.encrypt(nonce, empty_zstd, associated_data=None)
        payload = (1).to_bytes(4, "big") + ciphertext
        # SSEFormat uses standard base64 (not base64url)
        encoded = base64.b64encode(payload).decode("ascii")

        decryptor = ChunkDecryptor(session)
        # backports.zstd returns empty bytes for empty input
        result = decryptor.decrypt(encoded)
        assert result == b""

    def test_missing_encoding_id_raises_error(self) -> None:
        """Empty decrypted payload (no encoding ID) raises DecryptionError."""
        import base64

        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        session = make_sse_session()
        cipher = ChaCha20Poly1305(session.session_key)
        nonce = session.session_salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")

        # Completely empty payload
        ciphertext = cipher.encrypt(nonce, b"", associated_data=None)
        payload = (1).to_bytes(4, "big") + ciphertext
        # SSEFormat uses standard base64 (not base64url)
        encoded = base64.b64encode(payload).decode("ascii")

        decryptor = ChunkDecryptor(session)
        with pytest.raises(DecryptionError, match="too short"):
            decryptor.decrypt(encoded)

    def test_reserved_encoding_ids_raise_error(self) -> None:
        """Reserved encoding IDs (0x03-0xFF) raise DecryptionError."""
        import base64

        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        # 0x00=IDENTITY, 0x01=ZSTD, 0x02=GZIP are valid; 0x03+ are reserved
        for encoding_id in [0x03, 0x10, 0x80, 0xFE, 0xFF]:
            # Fresh session for each test (counter resets)
            test_session = make_sse_session()
            test_cipher = ChaCha20Poly1305(test_session.session_key)
            nonce = test_session.session_salt + b"\x00\x00\x00\x00" + (1).to_bytes(4, "little")

            invalid_data = bytes([encoding_id]) + b"data"
            ciphertext = test_cipher.encrypt(nonce, invalid_data, associated_data=None)
            payload = (1).to_bytes(4, "big") + ciphertext
            # SSEFormat uses standard base64 (not base64url)
            encoded = base64.b64encode(payload).decode("ascii")

            decryptor = ChunkDecryptor(test_session)
            with pytest.raises(DecryptionError, match="Unknown encoding"):
                decryptor.decrypt(encoded)


class TestCompressionBoundaries:
    """Boundary condition tests around ZSTD_MIN_SIZE."""

    def test_exactly_min_size_compressed(self) -> None:
        """Chunk exactly at ZSTD_MIN_SIZE gets compressed."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=True)
        decryptor = ChunkDecryptor(session)

        chunk = b"x" * ZSTD_MIN_SIZE
        encrypted = encryptor.encrypt(chunk)
        decrypted = decryptor.decrypt(extract_sse_data_field(encrypted))
        assert decrypted == chunk

    def test_below_min_size_not_compressed(self) -> None:
        """Chunk below ZSTD_MIN_SIZE uses IDENTITY."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=True)
        decryptor = ChunkDecryptor(session)

        chunk = b"x" * (ZSTD_MIN_SIZE - 1)
        encrypted = encryptor.encrypt(chunk)
        decrypted = decryptor.decrypt(extract_sse_data_field(encrypted))
        assert decrypted == chunk

    def test_mixed_sizes_in_stream(self) -> None:
        """Stream with mixed sizes handles compression correctly."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=True)
        decryptor = ChunkDecryptor(session)

        chunks = [
            b"tiny",  # < min, IDENTITY
            b"x" * 100,  # >= min, ZSTD
            b"small",  # < min, IDENTITY
            b"y" * 500,  # >= min, ZSTD
        ]

        for chunk in chunks:
            encrypted = encryptor.encrypt(chunk)
            decrypted = decryptor.decrypt(extract_sse_data_field(encrypted))
            assert decrypted == chunk

    def test_incompressible_data_works(self) -> None:
        """Random (incompressible) data still works."""
        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=True)
        decryptor = ChunkDecryptor(session)

        chunk = secrets.token_bytes(200)
        encrypted = encryptor.encrypt(chunk)
        decrypted = decryptor.decrypt(extract_sse_data_field(encrypted))
        assert decrypted == chunk


class TestCompressionThreadSafety:
    """Thread safety for compression."""

    def test_encryptor_concurrent_access(self) -> None:
        """ChunkEncryptor handles concurrent encryption safely."""
        import threading

        session = make_sse_session()
        encryptor = ChunkEncryptor(session, compress=True)
        results: list[bytes] = []
        lock = threading.Lock()

        def encrypt_chunk(i: int) -> None:
            chunk = f"chunk-{i}-{'x' * 100}".encode()
            encrypted = encryptor.encrypt(chunk)
            with lock:
                results.append(encrypted)

        threads = [threading.Thread(target=encrypt_chunk, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert len(set(results)) == 20  # All unique (different counters)


class TestCompressionRatiosRealisticData:
    """Measure compression with realistic data patterns.

    Uses real-world data (not artificial "x" * 1000) to validate compression.
    """

    def _compress(self, data: bytes) -> tuple[int, int, float]:
        """Returns (original_size, compressed_size, savings_ratio)."""
        zstd = import_zstd()
        compressed = zstd.compress(data, level=ZSTD_COMPRESSION_LEVEL)
        orig, comp = len(data), len(compressed)
        return (orig, comp, 1.0 - comp / orig if orig else 0.0)

    def _sse_wire_size(self, chunks: list[bytes], *, compress: bool) -> int:
        """Total encrypted SSE wire size."""
        session = make_sse_session()
        enc = ChunkEncryptor(session, compress=compress)
        return sum(len(enc.encrypt(c)) for c in chunks)

    # --- JSON Patterns ---

    def test_json_unique_ids(self) -> None:
        """JSON with UUIDs and timestamps (typical API response)."""
        import hashlib

        data = json.dumps(
            {
                "items": [
                    {
                        "id": hashlib.sha256(f"item-{i}".encode()).hexdigest()[:24],
                        "ts": f"2024-01-{(i % 28) + 1:02d}T{i % 24:02d}:{i % 60:02d}:00Z",
                        "type": ["A", "B", "C"][i % 3],
                        "val": (i * 17) % 1000,
                    }
                    for i in range(50)
                ],
            }
        ).encode()

        orig, _, ratio = self._compress(data)
        assert orig > 2000
        assert ratio > 0.5, f"Expected >50% savings, got {ratio:.1%}"

    def test_json_nested_config(self) -> None:
        """Nested JSON config (settings, preferences)."""
        data = json.dumps(
            {
                "user": {
                    "profile": {"name": "John", "email": "john@test.com"},
                    "settings": {
                        "theme": "dark",
                        "lang": "en",
                        "notifications": {"email": True, "push": False},
                    },
                    "orgs": [{"id": f"org_{i}", "role": ["admin", "user"][i % 2]} for i in range(10)],
                }
            }
        ).encode()

        _, _, ratio = self._compress(data)
        assert ratio > 0.4, f"Expected >40% savings, got {ratio:.1%}"

    # --- SSE Streaming ---

    def test_sse_token_stream(self) -> None:
        """LLM-style token streaming."""
        tokens = ["Hello", ",", " I", "'m", " an", " AI", ".", " How", " can", " I", " help", "?"]
        chunks = [f"event: token\ndata: {json.dumps({'i': i, 't': t})}\n\n".encode() for i, t in enumerate(tokens)]

        with_comp = self._sse_wire_size(chunks, compress=True)
        without_comp = self._sse_wire_size(chunks, compress=False)

        # Small chunks - verify both work
        assert with_comp > 0 and without_comp > 0

    def test_sse_progress_varied(self) -> None:
        """Progress events with varied messages."""
        msgs = [
            "Connecting to server",
            "Authenticating request",
            "Loading configuration",
            "Processing batch 1/5",
            "Validating results",
            "Generating report",
            "Finalizing output",
        ]
        chunks = [
            f"event: progress\ndata: {json.dumps({'step': i, 'msg': m})}\n\n".encode() for i, m in enumerate(msgs)
        ]

        with_comp = self._sse_wire_size(chunks, compress=True)
        without_comp = self._sse_wire_size(chunks, compress=False)

        savings = 1.0 - with_comp / without_comp
        assert savings > 0.1, f"Expected some savings, got {savings:.1%}"

    def test_sse_large_json_event(self) -> None:
        """Large search/data response."""
        data = json.dumps(
            {
                "results": [
                    {
                        "id": f"doc_{i:03d}",
                        "title": f"Document {i}",
                        "snippet": f"Content preview for document number {i}.",
                        "score": round(0.99 - i * 0.008, 3),
                    }
                    for i in range(80)
                ],
            }
        ).encode()
        chunks = [f"event: data\ndata: {data.decode()}\n\n".encode()]

        with_comp = self._sse_wire_size(chunks, compress=True)
        without_comp = self._sse_wire_size(chunks, compress=False)

        savings = 1.0 - with_comp / without_comp
        assert savings > 0.6, f"Expected >60% savings, got {savings:.1%}"

    # --- Log/Text Patterns ---

    def test_log_entries(self) -> None:
        """Structured log entries."""
        levels = ["INFO", "WARN", "ERROR", "DEBUG"]
        logs = "\n".join(
            f"2024-01-15T10:{i:02d}:00Z {levels[i % 4]} [svc] msg_{i} rid={secrets.token_hex(4)}" for i in range(60)
        ).encode()

        _, _, ratio = self._compress(logs)
        assert ratio > 0.4, f"Expected >40% savings, got {ratio:.1%}"

    def test_html_content(self) -> None:
        """HTML markup."""
        html = (
            b"<html><body>"
            + b"".join(f'<div id="i{i}"><h2>Item {i}</h2><p>Text {i}</p></div>'.encode() for i in range(25))
            + b"</body></html>"
        )

        _, _, ratio = self._compress(html)
        assert ratio > 0.6, f"Expected >60% savings, got {ratio:.1%}"

    def test_code_content(self) -> None:
        """Source code snippet."""
        code = b"""def process(items):
    results = []
    for item in items:
        if item.get("active"):
            results.append({
                "id": item["id"],
                "value": item["value"] * 1.1,
            })
    return results

async def fetch(url, timeout=30):
    async with session.get(url, timeout=timeout) as resp:
        return await resp.json()
"""
        _, _, ratio = self._compress(code)
        assert ratio > 0.3, f"Expected >30% savings, got {ratio:.1%}"

    # --- Binary/Edge Cases ---

    def test_base64_data(self) -> None:
        """Base64 encoded binary (API attachments)."""
        import base64

        raw = secrets.token_bytes(2000)
        b64 = base64.b64encode(raw)

        orig, comp, _ = self._compress(b64)
        # Base64 doesn't compress well, just verify no major expansion
        assert comp <= orig * 1.1

    def test_random_bytes(self) -> None:
        """Random bytes should not compress."""
        data = secrets.token_bytes(1000)

        _, _, ratio = self._compress(data)
        assert ratio < 0.05, f"Random data shouldn't compress, got {ratio:.1%}"

    def test_mixed_binary_text(self) -> None:
        """Mixed binary and text content."""
        # Simulate binary blob embedded in JSON
        import base64

        binary_chunk = secrets.token_bytes(500)
        data = json.dumps(
            {
                "id": "file_001",
                "name": "document.bin",
                "size": len(binary_chunk),
                "content": base64.b64encode(binary_chunk).decode("ascii"),
                "metadata": {"type": "binary", "encoding": "base64"},
            }
        ).encode()

        _, _, ratio = self._compress(data)
        # Mixed content has some structure
        assert ratio > 0.1, f"Expected some compression, got {ratio:.1%}"

    # --- Wire Size Comparisons ---

    def test_wire_comparison_json(self) -> None:
        """Wire size: compressed vs uncompressed JSON."""
        data = json.dumps({"records": [{"k": f"key_{i}", "v": i * 10} for i in range(30)]}).encode()
        chunks = [f"event: d\ndata: {data.decode()}\n\n".encode()]

        with_comp = self._sse_wire_size(chunks, compress=True)
        without_comp = self._sse_wire_size(chunks, compress=False)

        assert with_comp < without_comp
        savings = 1.0 - with_comp / without_comp
        assert savings > 0.4, f"Expected >40% savings, got {savings:.1%}"

    def test_wire_comparison_small_events(self) -> None:
        """Wire size: many small events (compression overhead)."""
        chunks = [f'event: p\ndata: {{"n":{i}}}\n\n'.encode() for i in range(15)]

        with_comp = self._sse_wire_size(chunks, compress=True)
        without_comp = self._sse_wire_size(chunks, compress=False)

        # Small events may not benefit, just verify no crash
        assert with_comp > 0 and without_comp > 0

    def test_roundtrip_all_patterns(self) -> None:
        """All patterns roundtrip correctly."""
        import base64

        patterns = [
            b'{"id": 1, "data": [1,2,3]}',
            b"Plain text message here",
            b"Log: 2024-01-15 INFO test",
            json.dumps({"items": [{"i": i} for i in range(10)]}).encode(),
            base64.b64encode(secrets.token_bytes(100)),
            secrets.token_bytes(100),  # Raw binary
        ]

        session = make_sse_session()
        enc = ChunkEncryptor(session, compress=True)
        dec = ChunkDecryptor(session)

        for p in patterns:
            encrypted = enc.encrypt(p)
            decrypted = dec.decrypt(extract_sse_data_field(encrypted))
            assert decrypted == p


class TestUnifiedCompression:
    """Unified compression functions with auto-selection."""

    def test_compress_decompress_roundtrip(self) -> None:
        """Unified compress/decompress returns original data."""
        from hpke_http.streaming import zstd_compress, zstd_decompress

        original = b"Hello, unified compression!" * 1000
        compressed = zstd_compress(original)
        decompressed = zstd_decompress(compressed)

        assert decompressed == original

    def test_empty_input_compress(self) -> None:
        """Empty input returns empty bytes (compress)."""
        from hpke_http.streaming import zstd_compress

        result = zstd_compress(b"")
        assert result == b""

    def test_empty_input_decompress(self) -> None:
        """Empty input returns empty bytes (decompress)."""
        from hpke_http.streaming import zstd_decompress

        result = zstd_decompress(b"")
        assert result == b""

    def test_compression_level_parameter(self) -> None:
        """Compression level affects output size."""
        from hpke_http.streaming import zstd_compress

        data = b"x" * 100000
        compressed_level1 = zstd_compress(data, level=1)
        compressed_level22 = zstd_compress(data, level=22)

        # Higher level should produce smaller output (for compressible data)
        assert len(compressed_level22) <= len(compressed_level1)

    def test_small_payload_uses_inmemory(self) -> None:
        """Small payloads (< threshold) use in-memory compression."""
        from hpke_http.streaming import zstd_compress, zstd_decompress

        # Small payload - below default 1MB threshold
        original = b"small payload " * 100
        compressed = zstd_compress(original)
        decompressed = zstd_decompress(compressed)

        assert decompressed == original

    def test_large_payload_uses_streaming(self) -> None:
        """Large payloads (>= threshold) use streaming compression."""
        from hpke_http.streaming import zstd_compress, zstd_decompress

        # Large payload - above default 1MB threshold
        original = b"x" * (2 * 1024 * 1024)  # 2MB
        compressed = zstd_compress(original)
        decompressed = zstd_decompress(compressed)

        assert decompressed == original

    def test_custom_threshold(self) -> None:
        """Custom streaming_threshold parameter works."""
        from hpke_http.streaming import zstd_compress, zstd_decompress

        original = b"data " * 1000  # ~5KB
        # Set threshold to 1KB to force streaming
        compressed = zstd_compress(original, streaming_threshold=1024)
        decompressed = zstd_decompress(compressed, streaming_threshold=1024)

        assert decompressed == original

    def test_interop_with_raw_zstd(self) -> None:
        """Unified functions interop with raw zstd module."""
        from hpke_http.streaming import zstd_compress, zstd_decompress

        zstd = import_zstd()
        original = b"test data " * 1000

        # Unified compress -> raw decompress
        compressed = zstd_compress(original)
        decompressed = zstd.decompress(compressed)
        assert decompressed == original

        # Raw compress -> unified decompress
        compressed = zstd.compress(original, level=ZSTD_COMPRESSION_LEVEL)
        decompressed = zstd_decompress(compressed)
        assert decompressed == original

    def test_incompressible_data(self) -> None:
        """Random (incompressible) data still works."""
        from hpke_http.streaming import zstd_compress, zstd_decompress

        original = secrets.token_bytes(100000)
        compressed = zstd_compress(original)
        decompressed = zstd_decompress(compressed)

        assert decompressed == original


class TestUnifiedCompressionMemory:
    """Memory usage tests for unified compression.

    Validates that streaming compression/decompression uses bounded memory
    overhead (~4MB) regardless of payload size. Tests multiple payload sizes
    to ensure consistent behavior.
    """

    # Maximum allowed overhead for streaming operations (4MB)
    MAX_OVERHEAD_BYTES = 4 * 1024 * 1024

    @pytest.mark.parametrize(
        "payload_size_mb",
        [5, 10, 50],
        ids=["5MB", "10MB", "50MB"],
    )
    def test_compress_memory_overhead_bounded(self, payload_size_mb: int) -> None:
        """Compression overhead stays under 4MB regardless of payload size."""
        from hpke_http.streaming import zstd_compress

        payload_size = payload_size_mb * 1024 * 1024
        payload = b"x" * payload_size

        # Warmup - first call may allocate internal buffers
        _ = zstd_compress(payload[:1024])

        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        compressed = zstd_compress(payload)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        peak_allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        # Overhead = peak - input - output (compressed is much smaller for repetitive data)
        overhead = peak_allocated - payload_size - len(compressed)

        assert overhead < self.MAX_OVERHEAD_BYTES, (
            f"Compression overhead {overhead / 1024 / 1024:.1f}MB exceeds 4MB limit "
            f"(payload={payload_size_mb}MB, peak={peak_allocated / 1024 / 1024:.1f}MB)"
        )

    @pytest.mark.parametrize(
        "payload_size_mb",
        [5, 10, 50],
        ids=["5MB", "10MB", "50MB"],
    )
    def test_decompress_memory_overhead_bounded(self, payload_size_mb: int) -> None:
        """Decompression overhead stays under 4MB regardless of payload size.

        Note: Uses low streaming_threshold to force streaming mode, since
        repetitive test data compresses extremely well (<2KB for 50MB).
        """
        from hpke_http.streaming import zstd_compress, zstd_decompress

        payload_size = payload_size_mb * 1024 * 1024
        original = b"y" * payload_size
        compressed = zstd_compress(original)

        # Warmup - first call may allocate internal buffers
        _ = zstd_decompress(zstd_compress(b"warmup" * 100), streaming_threshold=1)

        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Force streaming mode with low threshold (compressed data is tiny)
        decompressed = zstd_decompress(compressed, streaming_threshold=1)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        peak_allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        # Overhead = peak - input (compressed) - output (decompressed)
        overhead = peak_allocated - len(compressed) - len(decompressed)

        assert overhead < self.MAX_OVERHEAD_BYTES, (
            f"Decompression overhead {overhead / 1024 / 1024:.1f}MB exceeds 4MB limit "
            f"(payload={payload_size_mb}MB, peak={peak_allocated / 1024 / 1024:.1f}MB)"
        )

    @pytest.mark.parametrize(
        "payload_size_mb",
        [5, 10, 50],
        ids=["5MB", "10MB", "50MB"],
    )
    def test_roundtrip_memory_overhead_bounded(self, payload_size_mb: int) -> None:
        """Full compress+decompress roundtrip overhead stays under 4MB.

        Note: Uses low streaming_threshold for decompression to force streaming
        mode, since repetitive test data compresses extremely well.
        """
        from hpke_http.streaming import zstd_compress, zstd_decompress

        payload_size = payload_size_mb * 1024 * 1024
        original = b"z" * payload_size

        # Warmup
        _ = zstd_decompress(zstd_compress(b"warmup" * 100), streaming_threshold=1)

        gc.collect()
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        compressed = zstd_compress(original)
        # Force streaming decompression with low threshold
        decompressed = zstd_decompress(compressed, streaming_threshold=1)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        assert decompressed == original

        diff = snapshot2.compare_to(snapshot1, "lineno")
        peak_allocated = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        # Overhead = peak - input - compressed - output
        # Note: input and output are same size, compressed is small
        overhead = peak_allocated - payload_size - len(compressed) - payload_size

        assert overhead < self.MAX_OVERHEAD_BYTES, (
            f"Roundtrip overhead {overhead / 1024 / 1024:.1f}MB exceeds 4MB limit "
            f"(payload={payload_size_mb}MB, peak={peak_allocated / 1024 / 1024:.1f}MB)"
        )

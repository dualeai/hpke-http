"""
SSE (Server-Sent Events) streaming encryption.

Transparent encryption layer for SSE streams. Server sends normal SSE,
client receives normal SSE - encryption is invisible to application code.

Wire format v2 (encoding_id authenticated via AEAD AAD):
    event: enc
    data: <base64(counter_be32 || encoding_id_byte || ciphertext+tag)>

The ciphertext contains the raw SSE chunk exactly as the server sent it.
Perfect fidelity: comments, retry, id, events - everything preserved.

Reference: RFC-065 §6


Design rationale
================

Wire format v2 — encoding_id moved to AEAD AAD (was in plaintext-prefix in v1)
    Eliminates one full chunk-sized memcpy on the encrypt path (no more
    ``encoding_id || data`` plaintext prepend). encoding_id stays authenticated
    via AAD; tampering on the wire still fails AEAD verification (downgrade
    attack protection: e.g., flipping ZSTD 0x01 -> IDENTITY 0x00 to skip
    decompression). Wire size unchanged (25-byte fixed overhead per chunk).
    Trade-off: one-time wire format break (hard cutover; both ends must
    run matching wire-format version).

Combined buffer + reusable scratch (msgspec/picows pattern)
    Each ``Format`` instance owns a ``_scratch`` bytearray sized for full wire
    output. ``encrypt_chunk`` writes header + encoding_id + AEAD ciphertext
    directly into the scratch buffer in one allocation, then yields fresh bytes
    for ASGI emit. Steady-state encrypt allocates ~0.1 alloc/call (was 3-4
    pre-refactor). Trade-off: scratch grows once if a chunk exceeds initial
    size (rare; sized for ``MAX_CHUNK_WIRE_SIZE`` at construction).

Per-chunk return type: ``bytes | bytearray``
    Identity decrypt path returns ``bytearray`` (zero-copy ``decrypt_into``
    output); compressed paths return ``bytes`` (stdlib gzip/zstd output).
    Encrypt always returns ``bytes`` (RawFormat coerces; SSEFormat is naturally
    bytes via ``b"".join``). Considered + rejected: pure ``bytearray`` (forces
    extra copies on compressed paths); ``bytes | bytearray | memoryview``
    (memoryview lifetime confusion in async contexts); PEP 688 ``Buffer``
    (lacks ``__len__``, incompatible with cryptography lib's typed encrypt_into).

Granian PyO3 boundary: bytes vs bytearray
    Granian extracts ASGI ``body`` field as ``Cow<[u8]>``: ``bytes`` borrows
    (zero copy on FFI boundary), ``bytearray`` calls ``to_vec()`` (full copy).
    ``RawFormat.encrypt_chunk`` therefore coerces its scratch output to
    ``bytes`` before return. uvicorn's asyncio transport similarly copies
    ``bytearray`` on backpressure but borrows ``bytes`` — same conclusion.

ChunkEncryptor: thread-safe via internal lock
    ``threading.Lock`` guards counter increment + nonce_buf packing inside
    ``_prepare_chunk``. AEAD encrypt runs outside the lock (each call holds
    its own 12-byte nonce copy). Required under free-threaded CPython
    (PEP 703): without the GIL, ``self.counter += 1`` is non-atomic and
    concurrent encrypts can produce duplicate counters / reused nonces.

_ChunkStreamParser: memoryview yield with buffer-swap on feed
    Yields read-only memoryview slices into the parser's internal bytearray
    (was ``bytes(self._buffer[s:e])`` copy in v1, ~30x slower on 64KB chunks).
    Each ``feed()`` REBINDS ``self._buffer`` to a fresh bytearray; the old
    buffer (with outstanding memoryview exports) stays alive via Python ref
    counting until consumer drops references. Avoids ``BufferError`` from
    in-place ``extend()`` while exports exist. Trade-off: O(leftover) copy
    per feed call; leftover is bounded by chunk size and typically near-zero
    in steady state.

SSEFormat: no ``encrypt_chunk_into`` (RawFormat-only public API)
    SSEFormat finalizes via ``base64.b64encode`` + SSE event wrap. base64
    expansion (~4/3 size) is not in-place compatible — output goes to a
    different size than input. Public ``ChunkEncryptor.encrypt_into`` raises
    TypeError for SSEFormat. RawFormat is the bulk-transfer path where
    msgspec-style write-into-own-buffer makes sense.

Encoding_id range validation BEFORE AEAD call
    Decryptor validates ``encoding_id ∈ {0x00, 0x01, 0x02}`` before passing
    it as AAD to ``decrypt_into``. Gives clear ``DecryptionError("Unknown
    encoding")`` instead of misleading ``Authentication failed`` when sender
    supplied a bogus encoding (with matching AAD). Tampering on wire (sender's
    AAD differs from receiver's wire byte) still produces AEAD auth failure.

Single nonce buffer reused via ``_compute_chunk_nonce``
    ``_nonce_buf`` is a 12-byte bytearray initialized once per encryptor /
    decryptor with ``salt(4) || zeros(4) || zeros(4)``. ``_compute_chunk_nonce``
    packs the counter into bytes [8:12] in-place and returns ``bytes(nonce_buf)``
    (12-byte copy). The ``bytes()`` coerce is needed: ``cipher.encrypt_into``
    accepts the buffer protocol, but reusing the SAME bytearray across calls
    while the cipher holds a reference would cause undefined behavior under
    overlapping writes. Trade-off: 12-byte alloc per chunk vs lifetime safety.
"""

from __future__ import annotations

import base64
import gzip
import io
import secrets
import struct
import sys
import threading
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from hpke_http.constants import (
    CHACHA20_POLY1305_KEY_SIZE,
    CHACHA20_POLY1305_TAG_SIZE,
    GZIP_COMPRESSION_LEVEL,
    GZIP_STREAMING_CHUNK_SIZE,
    GZIP_STREAMING_THRESHOLD,
    MAX_CHUNK_WIRE_SIZE,
    MAX_DECOMPRESSED_CHUNK_SIZE,
    RAW_LENGTH_PREFIX_SIZE,
    SSE_COUNTER_SIZE,
    SSE_MAX_COUNTER,
    SSE_SESSION_KEY_LABEL,
    SSE_SESSION_SALT_SIZE,
    ZSTD_COMPRESSION_LEVEL,
    ZSTD_MIN_SIZE,
    ZSTD_STREAMING_CHUNK_SIZE,
    ZSTD_STREAMING_THRESHOLD,
    SSEEncodingId,
)
from hpke_http.exceptions import DecryptionError, ReplayAttackError, SessionExpiredError
from hpke_http.hpke import HPKEContext

# Cached zstd module (PEP 784 pattern)
_zstd_module: Any = None

# Pre-compiled struct for nonce counter (little-endian uint32)
# Used by ChunkEncryptor/ChunkDecryptor._compute_nonce for faster packing
_COUNTER_STRUCT = struct.Struct("<I")

# Pre-compiled struct for RawFormat header (big-endian: length + counter)
# Using pack_into() is ~2x faster than to_bytes() in hot paths
_RAW_HEADER_STRUCT = struct.Struct(">II")  # 4B length + 4B counter

# Pre-compiled struct for SSEFormat counter (big-endian uint32)
_SSE_COUNTER_STRUCT = struct.Struct(">I")

# Pre-defined encoding prefixes for zero-allocation concat (5x faster than bytearray)
_IDENTITY_PREFIX = bytes([SSEEncodingId.IDENTITY])  # b"\x00"
_ZSTD_PREFIX = bytes([SSEEncodingId.ZSTD])  # b"\x01"
_GZIP_PREFIX = bytes([SSEEncodingId.GZIP])  # b"\x02"


def _compute_chunk_nonce(nonce_buf: bytearray, counter: int) -> bytes:
    """Compute 12-byte chunk nonce from pre-initialized salt buffer.

    Layout: ``nonce_buf[0:4]`` = session_salt (set once at init), ``[4:8]`` = zeros,
    ``[8:12]`` = counter (little-endian, packed in-place each call).

    Shared by ChunkEncryptor and ChunkDecryptor; both pre-allocate ``_nonce_buf``
    via ``_init_chunk_state``.
    """
    _COUNTER_STRUCT.pack_into(nonce_buf, 8, counter)
    return bytes(nonce_buf)


def _init_chunk_state(session: StreamingSession) -> tuple[ChaCha20Poly1305, bytearray]:
    """Construct AEAD cipher + pre-allocated 12-byte nonce buffer with salt prefix.

    Shared init for ChunkEncryptor/ChunkDecryptor ``__post_init__``.

    Returns:
        ``(cipher, nonce_buf)`` where ``nonce_buf[:4]`` = session_salt,
        ``[4:12]`` = zeros (counter packed in-place via ``_compute_chunk_nonce``).
    """
    cipher = ChaCha20Poly1305(session.session_key)
    nonce_buf = bytearray(12)
    nonce_buf[:4] = session.session_salt
    return cipher, nonce_buf


def import_zstd() -> Any:
    """Import zstd module (PEP 784 pattern, cached).

    Uses Python 3.14+ native compression.zstd, or backports.zstd for earlier versions.

    Returns:
        The zstd module

    Raises:
        ImportError: If backports.zstd is not installed on Python < 3.14
    """
    global _zstd_module
    if _zstd_module is not None:
        return _zstd_module

    if sys.version_info >= (3, 14):
        from compression import zstd  # type: ignore[import-not-found]

        _zstd_module = zstd
    else:
        try:
            from backports import zstd  # type: ignore[import-not-found]

            _zstd_module = zstd  # type: ignore[reportUnknownVariableType]
        except ImportError as e:
            raise ImportError(
                "Zstd compression requires 'backports.zstd' package. Install with: pip install hpke-http[zstd]"
            ) from e
    return _zstd_module  # type: ignore[return-value]


def _zstd_compress_streaming(
    data: bytes | bytearray | memoryview,
    level: int,
    chunk_size: int,
) -> bytes:
    """Internal: streaming compression with ZstdFile."""
    zstd = import_zstd()
    output = io.BytesIO()

    mv = memoryview(data)
    with zstd.ZstdFile(output, mode="wb", level=level) as f:
        for offset in range(0, len(mv), chunk_size):
            f.write(mv[offset : offset + chunk_size])

    return output.getvalue()


def _zstd_decompress_streaming(
    data: bytes,
    chunk_size: int,
) -> bytes:
    """Internal: streaming decompression with ZstdFile.

    Uses BytesIO for output instead of list+join. This is a deliberate
    RAM/CPU tradeoff optimized for server middleware:

    RAM: Reduces peak memory from ~2x to ~1.3x decompressed size.
         For 50MB payload: 100MB -> 65MB (saves 35MB per request).

    CPU: Adds ~40% overhead due to incremental writes vs batch append.
         For 50MB payload: 10ms -> 15ms (adds 5ms per request).

    The tradeoff favors RAM because memory pressure affects all concurrent
    requests (OOM, swapping), while 5ms CPU is negligible vs network RTT.
    """
    zstd = import_zstd()
    input_buffer = io.BytesIO(data)
    output_buffer = io.BytesIO()

    with zstd.ZstdFile(input_buffer, mode="rb") as f:
        while chunk := f.read(chunk_size):
            output_buffer.write(chunk)

    return output_buffer.getvalue()


def zstd_compress(
    data: bytes | bytearray | memoryview,
    level: int = ZSTD_COMPRESSION_LEVEL,
    streaming_threshold: int = ZSTD_STREAMING_THRESHOLD,
) -> bytes:
    """
    Compress data, auto-selecting streaming for large payloads.

    For payloads >= streaming_threshold (default 1MB), uses streaming
    compression with ~4MB constant memory. Smaller payloads use faster
    in-memory compression.

    Args:
        data: Raw bytes to compress
        level: Compression level (1-22, default 3 = fast)
        streaming_threshold: Size threshold for streaming mode (default 1MB)

    Returns:
        Compressed bytes in Zstandard format

    Raises:
        ImportError: If backports.zstd not installed (Python < 3.14)

    Example:
        >>> compressed = zstd_compress(large_image_bytes)
        >>> # Auto-selects streaming for 50MB+ payloads
    """
    if not data:
        return b""

    if len(data) >= streaming_threshold:
        return _zstd_compress_streaming(data, level, ZSTD_STREAMING_CHUNK_SIZE)

    zstd = import_zstd()
    return zstd.compress(bytes(data) if not isinstance(data, bytes) else data, level=level)


def zstd_decompress(
    data: bytes,
    streaming_threshold: int = ZSTD_STREAMING_THRESHOLD,
) -> bytes:
    """
    Decompress data, auto-selecting streaming for large payloads.

    For compressed payloads >= streaming_threshold (default 1MB), uses
    streaming decompression with bounded memory. Smaller payloads use
    faster in-memory decompression.

    Args:
        data: Zstandard-compressed bytes
        streaming_threshold: Size threshold for streaming mode (default 1MB)

    Returns:
        Decompressed bytes

    Raises:
        ImportError: If backports.zstd not installed (Python < 3.14)
        zstd.ZstdError: If data is invalid or corrupted

    Example:
        >>> original = zstd_decompress(compressed_data)
        >>> # Auto-selects streaming for large compressed payloads
    """
    if not data:
        return b""

    if len(data) >= streaming_threshold:
        return _zstd_decompress_streaming(data, ZSTD_STREAMING_CHUNK_SIZE)

    zstd = import_zstd()
    return zstd.decompress(data)


# =============================================================================
# Gzip Compression (RFC 1952) - Stdlib fallback when zstd unavailable
# =============================================================================
#
# Why gzip compressor cannot be reused like zstd:
#
# Zstd's FLUSH_BLOCK mode produces independently decompressible blocks while
# maintaining the compression dictionary across blocks. This allows a single
# ZstdCompressor instance to be reused for multiple chunks with good compression.
#
# Gzip/deflate has no equivalent:
# - Z_SYNC_FLUSH: Flushes output but maintains dictionary (not independently
#   decompressible - requires prior blocks for context)
# - Z_FULL_FLUSH: Resets dictionary entirely (independently decompressible but
#   loses all compression benefit from reuse)
#
# For SSE per-chunk compression, each chunk MUST be independently decompressible
# so clients can decrypt and decompress chunks as they arrive. This requires
# Z_FULL_FLUSH semantics, which resets the dictionary - meaning no benefit from
# compressor reuse.
#
# Additionally, gzip format requires header (10B) + trailer (8B) per independent
# block for CRC validation. There's no way to produce valid independent gzip
# blocks without this 18-byte overhead per chunk.
#
# Performance impact is minimal:
# - gzip.compress(): ~25µs per 10KB chunk
# - Compressor creation overhead: ~5µs (negligible)
# - Zstd with reuse: ~15µs per 10KB chunk (only ~10µs faster)
#
# Given gzip is a fallback for when zstd is unavailable (increasingly rare with
# Python 3.14's native compression.zstd), the complexity of raw deflate with
# custom framing isn't justified.
#
# References:
# - https://www.bolet.org/~pornin/deflate-flush.html (Zlib flush modes)
# - https://docs.python.org/3/library/compression.zstd.html (Python 3.14 zstd)
# - RFC 1952 (gzip format specification)
# =============================================================================


def _gzip_compress_streaming(
    data: bytes | bytearray | memoryview,
    level: int,
    chunk_size: int,
) -> bytes:
    """Internal: streaming compression with GzipFile."""
    output = io.BytesIO()

    mv = memoryview(data)
    with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=level, mtime=0) as f:
        for offset in range(0, len(mv), chunk_size):
            f.write(mv[offset : offset + chunk_size])

    return output.getvalue()


def _gzip_decompress_streaming(
    data: bytes,
    chunk_size: int,
) -> bytes:
    """Internal: streaming decompression with GzipFile.

    Uses BytesIO for output instead of list+join. Same RAM/CPU tradeoff
    as zstd streaming: reduces peak memory from ~2x to ~1.3x decompressed size.
    """
    input_buffer = io.BytesIO(data)
    output_buffer = io.BytesIO()

    with gzip.GzipFile(fileobj=input_buffer, mode="rb") as f:
        while chunk := f.read(chunk_size):
            output_buffer.write(chunk)

    return output_buffer.getvalue()


def gzip_compress(
    data: bytes | bytearray | memoryview,
    level: int = GZIP_COMPRESSION_LEVEL,
    streaming_threshold: int = GZIP_STREAMING_THRESHOLD,
) -> bytes:
    """
    Compress data with gzip, auto-selecting streaming for large payloads.

    For payloads >= streaming_threshold (default 1MB), uses streaming
    compression with bounded memory. Smaller payloads use faster in-memory.

    Uses gzip module (not zlib) for proper gzip format with headers.
    mtime=0 for reproducible output across Python versions.

    Args:
        data: Raw bytes to compress
        level: Compression level (0-9, default 6 = balanced)
        streaming_threshold: Size threshold for streaming mode (default 1MB)

    Returns:
        Compressed bytes in gzip format (RFC 1952)

    Example:
        >>> compressed = gzip_compress(large_json_bytes)
        >>> # Auto-selects streaming for payloads >= 1MB
    """
    if not data:
        return b""

    if len(data) >= streaming_threshold:
        return _gzip_compress_streaming(data, level, GZIP_STREAMING_CHUNK_SIZE)

    return gzip.compress(bytes(data) if not isinstance(data, bytes) else data, compresslevel=level, mtime=0)


def gzip_decompress(
    data: bytes,
    streaming_threshold: int = GZIP_STREAMING_THRESHOLD,
) -> bytes:
    """
    Decompress gzip data, auto-selecting streaming for large payloads.

    For compressed payloads >= streaming_threshold (default 1MB), uses
    streaming decompression with bounded memory.

    Args:
        data: Gzip-compressed bytes
        streaming_threshold: Size threshold for streaming mode (default 1MB)

    Returns:
        Decompressed bytes

    Raises:
        OSError: If data is invalid or corrupted

    Example:
        >>> original = gzip_decompress(compressed_data)
        >>> # Auto-selects streaming for large compressed payloads
    """
    if not data:
        return b""

    if len(data) >= streaming_threshold:
        return _gzip_decompress_streaming(data, GZIP_STREAMING_CHUNK_SIZE)

    return gzip.decompress(data)


# =============================================================================
# Streaming Compressor Abstraction
# =============================================================================
#
# Abstract interface for streaming compression with dictionary context preservation.
#
# Unlike per-chunk compression (used for SSE where each chunk must be independently
# decompressible), streaming compression maintains dictionary context across chunks.
# This enables better compression ratios for whole-body compression where the server
# decrypts all chunks first, then decompresses as one stream.
#
# Implementations:
# - ZstdStreamingCompressor: Uses FLUSH_BLOCK mode (maintains dictionary)
# - GzipStreamingCompressor: Uses Z_SYNC_FLUSH mode (~32% overhead vs one-shot)
# - IdentityCompressor: No compression (passthrough)
# =============================================================================


class StreamingCompressor(ABC):
    """Abstract streaming compressor with dictionary context preservation.

    Maintains compression dictionary across chunks for better compression ratio.
    Each call to compress() produces output that must be concatenated and
    decompressed as a single stream.

    Use finalize() after all data is compressed to flush any remaining output.
    """

    @abstractmethod
    def compress(self, chunk: bytes) -> bytes:
        """Compress chunk, maintaining dictionary context across calls.

        Args:
            chunk: Raw bytes to compress

        Returns:
            Compressed bytes (may be empty if buffered internally)
        """
        ...

    @abstractmethod
    def finalize(self) -> bytes:
        """Finalize the compression stream.

        Must be called after all data is compressed to flush any remaining
        buffered output and write stream terminator.

        Returns:
            Final compressed bytes (may be empty)
        """
        ...

    @property
    @abstractmethod
    def encoding_id(self) -> bytes:
        """1-byte encoding ID for wire format v2.

        In wire format v2 the encoding_id byte is placed on the wire (outside
        the AEAD ciphertext) and authenticated via AAD. This property returns
        the byte for the format; ChunkEncryptor reads ``encoding_id[0]`` (int).

        Returns:
            Single-byte ``bytes`` identifying the compression format.
        """
        ...


class ZstdStreamingCompressor(StreamingCompressor):
    """Zstd streaming compressor with FLUSH_BLOCK mode.

    Maintains dictionary context across chunks for good compression ratio.
    Output can be decompressed as a single zstd stream.

    Uses FLUSH_BLOCK to produce output after each compress() call while
    maintaining the compression dictionary. Final FLUSH_FRAME closes the stream.
    """

    def __init__(self, level: int = ZSTD_COMPRESSION_LEVEL):
        """Initialize zstd streaming compressor.

        Args:
            level: Compression level (1-22, default 3 = fast)

        Raises:
            ImportError: If backports.zstd not installed (Python < 3.14)
        """
        zstd = import_zstd()
        self._compressor = zstd.ZstdCompressor(level=level)
        self._zstd = zstd

    def compress(self, chunk: bytes) -> bytes:
        """Compress chunk with FLUSH_BLOCK to maintain dictionary context."""
        return self._compressor.compress(chunk, mode=self._zstd.ZstdCompressor.FLUSH_BLOCK)

    def finalize(self) -> bytes:
        """Finalize with FLUSH_FRAME to close the zstd stream."""
        return self._compressor.compress(b"", mode=self._zstd.ZstdCompressor.FLUSH_FRAME)

    @property
    def encoding_id(self) -> bytes:
        return _ZSTD_PREFIX  # 0x01


class GzipStreamingCompressor(StreamingCompressor):
    """Gzip streaming compressor with Z_SYNC_FLUSH mode.

    Maintains dictionary context across chunks. Output can be decompressed
    as a single gzip stream.

    Uses Z_SYNC_FLUSH to produce output after each compress() call while
    maintaining the compression dictionary. Has ~32% overhead vs one-shot
    gzip.compress() due to flush overhead, but enables streaming.

    Note: For SSE per-chunk compression (where each chunk must decompress
    independently), use gzip_compress() directly instead. This class is for
    whole-body streaming where all chunks are concatenated before decompression.
    """

    def __init__(self, level: int = GZIP_COMPRESSION_LEVEL):
        """Initialize gzip streaming compressor.

        Args:
            level: Compression level (0-9, default 6 = balanced)
        """
        # wbits=31 = gzip format (16 + MAX_WBITS where MAX_WBITS=15)
        self._compressor = zlib.compressobj(level=level, wbits=31)

    def compress(self, chunk: bytes) -> bytes:
        """Compress chunk with Z_SYNC_FLUSH to maintain dictionary context."""
        out = self._compressor.compress(chunk)
        out += self._compressor.flush(zlib.Z_SYNC_FLUSH)
        return out

    def finalize(self) -> bytes:
        """Finalize with Z_FINISH to close the gzip stream."""
        return self._compressor.flush(zlib.Z_FINISH)

    @property
    def encoding_id(self) -> bytes:
        return _GZIP_PREFIX  # 0x02


class IdentityCompressor(StreamingCompressor):
    """Identity compressor (passthrough, no compression).

    Used when compression is disabled or chunk is too small to benefit.
    """

    def compress(self, chunk: bytes) -> bytes:
        """Return chunk unchanged."""
        return chunk

    def finalize(self) -> bytes:
        """No finalization needed for identity."""
        return b""

    @property
    def encoding_id(self) -> bytes:
        return _IDENTITY_PREFIX  # 0x00


__all__ = [
    "ChunkDecryptor",
    "ChunkEncryptor",
    "ChunkFormat",
    "GzipStreamingCompressor",
    "IdentityCompressor",
    "RawFormat",
    "SSEFormat",
    "StreamingCompressor",
    "StreamingSession",
    "ZstdStreamingCompressor",
    "create_session_from_context",
    "gzip_compress",
    "gzip_decompress",
    "import_zstd",
    "zstd_compress",
    "zstd_decompress",
]


# =============================================================================
# Chunk Format Strategy (for different wire formats)
# =============================================================================


class ChunkFormat(Protocol):
    """Strategy for encoding/decoding encrypted chunks (wire format v2).

    Wire format v2: encoding_id is authenticated via AAD (not prepended to plaintext).
    Wire layout per chunk:
    - RawFormat: length(4) || counter(4) || encoding_id(1) || ciphertext+tag
    - SSEFormat: event: enc\\ndata: <base64(counter || encoding_id || ciphertext+tag)>\\n\\n

    Format owns a reusable scratch bytearray to avoid per-call allocation
    (msgspec/picows pattern). Encrypts directly into the scratch buffer.
    """

    def encrypt_chunk(
        self,
        counter: int,
        encoding_id: int,
        plaintext: bytes | bytearray | memoryview,
        cipher: ChaCha20Poly1305,
        nonce: bytes,
    ) -> bytes:
        """Encrypt plaintext into combined-buffer wire output.

        Args:
            counter: Chunk counter (4 bytes, big-endian)
            encoding_id: Encoding ID byte (passed as AEAD AAD)
            plaintext: Raw plaintext to encrypt
            cipher: AEAD cipher (caller-owned)
            nonce: 12-byte nonce (counter-derived)

        Returns:
            Wire-formatted bytes ready for ASGI emit
        """
        ...

    def decode(self, data: bytes | bytearray | memoryview | str) -> tuple[int, int, memoryview]:
        """Parse wire data into (counter, encoding_id, ciphertext).

        Args:
            data: Wire-formatted data (any buffer-protocol provider, or str for SSE).

        Returns:
            Tuple of (counter, encoding_id, ciphertext_memoryview).
            Ciphertext is memoryview for zero-copy slicing.
        """
        ...


@dataclass
class SSEFormat:
    """SSE event format: event: enc\\ndata: <base64>\\n\\n (wire format v2).

    Used for Server-Sent Events streaming encryption.

    Uses standard base64 (RFC 4648 §4) instead of base64url because:
    - SSE data fields only forbid LF (0x0A) and CR (0x0D) per WHATWG spec
    - Base64 alphabet (+, /, =) contains neither forbidden character
    - Standard base64 is ~1.7x faster than base64url in Python stdlib
    - No URL encoding needed since SSE is not transmitted via URL

    Reference: https://html.spec.whatwg.org/multipage/server-sent-events.html
    """

    _PREFIX: ClassVar[bytes] = b"event: enc\ndata: "
    _SUFFIX: ClassVar[bytes] = b"\n\n"
    _scratch: bytearray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._scratch = bytearray(MAX_CHUNK_WIRE_SIZE)

    def encrypt_chunk(
        self,
        counter: int,
        encoding_id: int,
        plaintext: bytes | bytearray | memoryview,
        cipher: ChaCha20Poly1305,
        nonce: bytes,
    ) -> bytes:
        """Encrypt into reusable scratch buffer, then base64+wrap as SSE event."""
        pt_len = len(plaintext)
        ct_len = pt_len + CHACHA20_POLY1305_TAG_SIZE
        # Pre-base64 layout: counter(4) || encoding_id(1) || ct+tag
        payload_size = SSE_COUNTER_SIZE + 1 + ct_len
        if payload_size > len(self._scratch):
            self._scratch = bytearray(payload_size)
        _SSE_COUNTER_STRUCT.pack_into(self._scratch, 0, counter)
        self._scratch[SSE_COUNTER_SIZE] = encoding_id
        ct_start = SSE_COUNTER_SIZE + 1
        cipher.encrypt_into(
            nonce,
            plaintext,
            bytes((encoding_id,)),
            memoryview(self._scratch)[ct_start : ct_start + ct_len],
        )
        # base64 always returns bytes; final SSE wire must be bytes
        b64 = base64.b64encode(memoryview(self._scratch)[:payload_size])
        return b"".join((self._PREFIX, b64, self._SUFFIX))

    def decode(self, data: bytes | bytearray | memoryview | str) -> tuple[int, int, memoryview]:
        """Decode base64 payload: returns (counter, encoding_id, ciphertext_mv)."""
        data_bytes = data.encode("ascii") if isinstance(data, str) else data
        payload = base64.b64decode(data_bytes)
        mv = memoryview(payload)
        counter = int.from_bytes(mv[:SSE_COUNTER_SIZE], "big")
        encoding_id = mv[SSE_COUNTER_SIZE]
        return counter, encoding_id, mv[SSE_COUNTER_SIZE + 1 :]


@dataclass
class RawFormat:
    """Binary format (wire format v2): length(4B) || counter(4B) || encoding_id(1B) || ciphertext+tag.

    Used for standard HTTP response encryption.

    The length prefix enables O(1) chunk boundary detection when multiple
    chunks are concatenated in a response body. Length is the size of
    counter + encoding_id + ciphertext+tag (excludes the length field itself).
    """

    _scratch: bytearray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._scratch = bytearray(MAX_CHUNK_WIRE_SIZE)

    def encrypt_chunk(
        self,
        counter: int,
        encoding_id: int,
        plaintext: bytes | bytearray | memoryview,
        cipher: ChaCha20Poly1305,
        nonce: bytes,
    ) -> bytes:
        """Encrypt into reusable internal scratch, return fresh bytes for ASGI emit.

        Layout: length(4) || counter(4) || encoding_id(1) || ct+tag.
        """
        # Auto-grow scratch if a larger plaintext arrives than initial sizing
        wire_size = RAW_LENGTH_PREFIX_SIZE + SSE_COUNTER_SIZE + 1 + len(plaintext) + CHACHA20_POLY1305_TAG_SIZE
        if wire_size > len(self._scratch):
            self._scratch = bytearray(wire_size)
        n = self.encrypt_chunk_into(self._scratch, 0, counter, encoding_id, plaintext, cipher, nonce)
        # Coerce to fresh bytes for ASGI emit (granian PyO3 borrows bytes; copies bytearray)
        return bytes(memoryview(self._scratch)[:n])

    def encrypt_chunk_into(
        self,
        dest: bytearray,
        offset: int,
        counter: int,
        encoding_id: int,
        plaintext: bytes | bytearray | memoryview,
        cipher: ChaCha20Poly1305,
        nonce: bytes,
    ) -> int:
        """Encrypt directly into caller-owned buffer at given offset.

        Truly zero-copy: skips internal scratch entirely. Wire layout written
        in-place to ``dest[offset:offset+total]``:

        - ``dest[offset:offset+4]``     = length prefix (BE uint32)
        - ``dest[offset+4:offset+8]``   = counter (BE uint32)
        - ``dest[offset+8]``            = encoding_id byte
        - ``dest[offset+9:offset+9+N+16]`` = AEAD ciphertext + 16-byte tag

        ``ChunkEncryptor.encrypt`` uses this internally with ``dest = self._scratch,
        offset = 0`` (combined-buffer path). Public users call via
        ``ChunkEncryptor.encrypt_into(plaintext, dest, offset)`` for msgspec-style
        write-into-own-buffer.

        Args:
            dest: Pre-allocated bytearray. Must be >= ``offset + (4 + 4 + 1 + len(plaintext) + 16)``.
            offset: Write position in dest.
            counter: Chunk counter (4 bytes BE).
            encoding_id: Encoding ID byte (passed as AEAD AAD).
            plaintext: Raw plaintext (bytes/bytearray/memoryview).
            cipher: AEAD cipher (caller-owned).
            nonce: 12-byte nonce (counter-derived).

        Returns:
            Total bytes written: ``4 + 4 + 1 + len(plaintext) + 16``.

        Raises:
            ValueError: If dest is too small at the given offset.
        """
        pt_len = len(plaintext)
        ct_len = pt_len + CHACHA20_POLY1305_TAG_SIZE
        chunk_len = SSE_COUNTER_SIZE + 1 + ct_len
        total = RAW_LENGTH_PREFIX_SIZE + chunk_len
        if offset + total > len(dest):
            raise ValueError(f"dest buffer too small: need {offset + total}, got {len(dest)}")
        _RAW_HEADER_STRUCT.pack_into(dest, offset, chunk_len, counter)
        eid_offset = offset + RAW_LENGTH_PREFIX_SIZE + SSE_COUNTER_SIZE
        dest[eid_offset] = encoding_id
        ct_start = eid_offset + 1
        cipher.encrypt_into(
            nonce,
            plaintext,
            bytes((encoding_id,)),
            memoryview(dest)[ct_start : ct_start + ct_len],
        )
        return total

    def decode(self, data: bytes | bytearray | memoryview | str) -> tuple[int, int, memoryview]:
        """Decode raw wire: returns (counter, encoding_id, ciphertext_mv)."""
        raw = memoryview(data if isinstance(data, (bytes, bytearray, memoryview)) else data.encode("latin-1"))
        counter_start = RAW_LENGTH_PREFIX_SIZE
        counter_end = RAW_LENGTH_PREFIX_SIZE + SSE_COUNTER_SIZE
        counter = int.from_bytes(raw[counter_start:counter_end], "big")
        encoding_id = raw[counter_end]
        return counter, encoding_id, raw[counter_end + 1 :]


@dataclass
class StreamingSession:
    """
    SSE streaming session parameters.

    Created by server after decrypting initial request.
    Sent to client in X-HPKE-Stream header.
    """

    session_key: bytes
    """32-byte key derived from HPKE context."""

    session_salt: bytes
    """4-byte random salt for nonce construction."""

    @classmethod
    def create(cls, session_key: bytes) -> StreamingSession:
        """
        Create a new streaming session with random salt.

        Args:
            session_key: 32-byte key (from HPKE context.export())

        Returns:
            New StreamingSession
        """
        return cls(
            session_key=session_key,
            session_salt=secrets.token_bytes(SSE_SESSION_SALT_SIZE),
        )

    def serialize(self) -> bytes:
        """
        Serialize session parameters for transmission.

        Returns:
            session_salt (4 bytes) - key is derived, not transmitted
        """
        # Only transmit salt; key is derived from HPKE context
        return self.session_salt

    @classmethod
    def deserialize(cls, data: bytes, session_key: bytes) -> StreamingSession:
        """
        Deserialize session parameters.

        Args:
            data: Serialized session (4 bytes salt)
            session_key: Key derived from HPKE context

        Returns:
            StreamingSession
        """
        if len(data) != SSE_SESSION_SALT_SIZE:
            raise ValueError(f"Invalid session data length: {len(data)}")
        return cls(session_key=session_key, session_salt=data)


def create_session_from_context(ctx: HPKEContext) -> StreamingSession:
    """
    Create SSE streaming session from HPKE context.

    Uses HPKE export secret to derive session key.

    Args:
        ctx: HPKE context (sender or recipient)

    Returns:
        StreamingSession ready for encryption/decryption
    """
    session_key = ctx.export(SSE_SESSION_KEY_LABEL, CHACHA20_POLY1305_KEY_SIZE)
    return StreamingSession.create(session_key)


@dataclass
class ChunkEncryptor:
    """
    Chunk encryptor with counter-based nonces (wire format v2).

    Thread-safe: counter increment + nonce packing in ``_prepare_chunk`` is
    guarded by an internal ``threading.Lock``. Required under free-threaded
    CPython (PEP 703) where ``+=`` is non-atomic; concurrent encrypts would
    otherwise produce duplicate counters and reused nonces.

    Wire format v2: encoding_id is authenticated via AAD on encrypt_into. Format
    instances own a reusable scratch bytearray; encrypt_into writes ciphertext
    directly into the wire output buffer (combined-buffer / msgspec pattern).

    Wire format is determined by the ChunkFormat strategy:
    - SSEFormat (default): SSE events with base64 payload
    - RawFormat: Binary length || counter || encoding_id || ct+tag

    Compression: Set compress=True to enable (zstd preferred, gzip fallback).
    """

    session: StreamingSession
    format: ChunkFormat = field(default_factory=SSEFormat)
    compress: bool = False
    counter: int = field(default=1)  # Start at 1 (0 reserved)
    _cipher: ChaCha20Poly1305 = field(init=False, repr=False)
    _nonce_buf: bytearray = field(init=False, repr=False)  # Pre-allocated 12-byte nonce buffer
    _compressor: StreamingCompressor | None = field(init=False, repr=False, default=None)
    _lock: threading.Lock = field(init=False, repr=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._cipher, self._nonce_buf = _init_chunk_state(self.session)

        if self.compress:
            try:
                self._compressor = ZstdStreamingCompressor()
            except ImportError:
                self._compressor = GzipStreamingCompressor()

    def _compute_nonce(self, counter: int) -> bytes:
        return _compute_chunk_nonce(self._nonce_buf, counter)

    def _prepare_chunk(
        self,
        chunk: bytes | bytearray | memoryview,
    ) -> tuple[bytes | bytearray | memoryview, int, int, bytes]:
        """Compute (plaintext, encoding_id, current_counter, nonce) for a chunk.

        Shared prep for ``encrypt()`` and ``encrypt_into()``:

        1. Decide compression: if compressor configured and ``len(chunk) >= ZSTD_MIN_SIZE``,
           run compressor (yields bytes) and pick its encoding_id (ZSTD or GZIP).
           Else: pass chunk through (zero-copy) with ``encoding_id = IDENTITY``.
        2. Bounds-check counter against ``SSE_MAX_COUNTER`` (raise on exhaustion).
        3. Increment counter under ``self._lock`` (free-threaded safety).
        4. Compute 12-byte nonce: ``salt(4) || zeros(4) || counter_le32(4)`` via
           ``_compute_chunk_nonce`` into reused ``self._nonce_buf``.

        Args:
            chunk: Raw chunk (bytes/bytearray/memoryview).

        Returns:
            Tuple of (plaintext_buffer, encoding_id_int, counter_value, nonce_bytes).

        Raises:
            SessionExpiredError: If counter would exceed SSE_MAX_COUNTER.
        """
        if self._compressor is not None and len(chunk) >= ZSTD_MIN_SIZE:
            # Compressor needs bytes input; tolerate memoryview/bytearray
            chunk_bytes = bytes(chunk) if not isinstance(chunk, bytes) else chunk
            plaintext: bytes | bytearray | memoryview = self._compressor.compress(chunk_bytes)
            encoding_id = self._compressor.encoding_id[0]
        else:
            # Pass memoryview/bytes/bytearray directly — encrypt_into accepts buffer protocol
            plaintext = chunk
            encoding_id = SSEEncodingId.IDENTITY

        if self.counter > SSE_MAX_COUNTER:
            raise SessionExpiredError("Session counter exhausted")

        current_counter = self.counter
        self.counter += 1
        nonce = self._compute_nonce(current_counter)
        return plaintext, encoding_id, current_counter, nonce

    def encrypt(self, chunk: bytes | bytearray | memoryview) -> bytes:
        """
        Encrypt a chunk (wire format v2).

        Pipeline:
        1. ``_prepare_chunk(chunk)`` decides compression, increments counter,
           computes nonce, returns ``(plaintext, encoding_id, counter, nonce)``.
        2. Delegate to ``Format.encrypt_chunk(...)`` which combines header +
           encoding_id byte + ``cipher.encrypt_into(...)`` in a single buffer
           and returns wire-formatted bytes.

        encoding_id is authenticated via AEAD AAD; tampering on the wire is
        detected on receiver decrypt (downgrade-attack resistance).

        Args:
            chunk: Raw chunk as bytes, bytearray, or memoryview (zero-copy
                slicing supported when not compressed).

        Returns:
            Encrypted chunk in wire format. Always ``bytes``: RawFormat coerces
            combined-buffer to bytes for ASGI emit (granian PyO3 borrows bytes
            but copies bytearray); SSEFormat output is base64+wrap, naturally
            bytes via ``b"".join``.

        Raises:
            SessionExpiredError: If counter > SSE_MAX_COUNTER (2**32 - 1).
        """
        # Lock spans full pipeline: counter increment + nonce_buf packing in
        # _prepare_chunk AND format.encrypt_chunk's reusable scratch bytearray.
        # All three are shared mutable state; under free-threaded CPython
        # concurrent encrypts would corrupt scratch and reuse nonces.
        with self._lock:
            plaintext, encoding_id, current_counter, nonce = self._prepare_chunk(chunk)
            return self.format.encrypt_chunk(current_counter, encoding_id, plaintext, self._cipher, nonce)

    def encrypt_into(
        self,
        chunk: bytes | bytearray | memoryview,
        dest: bytearray,
        offset: int = -1,
    ) -> int:
        """Encrypt a chunk directly into caller-owned buffer (msgspec.Encoder.encode_into pattern).

        Skips internal Format scratch buffer for users managing their own memory layout.
        Only supports RawFormat (binary wire); SSEFormat raises TypeError.

        Args:
            chunk: Raw chunk bytes/bytearray/memoryview.
            dest: Pre-allocated bytearray to write wire output into.
            offset: Write position. ``-1`` (default) appends; resizes dest. ``>= 0`` writes
                at given offset; caller must size dest beforehand (>= offset + wire_size).

        Returns:
            Number of bytes written to dest.

        Raises:
            SessionExpiredError: If counter exhausted.
            TypeError: If format is not RawFormat.
            ValueError: If dest is too small at the given offset.
        """
        if not isinstance(self.format, RawFormat):
            raise TypeError(
                f"encrypt_into requires RawFormat; got {type(self.format).__name__}. "
                "SSEFormat needs base64 expansion which is not in-place compatible."
            )

        # Lock guards counter + nonce_buf; encrypt_chunk_into writes into
        # caller-owned dest (no shared scratch), but _prepare_chunk's mutable
        # state still needs protection under free-threaded CPython.
        with self._lock:
            plaintext, encoding_id, current_counter, nonce = self._prepare_chunk(chunk)

            # Compute total wire size to either append-resize or validate offset
            pt_len = len(plaintext)
            ct_len = pt_len + CHACHA20_POLY1305_TAG_SIZE
            wire_size = RAW_LENGTH_PREFIX_SIZE + SSE_COUNTER_SIZE + 1 + ct_len

            if offset < 0:
                offset = len(dest)
                dest.extend(b"\x00" * wire_size)

            return self.format.encrypt_chunk_into(
                dest, offset, current_counter, encoding_id, plaintext, self._cipher, nonce
            )


@dataclass
class ChunkDecryptor:
    """
    Chunk decryptor with counter validation (wire format v2).

    Decrypts chunks and validates counter monotonicity for replay protection.
    Returns the exact raw chunk the server originally sent.

    Wire format v2: encoding_id parsed from wire (length-prefixed position),
    passed as AAD to decrypt_into. Tampered encoding_id → AEAD authentication
    failure (downgrade-attack resistance).

    Wire format determined by ChunkFormat strategy:
    - SSEFormat (default): Base64-encoded SSE data field
    - RawFormat: Binary length || counter || encoding_id || ct+tag

    Automatically handles decompression based on encoding_id from wire.
    """

    session: StreamingSession
    format: ChunkFormat = field(default_factory=SSEFormat)
    expected_counter: int = field(default=1)  # Expect counter starting at 1
    _cipher: ChaCha20Poly1305 = field(init=False, repr=False)
    _decompressor: Any = field(init=False, repr=False, default=None)  # Lazy init, reused
    _nonce_buf: bytearray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cipher, self._nonce_buf = _init_chunk_state(self.session)

    def _compute_nonce(self, counter: int) -> bytes:
        return _compute_chunk_nonce(self._nonce_buf, counter)

    def _get_decompressor(self) -> Any:
        """Get or create decompressor (lazy, reused per session)."""
        if self._decompressor is None:
            zstd = import_zstd()
            self._decompressor = zstd.ZstdDecompressor()
        return self._decompressor

    def decrypt(self, data: bytes | bytearray | memoryview | str) -> bytes | bytearray:
        """
        Decrypt a chunk to recover the original data.

        Pipeline (wire format v2):
        1. ``Format.decode(data)`` extracts ``(counter, encoding_id, ciphertext_mv)``.
        2. Validate ciphertext length >= AEAD tag size.
        3. Validate counter == ``expected_counter`` (replay protection).
        4. Validate encoding_id ∈ {IDENTITY, ZSTD, GZIP} (clear error before AEAD).
        5. ``cipher.decrypt_into(nonce, ciphertext, aad=encoding_id_byte, pt_buf)``.
        6. Decompress if encoding_id != IDENTITY (cap at MAX_DECOMPRESSED_CHUNK_SIZE).
        7. Increment ``expected_counter``.

        Args:
            data: Encrypted wire data. Bytes/bytearray/memoryview for binary
                (RawFormat); str or bytes for SSE base64 data field.

        Returns:
            Original raw chunk. Returns ``bytearray`` for IDENTITY (zero-copy
            decrypt_into output); returns ``bytes`` for compressed paths
            (gzip/zstd decompress allocates fresh bytes).

        Raises:
            ReplayAttackError: Counter mismatch (replay or out-of-order).
            DecryptionError: Failed format decode, ciphertext too short,
                unknown encoding_id, AEAD authentication failure, decompression
                failure, or decompressed size exceeds DoS cap.
        """
        # Parse via format strategy: returns (counter, encoding_id, ciphertext_mv)
        try:
            counter, encoding_id, ciphertext = self.format.decode(data)
        except Exception as e:
            raise DecryptionError("Failed to decode chunk") from e

        if len(ciphertext) < CHACHA20_POLY1305_TAG_SIZE:
            raise DecryptionError("Ciphertext too short")

        if counter != self.expected_counter:
            raise ReplayAttackError(self.expected_counter, counter)

        # Validate encoding_id range BEFORE AEAD call for clear error message.
        # If sender supplied an unknown encoding_id (with matching AAD), reject early
        # rather than getting a misleading "Authentication failed".
        if encoding_id not in (SSEEncodingId.IDENTITY, SSEEncodingId.ZSTD, SSEEncodingId.GZIP):
            raise DecryptionError(f"Unknown encoding: 0x{encoding_id:02x}")

        nonce = self._compute_nonce(counter)
        # Decrypt directly into pre-allocated bytearray (zero-copy plaintext alloc)
        pt_len = len(ciphertext) - CHACHA20_POLY1305_TAG_SIZE
        pt_buf = bytearray(pt_len)
        try:
            self._cipher.decrypt_into(nonce, ciphertext, bytes((encoding_id,)), pt_buf)
        except Exception as e:
            raise DecryptionError("Decryption failed") from e

        plaintext: bytes | bytearray
        match encoding_id:
            case SSEEncodingId.ZSTD:
                try:
                    plaintext = self._get_decompressor().decompress(bytes(pt_buf))
                except Exception as e:
                    raise DecryptionError("Zstd decompression failed") from e
            case SSEEncodingId.GZIP:
                try:
                    plaintext = gzip_decompress(bytes(pt_buf))
                except Exception as e:
                    raise DecryptionError("Gzip decompression failed") from e
            case _:  # IDENTITY (other values rejected above)
                plaintext = pt_buf

        # DoS protection: cap decompressed chunk size (zip-bomb prevention)
        # Only applies to compressed data — identity encoding is already bounded by CHUNK_SIZE
        if encoding_id != SSEEncodingId.IDENTITY and len(plaintext) > MAX_DECOMPRESSED_CHUNK_SIZE:
            raise DecryptionError(f"Decompressed chunk too large: {len(plaintext)} > {MAX_DECOMPRESSED_CHUNK_SIZE}")

        self.expected_counter += 1
        return plaintext

"""
SSE (Server-Sent Events) streaming encryption.

Transparent encryption layer for SSE streams. Server sends normal SSE,
client receives normal SSE - encryption is invisible to application code.

Wire format:
    event: enc
    data: <base64(counter_be32 || ciphertext)>

The ciphertext contains the raw SSE chunk exactly as the server sent it.
Perfect fidelity: comments, retry, id, events - everything preserved.

Reference: RFC-065 §6
"""

from __future__ import annotations

import base64
import io
import secrets
import struct
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from hpke_http.constants import (
    CHACHA20_POLY1305_KEY_SIZE,
    CHACHA20_POLY1305_TAG_SIZE,
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

# Pre-defined encoding prefixes for zero-allocation concat (5x faster than bytearray)
_IDENTITY_PREFIX = bytes([SSEEncodingId.IDENTITY])  # b"\x00"
_ZSTD_PREFIX = bytes([SSEEncodingId.ZSTD])  # b"\x01"


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
    data: bytes,
    level: int,
    chunk_size: int,
) -> bytes:
    """Internal: streaming compression with ZstdFile."""
    zstd = import_zstd()
    output = io.BytesIO()

    with zstd.ZstdFile(output, mode="wb", level=level) as f:
        for offset in range(0, len(data), chunk_size):
            f.write(data[offset : offset + chunk_size])

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
    data: bytes,
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
    return zstd.compress(data, level=level)


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


__all__ = [
    "ChunkDecryptor",
    "ChunkEncryptor",
    "ChunkFormat",
    "RawFormat",
    "SSEFormat",
    "StreamingSession",
    "create_session_from_context",
    "import_zstd",
    "zstd_compress",
    "zstd_decompress",
]


# =============================================================================
# Chunk Format Strategy (for different wire formats)
# =============================================================================


class ChunkFormat(Protocol):
    """Strategy for encoding/decoding encrypted chunks.

    Implementations define how counter + ciphertext are formatted for wire
    transmission. This allows the same encryption logic to work with different
    output formats (SSE events, raw binary, WebSocket frames, etc.).
    """

    def encode(self, counter: int, ciphertext: bytes) -> bytes:
        """Format counter + ciphertext for wire transmission.

        Args:
            counter: Chunk counter (4 bytes, big-endian)
            ciphertext: Encrypted payload with 16-byte auth tag

        Returns:
            Wire-formatted bytes
        """
        ...

    def decode(self, data: bytes | str) -> tuple[int, bytes | memoryview]:
        """Parse wire data into (counter, ciphertext).

        Args:
            data: Wire-formatted data

        Returns:
            Tuple of (counter, ciphertext) - ciphertext may be memoryview for zero-copy
        """
        ...


class SSEFormat:
    """SSE event format: event: enc\\ndata: <base64>\\n\\n

    Used for Server-Sent Events streaming encryption.

    Uses standard base64 (RFC 4648 §4) instead of base64url because:
    - SSE data fields only forbid LF (0x0A) and CR (0x0D) per WHATWG spec
    - Base64 alphabet (+, /, =) contains neither forbidden character
    - Standard base64 is ~1.7x faster than base64url in Python stdlib
    - No URL encoding needed since SSE is not transmitted via URL

    Reference: https://html.spec.whatwg.org/multipage/server-sent-events.html
    """

    _PREFIX: bytes = b"event: enc\ndata: "
    _SUFFIX: bytes = b"\n\n"

    def encode(self, counter: int, ciphertext: bytes) -> bytes:
        """Encode as SSE event with base64 payload."""
        payload = counter.to_bytes(SSE_COUNTER_SIZE, "big") + ciphertext
        encoded = base64.b64encode(payload)
        return self._PREFIX + encoded + self._SUFFIX

    def decode(self, data: bytes | str) -> tuple[int, memoryview]:
        """Decode base64 payload from SSE data field.

        Returns memoryview for zero-copy ciphertext access.
        """
        data_bytes = data.encode("ascii") if isinstance(data, str) else data
        payload = base64.b64decode(data_bytes)
        mv = memoryview(payload)
        return int.from_bytes(mv[:SSE_COUNTER_SIZE], "big"), mv[SSE_COUNTER_SIZE:]


class RawFormat:
    """Binary format: length(4B) || counter(4B) || ciphertext

    Used for standard HTTP response encryption.

    The length prefix enables O(1) chunk boundary detection when multiple
    chunks are concatenated in a response body. Length is the size of
    counter + ciphertext (excludes the length field itself).
    """

    # Precompute offsets for wire format parsing
    _COUNTER_START: int = RAW_LENGTH_PREFIX_SIZE
    _COUNTER_END: int = RAW_LENGTH_PREFIX_SIZE + SSE_COUNTER_SIZE
    _CIPHERTEXT_START: int = RAW_LENGTH_PREFIX_SIZE + SSE_COUNTER_SIZE

    def encode(self, counter: int, ciphertext: bytes) -> bytes:
        """Encode as raw binary: length || counter || ciphertext.

        Uses single allocation instead of two concatenations for better performance.
        """
        # Single allocation: length + counter + ciphertext
        total_len = SSE_COUNTER_SIZE + len(ciphertext)
        result = bytearray(RAW_LENGTH_PREFIX_SIZE + total_len)
        result[0:RAW_LENGTH_PREFIX_SIZE] = total_len.to_bytes(RAW_LENGTH_PREFIX_SIZE, "big")
        result[self._COUNTER_START : self._COUNTER_END] = counter.to_bytes(SSE_COUNTER_SIZE, "big")
        result[self._CIPHERTEXT_START :] = ciphertext
        return bytes(result)

    def decode(self, data: bytes | str) -> tuple[int, memoryview]:
        """Decode raw binary: length(4B) || counter(4B) || ciphertext.

        Returns memoryview for zero-copy slicing of ciphertext.
        The length prefix is read and validated, then counter and ciphertext
        are extracted from the remaining bytes.
        """
        raw = memoryview(data if isinstance(data, bytes) else data.encode("latin-1"))
        # Skip length prefix, read counter and ciphertext
        return int.from_bytes(raw[self._COUNTER_START : self._COUNTER_END], "big"), raw[self._CIPHERTEXT_START :]


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
    Chunk encryptor with counter-based nonces.

    Encrypts chunks with monotonic counter for replay protection.
    Thread-safe: uses a lock to protect counter operations.

    Wire format is determined by the ChunkFormat strategy:
    - SSEFormat (default): SSE events with base64url payload
    - RawFormat: Binary length || counter || ciphertext

    Optional compression (RFC 8878 Zstd) can be enabled via compress=True.
    Compressed chunks are prefixed with encoding ID for client detection.
    """

    session: StreamingSession
    format: ChunkFormat = field(default_factory=SSEFormat)
    compress: bool = False
    counter: int = field(default=1)  # Start at 1 (0 reserved)
    _cipher: ChaCha20Poly1305 = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False, default_factory=threading.Lock)
    _compressor: Any = field(init=False, repr=False, default=None)  # Reused per session
    _nonce_buf: bytearray = field(init=False, repr=False)  # Pre-allocated 12-byte nonce buffer

    def __post_init__(self) -> None:
        self._cipher = ChaCha20Poly1305(self.session.session_key)
        # Pre-allocate nonce buffer: salt(4B) + zeros(4B) + counter(4B)
        self._nonce_buf = bytearray(12)
        self._nonce_buf[:4] = self.session.session_salt
        # Bytes 4-7 remain zeros (default bytearray value)
        if self.compress:
            zstd = import_zstd()
            self._compressor = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL)

    def _compute_nonce(self, counter: int) -> memoryview:
        """
        Compute 12-byte nonce from salt and counter (zero-copy).

        nonce = session_salt (4B) || zero_pad (4B) || counter_le32 (4B)

        Uses pre-allocated buffer and struct.pack_into for ~2x faster
        than counter.to_bytes() in hot paths.
        """
        _COUNTER_STRUCT.pack_into(self._nonce_buf, 8, counter)
        return memoryview(self._nonce_buf)

    def encrypt(self, chunk: bytes) -> bytes:
        """
        Encrypt a chunk.

        Args:
            chunk: Raw chunk as bytes.

        Returns:
            Encrypted chunk formatted according to the format strategy.

        Raises:
            SessionExpiredError: If counter exhausted
        """
        # Build payload outside lock: encoding_id (1B) || data
        # Uses bytes concat (5x faster than bytearray + slice copy for 64KB chunks)
        if self._compressor is not None and len(chunk) >= ZSTD_MIN_SIZE:
            zstd = import_zstd()
            compressed = self._compressor.compress(chunk, mode=zstd.ZstdCompressor.FLUSH_BLOCK)
            data = _ZSTD_PREFIX + compressed
        else:
            data = _IDENTITY_PREFIX + chunk

        # Lock only for counter operations and nonce computation
        # (nonce buffer is shared, so must be protected)
        with self._lock:
            if self.counter > SSE_MAX_COUNTER:
                raise SessionExpiredError("Session counter exhausted")

            current_counter = self.counter
            self.counter += 1

            # Compute nonce inside lock (modifies shared _nonce_buf)
            nonce = self._compute_nonce(current_counter)
            # Convert to bytes so we can release lock before encryption
            nonce_bytes = bytes(nonce)

        # Encrypt outside lock (ChaCha20Poly1305 is thread-safe with unique nonces)
        ciphertext = self._cipher.encrypt(nonce_bytes, data, associated_data=None)

        # Format output via strategy
        return self.format.encode(current_counter, ciphertext)


@dataclass
class ChunkDecryptor:
    """
    Chunk decryptor with counter validation.

    Decrypts chunks and validates counter monotonicity for replay protection.
    Returns the exact raw chunk the server originally sent.

    Wire format is determined by the ChunkFormat strategy:
    - SSEFormat (default): Base64url-encoded SSE data field
    - RawFormat: Binary length || counter || ciphertext

    Automatically handles decompression based on encoding ID prefix.
    """

    session: StreamingSession
    format: ChunkFormat = field(default_factory=SSEFormat)
    expected_counter: int = field(default=1)  # Expect counter starting at 1
    _cipher: ChaCha20Poly1305 = field(init=False, repr=False)
    _decompressor: Any = field(init=False, repr=False, default=None)  # Lazy init, reused
    _nonce_buf: bytearray = field(init=False, repr=False)  # Pre-allocated 12-byte nonce buffer

    def __post_init__(self) -> None:
        self._cipher = ChaCha20Poly1305(self.session.session_key)
        # Pre-allocate nonce buffer: salt(4B) + zeros(4B) + counter(4B)
        self._nonce_buf = bytearray(12)
        self._nonce_buf[:4] = self.session.session_salt
        # Bytes 4-7 remain zeros (default bytearray value)

    def _compute_nonce(self, counter: int) -> memoryview:
        """Compute 12-byte nonce from salt and counter (zero-copy)."""
        _COUNTER_STRUCT.pack_into(self._nonce_buf, 8, counter)
        return memoryview(self._nonce_buf)

    def _get_decompressor(self) -> Any:
        """Get or create decompressor (lazy, reused per session)."""
        if self._decompressor is None:
            zstd = import_zstd()
            self._decompressor = zstd.ZstdDecompressor()
        return self._decompressor

    def decrypt(self, data: bytes | str) -> bytes:
        """
        Decrypt a chunk to recover the original data.

        Args:
            data: Encrypted data in format-specific encoding

        Returns:
            Original raw chunk as bytes

        Raises:
            ReplayAttackError: If counter is out of order
            DecryptionError: If decryption fails or unknown encoding
        """
        # Parse via format strategy
        try:
            counter, ciphertext = self.format.decode(data)
        except Exception as e:
            raise DecryptionError("Failed to decode chunk") from e

        if len(ciphertext) < CHACHA20_POLY1305_TAG_SIZE:  # Minimum ciphertext (tag only)
            raise DecryptionError("Ciphertext too short")

        # Validate counter monotonicity
        if counter != self.expected_counter:
            raise ReplayAttackError(self.expected_counter, counter)

        # Decrypt
        # TODO(perf): When cryptography >= 47.0.0 is available, use decrypt_into()
        # to eliminate per-chunk allocation. Example:
        #   buffer = bytearray(len(ciphertext) - 16)
        #   self._cipher.decrypt_into(nonce, ciphertext, None, buffer)
        # See: https://cryptography.io/en/latest/hazmat/primitives/aead/
        nonce = self._compute_nonce(counter)
        try:
            payload = self._cipher.decrypt(nonce, ciphertext, associated_data=None)
        except Exception as e:
            raise DecryptionError("Decryption failed") from e

        # Parse encoding_id and decompress if needed
        if len(payload) < 1:
            raise DecryptionError("Decrypted payload too short (missing encoding ID)")

        encoding_id = payload[0]
        encoded_payload = payload[1:]

        if encoding_id == SSEEncodingId.ZSTD:
            try:
                plaintext = self._get_decompressor().decompress(encoded_payload)
            except Exception as e:
                raise DecryptionError("Zstd decompression failed") from e
        elif encoding_id == SSEEncodingId.IDENTITY:
            plaintext = encoded_payload
        else:
            raise DecryptionError(f"Unknown encoding: 0x{encoding_id:02x}")

        # Increment expected counter
        self.expected_counter += 1

        return plaintext

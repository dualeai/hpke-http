"""
SSE (Server-Sent Events) streaming encryption.

Transparent encryption layer for SSE streams. Server sends normal SSE,
client receives normal SSE - encryption is invisible to application code.

Wire format:
    event: enc
    data: <base64url(counter_be32 || ciphertext)>

The ciphertext contains the raw SSE chunk exactly as the server sent it.
Perfect fidelity: comments, retry, id, events - everything preserved.

Reference: RFC-065 §6
"""

from __future__ import annotations

import io
import secrets
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from hpke_http.constants import (
    CHACHA20_POLY1305_KEY_SIZE,
    CHACHA20_POLY1305_TAG_SIZE,
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
from hpke_http.headers import b64url_decode, b64url_encode
from hpke_http.hpke import HPKEContext

# Cached zstd module (PEP 784 pattern)
_zstd_module: Any = None


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
    """Internal: streaming decompression with ZstdFile."""
    zstd = import_zstd()
    input_buffer = io.BytesIO(data)
    output_chunks: list[bytes] = []

    with zstd.ZstdFile(input_buffer, mode="rb") as f:
        while chunk := f.read(chunk_size):
            output_chunks.append(chunk)

    return b"".join(output_chunks)


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

    def decode(self, data: bytes | str) -> tuple[int, bytes]:
        """Parse wire data into (counter, ciphertext).

        Args:
            data: Wire-formatted data

        Returns:
            Tuple of (counter, ciphertext)
        """
        ...


class SSEFormat:
    """SSE event format: event: enc\\ndata: <base64url>\\n\\n

    Used for Server-Sent Events streaming encryption.
    """

    _PREFIX: bytes = b"event: enc\ndata: "
    _SUFFIX: bytes = b"\n\n"

    def encode(self, counter: int, ciphertext: bytes) -> bytes:
        """Encode as SSE event with base64url payload."""
        payload = counter.to_bytes(4, "big") + ciphertext
        encoded = b64url_encode(payload)
        return self._PREFIX + encoded.encode("ascii") + self._SUFFIX

    def decode(self, data: bytes | str) -> tuple[int, bytes]:
        """Decode base64url payload from SSE data field."""
        data_str = data.decode("ascii") if isinstance(data, bytes) else data
        payload = bytes(b64url_decode(data_str))
        return int.from_bytes(payload[:4], "big"), payload[4:]


class RawFormat:
    """Binary format: length(4B) || counter(4B) || ciphertext

    Used for standard HTTP response encryption.

    The length prefix enables O(1) chunk boundary detection when multiple
    chunks are concatenated in a response body. Length is the size of
    counter + ciphertext (excludes the length field itself).
    """

    def encode(self, counter: int, ciphertext: bytes) -> bytes:
        """Encode as raw binary: length || counter || ciphertext."""
        chunk = counter.to_bytes(4, "big") + ciphertext
        length = len(chunk)
        return length.to_bytes(4, "big") + chunk

    def decode(self, data: bytes | str) -> tuple[int, bytes]:
        """Decode raw binary: length(4B) || counter(4B) || ciphertext.

        The length prefix is read and validated, then counter and ciphertext
        are extracted from the remaining bytes.
        """
        raw = data if isinstance(data, bytes) else data.encode("latin-1")
        # Skip length prefix (4 bytes), read counter and ciphertext
        return int.from_bytes(raw[4:8], "big"), raw[8:]


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

    def __post_init__(self) -> None:
        self._cipher = ChaCha20Poly1305(self.session.session_key)
        if self.compress:
            zstd = import_zstd()
            self._compressor = zstd.ZstdCompressor(level=ZSTD_COMPRESSION_LEVEL)

    def _compute_nonce(self, counter: int) -> bytes:
        """
        Compute 12-byte nonce from salt and counter.

        nonce = session_salt (4B) || zero_pad (4B) || counter_le32 (4B)
        """
        return self.session.session_salt + b"\x00\x00\x00\x00" + counter.to_bytes(4, "little")

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
        with self._lock:
            if self.counter > SSE_MAX_COUNTER:
                raise SessionExpiredError("Session counter exhausted")

            # Build payload: encoding_id (1B) || data
            # Compress if enabled and chunk is large enough
            if self._compressor is not None and len(chunk) >= ZSTD_MIN_SIZE:
                zstd = import_zstd()
                compressed = self._compressor.compress(chunk, mode=zstd.ZstdCompressor.FLUSH_BLOCK)
                data = bytes([SSEEncodingId.ZSTD]) + compressed
            else:
                data = bytes([SSEEncodingId.IDENTITY]) + chunk

            # Encrypt with counter nonce
            nonce = self._compute_nonce(self.counter)
            ciphertext = self._cipher.encrypt(nonce, data, associated_data=None)

            # Format output via strategy
            result = self.format.encode(self.counter, ciphertext)

            # Increment counter for next chunk
            self.counter += 1

        return result


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

    def __post_init__(self) -> None:
        self._cipher = ChaCha20Poly1305(self.session.session_key)

    def _compute_nonce(self, counter: int) -> bytes:
        """Compute 12-byte nonce from salt and counter."""
        return self.session.session_salt + b"\x00\x00\x00\x00" + counter.to_bytes(4, "little")

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

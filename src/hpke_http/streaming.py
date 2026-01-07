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
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from hpke_http.constants import (
    CHACHA20_POLY1305_KEY_SIZE,
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
    "SSEDecryptor",
    "SSEEncryptor",
    "StreamingSession",
    "create_session_from_context",
    "import_zstd",
    "zstd_compress",
    "zstd_decompress",
]


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
class SSEEncryptor:
    """
    Server-side SSE encryptor.

    Encrypts raw SSE chunks with monotonic counter for replay protection.
    Thread-safe: uses a lock to protect counter operations.

    The encryption is transparent - any valid SSE chunk (events, comments,
    retry directives) is encrypted as-is and will be decrypted identically.

    Optional compression (RFC 8878 Zstd) can be enabled via compress=True.
    Compressed chunks are prefixed with encoding ID for client detection.
    """

    session: StreamingSession
    counter: int = field(default=1)  # Start at 1 (0 reserved)
    compress: bool = False  # Enable Zstd compression
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

    # Pre-computed wire format parts (avoid repeated allocations)
    _WIRE_PREFIX: bytes = b"event: enc\ndata: "
    _WIRE_SUFFIX: bytes = b"\n\n"

    def encrypt(self, chunk: bytes) -> bytes:
        """
        Encrypt a raw SSE chunk.

        Args:
            chunk: Raw SSE chunk as bytes, exactly as server would send it.
                   Can be event, comment, retry directive, or any valid SSE.

        Returns:
            Encrypted SSE event as bytes (event: enc, data: <encrypted>)

        Raises:
            SessionExpiredError: If counter exhausted
        """
        with self._lock:
            if self.counter > SSE_MAX_COUNTER:
                raise SessionExpiredError("SSE session counter exhausted")

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

            # Wire format: counter_be32 || ciphertext
            payload = self.counter.to_bytes(SSE_COUNTER_SIZE, "big") + ciphertext
            encoded = b64url_encode(payload)

            # Increment counter for next chunk
            self.counter += 1

        # Format as encrypted SSE event (zero-copy concatenation)
        return self._WIRE_PREFIX + encoded.encode("ascii") + self._WIRE_SUFFIX


@dataclass
class SSEDecryptor:
    """
    Client-side SSE decryptor.

    Decrypts SSE chunks and validates counter monotonicity for replay protection.
    Returns the exact raw SSE chunk the server originally sent.

    Automatically handles decompression based on encoding ID prefix.
    """

    session: StreamingSession
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

    def decrypt(self, sse_data: str) -> bytes:
        """
        Decrypt an SSE data field to recover the original chunk.

        Args:
            sse_data: base64url-encoded payload from SSE data field

        Returns:
            Original raw SSE chunk as bytes exactly as server sent it

        Raises:
            ReplayAttackError: If counter is out of order
            DecryptionError: If decryption fails or unknown encoding
        """
        # Decode payload
        try:
            payload = b64url_decode(sse_data)
        except Exception as e:
            raise DecryptionError("Invalid base64url encoding") from e

        if len(payload) < SSE_COUNTER_SIZE + 16:  # Counter + minimum ciphertext (tag only)
            raise DecryptionError("Payload too short")

        # Extract counter and ciphertext (zero-copy slicing via memoryview)
        counter = int.from_bytes(payload[:SSE_COUNTER_SIZE], "big")
        ciphertext = payload[SSE_COUNTER_SIZE:]

        # Validate counter monotonicity
        if counter != self.expected_counter:
            raise ReplayAttackError(self.expected_counter, counter)

        # Decrypt
        nonce = self._compute_nonce(counter)
        try:
            data = self._cipher.decrypt(nonce, ciphertext, associated_data=None)
        except Exception as e:
            raise DecryptionError("SSE decryption failed") from e

        # Parse encoding_id and decompress if needed
        if len(data) < 1:
            raise DecryptionError("Decrypted payload too short (missing encoding ID)")

        encoding_id = data[0]
        encoded_payload = data[1:]

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

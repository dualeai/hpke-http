"""
RFC 9180 HPKE constants and algorithm identifiers.

Cipher Suite:
- KEM: DHKEM(X25519, HKDF-SHA256) - 0x0020
- KDF: HKDF-SHA256 - 0x0001
- AEAD: ChaCha20-Poly1305 - 0x0003
- Mode: PSK - 0x01

References:
- RFC 9180 §7.1 (KEM IDs)
- RFC 9180 §7.2 (KDF IDs)
- RFC 9180 §7.3 (AEAD IDs)
- RFC 9180 §5.1.2 (PSK mode)
"""

from enum import IntEnum
from typing import Final

# =============================================================================
# HPKE Version
# =============================================================================

VERSION: Final[bytes] = b"HPKE-v1"
"""HPKE version string used in labeled operations (RFC 9180 §4)."""

# =============================================================================
# KEM Identifiers (RFC 9180 §7.1)
# =============================================================================


class KemId(IntEnum):
    """Key Encapsulation Mechanism identifiers."""

    DHKEM_X25519_HKDF_SHA256 = 0x0020
    """DHKEM(X25519, HKDF-SHA256) - RFC 9180 recommended."""

    # Future: X-Wing (X25519 + ML-KEM-768) - draft-connolly-cfrg-xwing-kem
    # X_WING = 0x647A


# Default KEM for this implementation
KEM_ID: Final[int] = KemId.DHKEM_X25519_HKDF_SHA256

# X25519 key sizes
X25519_PUBLIC_KEY_SIZE: Final[int] = 32
X25519_PRIVATE_KEY_SIZE: Final[int] = 32
X25519_SHARED_SECRET_SIZE: Final[int] = 32
X25519_ENC_SIZE: Final[int] = 32  # Encapsulated key size

# =============================================================================
# KDF Identifiers (RFC 9180 §7.2)
# =============================================================================


class KdfId(IntEnum):
    """Key Derivation Function identifiers."""

    HKDF_SHA256 = 0x0001
    """HKDF-SHA256 - RFC 5869."""


# Default KDF for this implementation
KDF_ID: Final[int] = KdfId.HKDF_SHA256

# HKDF-SHA256 parameters
HKDF_SHA256_HASH_SIZE: Final[int] = 32
HKDF_SHA256_N_SECRET: Final[int] = 32  # Output size of Extract
HKDF_SHA256_N_H: Final[int] = 32  # Hash output size

# =============================================================================
# AEAD Identifiers (RFC 9180 §7.3)
# =============================================================================


class AeadId(IntEnum):
    """Authenticated Encryption with Associated Data identifiers."""

    AES_128_GCM = 0x0001
    AES_256_GCM = 0x0002
    CHACHA20_POLY1305 = 0x0003
    """ChaCha20-Poly1305 - RFC 8439."""


# Default AEAD for this implementation
AEAD_ID: Final[int] = AeadId.CHACHA20_POLY1305

# ChaCha20-Poly1305 parameters (RFC 8439)
CHACHA20_POLY1305_KEY_SIZE: Final[int] = 32
CHACHA20_POLY1305_NONCE_SIZE: Final[int] = 12
CHACHA20_POLY1305_TAG_SIZE: Final[int] = 16

# =============================================================================
# HPKE Mode Identifiers (RFC 9180 §5)
# =============================================================================


class HpkeMode(IntEnum):
    """HPKE operation modes."""

    BASE = 0x00
    """Base mode - no additional keying material."""
    PSK = 0x01
    """PSK mode - Pre-Shared Key for additional authentication."""
    AUTH = 0x02
    """Auth mode - Sender authentication via static key."""
    AUTH_PSK = 0x03
    """AuthPSK mode - Both sender auth and PSK."""


# Default mode for this implementation
MODE_PSK: Final[int] = HpkeMode.PSK

# PSK minimum size for cryptographic security (256 bits)
PSK_MIN_SIZE: Final[int] = 32

# =============================================================================
# Suite ID Construction (RFC 9180 §5.1)
# =============================================================================


def build_suite_id(kem_id: int = KEM_ID, kdf_id: int = KDF_ID, aead_id: int = AEAD_ID) -> bytes:
    """
    Build the suite_id for HPKE operations.

    suite_id = "HPKE" || I2OSP(kem_id, 2) || I2OSP(kdf_id, 2) || I2OSP(aead_id, 2)

    Args:
        kem_id: KEM algorithm identifier
        kdf_id: KDF algorithm identifier
        aead_id: AEAD algorithm identifier

    Returns:
        10-byte suite_id
    """
    return b"HPKE" + kem_id.to_bytes(2, "big") + kdf_id.to_bytes(2, "big") + aead_id.to_bytes(2, "big")


# Default suite ID for DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, ChaCha20-Poly1305
SUITE_ID: Final[bytes] = build_suite_id()

# =============================================================================
# HTTP Header Names
# =============================================================================

HEADER_HPKE_ENC: Final[str] = "X-HPKE-Enc"
"""Header containing base64url-encoded encapsulated key."""

HEADER_HPKE_STREAM: Final[str] = "X-HPKE-Stream"
"""Header containing encrypted SSE session parameters."""

HEADER_HPKE_ENCODING: Final[str] = "X-HPKE-Encoding"
"""Header specifying compression algorithm for request body (RFC 8878)."""

RESPONSE_KEY_LABEL: Final[bytes] = b"response-key"
"""Export label for deriving response encryption key from HPKE context."""

REQUEST_KEY_LABEL: Final[bytes] = b"request-stream-key"
"""Export label for deriving request encryption key from HPKE context."""

CHUNK_SIZE: Final[int] = 64 * 1024  # 64KB
"""Chunk size for streaming encryption (age/libsodium standard)."""

# =============================================================================
# ASGI Scope Keys
# =============================================================================

SCOPE_HPKE_CONTEXT: Final[str] = "hpke_context"
"""ASGI scope key for storing HPKE recipient context after request decryption."""

# =============================================================================
# SSE Streaming Constants
# =============================================================================

SSE_SESSION_SALT_SIZE: Final[int] = 4
"""Random salt size for SSE session nonces (4 bytes)."""

SSE_COUNTER_SIZE: Final[int] = 4
"""Counter size in wire format (4 bytes, big-endian)."""

RAW_LENGTH_PREFIX_SIZE: Final[int] = 4
"""Length prefix size in RawFormat wire encoding (4 bytes, big-endian).

Used for O(1) chunk boundary detection when parsing concatenated chunks.
"""

SSE_MAX_COUNTER: Final[int] = 2**32 - 1
"""Maximum counter value (4 billion events per session)."""

SSE_MAX_EVENT_SIZE: Final[int] = 64 * 1024 * 1024
"""Default maximum buffered SSE event size (64MB).

This is a DoS protection limit for incomplete events without proper \\n\\n boundaries.
Can be overridden via HPKEMiddleware's max_sse_event_size parameter.

Note: SSE is text-only (UTF-8). Binary data (images, documents) must be
base64-encoded, which adds ~33% overhead. A 48MB file becomes ~64MB in base64.
"""

# Export label for deriving SSE session key from HPKE context
SSE_SESSION_KEY_LABEL: Final[bytes] = b"sse-session-key"

# =============================================================================
# Key Discovery
# =============================================================================

DISCOVERY_PATH: Final[str] = "/.well-known/hpke-keys"
"""Path for HPKE key discovery endpoint."""

DISCOVERY_CACHE_MAX_AGE: Final[int] = 86400
"""Default cache max-age for discovery response (24 hours)."""

# =============================================================================
# Compression Constants (RFC 8878 - Zstandard)
# =============================================================================


class SSEEncodingId(IntEnum):
    """Encoding algorithm identifiers for SSE payloads.

    First byte of decrypted SSE payload indicates compression algorithm.
    Extensible for future algorithms (brotli, etc.).
    """

    IDENTITY = 0x00
    """No compression - raw plaintext."""
    ZSTD = 0x01
    """Zstandard compression (RFC 8878)."""
    # Reserved for future:
    # BROTLI = 0x02  # Brotli (RFC 7932)


ZSTD_COMPRESSION_LEVEL: Final[int] = 3
"""Zstd compression level (1-22). Level 3 = fast compression."""

ZSTD_MIN_SIZE: Final[int] = 64
"""Minimum payload size for compression. Smaller payloads skip compression."""

ZSTD_STREAMING_THRESHOLD: Final[int] = 1024 * 1024  # 1MB
"""Threshold for using streaming compression vs in-memory.

Payloads >= this size use streaming compression with constant memory (~4MB).
Smaller payloads use in-memory compression for better performance.
"""

ZSTD_STREAMING_CHUNK_SIZE: Final[int] = 4 * 1024 * 1024  # 4MB
"""Chunk size for streaming compression write operations.

Determines how much data is written to ZstdFile at once.
Affects peak memory usage during compression.
"""

ZSTD_DECOMPRESS_STREAMING_THRESHOLD: Final[int] = 1024  # 1KB
"""Threshold for using streaming decompression in request handling.

For compressed request bodies, use streaming decompression when compressed
size >= this threshold. This is a RAM/CPU tradeoff:

- RAM: Streaming uses ~1.3x decompressed size vs ~2x for list+join.
       For 50MB: saves ~35MB per request.
- CPU: Streaming adds ~40% overhead (10ms -> 15ms for 50MB).

Set low (1KB) because compressed JSON often expands 100-1000x, so even
tiny compressed payloads can decompress to large sizes where RAM savings
outweigh CPU cost.
"""

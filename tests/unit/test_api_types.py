"""API type contract tests.

These tests verify that public APIs maintain their type signatures.
Uses typing_extensions.assert_type for STATIC type checking by pyright.

If someone changes a return type, pyright will fail BEFORE tests run.
This prevents the anti-pattern of changing tests to match broken code.

Wire format v2 widens per-chunk return type to bytes | bytearray (the
combined-buffer encrypt_into path returns bytes for RawFormat and SSEFormat,
but the decryptor returns bytearray for IDENTITY path and bytes for
compressed paths).
"""

from typing_extensions import assert_type

from hpke_http.streaming import ChunkDecryptor, ChunkEncryptor, StreamingSession


class TestChunkEncryptorTypes:
    """Verify ChunkEncryptor type contracts."""

    def test_encrypt_returns_bytes(self) -> None:
        """ChunkEncryptor.encrypt returns bytes (both formats coerce to bytes for ASGI emit)."""
        session = StreamingSession(session_key=b"k" * 32, session_salt=b"salt")
        encryptor = ChunkEncryptor(session)

        result = encryptor.encrypt(b"event: test\n\n")

        # Static assertion - pyright validates at type-check time
        assert_type(result, bytes)
        # Runtime assertion - pytest validates at test time
        assert isinstance(result, bytes)

    def test_encrypt_accepts_bytes(self) -> None:
        """ChunkEncryptor.encrypt must accept bytes input."""
        session = StreamingSession(session_key=b"k" * 32, session_salt=b"salt")
        encryptor = ChunkEncryptor(session)

        chunk: bytes = b"event: test\n\n"
        result = encryptor.encrypt(chunk)
        assert_type(result, bytes)

    def test_encrypt_accepts_bytearray(self) -> None:
        """ChunkEncryptor.encrypt accepts bytearray input."""
        session = StreamingSession(session_key=b"k" * 32, session_salt=b"salt")
        encryptor = ChunkEncryptor(session)

        chunk: bytearray = bytearray(b"event: test\n\n")
        result = encryptor.encrypt(chunk)
        assert_type(result, bytes)

    def test_encrypt_accepts_memoryview(self) -> None:
        """ChunkEncryptor.encrypt accepts memoryview input (zero-copy slicing)."""
        session = StreamingSession(session_key=b"k" * 32, session_salt=b"salt")
        encryptor = ChunkEncryptor(session)

        chunk: memoryview = memoryview(b"event: test\n\n")
        result = encryptor.encrypt(chunk)
        assert_type(result, bytes)


class TestChunkDecryptorTypes:
    """Verify ChunkDecryptor type contracts."""

    def test_decrypt_returns_bytes_or_bytearray(self) -> None:
        """ChunkDecryptor.decrypt returns bytes | bytearray."""
        session = StreamingSession(session_key=b"k" * 32, session_salt=b"salt")
        encryptor = ChunkEncryptor(session)
        decryptor = ChunkDecryptor(session)

        # Create valid encrypted data
        encrypted = encryptor.encrypt(b"event: test\n\n")
        data_field = bytes(encrypted).decode("ascii").split("\n")[1][6:]  # Extract data: field

        result = decryptor.decrypt(data_field)

        # Static assertion - pyright validates at type-check time
        assert_type(result, bytes | bytearray)
        # Runtime assertion - pytest validates at test time
        assert isinstance(result, (bytes, bytearray))

    def test_decrypt_accepts_str(self) -> None:
        """ChunkDecryptor.decrypt accepts str input (base64-encoded)."""
        session = StreamingSession(session_key=b"k" * 32, session_salt=b"salt")
        encryptor = ChunkEncryptor(session)
        decryptor = ChunkDecryptor(session)

        encrypted = encryptor.encrypt(b"test")
        data_field: str = bytes(encrypted).decode("ascii").split("\n")[1][6:]

        result = decryptor.decrypt(data_field)
        assert_type(result, bytes | bytearray)
